"""SemanticDedup — near-duplicate content block elimination FusionStage.

Detects and eliminates repeated content blocks within a single text using
3-word shingle fingerprinting (no external dependencies).  Blocks with
Jaccard similarity > 0.8 are considered near-duplicates; only the first
occurrence is kept, later ones are replaced with a compact reference.

Also exposes ``dedup_across_messages`` for cross-message deduplication in
a chat message list.

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from claw_compactor.fusion.base import FusionContext, FusionResult, FusionStage
from claw_compactor.tokens import estimate_tokens

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum block length (chars) to be considered for deduplication.
_MIN_BLOCK_CHARS = 50

# Jaccard similarity threshold above which two blocks are "near-duplicate".
_SIM_THRESHOLD = 0.80

# Shingle size (number of consecutive words).
_SHINGLE_N = 3

# Minimum shingle set size; blocks with fewer shingles are not fingerprinted.
_MIN_SHINGLES = 2

# Template used to replace a duplicate block in-place.
_REF_TEMPLATE = "[duplicate of block {n} — omitted]"

# Template used for cross-message references.
_MSG_REF_TEMPLATE = "[content similar to message {idx} — omitted]"


# ---------------------------------------------------------------------------
# Fingerprinting helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Split text into lowercase word tokens (letters + digits only)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _shingles(tokens: list[str], n: int = _SHINGLE_N) -> frozenset[tuple[str, ...]]:
    """Return the set of n-gram shingles from *tokens*."""
    if len(tokens) < n:
        return frozenset()
    return frozenset(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Return the Jaccard similarity of two sets."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------

@dataclass
class _Block:
    """A single logical block extracted from the source text."""
    text: str
    # Offsets into the *original* text for reconstruction.
    start: int
    end: int
    is_code: bool = False
    shingles: frozenset = field(default_factory=frozenset)
    kept: bool = True
    ref_to: int | None = None  # 1-based index of the first occurrence


_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _split_blocks(text: str) -> list[_Block]:
    """
    Split *text* into logical blocks.

    Rules (applied in order):
    1. Fenced code blocks (``` ... ```) are treated as atomic units.
    2. All remaining text is split on blank lines (one or more empty lines).
    """
    blocks: list[_Block] = []
    # Find code fence spans so we can protect them.
    fence_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in _CODE_FENCE_RE.finditer(text)
    ]

    def _in_fence(start: int, end: int) -> bool:
        return any(fs <= start and end <= fe for fs, fe in fence_spans)

    # Add fenced code blocks as atomic blocks first.
    for fs, fe in fence_spans:
        block_text = text[fs:fe]
        sh = _shingles(_tokenise(block_text))
        blocks.append(_Block(
            text=block_text,
            start=fs,
            end=fe,
            is_code=True,
            shingles=sh,
        ))

    # Build a set of positions covered by fences.
    fence_positions: set[int] = set()
    for fs, fe in fence_spans:
        fence_positions.update(range(fs, fe))

    # Split the non-fence remainder on blank lines.
    # We iterate over the text, collecting runs of non-fence characters.
    # Then split those runs by blank-line boundaries.
    non_fence_segments: list[tuple[int, str]] = []
    i = 0
    while i < len(text):
        if i in fence_positions:
            i += 1
            continue
        seg_start = i
        buf: list[str] = []
        while i < len(text) and i not in fence_positions:
            buf.append(text[i])
            i += 1
        segment = "".join(buf)
        if segment.strip():
            non_fence_segments.append((seg_start, segment))

    for seg_start, segment in non_fence_segments:
        # Split by blank lines (2+ newlines or line with only whitespace).
        para_re = re.compile(r"\n\s*\n")
        last = 0
        for m in para_re.finditer(segment):
            chunk = segment[last : m.start()]
            if chunk.strip():
                abs_start = seg_start + last
                abs_end = seg_start + m.start()
                sh = _shingles(_tokenise(chunk))
                blocks.append(_Block(
                    text=chunk,
                    start=abs_start,
                    end=abs_end,
                    is_code=False,
                    shingles=sh,
                ))
            last = m.end()
        # Trailing chunk after last separator.
        chunk = segment[last:]
        if chunk.strip():
            abs_start = seg_start + last
            abs_end = seg_start + len(segment)
            sh = _shingles(_tokenise(chunk))
            blocks.append(_Block(
                text=chunk,
                start=abs_start,
                end=abs_end,
                is_code=False,
                shingles=sh,
            ))

    # Sort by position in original text.
    blocks.sort(key=lambda b: b.start)
    return blocks


# ---------------------------------------------------------------------------
# Core dedup logic
# ---------------------------------------------------------------------------

@dataclass
class DedupStats:
    """Statistics returned from a dedup run."""
    blocks_total: int = 0
    blocks_kept: int = 0
    blocks_deduped: int = 0
    chars_removed: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def blocks_skipped_too_short(self) -> int:
        return self.blocks_total - self.blocks_kept - self.blocks_deduped

    def as_dict(self) -> dict:
        return {
            "blocks_total": self.blocks_total,
            "blocks_kept": self.blocks_kept,
            "blocks_deduped": self.blocks_deduped,
            "chars_removed": self.chars_removed,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
        }


def _run_dedup(text: str) -> tuple[str, DedupStats]:
    """
    Run within-text block deduplication.

    Returns the rewritten text and statistics.
    """
    stats = DedupStats(tokens_before=estimate_tokens(text))

    blocks = _split_blocks(text)
    stats.blocks_total = len(blocks)

    if not blocks:
        stats.tokens_after = stats.tokens_before
        return text, stats

    # Assign 1-based sequential numbers for use in references.
    # We'll use the position in the sorted block list as the "block number".
    # Blocks that are too short to consider receive no shingle set.

    # First pass: mark duplicates.
    # kept_blocks: list of (block_number, shingles) for blocks we are keeping.
    kept_blocks: list[tuple[int, frozenset]] = []

    for idx, block in enumerate(blocks):
        block_num = idx + 1  # 1-based
        short = len(block.text.strip()) < _MIN_BLOCK_CHARS
        no_shingles = len(block.shingles) < _MIN_SHINGLES

        if short or no_shingles:
            # Too short / no shingles — always keep, never dedup.
            block.kept = True
            block.ref_to = None
            continue

        # Compare against all previously kept blocks.
        duplicate_of: int | None = None
        for prev_num, prev_sh in kept_blocks:
            sim = _jaccard(block.shingles, prev_sh)
            if sim >= _SIM_THRESHOLD:
                duplicate_of = prev_num
                break

        if duplicate_of is not None:
            block.kept = False
            block.ref_to = duplicate_of
        else:
            block.kept = True
            kept_blocks.append((block_num, block.shingles))

    # Second pass: reconstruct the text.
    # We rebuild from the original text, replacing duplicate block spans with
    # compact references.  Because blocks may not cover the full text (gaps
    # between them contain separators / fences), we work by scanning through
    # the original text character by character.

    result_parts: list[str] = []
    pos = 0
    blocks_kept = 0
    blocks_deduped = 0
    chars_removed = 0

    for block in blocks:
        # Append any gap before this block.
        if block.start > pos:
            result_parts.append(text[pos : block.start])
        pos = block.end

        if block.kept:
            result_parts.append(block.text)
            blocks_kept += 1
        else:
            ref = _REF_TEMPLATE.format(n=block.ref_to)
            result_parts.append(ref)
            chars_removed += len(block.text) - len(ref)
            blocks_deduped += 1

    # Append any trailing text after the last block.
    if pos < len(text):
        result_parts.append(text[pos:])

    output = "".join(result_parts)

    stats.blocks_kept = blocks_kept
    stats.blocks_deduped = blocks_deduped
    stats.chars_removed = max(0, chars_removed)
    stats.tokens_after = estimate_tokens(output)

    return output, stats


# ---------------------------------------------------------------------------
# FusionStage
# ---------------------------------------------------------------------------

class SemanticDedup(FusionStage):
    """Near-duplicate content block eliminator.

    Splits text into blocks (paragraphs + fenced code blocks), fingerprints
    each with 3-word shingles, and replaces near-duplicate blocks
    (Jaccard >= 0.8) with compact back-references.
    """

    name = "semantic_dedup"
    order = 12  # After Cortex(5), after any RLE-style stages(10), before Ionizer(15)

    def should_apply(self, ctx: FusionContext) -> bool:
        """Apply to any content longer than 200 characters."""
        return len(ctx.content) > 200

    def apply(self, ctx: FusionContext) -> FusionResult:
        original_tokens = estimate_tokens(ctx.content)
        output, stats = _run_dedup(ctx.content)
        compressed_tokens = estimate_tokens(output)

        markers: list[str] = []
        if stats.blocks_deduped > 0:
            markers.append(
                f"semantic_dedup:{stats.blocks_deduped}_blocks_removed"
                f":{stats.tokens_before}->{compressed_tokens}_tokens"
            )

        return FusionResult(
            content=output,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            markers=markers,
        )


# ---------------------------------------------------------------------------
# Cross-message deduplication
# ---------------------------------------------------------------------------

def dedup_across_messages(
    messages: list[dict],
) -> tuple[list[dict], dict]:
    """Deduplicate repeated content across multiple chat messages.

    If message B's content is >80% similar to a prior message A's content,
    B's content is replaced with a compact reference to A.

    Only processes messages whose ``content`` value is a non-empty string.
    Messages with list-valued content (multi-part) are passed through
    unchanged.

    Args:
        messages: List of message dicts, each with at least a ``"content"``
            key (and typically a ``"role"`` key).

    Returns:
        A 2-tuple of (deduped_messages, stats).
        ``deduped_messages`` is a new list — the originals are not mutated.
        ``stats`` is a plain dict with keys:
            - ``messages_total``
            - ``messages_deduped``
            - ``tokens_before``
            - ``tokens_after``
    """
    if not messages:
        return [], {
            "messages_total": 0,
            "messages_deduped": 0,
            "tokens_before": 0,
            "tokens_after": 0,
        }

    tokens_before = sum(
        estimate_tokens(m["content"])
        for m in messages
        if isinstance(m.get("content"), str)
    )

    # Build fingerprints for messages that are eligible for comparison.
    # A message is eligible when its content is a non-empty string and has
    # enough shingles.
    kept: list[tuple[int, frozenset]] = []  # (0-based index, shingles)
    deduped_messages: list[dict] = []
    deduped_count = 0

    for idx, msg in enumerate(messages):
        content = msg.get("content")

        # Non-string or empty content — pass through unchanged.
        if not isinstance(content, str) or not content.strip():
            deduped_messages.append(dict(msg))
            continue

        sh = _shingles(_tokenise(content))
        too_short = len(content.strip()) < _MIN_BLOCK_CHARS
        no_shingles = len(sh) < _MIN_SHINGLES

        if too_short or no_shingles:
            deduped_messages.append(dict(msg))
            kept.append((idx, sh))
            continue

        # Compare against all previously kept messages.
        duplicate_of: int | None = None
        for prev_idx, prev_sh in kept:
            sim = _jaccard(sh, prev_sh)
            if sim >= _SIM_THRESHOLD:
                duplicate_of = prev_idx
                break

        if duplicate_of is not None:
            new_msg = dict(msg)
            new_msg["content"] = _MSG_REF_TEMPLATE.format(idx=duplicate_of)
            deduped_messages.append(new_msg)
            deduped_count += 1
        else:
            deduped_messages.append(dict(msg))
            kept.append((idx, sh))

    tokens_after = sum(
        estimate_tokens(m["content"])
        for m in deduped_messages
        if isinstance(m.get("content"), str)
    )

    stats = {
        "messages_total": len(messages),
        "messages_deduped": deduped_count,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
    }
    return deduped_messages, stats
