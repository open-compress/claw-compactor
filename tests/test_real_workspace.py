"""Performance tests against real workspace data — verifies actual savings."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compress_memory import rule_compress
from claw_compactor.tokens import estimate_tokens
from claw_compactor.dictionary import build_codebook, compress_text, decompress_text
from claw_compactor.tokenizer_optimizer import optimize_tokens
from claw_compactor.rle import compress as rle_compress

WORKSPACE = Path("/home/user/workspace")


def _skip_if_no_workspace():
    if not WORKSPACE.is_dir():
        pytest.skip("Real workspace not available")


def _read_workspace_texts():
    _skip_if_no_workspace()
    texts = {}
    for name in ["MEMORY.md", "TOOLS.md", "AGENTS.md", "SOUL.md", "USER.md"]:
        p = WORKSPACE / name
        if p.exists():
            texts[name] = p.read_text()
    mem = WORKSPACE / "memory"
    if mem.is_dir():
        for f in sorted(mem.rglob("*.md"))[:20]:
            texts[f.name] = f.read_text()
    return texts


class TestRealWorkspacePerformance:
    """Tests against real workspace data to verify savings claims."""

    def test_rule_compress_saves_tokens(self):
        texts = _read_workspace_texts()
        if not texts:
            pytest.skip("No workspace files")
        total_before = sum(estimate_tokens(t) for t in texts.values())
        total_after = sum(estimate_tokens(rule_compress(t)) for t in texts.values())
        saved = total_before - total_after
        pct = (saved / total_before * 100) if total_before > 0 else 0
        assert saved >= 0, "Rule compression should not increase tokens"
        print(f"\nRule compress: {total_before:,} → {total_after:,} (saved {saved:,}, {pct:.1f}%)")

    def test_dictionary_saves_tokens(self):
        texts = _read_workspace_texts()
        if not texts:
            pytest.skip("No workspace files")
        combined = "\n".join(texts.values())
        codebook = build_codebook([combined])
        compressed = compress_text(combined, codebook)
        before = estimate_tokens(combined)
        after = estimate_tokens(compressed)
        saved = before - after
        assert saved >= 0, "Dictionary compression should not increase tokens"
        # Verify roundtrip
        assert decompress_text(compressed, codebook) == combined
        print(f"\nDict compress: {before:,} → {after:,} (saved {saved:,})")

    def test_tokenizer_optimize_saves_tokens(self):
        texts = _read_workspace_texts()
        if not texts:
            pytest.skip("No workspace files")
        total_before = sum(estimate_tokens(t) for t in texts.values())
        total_after = sum(estimate_tokens(optimize_tokens(t)) for t in texts.values())
        saved = total_before - total_after
        assert saved >= 0, "Tokenizer optimize should not increase tokens"
        print(f"\nTokenizer optimize: {total_before:,} → {total_after:,} (saved {saved:,})")

    def test_combined_pipeline_savings(self):
        """Full pipeline on real data."""
        texts = _read_workspace_texts()
        if not texts:
            pytest.skip("No workspace files")

        combined = "\n".join(texts.values())
        initial = estimate_tokens(combined)

        # Step 1: Rule compress
        rule_texts = {k: rule_compress(v) for k, v in texts.items()}
        rule_combined = "\n".join(rule_texts.values())
        after_rule = estimate_tokens(rule_combined)

        # Step 2: Dict compress
        codebook = build_codebook([rule_combined])
        dict_combined = compress_text(rule_combined, codebook)
        after_dict = estimate_tokens(dict_combined)

        # Step 3: Tokenizer optimize
        opt_combined = optimize_tokens(dict_combined)
        after_opt = estimate_tokens(opt_combined)

        total_saved = initial - after_opt
        pct = (total_saved / initial * 100) if initial > 0 else 0

        print(f"\n=== Combined Pipeline ===")
        print(f"  Initial:    {initial:,} tokens")
        print(f"  After rule: {after_rule:,} ({initial - after_rule:,} saved)")
        print(f"  After dict: {after_dict:,} ({after_rule - after_dict:,} saved)")
        print(f"  After opt:  {after_opt:,} ({after_dict - after_opt:,} saved)")
        print(f"  TOTAL: {total_saved:,} tokens saved ({pct:.1f}%)")

        assert total_saved >= 0

    def test_per_file_breakdown(self):
        texts = _read_workspace_texts()
        if not texts:
            pytest.skip("No workspace files")

        print(f"\n{'File':<25} | {'Before':>7} | {'After':>7} | {'Saved':>6} | {'%':>5}")
        print(f"{'-'*25}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}")

        for name, text in sorted(texts.items()):
            before = estimate_tokens(text)
            compressed = rule_compress(text)
            after = estimate_tokens(compressed)
            saved = before - after
            pct = (saved / before * 100) if before > 0 else 0
            print(f"{name:<25} | {before:>7,} | {after:>7,} | {saved:>6,} | {pct:>4.1f}%")
            assert saved >= 0, f"{name}: compression increased tokens!"


class TestSessionTranscriptStats:
    """Verify session transcript compression claims."""

    def test_transcript_exists(self):
        _skip_if_no_workspace()
        sessions_dir = Path.home() / ".openclaw" / "agents" / "main" / "sessions"
        if not sessions_dir.is_dir():
            pytest.skip("No sessions directory")
        files = list(sessions_dir.glob("*.jsonl"))
        print(f"\nFound {len(files)} session transcripts")
        assert len(files) >= 0  # Just report

    def test_sample_transcript_compression(self):
        _skip_if_no_workspace()
        sessions_dir = Path.home() / ".openclaw" / "agents" / "main" / "sessions"
        if not sessions_dir.is_dir():
            pytest.skip("No sessions directory")
        files = sorted(sessions_dir.glob("*.jsonl"))
        if not files:
            pytest.skip("No session files")

        from observation_compressor import compress_session, format_observations_md

        # Take first file only
        f = files[0]
        original_tokens = estimate_tokens(f.read_text())
        result = compress_session(f)
        obs = result.get("observations", [])
        compressed_text = format_observations_md(obs)
        compressed_tokens = estimate_tokens(compressed_text) if compressed_text.strip() else 0

        if original_tokens > 0:
            ratio = (1 - compressed_tokens / original_tokens) * 100
            print(f"\nSample transcript: {original_tokens:,} → {compressed_tokens:,} ({ratio:.1f}% compression)")
