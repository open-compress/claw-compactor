# Claw Compactor — Architecture

## Overview

Claw Compactor is a 14-stage Fusion Pipeline for LLM token compression. It operates on raw text or structured chat messages, applies a sequence of specialized compression stages, and produces output that is semantically equivalent but substantially smaller — achieving a weighted average of **53.9% token reduction** across real-world workloads.

The pipeline is content-aware. An early detection stage (Cortex) identifies the content type and programming language before any compression occurs, ensuring that subsequent stages fire only when appropriate. Each stage is independent, stateless with respect to the pipeline, and operates on an immutable context object. The output of one stage becomes the input of the next.

The system also supports **reversible compression** via the Rewind engine: content that cannot be losslessly inferred from context (e.g., large JSON arrays) is stored in a hash-addressed store, and the LLM can retrieve the original via a tool call. This makes aggressive compression safe in agentic settings.

---

## ASCII Pipeline Flow

```
  Input text / messages
         |
         v
  +--------------+
  |  FusionEngine |  compress() / compress_messages()
  +--------------+
         |
         v
  +--------------------+
  | Cross-msg SemanticDedup |  (messages only — dedup across turns before per-msg compression)
  +--------------------+
         |
         v
  +==================================================+
  |              F U S I O N   P I P E L I N E       |
  |                                                  |
  |  [1] QuantumLock      order=3   KV-cache align   |
  |   |                                              |
  |  [2] Cortex           order=5   Type/lang detect |
  |   |                                              |
  |  [3] Photon           order=8   Image/base64     |
  |   |                                              |
  |  [4] RLE              order=10  Path/IP/enum     |
  |   |                                              |
  |  [5] SemanticDedup    order=12  Simhash dedup    |
  |   |                                              |
  |  [6] Ionizer          order=15  JSON sampling    |
  |   |                                              |
  |  [7] LogCrunch        order=16  Log folding      |
  |   |                                              |
  |  [8] SearchCrunch     order=17  Search dedup     |
  |   |                                              |
  |  [9] DiffCrunch       order=18  Diff folding     |
  |   |                                              |
  | [10] StructuralCollapse order=20 Import collapse |
  |   |                                              |
  | [11] Neurosyntax      order=25  AST compression  |
  |   |                                              |
  | [12] Nexus            order=35  ML token-level   |
  |   |                                              |
  | [13] TokenOpt         order=40  Format cleanup   |
  |   |                                              |
  | [14] AbbrevStage      order=45  NL abbreviation  |
  +==================================================+
         |
         v
  Compressed output + RewindStore (hash markers for reversible content)
```

---

## Design Philosophy

### Immutable Data Flow

Every object that crosses a stage boundary is frozen. `FusionContext` is a frozen dataclass carrying the current text, detected content type, detected language, and accumulated metadata. `FusionResult` is a frozen dataclass carrying the output text, bytes saved, and a flag indicating whether the stage fired. Neither is mutated in place — each stage receives a context, returns a result, and the pipeline constructs the next context from that result.

This eliminates an entire class of bugs: no stage can corrupt the state seen by a later stage, no side effect can propagate unexpectedly, and any stage's output can be inspected in isolation for debugging or testing.

### Stage Independence

Stages do not call each other. They do not share mutable state. A stage knows only what is in `FusionContext` and what its own constructor was given at initialization time. This means:

- Stages can be unit-tested by passing a constructed `FusionContext` directly.
- Stages can be reordered by changing their `order` value without modifying any other code.
- A stage can be disabled by removing it from the pipeline without affecting any other stage.
- New stages can be inserted at any `order` position without touching existing stages.

### Gate Before Compress

Every stage implements `should_apply(context: FusionContext) -> bool` before `apply()`. This gate is cheap — typically an `O(1)` check on the content type or a substring scan — and it prevents expensive compression logic from running on content where it would have no effect or could cause harm. For example, `AbbrevStage` gates on `content_type == "text"` and will never touch code. `DiffCrunch` gates on the presence of unified diff headers.

The gate-before-compress pattern also makes performance profiling straightforward: `timed_apply()` records both gate evaluation time and compression time separately.

### Compression Depth vs. Safety

Aggressive compression (Ionizer, Neurosyntax, Nexus) can discard information. Claw Compactor's safety boundary is:

1. **Lossless stages** (QuantumLock, Cortex, RLE, SemanticDedup, StructuralCollapse, TokenOpt, AbbrevStage) — output is fully recoverable from the compressed form.
2. **Lossy-with-rewind stages** (Ionizer, LogCrunch, SearchCrunch, DiffCrunch) — discarded content is stored in RewindStore; the LLM can retrieve it on demand.
3. **Semantic-preserving lossy stages** (Neurosyntax, Nexus) — structure is compressed but semantics are maintained; no rewind needed.

---

## Stage Reference

### [1] QuantumLock — order 3

**Purpose:** Prepare system messages for KV-cache alignment before any compression occurs.

**When it fires:** Always, on system-role messages or the first segment of a prompt.

**Mechanism:** Identifies the stable (cacheable) prefix of a system message and isolates the dynamic suffix — the part that changes per-request (e.g., injected tool schemas, current date, user-specific context). It inserts a cache boundary marker between the stable and dynamic sections. Downstream stages then treat the stable prefix as protected, preserving its token layout so that the KV-cache hit rate at inference time is maximized. Without this stage, even a single token change anywhere in the system prompt invalidates the entire cache.

---

### [2] Cortex — order 5

**Purpose:** Auto-detect content type and programming language so that all subsequent stages can gate correctly.

**When it fires:** Always — it is the classification backbone of the pipeline.

**Mechanism:** Applies a cascade of heuristics and pattern matchers. Content types detected: `code`, `json`, `log`, `diff`, `search`, `text`. Languages detected (16): Python, JavaScript, TypeScript, Java, C, C++, C#, Go, Rust, Ruby, PHP, Swift, Kotlin, Scala, Shell, SQL. Detection priority is: structural markers (unified diff headers, JSON root token, log timestamp patterns) > file extension hints in context metadata > lexical heuristics (keyword density, indent character). The detected type and language are written into `FusionContext` and are immutable for the remainder of the pipeline.

---

### [3] Photon — order 8

**Purpose:** Compress image data and base64-encoded binary blobs embedded in messages.

**When it fires:** When the content contains `data:image/` URIs or raw base64 blocks above a minimum size threshold.

**Mechanism:** For vision-model inputs, Photon re-encodes images at a lower quality level sufficient for the model's perception task. For base64 blobs that are not images (e.g., embedded files), it replaces the blob with a typed placeholder and stores the original in RewindStore. This stage alone can reduce message size by 40-70% when images are present.

---

### [4] RLE — order 10

**Purpose:** Compact highly repetitive structured tokens: file paths, IP addresses, enumerated constants.

**When it fires:** On `code`, `log`, and `json` content types.

**Mechanism:**
- **Path shorthand**: Repeated path prefixes are replaced with sigils (e.g., `/Users/duke_nukem_opcdbase/.openclaw/workspace` becomes `$WS`). A legend is prepended to the block.
- **IP prefix compaction**: Repeated octets in IP addresses within log blocks are replaced with positional references.
- **Enum compaction**: Sequences of lines sharing a common prefix/suffix pattern are collapsed into a run-length encoded form.

RLE is intentionally placed after Cortex (which provides type context) and before semantic stages (which assume the text has already been structurally reduced).

---

### [5] SemanticDedup — order 12

**Purpose:** Remove near-duplicate content blocks within a single message.

**When it fires:** On all content types when the message exceeds a minimum length threshold.

**Mechanism:** Computes a 64-bit simhash fingerprint for each paragraph or code block. Pairs with Hamming distance <= 3 are considered near-duplicates. The first occurrence is kept; subsequent occurrences are replaced with a back-reference token. This is distinct from cross-message dedup (handled before the pipeline at the `FusionEngine` level) — this stage handles within-message redundancy such as repeated boilerplate in multi-section documents.

---

### [6] Ionizer — order 15

**Purpose:** Compress large JSON arrays by statistical sampling and schema extraction.

**When it fires:** On `json` content type when an array contains more than a configurable number of elements (default: 20).

**Mechanism:** Performs schema discovery on the array elements (union of all keys, type inference per key). Retains a representative statistical sample of elements (configurable, default: 5). Stores the full array in RewindStore under a content hash. The compressed output contains: the inferred schema, the sample, summary statistics (count, numeric field distributions), and a Rewind retrieval marker. The LLM can request the full array via the Rewind tool call if it needs element-level detail.

---

### [7] LogCrunch — order 16

**Purpose:** Fold repetitive or low-information lines in build and test logs.

**When it fires:** On `log` content type.

**Mechanism:** Identifies line groups with identical or near-identical structure (same log level, same logger prefix, same message template with varying parameters). Consecutive duplicate or near-duplicate lines are folded into a single representative line with a repeat count annotation. Lines containing error, failure, or exception markers are always preserved verbatim regardless of repetition. Folded content is stored in RewindStore.

---

### [8] SearchCrunch — order 17

**Purpose:** Deduplicate and truncate search result sets.

**When it fires:** On `search` content type (structured search result blocks).

**Mechanism:** Identifies the title, URL, and snippet fields of each search result. Results with near-duplicate snippets (simhash, same threshold as SemanticDedup) are merged. Results beyond a configurable cutoff rank are summarized (title + URL only, no snippet). The original full result set is stored in RewindStore.

---

### [9] DiffCrunch — order 18

**Purpose:** Fold unchanged context lines in git diffs, retaining only the changed lines and minimal surrounding context.

**When it fires:** On `diff` content type (unified diff format).

**Mechanism:** Parses unified diff hunks. Within each hunk, runs of unchanged context lines exceeding a configurable window (default: 3 lines) are replaced with a fold marker showing the line count. Changed lines (`+`/`-`) and their immediate context window are always preserved. This can reduce large diffs by 60-80% when most of the diff is context. The full diff is not stored in RewindStore because the original is typically available from the VCS; a reference marker is emitted instead.

---

### [10] StructuralCollapse — order 20

**Purpose:** Collapse repetitive structural patterns in code: import blocks, assertion sequences, repeated boilerplate.

**When it fires:** On `code` content type.

**Mechanism:**
- **Import collapse**: Consecutive import/require/use statements are grouped and replaced with a compact multi-import form plus a count annotation.
- **Assertion collapse**: Sequences of structurally identical assertions (same function, varying arguments) are folded into a template with an argument list.
- **Repeated pattern collapse**: Any block of lines that matches a generalized template more than a configurable threshold times is collapsed into a template instantiation.

StructuralCollapse operates on the surface text without an AST. Neurosyntax (the next semantic stage) handles deeper structural compression with full parse tree access.

---

### [11] Neurosyntax — order 25

**Purpose:** AST-aware code compression — the deepest structural compression stage.

**When it fires:** On `code` content type when a supported language is detected.

**Mechanism:** Uses tree-sitter to parse the source into a concrete syntax tree. Applies a set of structure-preserving transformations:
- Dead code path elimination (unreachable branches after constant folding)
- Identifier shortening for local variables (not exported symbols)
- Whitespace and comment normalization
- Redundant type annotation removal in dynamically-typed languages

When tree-sitter parsing fails or the language is unsupported, Neurosyntax falls back to a set of safe regex-based transformations that cannot corrupt syntax: blank line normalization, trailing whitespace removal, and comment stripping (with a flag to preserve doc comments).

This stage is the primary driver of the 3.4x improvement over legacy regex on Python source.

---

### [12] Nexus — order 35

**Purpose:** ML token-level compression — removes low-information tokens across any content type.

**When it fires:** On all content types after structural compression has been applied.

**Mechanism:** Nexus operates at the token level rather than the character or line level. It applies a learned model (or, in fallback mode, a curated stopword list) to identify tokens that contribute minimal information given their surrounding context. Filler words, redundant determiners, and verbose connectives are removed or contracted. Nexus is content-type-aware: it applies conservative settings on `code` (only removing clearly safe tokens) and aggressive settings on `text` and `log`. The fallback stopword mode handles environments where the ML model is unavailable.

---

### [13] TokenOpt — order 40

**Purpose:** Final tokenizer-level format optimization — ensure the output tokenizes as efficiently as possible for the target model's BPE vocabulary.

**When it fires:** Always, as a final cleanup pass.

**Mechanism:**
- Removes bold/italic markdown decorators that add tokens without information value (e.g., `**word**` -> `word` in contexts where emphasis is not meaningful).
- Collapses markdown tables with redundant columns.
- Normalizes whitespace sequences to single spaces or single newlines, preserving indentation structure in code blocks.
- Removes Unicode whitespace variants and zero-width characters.

TokenOpt does not change meaning; it only changes surface form to improve tokenization efficiency.

---

### [14] AbbrevStage — order 45

**Purpose:** Natural language abbreviation — the final compression stage, applied only to prose.

**When it fires:** On `text` content type only. Never fires on code, JSON, logs, or diffs.

**Mechanism:** Applies a curated dictionary of safe natural language abbreviations: common phrases contracted to standard shortened forms, verbose constructions replaced with concise equivalents. All substitutions are reversible by a reader — no information is destroyed, only verbosity. The abbreviation dictionary is domain-aware: technical writing, academic writing, and conversational text have different abbreviation profiles.

---

## The Rewind System

Rewind is the mechanism by which Claw Compactor achieves aggressive compression without permanent information loss.

### Architecture

```
  Compression stage (e.g., Ionizer)
         |
         | -- stores original content -->  RewindStore
         |                                      |
         | -- emits compressed text             | hash-addressed
         |    with marker:                      | blob storage
         |    [[REWIND:sha256:abc123]]          |
         v                                      |
  Compressed output in context                  |
                                                |
  LLM tool call: rewind_retrieve("abc123") ---->+
                                                |
                                           returns original
```

### RewindStore

`RewindStore` is a hash-addressed in-memory (or optionally persistent) store. Keys are SHA-256 hashes of the original content. Values are the original uncompressed content. The store is append-only — content is never modified or deleted during a session.

When a stage stores content in RewindStore, it emits a marker token in the compressed output: `[[REWIND:sha256:<hash>]]`. This marker is compact (approximately 75 characters) and uniquely identifies the stored content.

### LLM Retrieval

The Rewind tool is exposed as a standard tool call in the OpenAI function-calling format. The LLM can invoke it when it needs the full content behind a marker:

```json
{
  "name": "rewind_retrieve",
  "parameters": {
    "hash": "abc123..."
  }
}
```

The proxy intercepts this tool call, looks up the hash in RewindStore, and injects the result as a tool response message. From the LLM's perspective, it requested data and received it — no special handling is required on the model side.

### Which Stages Use Rewind

| Stage | What is stored |
|---|---|
| Photon | Non-image base64 blobs |
| Ionizer | Full JSON arrays (beyond sample) |
| LogCrunch | Folded log line groups |
| SearchCrunch | Full search result snippets beyond cutoff |

---

## Cross-Message Semantic Deduplication

Before the per-message pipeline runs, `FusionEngine.compress_messages()` performs a cross-message deduplication pass across the entire conversation history.

### Motivation

In long agentic conversations, the same context frequently appears in multiple turns: tool call results repeated in assistant summaries, the same file content re-pasted across user messages, or repeated system-prompt fragments. Per-message compression cannot see this redundancy because it processes one message at a time.

### Mechanism

```
  messages = [msg_0, msg_1, msg_2, ..., msg_n]
       |
       v
  CrossMsgDedup:
    1. Compute simhash fingerprint for each paragraph/block in each message
    2. Build a global fingerprint index keyed by hash
    3. For each message (in order), for each block:
       - If fingerprint seen before: replace block with back-reference token
         [[DEDUP_REF:msg_idx:block_idx]]
       - Else: record fingerprint, keep block verbatim
       |
       v
  Deduplicated messages -> per-message FusionPipeline
```

Back-references are resolved by the proxy when assembling the final API request, so the model always receives coherent content. The deduplication is transparent to the model.

### Ordering Guarantee

Cross-message dedup runs before any per-message compression. This ensures that when a block is kept (first occurrence), the per-message pipeline can still compress it. When a block is replaced with a reference, no compression effort is wasted on it.

---

## Data Flow

### FusionContext

```
FusionContext (frozen)
  .text          : str       — current text at this pipeline position
  .content_type  : str       — detected by Cortex: code/json/log/diff/search/text
  .language      : str|None  — detected by Cortex: python/javascript/...
  .metadata      : dict      — arbitrary stage annotations (frozen copy-on-write)
  .original_len  : int       — byte length before any compression
  .rewind_store  : RewindStore — shared store for this pipeline run
```

### FusionResult

```
FusionResult (frozen)
  .text          : str       — output text from this stage
  .bytes_saved   : int       — reduction in byte length
  .stage_fired   : bool      — whether should_apply() returned True
  .stage_name    : str       — name of the stage that produced this result
  .elapsed_ms    : float     — wall time for gate + compress
```

### Pipeline Execution

```
  FusionContext(text=input, content_type=None, ...)
         |
         v
  stage_1.should_apply(ctx) -> True
  stage_1.apply(ctx) -> FusionResult(text=t1, ...)
         |
         | ctx = FusionContext(text=t1, content_type=ctx.content_type, ...)
         v
  stage_2.should_apply(ctx) -> False   [gate rejects]
  FusionResult(text=t1, stage_fired=False, bytes_saved=0)
         |
         | ctx unchanged (text=t1)
         v
  stage_3.should_apply(ctx) -> True
  stage_3.apply(ctx) -> FusionResult(text=t3, ...)
         |
         v
  ...
         |
         v
  Final FusionResult -> compressed output
```

Each stage receives a freshly constructed `FusionContext` derived from the previous result. The `content_type` and `language` fields are propagated unchanged (only Cortex writes them). The `metadata` dict accumulates annotations from all stages that fired.

---

## Extending the Pipeline

### Adding a Custom Stage

1. Create a new file in `scripts/lib/fusion/`.

2. Subclass `FusionStage`:

```python
from .base import FusionStage, FusionContext, FusionResult
import dataclasses

class MyStage(FusionStage):
    order = 22  # insert between StructuralCollapse (20) and Neurosyntax (25)
    name = "MyStage"

    def should_apply(self, ctx: FusionContext) -> bool:
        # Return False immediately if this stage is irrelevant.
        # Keep this cheap — it runs on every message.
        return ctx.content_type == "code" and "my_pattern" in ctx.text

    def apply(self, ctx: FusionContext) -> FusionResult:
        # ctx.text is immutable — build a new string.
        compressed = ctx.text.replace("verbose_pattern", "v_pat")
        return dataclasses.replace(
            FusionResult.empty(self.name),
            text=compressed,
            bytes_saved=len(ctx.text) - len(compressed),
            stage_fired=True,
        )
```

3. Register the stage in `FusionPipeline`:

```python
from .my_stage import MyStage

pipeline = FusionPipeline(stages=[
    QuantumLockStage(),
    CortexStage(),
    # ... existing stages ...
    MyStage(),          # pipeline sorts by .order automatically
    # ...
])
```

The pipeline sorts stages by `order` at construction time, so insertion order in the list does not matter.

### Guidelines for New Stages

- **Gate strictly.** A stage that fires on content where it has no effect wastes CPU and can interfere with accurate `bytes_saved` accounting.
- **Never mutate `ctx`.** Use `dataclasses.replace()` or construct a new object.
- **Store discarded content in RewindStore** if the stage is lossy. Use `ctx.rewind_store.put(content)` to get a hash, then embed `[[REWIND:sha256:<hash>]]` in the output.
- **Write tests first.** The test suite (`tests/`) has 1676 tests and a helper fixture `make_context(text, content_type, language)` for constructing `FusionContext` instances in unit tests.
- **Choose an `order` value** that reflects where the stage logically belongs:
  - 1-9: Pre-processing and classification
  - 10-19: Structural/token normalization
  - 20-29: Syntax-aware compression
  - 30-44: Semantic/ML compression
  - 45+: Final format polish

---

## Performance Characteristics

### Benchmark Results

The following measurements use real-world content. "Old (regex)" refers to the legacy single-pass regex compressor. Compression ratio is `(original - compressed) / original`.

| Scenario | Old (regex) | FusionEngine | Improvement |
|---|---|---|---|
| Python source | 7.3% | 25.0% | 3.4x |
| JSON (100 items) | 12.6% | 81.9% | 6.5x |
| Build logs | 5.5% | 24.1% | 4.4x |
| Agent conversation | 5.7% | 31.0% | 5.4x |
| Git diff | 6.2% | 15.0% | 2.4x |
| Search results | 5.3% | 40.7% | 7.7x |
| Weighted average | 9.2% | 53.9% | 5.9x |

### Stage Contribution (approximate, Python source)

```
  Input: 100%
  After QuantumLock:        100%   (no size change, cache alignment only)
  After Cortex:             100%   (classification only)
  After RLE:                 97%
  After SemanticDedup:       95%
  After StructuralCollapse:  88%
  After Neurosyntax:         78%
  After Nexus:               76%
  After TokenOpt:            75%
  Final: 75% of original = 25% compression
```

### Latency

Pipeline execution time scales with input length, not pipeline depth. Most stages operate in `O(n)` with respect to character count. Neurosyntax tree-sitter parsing is `O(n log n)`. For typical LLM context windows (8K-128K tokens), total pipeline latency is 10-80ms on a modern CPU.

Gate evaluations are `O(1)` or `O(k)` for small constant `k`. On a 32K-token message where 10 of 14 stages do not fire, the overhead of the non-firing stages is negligible.

---

## Repository Layout

```
/
  scripts/
    lib/
      fusion/           14-stage pipeline implementation
        base.mjs        FusionContext, FusionResult, FusionStage ABC
        pipeline.mjs    FusionPipeline (ordered chain executor)
        engine.mjs      FusionEngine (public entry point)
        quantum_lock.mjs
        cortex.mjs
        photon.mjs
        rle.mjs
        semantic_dedup.mjs
        ionizer.mjs
        log_crunch.mjs
        search_crunch.mjs
        diff_crunch.mjs
        structural_collapse.mjs
        neurosyntax.mjs
        nexus.mjs
        token_opt.mjs
        abbrev_stage.mjs
      rewind/           Rewind engine
        store.mjs       RewindStore (hash-addressed storage)
        tool.mjs        rewind_retrieve tool call handler
      rle.mjs           Legacy RLE utility (used by RLE stage)
      dictionary.mjs    Legacy dictionary compressor
      tokenizer_optimizer.mjs  Legacy tokenizer format optimizer
  tests/                1676 tests
  proxy/                Node.js OpenAI-compatible proxy
```

---

## Key Invariants

1. `FusionContext` is never mutated after construction.
2. `FusionResult` is never mutated after construction.
3. `should_apply()` has no side effects.
4. `apply()` is the only method that writes to `RewindStore`.
5. Stages are sorted by `order` at pipeline construction time; the list passed to `FusionPipeline` may be in any order.
6. `content_type` and `language` are set once by Cortex and are read-only for all subsequent stages.
7. Cross-message deduplication always precedes per-message pipeline execution.
8. Rewind markers are opaque to all compression stages — no stage modifies or removes a marker emitted by a prior stage.
