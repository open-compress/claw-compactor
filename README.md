<!--
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "Claw Compactor",
  "description": "14-stage Fusion Pipeline for LLM token compression with reversible compression, AST-aware code analysis, and intelligent content routing",
  "applicationCategory": "DeveloperApplication",
  "operatingSystem": "Cross-platform",
  "softwareVersion": "7.0.0",
  "license": "https://opensource.org/licenses/MIT",
  "url": "https://github.com/open-compress/claw-compactor",
  "downloadUrl": "https://github.com/open-compress/claw-compactor",
  "author": {
    "@type": "Organization",
    "name": "OpenClaw",
    "url": "https://openclaw.ai"
  },
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "USD"
  },
  "keywords": "token compression, LLM, AI agent, fusion pipeline, reversible compression, AST code analysis, context window optimization"
}
</script>
-->

<div align="center">

# Claw Compactor

### 14-Stage Fusion Pipeline for LLM Token Compression

![Claw Compactor Banner](assets/banner.png)

[![CI](https://github.com/open-compress/claw-compactor/actions/workflows/ci.yml/badge.svg)](https://github.com/open-compress/claw-compactor/actions)
[![Tests](https://img.shields.io/badge/tests-1676%20passed-brightgreen)](https://github.com/open-compress/claw-compactor)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-purple)](LICENSE)
[![Stars](https://img.shields.io/github/stars/open-compress/claw-compactor?style=social)](https://github.com/open-compress/claw-compactor)

**54% average compression &middot; Zero LLM inference cost &middot; Reversible &middot; 1676 tests**

[Architecture](ARCHITECTURE.md) &middot; [Benchmarks](#benchmarks) &middot; [Quick Start](#quick-start) &middot; [API](#api)

</div>

---

## What is Claw Compactor?

Claw Compactor is an open-source **LLM token compression engine** built around a 14-stage **Fusion Pipeline**. Each stage is a specialized compressor — from AST-aware code analysis to JSON statistical sampling to simhash-based deduplication — chained through an immutable data flow architecture where each stage's output feeds the next.

```
Input
  |
  v
┌─────────────────────────────────────────────────────────────────────────┐
│                         FUSION PIPELINE                                 │
│                                                                         │
│  QuantumLock ─> Cortex ─> Photon ─> RLE ─> SemanticDedup ─> Ionizer    │
│       |            |         |        |          |              |        │
│   KV-cache    auto-detect  base64   path     simhash       JSON         │
│   alignment   16 languages  strip  shorten   dedup        sampling      │
│                                                                         │
│  ─> LogCrunch ─> SearchCrunch ─> DiffCrunch ─> StructuralCollapse      │
│        |              |              |                |                  │
│    log folding    result dedup   context fold    import merge            │
│                                                                         │
│  ─> Neurosyntax ─> Nexus ─> TokenOpt ─> Abbrev ─────────> Output       │
│        |             |          |           |                            │
│    AST compress   ML token   format     NL shorten                      │
│    (tree-sitter)  classify   optimize   (text only)                     │
│                                                                         │
│  [ RewindStore ] ── hash-addressed LRU for reversible retrieval         │
└─────────────────────────────────────────────────────────────────────────┘
```

Key design principles:

- **Immutable data flow** — `FusionContext` is a frozen dataclass. Every stage produces a new `FusionResult`; nothing is mutated in-place.
- **Gate-before-compress** — Each stage has `should_apply()` that inspects context type, language, and role before doing any work. Stages that don't apply are skipped at zero cost.
- **Content-aware routing** — Cortex auto-detects content type (code, JSON, logs, diffs, search results) and language (Python, Go, Rust, TypeScript, etc.), then downstream stages make type-aware compression decisions.
- **Reversible compression** — Ionizer stores originals in a hash-addressed `RewindStore`. The LLM can call a tool to retrieve any compressed section by its marker ID.

---

## Benchmarks

### Real-World Compression (FusionEngine v7 vs Legacy Regex)

| Content Type | Legacy | FusionEngine | Improvement |
|:-------------|-------:|-------------:|:-----------:|
| Python source | 7.3% | **25.0%** | 3.4x |
| JSON (100 items) | 12.6% | **81.9%** | 6.5x |
| Build logs | 5.5% | **24.1%** | 4.4x |
| Agent conversation | 5.7% | **31.0%** | 5.4x |
| Git diff | 6.2% | **15.0%** | 2.4x |
| Search results | 5.3% | **40.7%** | 7.7x |
| **Weighted average** | **9.2%** | **53.9%** | **5.9x** |

### SWE-bench Real Tasks

Tested on real SWE-bench instances with actual repository code:

| Instance | Size | Compression |
|:---------|-----:|------------:|
| django__django-11620 | 4.5K | **14.5%** |
| sympy__sympy-14396 | 5.5K | **19.1%** |
| scikit-learn-25747 | 11.8K | **15.9%** |
| scikit-learn-13554 | 73K | **11.8%** |
| scikit-learn-25308 | 81K | **14.4%** |

### vs LLMLingua-2 (ROUGE-L Fidelity)

| Compression Rate | Claw Compactor | LLMLingua-2 | Delta |
|:-----------------|---------------:|------------:|------:|
| 0.3 (aggressive) | **0.653** | 0.346 | +88.2% |
| 0.5 (balanced) | **0.723** | 0.570 | +26.8% |

Claw Compactor preserves more semantic content at the same compression ratio, with zero LLM inference cost.

---

## Quick Start

```bash
git clone https://github.com/open-compress/claw-compactor.git
cd claw-compactor

# Benchmark your workspace (non-destructive)
python3 scripts/mem_compress.py /path/to/workspace benchmark

# Full compression pipeline
python3 scripts/mem_compress.py /path/to/workspace full
```

**Requirements:** Python 3.9+. Optional: `pip install tiktoken` for exact token counts.

---

## API

### FusionEngine — Single Text

```python
from scripts.lib.fusion.engine import FusionEngine

engine = FusionEngine()

result = engine.compress(
    text="def hello():\n    # greeting function\n    print('hello')",
    content_type="code",    # or let Cortex auto-detect
    language="python",      # optional hint
)

print(result["compressed"])     # compressed output
print(result["stats"])          # per-stage timing + token counts
print(result["markers"])        # Rewind markers for reversibility
```

### FusionEngine — Chat Messages

```python
messages = [
    {"role": "system", "content": "You are a coding assistant..."},
    {"role": "user", "content": "Fix the auth bug in login.py"},
    {"role": "assistant", "content": "I found the issue. Here's the fix:\n```python\n..."},
    {"role": "tool", "content": '{"results": [{"file": "login.py", ...}, ...]}'},
]

result = engine.compress_messages(messages)

# Cross-message dedup runs first, then per-message pipeline
print(result["stats"]["reduction_pct"])   # aggregate compression %
print(result["per_message"])              # per-message breakdown
```

### Rewind — Reversible Retrieval

```python
engine = FusionEngine(enable_rewind=True)
result = engine.compress(large_json, content_type="json")

# LLM sees compressed output with markers like [rewind:abc123...]
# When the LLM needs the original, it calls the Rewind tool:
original = engine.rewind_store.retrieve("abc123def456...")
```

### Custom Stage

```python
from scripts.lib.fusion.base import FusionStage, FusionContext, FusionResult

class MyStage(FusionStage):
    name = "my_compressor"
    order = 22  # runs between StructuralCollapse (20) and Neurosyntax (25)

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_type == "log"

    def apply(self, ctx: FusionContext) -> FusionResult:
        compressed = my_compression_logic(ctx.content)
        return FusionResult(
            content=compressed,
            original_tokens=estimate_tokens(ctx.content),
            compressed_tokens=estimate_tokens(compressed),
        )

# Add to pipeline
pipeline = engine.pipeline.add(MyStage())
```

---

## The 14 Stages

| # | Stage | Order | Purpose | Applies To |
|:-:|:------|:-----:|:--------|:-----------|
| 1 | **QuantumLock** | 3 | Isolates dynamic content in system prompts to maximize KV-cache hit rate | system messages |
| 2 | **Cortex** | 5 | Auto-detects content type and programming language (16 languages) | untyped content |
| 3 | **Photon** | 8 | Detects and compresses base64-encoded images | all |
| 4 | **RLE** | 10 | Path shorthand (`$WS`), IP prefix compression, enum compaction | all |
| 5 | **SemanticDedup** | 12 | SimHash fingerprint deduplication across content blocks | all |
| 6 | **Ionizer** | 15 | JSON array statistical sampling with schema discovery + error preservation | json |
| 7 | **LogCrunch** | 16 | Folds repeated log lines with occurrence counts | log |
| 8 | **SearchCrunch** | 17 | Deduplicates search/grep results | search |
| 9 | **DiffCrunch** | 18 | Folds unchanged context lines in git diffs | diff |
| 10 | **StructuralCollapse** | 20 | Merges import blocks, collapses repeated assertions/patterns | code |
| 11 | **Neurosyntax** | 25 | AST-aware code compression via tree-sitter (safe regex fallback). Never shortens identifiers. | code |
| 12 | **Nexus** | 35 | ML token-level compression (stopword removal fallback without model) | text |
| 13 | **TokenOpt** | 40 | Tokenizer format optimization — strips bold/italic markers, normalizes whitespace | all |
| 14 | **Abbrev** | 45 | Natural language abbreviation. Only fires on text — never touches code, JSON, or structured data. | text |

Each stage is independent and stateless. Stages communicate only through the immutable `FusionContext` that flows forward through the pipeline.

---

## Workspace Commands

```bash
python3 scripts/mem_compress.py <workspace> <command> [options]
```

| Command | Description |
|:--------|:-----------|
| `full` | Run complete compression pipeline |
| `benchmark` | Dry-run compression report |
| `compress` | Rule-based compression only |
| `dict` | Dictionary encoding with auto-learned codebook |
| `observe` | Session transcript JSONL to structured observations |
| `tiers` | Generate L0/L1/L2 tiered summaries |
| `dedup` | Cross-file duplicate detection |
| `estimate` | Token count report |
| `audit` | Workspace health check |
| `optimize` | Tokenizer-level format optimization |
| `auto` | Watch mode — compress on file changes |

Options: `--json`, `--dry-run`, `--since YYYY-MM-DD`, `--quiet`

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical deep-dive:
- Immutable data flow design
- Stage execution model and gating
- Rewind reversible compression protocol
- Cross-message semantic deduplication
- How to extend the pipeline

```
12,000+ lines Python  ·  1,676 tests  ·  14 fusion stages  ·  0 external ML dependencies
```

---

## Installation

```bash
# Clone
git clone https://github.com/open-compress/claw-compactor.git
cd claw-compactor

# Optional: exact token counting
pip install tiktoken

# Optional: AST-aware code compression (Neurosyntax)
pip install tree-sitter-language-pack

# Development
pip install -e ".[dev,accurate]"
```

**Zero required dependencies.** tiktoken and tree-sitter are optional enhancements — the pipeline runs with built-in heuristic fallbacks for both.

---

## Project Stats

| Metric | Value |
|:-------|:------|
| Tests | 1,676 passed |
| Python source | 12,000+ lines |
| Fusion stages | 14 |
| Languages detected | 16 |
| Required dependencies | 0 |
| Compression (weighted avg) | 53.9% |
| Compression (JSON peak) | 81.9% |
| ROUGE-L @ 0.3 rate | 0.653 |
| License | MIT |

---

## Related

- [OpenClaw](https://openclaw.ai) — AI agent platform
- [ClawhubAI](https://clawhub.com) — Agent skills marketplace
- [OpenClaw Discord](https://discord.com/invite/clawd) — Community
- [OpenClaw Docs](https://docs.openclaw.ai) — Documentation

---

`token-compression` `llm-tools` `fusion-pipeline` `reversible-compression` `ast-code-analysis` `context-compression` `ai-agent` `openclaw` `python` `developer-tools`

## License

[MIT](LICENSE)
