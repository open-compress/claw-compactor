# Changelog

All notable changes to Claw Compactor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [7.0.0] - 2026-03-17

### Architecture
- **14-stage Fusion Pipeline** replacing the legacy 5-layer sequential approach
- **Immutable data flow** — all pipeline state carried via frozen `FusionContext` / `FusionResult` dataclasses
- **Stage gate mechanism** — `should_apply()` lets each stage skip at zero cost when content type doesn't match
- **FusionEngine** — unified entry point with `compress()` and `compress_messages()` API

### New Compression Stages
- **QuantumLock** (order=3) — KV-cache alignment: isolates dynamic content in system prompts to maximize cache hit rate
- **Cortex** (order=5) — intelligent content router auto-detecting 8 content types and 16 programming languages
- **Photon** (order=8) — base64 image detection and compression
- **SemanticDedup** (order=12) — SimHash fingerprint near-duplicate block elimination (intra + cross-message)
- **Ionizer** (order=15) — JSON array statistical sampling with schema discovery and error preservation
- **LogCrunch** (order=16) — build/test log line folding with occurrence counts
- **SearchCrunch** (order=17) — search/grep result deduplication
- **DiffCrunch** (order=18) — git diff context line folding
- **StructuralCollapse** (order=20) — import merging, assertion collapse, repeated pattern compression
- **Neurosyntax** (order=25) — AST-aware code compression via tree-sitter (safe regex fallback). Never shortens identifiers.
- **Nexus** (order=35) — ML token-level compressor with stopword removal fallback

### Rewind (Reversible Compression)
- Hash-addressed LRU store for original text retrieval
- Marker embedding in compressed output — LLM tool-calls to retrieve originals
- Integrated with Ionizer for JSON array reversal

### Performance
- **5.9x improvement** over legacy regex path (weighted average)
- **53.9% average compression** across 6 content types
- **81.9% peak** on JSON arrays (Ionizer)
- **25.0%** on Python source (Neurosyntax + StructuralCollapse)
- **1,676 tests** (up from 848), 0 failures

### Benchmark
- SWE-bench tasks: **12-19% compression** on real repository code
- ROUGE-L fidelity maintained at 0.653 @ rate=0.3

---

## [1.0.0] - 2026-03-09

### Added
- **5-layer deterministic token compression pipeline** — rule engine, dictionary encoding, observation compression, RLE patterns, and compressed context protocol
- **Engram (Layer 6)** — real-time LLM-driven Observational Memory with Observer/Reflector architecture
- **Auto-compress hook** (v7.0+) — compress on every file change with zero config
- **Full CJK support** — Chinese/Japanese/Korean token estimation and punctuation normalization
- **Benchmark suite** — ROUGE-L and IR-F1 evaluation against LLMLingua-2 and other baselines
- **OpenClaw skill integration** — native skill with triggers and auto-activation
- **Dictionary encoding** — auto-learned codebook with `$XX` substitution and lossless roundtrip
- **RLE patterns** — path shorthand (`$WS`), IP prefix compression, enum compaction
- **Tiered summaries** — L0 (~200 tokens), L1 (~500 tokens), L2 (full) progressive loading
- **Cross-file deduplication** — shingle-based similarity detection with auto-merge
- **Engram daemon mode** — real-time streaming via stdin JSONL
- **Engram auto-mode** — concurrent multi-thread processing (4 workers)
- **CLI interface** — 11 commands: full, benchmark, compress, dict, observe, tiers, dedup, estimate, audit, optimize, auto
- **848 tests passing** across all compression layers

### Benchmarks
- ROUGE-L **0.653** at rate=0.3 (vs LLMLingua-2 0.346, **+88.2%**)
- ROUGE-L **0.723** at rate=0.5 (vs LLMLingua-2 0.570, **+26.8%**)
- Up to **97% token reduction** on session transcripts
- **50–70% token savings** on first run across unoptimized workspaces

[7.0.0]: https://github.com/open-compress/claw-compactor/releases/tag/v7.0.0
[1.0.0]: https://github.com/open-compress/claw-compactor/releases/tag/v1.0.0
