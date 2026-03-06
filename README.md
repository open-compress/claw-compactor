# 🗜️ Claw Compactor

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/aeromomo/claw-compactor)
[![Tests](https://img.shields.io/badge/tests-848%20passed-brightgreen)](https://github.com/aeromomo/claw-compactor)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-purple)](LICENSE)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-skill-orange)](https://openclaw.ai)
[![Release](https://img.shields.io/github/v/release/aeromomo/claw-compactor?color=blue)](https://github.com/aeromomo/claw-compactor/releases)

> **Cut your AI agent's token spend by 50–97%.**  
> One command compresses your entire workspace — memory files, session transcripts,
> sub-agent context — using 6 layered techniques, from deterministic rule-engines
> to a real-time LLM-driven memory system called **Engram**.

*"Cut your tokens. Keep your facts."*

---

## ✨ Why Claw Compactor?

Running long-lived AI agents is expensive. Context windows fill up. Memory files bloat. Session transcripts grow to megabytes. Claw Compactor solves all three:

- **5 deterministic layers** — zero LLM cost, instant, fully reversible
- **Layer 6: Engram** — the real magic: a live, LLM-powered memory engine that observes conversations as they happen and compresses them into structured, priority-annotated knowledge
- **~97% savings** on raw session transcripts  
- **One command** (`full`) to run everything

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/aeromomo/claw-compactor.git
cd claw-compactor

# 2. Dry-run benchmark (non-destructive, no side effects)
python3 scripts/mem_compress.py /path/to/workspace benchmark

# 3. Compress everything
python3 scripts/mem_compress.py /path/to/workspace full
```

**Requirements:** Python 3.9+. No mandatory dependencies.  
Optional: `pip install tiktoken` for exact token counts (falls back to CJK-aware heuristic).

For Engram (Layer 6): configure `engram.yaml` (or set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` env vars).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    mem_compress.py                                  │
│              (unified CLI entry point / dispatcher)                 │
└───┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────┘
    │      │      │      │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
estimate compress dict  dedup observe tiers audit optimize engram
    │      │      │      │      │      │      │      │      │
    └──────┴──────┴──────┴──────┴──────┴──────┴──────┘      │
                          │                                   │
                          ▼                                   ▼
             ┌────────────────────────┐        ┌─────────────────────────┐
             │   scripts/lib/         │        │   ENGRAM ENGINE         │
             │                        │        │  (Layer 6, real-time)   │
             │  tokens.py             │        │                         │
             │    ↳ tiktoken fallback │        │  engram.py              │
             │  markdown.py           │        │    ↳ EngramEngine       │
             │    ↳ section parsing   │        │    ↳ Observer Agent     │
             │  dedup.py              │        │    ↳ Reflector Agent    │
             │    ↳ shingle hashing   │        │                         │
             │  dictionary.py         │        │  engram_storage.py      │
             │    ↳ $XX codebook      │        │    ↳ atomic file I/O    │
             │  rle.py                │        │    ↳ JSONL pending log  │
             │    ↳ path/IP/enum      │        │                         │
             │  tokenizer_optimizer   │        │  engram_prompts.py      │
             │    ↳ format fixes      │        │    ↳ Observer prompt    │
             │  config.py             │        │    ↳ Reflector prompt   │
             │  exceptions.py         │        └─────────────────────────┘
             └────────────────────────┘

Engram Memory Layout (per thread / 每个会话线程):
  memory/engram/{thread_id}/
    ├── pending.jsonl     ← raw messages buffer (auto-cleared after observe)
    ├── observations.md   ← Observer output, append-only structured log
    ├── reflections.md    ← Reflector output, compressed long-term memory
    └── meta.json         ← timestamps, token counts
```

---

## 📊 Compression Layers

| Layer | Name | Technique | Typical Savings | Notes |
|-------|------|-----------|-----------------|-------|
| 1 | **Rule Engine** | Dedup lines, strip markdown filler, merge sections | 4–8% | Lossless |
| 2 | **Dictionary Encoding** | Auto-learned codebook, `$XX` substitution | 4–5% | Lossless |
| 3 | **Observation Compression** | Session JSONL → structured summaries | ~97% | Lossy* |
| 4 | **RLE Patterns** | Path shorthand (`$WS`), IP prefix, enum compaction | 1–2% | Lossless |
| 5 | **Compressed Context Protocol** | ultra/medium/light abbreviation | 20–60% | Lossy* |
| 6 | **Engram** | LLM-driven real-time Observational Memory | 3–6× | LLM required† |

\*Lossy techniques preserve all facts and decisions; only verbose formatting is removed.  
†Layer 6 requires `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

---

## 🧠 Engram — Layer 6: Observational Memory

> Engram is the flagship feature of Claw Compactor. While the first 5 layers
> batch-process static files, **Engram works live** — it watches conversations
> as they unfold and continuously compresses them into a structured,
> priority-annotated knowledge base.

### How It Works

```
  Incoming Messages
        │
        ▼
  ┌─────────────┐       threshold exceeded
  │  pending    │  ──────────────────────►  ┌─────────────────┐
  │  queue      │       (30K tokens)         │  Observer LLM   │
  │  (JSONL)    │                            │  Agent          │
  └─────────────┘                            │                 │
                                             │  Raw messages   │
                                             │  → structured   │
                                             │  observation    │
                                             │  log 🔴🟡🟢     │
                                             └────────┬────────┘
                                                      │
                                             append to observations.md
                                             clear pending queue
                                                      │
                                                      ▼
                                          observation_tokens > 40K?
                                                      │
                                                      ▼ yes
                                             ┌─────────────────┐
                                             │  Reflector LLM  │
                                             │  Agent          │
                                             │                 │
                                             │  All observations│
                                             │  → compressed   │
                                             │  long-term      │
                                             │  reflection     │
                                             └────────┬────────┘
                                                      │
                                             overwrite reflections.md
```

### Auto-Trigger Mechanism

Auto-triggering happens **every time `add_message()` is called**:

1. **Observer** fires when `pending.jsonl` exceeds `OM_OBSERVER_THRESHOLD` (default: 30,000 tokens)
2. **Reflector** fires when `observations.md` exceeds `OM_REFLECTOR_THRESHOLD` (default: 40,000 tokens)

Both thresholds can be tuned per-session — set lower for faster compression, higher to batch more context.

### Daemon Mode

For real-time streaming integration, run Engram as a daemon that reads JSONL from stdin:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Start daemon — pipe messages in real-time
python3 scripts/engram_cli.py /workspace daemon --thread live-session

# In another process, pipe messages:
echo '{"role":"user","content":"Building OpenCompress today"}' | \
    python3 scripts/engram_cli.py /workspace daemon --thread live-session

# Control commands (also via stdin):
echo '{"__cmd":"observe"}'  # force observe now
echo '{"__cmd":"reflect"}'  # force reflect now
echo '{"__cmd":"status"}'   # print thread status
echo '{"__cmd":"quit"}'     # exit daemon
```

### Observation Format

Observations use a priority-annotated, bilingual (EN/中文) format:

```
Date: 2026-03-05
- 🔴 12:10 User building OpenCompress; deadline within one week / 用户在构建 OpenCompress，deadline 一周内
  - 🔴 12:10 Using ModernBERT-large for inference / 使用 ModernBERT-large 做推理
  - 🟡 12:12 Discussed training data annotation strategy / 讨论了训练数据标注策略
  - 🟢 12:15 Mentioned benchmark results are promising / 提到 benchmark 结果不错
- 🟡 12:30 Switched to deployment pipeline discussion on M3 Ultra
- 🟢 12:45 User prefers concise, structured replies
```

**Priority legend (优先级):**
- 🔴 **Critical** — user goals, hard deadlines, blockers, key decisions (never dropped)
- 🟡 **Important** — technical details, ongoing work, significant context
- 🟢 **Useful** — background info, mentions, soft context (pruned after 7 days if unreferenced)

### Engram Quick Start

```bash
# Set API key (Anthropic preferred / OpenAI also supported)
export ANTHROPIC_API_KEY=sk-ant-...

# Check all thread statuses
python3 scripts/engram_cli.py /workspace status

# Add a message (auto-triggers observe/reflect at threshold)
python3 -c "
from scripts.lib.engram import EngramEngine
eng = EngramEngine('/path/to/workspace')
eng.add_message('my-thread', role='user', content='Building OpenCompress today')
ctx = eng.build_system_context('my-thread')
print(ctx)
"

# Force observe for a thread
python3 scripts/engram_cli.py /workspace observe --thread my-thread

# Force reflect
python3 scripts/engram_cli.py /workspace reflect --thread my-thread

# Import conversation from JSONL file
python3 scripts/engram_cli.py /workspace ingest \
    --thread my-thread --input conversation.jsonl

# Print injectable system context
python3 scripts/engram_cli.py /workspace context --thread my-thread

# Via unified entry point (mem_compress.py)
python3 scripts/mem_compress.py /workspace engram status
python3 scripts/mem_compress.py /workspace engram observe --thread my-thread
```

### Engram Auto-Mode (Multi-Channel, Concurrent)

The recommended way to run Engram in production is via the **auto-mode scripts**, which
automatically detect all active threads, process them concurrently, and apply observe/reflect
based on threshold triggers.

```bash
# Single run — auto-detects all threads in workspace
python3 scripts/engram_auto.py --workspace ~/.openclaw/workspace

# Via shell wrapper (same, with logging)
bash scripts/engram-auto.sh

# CLI equivalents
python3 scripts/engram_cli.py <workspace> auto --config engram.yaml
python3 scripts/engram_cli.py <workspace> status --thread openclaw-main
python3 scripts/engram_cli.py <workspace> observe --thread openclaw-main
python3 scripts/engram_cli.py <workspace> reflect --thread openclaw-main
```

**Auto-detected thread sources:**
- Discord channels: `#general`, `#open-compress`, `#aimm`, and others
- OpenClaw internals: `cron`, `subagent`, `openclaw-main`
- Any JSONL session files found in `sessions.scan_dir`

**Concurrency:** Uses `ThreadPoolExecutor` with 4 workers (configurable via `concurrency.max_workers`
in `engram.yaml`) for parallel thread processing.

**Retry mechanism:** All LLM HTTP calls use automatic exponential-backoff retry:
- Retries on: `429`, `500`, `502`, `503`, `504`
- Delays: `2s → 4s → 8s` (max 3 attempts)
- No retry on: `400`, `401`, `403` (configuration errors — fail fast)

### Engram Configuration (`engram.yaml`)

The preferred way to configure Engram is via `engram.yaml` in the project root:

```yaml
llm:
  provider: openai-compatible        # "anthropic" or "openai-compatible"
  base_url: http://localhost:8403    # endpoint for openai-compatible provider
  api_key_env: OPENAI_API_KEY        # env var holding the API key
  model: claude-code/sonnet          # model identifier
  max_tokens: 4096                   # max output tokens per LLM call

threads:
  default:
    observer_threshold: 30000        # pending-message tokens before Observer fires
    reflector_threshold: 40000       # observation tokens before Reflector fires

  # Optional per-thread overrides:
  # discord-general:
  #   observer_threshold: 20000

sessions:
  scan_dir: ~/.openclaw/agents/main/sessions   # where to find session JSONL files
  max_age_hours: 48                            # ignore sessions older than this

storage:
  base_dir: ~/.openclaw/workspace/memory/engram  # Engram memory root

concurrency:
  max_workers: 4                     # parallel thread workers

logging:
  level: INFO
```

**Environment variables (alternative to `engram.yaml`):**
```bash
ANTHROPIC_API_KEY=sk-ant-...        # Preferred LLM provider (Anthropic)
OPENAI_API_KEY=sk-...               # OpenAI-compatible API key
OPENAI_BASE_URL=https://...         # Custom endpoint (local LLM, etc.)
OM_OBSERVER_THRESHOLD=30000         # Tokens before auto-observe (default: 30000)
OM_REFLECTOR_THRESHOLD=40000        # Tokens before auto-reflect (default: 40000)
OM_MODEL=claude-opus-4-5            # Override LLM model
```

### Threshold Tuning Guide

Thresholds control how often Observer/Reflector fire. Lower = more frequent (fresher memory,
higher LLM cost). Higher = less frequent (lower cost, slightly stale context).

**Real-world channel token production (measured):**

| Channel | Daily Tokens | Observer @30K | Observer @10K |
|---------|-------------|---------------|---------------|
| #aimm | ~149K | ~5×/day | ~15×/day |
| openclaw-main | ~138K | ~4.5×/day | ~14×/day |
| #open-compress | ~68K | ~2.3×/day | ~7×/day |
| #general | ~62K | ~2×/day | ~6×/day |
| subagent | ~43K | ~1.4×/day | ~4×/day |
| cron | ~9K | ~0.3×/day | ~1×/day |
| **Total** | **~470K/day** | **~16×/day** | **~47×/day** |

**LLM cost at different thresholds** (each Observer call ≈ 2K output tokens, Sonnet):

| Threshold | Observer Calls/Day | Est. Output Tokens/Day |
|-----------|-------------------|------------------------|
| 30K (default) | ~16 | ~32K |
| 10K | ~47 | ~94K |

**Recommendation:** Start at `observer_threshold: 30000`. Tune down if context feels stale;
tune up to reduce LLM spend. Reflector threshold can stay at `40000` (fires less often).

**Python API:**
```python
from scripts.lib.engram import EngramEngine

engine = EngramEngine(
    workspace_path="/path/to/workspace",
    observer_threshold=30_000,       # tune lower for fresher context
    reflector_threshold=40_000,
    anthropic_api_key="sk-ant-...",  # or set env var
    model="claude-opus-4-5",         # optional model override
)

# Add messages — auto-triggers observe/reflect as needed
status = engine.add_message("thread-1", role="user", content="Hello!")
# status = {"observed": bool, "reflected": bool, "pending_tokens": int, ...}

# Build injectable context string for system prompt
ctx_str = engine.build_system_context("thread-1")
```

### Engram vs. `observe` (Layer 3)

| Feature | `observe` (Layer 3) | Engram (Layer 6) |
|---------|---------------------|------------------|
| Input source | OpenClaw session `.jsonl` files | Any messages via Python API |
| Trigger | Manual / cron | Auto on `add_message()` |
| LLM required | No (rule-based) | Yes (Anthropic / OpenAI) |
| Output format | Plain observation MD | Structured 🔴🟡🟢 priority log |
| Storage location | `memory/observations/` | `memory/engram/{thread}/` |
| Reflector stage | No | Yes (long-term compression) |
| Daemon mode | No | Yes (stdin JSONL streaming) |

---

## 📋 All Commands

```
python3 scripts/mem_compress.py <workspace> <command> [options]
```

| Command | Description | Typical Savings |
|---------|-------------|-----------------|
| `full` | Complete pipeline (all steps in order) | 50%+ combined |
| `benchmark` | Dry-run performance report (non-destructive) | — |
| `compress` | Rule-based compression (Layer 1) | 4–8% |
| `dict` | Dictionary encoding with auto-codebook (Layer 2) | 4–5% |
| `observe` | Session transcript → observations (Layer 3) | ~97% |
| `tiers` | Generate L0/L1/L2 summaries | 88–95% on sub-agent loads |
| `dedup` | Cross-file duplicate detection | varies |
| `estimate` | Token count report | — |
| `audit` | Workspace health check | — |
| `optimize` | Tokenizer-level format fixes (Layer 5) | 1–3% |
| `engram` | LLM Observational Memory (Layer 6) | 3–6× compression |

**Global options:**
- `--json` — Machine-readable JSON output
- `--dry-run` — Preview changes without writing
- `--since YYYY-MM-DD` — Filter sessions by date
- `--auto-merge` — Auto-merge duplicates (dedup)

---

## 📈 Benchmarks

Real-world compression results on OpenClaw agent workspaces:

| Input Type | Technique | Savings |
|------------|-----------|---------|
| Session transcripts (JSONL) | `observe` (Layer 3) | **~97%** |
| Verbose / new workspace | `full` pipeline | **50–70%** |
| Regular maintenance | `full` pipeline | **10–20%** |
| Already-optimized workspace | `full` pipeline | **3–12%** |
| Long-running agent memory | Engram (Layer 6) | **3–6×** |

**Combined effect:** 50% token reduction + prompt caching (90% cost discount) = **~95% effective cost reduction** on repeated context.

### Engram vs. Other Strategies (Benchmark)

Averaged across 5 conversation samples (DevOps, trading, ML, mixed-long, sysadmin):

| Strategy | Ratio | Token Savings | ROUGE-L | IR-F1 | Latency | LLM Calls |
|----------|-------|--------------|---------|-------|---------|-----------|
| **Engram (L6)** | 0.125 | **87.5%** | 0.038 | 0.414 | ~35s | 2 |
| RandomDrop | 0.785 | 21.5% | 0.852 | 0.911 | ~0ms | 0 |
| RuleCompressor (L1–5) | 0.910 | 9.0% | 0.923 | 0.958 | ~6ms | 0 |
| NoCompression | 1.000 | 0% | 1.000 | 1.000 | ~0ms | 0 |

**Key insights:**
- **Engram ROUGE-L is low** because it *semantically restructures* rather than copying verbatim — it preserves intent, not wording
- **RuleCompressor (L1–5)** is best for instant prompt compression (zero latency, zero LLM cost)
- **Engram** is best for long-term memory compression (87.5% space savings, meaning you keep 8× more history in the same token budget)
- Full per-sample results → [`benchmark/RESULTS.md`](benchmark/RESULTS.md)

---

## 🔗 OpenClaw Integration

Claw Compactor ships as an installable OpenClaw skill. When installed, the agent runs compression automatically:

### Auto Mode (every session start)

```bash
python3 scripts/mem_compress.py <workspace> auto
```

This: compresses all workspace files, tracks token counts between runs, and reports savings at session start.

### Heartbeat / Cron Automation

```markdown
## Memory Maintenance (weekly)
- Run: python3 skills/claw-compactor/scripts/mem_compress.py <workspace> benchmark
- If savings > 5%: run full pipeline
- If pending transcripts: run observe
```

Cron example (weekly, Sunday 3am):
```bash
0 3 * * 0 cd /path/to/skills/claw-compactor && \
  python3 scripts/mem_compress.py /path/to/workspace full
```

### cacheRetention — Complementary Optimization

Enable **prompt caching** for a 90% discount on cached tokens:

```json
{
  "agents": {
    "defaults": {
      "models": {
        "anthropic/claude-opus-4-6": {
          "params": { "cacheRetention": "long" }
        }
      }
    }
  }
}
```

> Compression reduces token **count**; caching reduces cost **per token**.  
> Together: 50% compression × 90% cache discount = **95% effective cost reduction**.

---

## ⚙️ Configuration

Optional `claw-compactor-config.json` in workspace root:

```json
{
  "chars_per_token": 4,
  "level0_max_tokens": 200,
  "level1_max_tokens": 500,
  "dedup_similarity_threshold": 0.6,
  "dedup_shingle_size": 3
}
```

All fields optional — sensible defaults apply when absent.

---

## 📁 Output Artifacts

| File | Description |
|------|-------------|
| `memory/.codebook.json` | Dictionary codebook (must travel with memory files) |
| `memory/.observed-sessions.json` | Tracks which transcripts have been processed |
| `memory/observations/` | Layer 3 compressed session summaries |
| `memory/engram/{thread}/observations.md` | Engram: Observer output (append-only) |
| `memory/engram/{thread}/reflections.md` | Engram: Reflector output (latest) |
| `memory/engram/{thread}/pending.jsonl` | Engram: unobserved message buffer |
| `memory/MEMORY-L0.md` | Level 0 summary (~200 tokens) |
| `memory/MEMORY-L1.md` | Level 1 summary (~500 tokens) |

---

## ❓ FAQ

**Q: Will compression lose my data?**  
A: Rule engine, dictionary, RLE, and tokenizer optimization are fully lossless. Observation compression (Layer 3) and Engram (Layer 6) are lossy but preserve all facts and decisions — only verbose formatting is removed.

**Q: How does dictionary decompression work?**  
A: `decompress_text(text, codebook)` expands all `$XX` codes back. The codebook JSON must be present (`memory/.codebook.json`).

**Q: Can I run individual steps?**  
A: Yes — every command is independent: `compress`, `dict`, `observe`, `tiers`, `dedup`, `optimize`, `engram`.

**Q: What if tiktoken isn't installed?**  
A: Falls back to a CJK-aware heuristic (`chars ÷ 4`). Results are ~90% accurate.

**Q: Does it handle Chinese/Japanese/Unicode?**  
A: Yes. Full CJK support including character-aware token estimation and Chinese punctuation normalization.

**Q: How does Engram differ from just running `observe`?**  
A: `observe` (Layer 3) is a batch processor for static JSONL files. Engram is a live engine: it works on any messages, triggers automatically, adds a Reflector stage for long-term distillation, and outputs structured priority logs.

**Q: Can I use Engram with a local LLM?**  
A: Yes — set `OPENAI_BASE_URL` to your local OpenAI-compatible endpoint (e.g., LM Studio, Ollama).

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` on workspace | Ensure path points to workspace root (contains `memory/` or `MEMORY.md`) |
| Dictionary decompression fails | Check `memory/.codebook.json` exists and is valid JSON |
| Zero savings on `benchmark` | Workspace is already optimized |
| `observe` finds no transcripts | Check `sessions/` directory for `.jsonl` files |
| Token count seems wrong | Install tiktoken: `pip3 install tiktoken` |
| Engram: "no API key configured" | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |
| Engram Observer returns None | No pending messages for that thread yet |

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Ensure tests pass: `python3 -m pytest tests/ -v`
4. Commit with a clear message
5. Open a Pull Request

---

## 🙏 Credits

- Inspired by [claude-mem](https://github.com/thedotmack/claude-mem) by thedotmack
- Built by Bot777 for [OpenClaw](https://openclaw.ai)

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
