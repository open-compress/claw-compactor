---
name: claw-compactor
version: "1.0.0"
description: >
  Claw Compactor — 6-layer token compression skill for OpenClaw agents.
  Cuts workspace token spend by 50–97% using deterministic rule-engines plus
  Engram: a real-time, LLM-driven Observational Memory system.
  Run at session start for automatic savings reporting.

triggers:
  - "compress memory"
  - "compress workspace"
  - "save tokens"
  - "token savings"
  - "compress context"
  - "run engram"
  - "engram observe"
  - "engram reflect"
  - "memory compression"
  - "benchmark compression"
---

# Claw Compactor — OpenClaw Skill Reference

## Overview

Claw Compactor reduces token usage across the full OpenClaw workspace using
6 compression layers:

| Layer | Name | Cost | Notes |
|-------|------|------|-------|
| 1 | Rule Engine | Free | Dedup, strip filler, merge sections |
| 2 | Dictionary Encoding | Free | Auto-codebook, `$XX` substitution |
| 3 | Observation Compression | Free | Session JSONL → structured summaries |
| 4 | RLE Patterns | Free | Path/IP/enum shorthand |
| 5 | Compressed Context Protocol | Free | Format abbreviations |
| **6** | **Engram** | LLM API | Real-time Observational Memory |

**Skill location:** `skills/claw-compactor/`  
**Entry point:** `scripts/mem_compress.py`  
**Engram CLI:** `scripts/engram_cli.py`

---

## Auto Mode (Recommended — Run at Session Start)

```bash
python3 skills/claw-compactor/scripts/mem_compress.py <workspace> auto
```

Automatically compresses all workspace files, tracks token counts between
runs, and reports savings. Run this at the start of every session.

---

## Core Commands

### Full Pipeline (All Layers)
```bash
python3 scripts/mem_compress.py <workspace> full
```
Runs all 5 deterministic layers in optimal order. Typical: 50%+ combined savings.

### Benchmark (Non-Destructive)
```bash
python3 scripts/mem_compress.py <workspace> benchmark
# JSON output:
python3 scripts/mem_compress.py <workspace> benchmark --json
```
Dry-run report showing potential savings without writing any files.

### Individual Layers
```bash
# Layer 1: Rule-based compression
python3 scripts/mem_compress.py <workspace> compress

# Layer 2: Dictionary encoding
python3 scripts/mem_compress.py <workspace> dict

# Layer 3: Observation compression (session JSONL → summaries)
python3 scripts/mem_compress.py <workspace> observe

# Layer 4: RLE pattern encoding (runs inside `compress`)
# Layer 5: Tokenizer optimization
python3 scripts/mem_compress.py <workspace> optimize

# Tiered summaries (L0/L1/L2)
python3 scripts/mem_compress.py <workspace> tiers

# Cross-file deduplication
python3 scripts/mem_compress.py <workspace> dedup

# Token count report
python3 scripts/mem_compress.py <workspace> estimate

# Workspace health check
python3 scripts/mem_compress.py <workspace> audit
```

### Global Options
```
--json          Machine-readable JSON output
--dry-run       Preview without writing files
--since DATE    Filter sessions by date (YYYY-MM-DD)
--auto-merge    Auto-merge duplicates (dedup command)
```

---

## Engram — Layer 6: Real-Time Observational Memory

Engram is the flagship layer. It operates as a live engine alongside conversations,
automatically compressing messages into structured, priority-annotated knowledge.

### Prerequisites

Configure via `engram.yaml` (recommended) or environment variables:

```yaml
# engram.yaml — place in claw-compactor root
llm:
  provider: openai-compatible
  base_url: http://localhost:8403
  model: claude-code/sonnet
  max_tokens: 4096

threads:
  default:
    observer_threshold: 30000    # pending tokens before Observer fires
    reflector_threshold: 40000   # observation tokens before Reflector fires

concurrency:
  max_workers: 4                 # parallel thread workers
```

```bash
# Alternative: environment variables
export ANTHROPIC_API_KEY=sk-ant-...   # Preferred
# or
export OPENAI_API_KEY=sk-...          # OpenAI-compatible fallback
export OPENAI_BASE_URL=https://...    # Optional: custom endpoint (local LLM, etc.)
```

### Engram Auto-Mode (Recommended for Production)

Auto-detects all active threads and processes them concurrently (4 workers):

```bash
# Single run — auto-detects all threads
python3 scripts/engram_auto.py --workspace ~/.openclaw/workspace

# Via shell wrapper
bash scripts/engram-auto.sh

# Via CLI
python3 scripts/engram_cli.py <workspace> auto --config engram.yaml
python3 scripts/engram_cli.py <workspace> status --thread openclaw-main
python3 scripts/engram_cli.py <workspace> observe --thread openclaw-main
python3 scripts/engram_cli.py <workspace> reflect --thread openclaw-main
```

**Retry:** LLM calls retry on 429/5xx with exponential backoff (2s→4s→8s, max 3 attempts).
No retry on 400/401/403 (fail fast on config errors).

### Engram via Unified Entry Point
```bash
# Check all thread statuses
python3 scripts/mem_compress.py <workspace> engram status

# Force Observer for a thread
python3 scripts/mem_compress.py <workspace> engram observe --thread <thread-id>

# Force Reflector for a thread
python3 scripts/mem_compress.py <workspace> engram reflect --thread <thread-id>

# Print injectable context
python3 scripts/mem_compress.py <workspace> engram context --thread <thread-id>
```

### Engram via Dedicated CLI
```bash
# Status: all threads
python3 scripts/engram_cli.py <workspace> status

# Status: single thread
python3 scripts/engram_cli.py <workspace> status --thread <thread-id>

# Force observe
python3 scripts/engram_cli.py <workspace> observe --thread <thread-id>

# Force reflect
python3 scripts/engram_cli.py <workspace> reflect --thread <thread-id>

# Import conversation from file (JSON array or JSONL)
python3 scripts/engram_cli.py <workspace> ingest \
    --thread <thread-id> --input /path/to/conversation.jsonl

# Get injectable context string (ready for system prompt)
python3 scripts/engram_cli.py <workspace> context --thread <thread-id>

# JSON output for any command
python3 scripts/engram_cli.py <workspace> status --json
python3 scripts/engram_cli.py <workspace> context --thread <id> --json
```

### Engram Daemon Mode (Real-Time Streaming)
```bash
# Start daemon, pipe JSONL messages via stdin
python3 scripts/engram_cli.py <workspace> daemon --thread <thread-id>

# Pipe a message:
echo '{"role":"user","content":"Hello!","timestamp":"12:00"}' | \
    python3 scripts/engram_cli.py <workspace> daemon --thread <thread-id>

# Control commands (send as JSONL):
echo '{"__cmd":"observe"}'   # force observe now
echo '{"__cmd":"reflect"}'   # force reflect now
echo '{"__cmd":"status"}'    # print thread status JSON
echo '{"__cmd":"quit"}'      # exit daemon

# Quiet mode (suppress startup messages on stderr)
python3 scripts/engram_cli.py <workspace> daemon --thread <id> --quiet
```

### Engram Python API
```python
from scripts.lib.engram import EngramEngine

engine = EngramEngine(
    workspace_path="/path/to/workspace",
    observer_threshold=30_000,     # tokens before auto-observe
    reflector_threshold=40_000,    # tokens before auto-reflect
    anthropic_api_key="sk-ant-...", # or set ANTHROPIC_API_KEY env
)

# Add a message — auto-triggers observe/reflect when thresholds exceeded
status = engine.add_message("thread-id", role="user", content="Hello!")
# Returns: {"observed": bool, "reflected": bool, "pending_tokens": int, ...}

# Manual trigger regardless of thresholds
obs_text = engine.observe("thread-id")    # returns None if no pending msgs
ref_text = engine.reflect("thread-id")   # returns None if no observations

# Get full context dict
ctx = engine.get_context("thread-id")
# Returns: {"thread_id", "observations", "reflection", "recent_messages", "stats", "meta"}

# Build injectable system context string
ctx_str = engine.build_system_context("thread-id")
# Ready to prepend to system prompt
```

### Engram Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (preferred) |
| `OPENAI_API_KEY` | — | OpenAI-compatible API key |
| `OPENAI_BASE_URL` | `https://api.openai.com` | Custom endpoint for local LLMs |
| `OM_OBSERVER_THRESHOLD` | `30000` | Pending tokens before auto-observe |
| `OM_REFLECTOR_THRESHOLD` | `40000` | Observation tokens before auto-reflect |
| `OM_MODEL` | `claude-opus-4-5` | LLM model override |

### Threshold Tuning Quick Reference

Each Observer call ≈ 2K output tokens (Sonnet). Daily volume at default 30K threshold:

| Channel | Daily Tokens | @30K threshold | @10K threshold |
|---------|-------------|----------------|----------------|
| #aimm | ~149K | ~5×/day | ~15×/day |
| openclaw-main | ~138K | ~4.5×/day | ~14×/day |
| #open-compress | ~68K | ~2.3×/day | ~7×/day |
| #general | ~62K | ~2×/day | ~6×/day |
| subagent | ~43K | ~1.4×/day | ~4×/day |
| cron | ~9K | ~0.3×/day | ~1×/day |
| **Total** | **~470K/day** | **~16×/day (~32K output tokens)** | **~47×/day (~94K output tokens)** |

Start at `observer_threshold: 30000`. Tune down for fresher context; tune up to reduce cost.

### Engram Benchmark Summary

| Strategy | Token Savings | ROUGE-L | IR-F1 | Latency | LLM Calls |
|----------|--------------|---------|-------|---------|-----------|
| **Engram (L6)** | **87.5%** | 0.038 | 0.414 | ~35s | 2 |
| RuleCompressor (L1–5) | 9.0% | 0.923 | 0.958 | ~6ms | 0 |
| RandomDrop | 21.5% | 0.852 | 0.911 | ~0ms | 0 |

- Engram low ROUGE-L = semantic restructuring, not verbatim copy — intent is preserved
- Use RuleCompressor for instant prompt compression; Engram for long-term memory
- Full results → `benchmark/RESULTS.md`

### Observation Format

Engram produces structured, bilingual (EN/中文) priority-annotated logs:

```
Date: 2026-03-05
- 🔴 12:10 User building OpenCompress; deadline one week / 用户在构建 OpenCompress，deadline 一周内
  - 🔴 12:10 Using ModernBERT-large / 使用 ModernBERT-large
  - 🟡 12:12 Discussed annotation strategy / 讨论了标注策略
- 🟡 12:30 Deployment pipeline discussion on M3 Ultra
- 🟢 12:45 User prefers concise replies
```

- 🔴 **Critical** — goals, deadlines, blockers, key decisions (never dropped)
- 🟡 **Important** — technical details, ongoing work, preferences
- 🟢 **Useful** — background, mentions, soft context

### Memory Storage Layout

```
memory/engram/{thread_id}/
├── pending.jsonl      # Unobserved message buffer (auto-cleared after observe)
├── observations.md    # Observer output — append-only structured log
├── reflections.md     # Reflector output — compressed long-term memory (overwrites)
└── meta.json          # Timestamps and token counts
```

---

## Integration with OpenClaw Memory System

### System Prompt Injection

Inject Engram context at the start of each session:

```python
from scripts.lib.engram import EngramEngine

engine = EngramEngine(workspace_path)
ctx_str = engine.build_system_context("my-session")
if ctx_str:
    system_prompt = ctx_str + "\n\n" + base_system_prompt
```

The `build_system_context()` output structure:
```
## Long-Term Memory (Reflections)
<Reflector output — long-term compressed context>

## Recent Observations
<Last 200 lines of Observer output>

<!-- engram_tokens: 1234 -->
```

### Combining Engram with Deterministic Layers

After an Engram session, run the deterministic pipeline on the output files:

```bash
# Engram produces observations.md and reflections.md
# Then apply deterministic compression to further reduce those:
python3 scripts/mem_compress.py <workspace> full
```

### Recommended Workflow for Long-Running Agent Sessions

1. **Session start:** inject `build_system_context()` into system prompt
2. **Each message:** call `engine.add_message()` — auto-triggers observe/reflect
3. **Session end / weekly cron:** run `full` pipeline on workspace
4. **Multi-session continuity:** context persists in `memory/engram/{thread}/`

---

## OpenClaw Skill Installation

To install as an OpenClaw skill, ensure the skill directory is available at:
```
~/.openclaw/workspace/skills/claw-compactor/
```
or configure the path in your OpenClaw skill registry.

SKILL.md is read by the OpenClaw agent dispatcher. The `description` and
`triggers` fields above control when this skill is automatically activated.

---

## Heartbeat / Cron Automation

```markdown
## Memory Maintenance (weekly)
- python3 skills/claw-compactor/scripts/mem_compress.py <workspace> benchmark
- If savings > 5%: run full pipeline
- If pending Engram messages: run engram observe --thread <id>
```

Cron (Sunday 3am):
```bash
0 3 * * 0 cd /path/to/skills/claw-compactor && \
  python3 scripts/mem_compress.py /path/to/workspace full
```

---

## Output Artifacts Reference

| Artifact | Location | Description |
|----------|----------|-------------|
| Dictionary codebook | `memory/.codebook.json` | Must travel with memory files |
| Observed session log | `memory/.observed-sessions.json` | Tracks processed transcripts |
| Layer 3 summaries | `memory/observations/` | Observation compression output |
| Engram observations | `memory/engram/{thread}/observations.md` | Live Observer log |
| Engram reflections | `memory/engram/{thread}/reflections.md` | Distilled long-term memory |
| Level 0 summary | `memory/MEMORY-L0.md` | ~200 token ultra-compressed summary |
| Level 1 summary | `memory/MEMORY-L1.md` | ~500 token compressed summary |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` on workspace | Point path to workspace root containing `memory/` |
| Dictionary decompression fails | Check `memory/.codebook.json` is valid JSON |
| Zero savings on `benchmark` | Workspace already optimized |
| `observe` finds no transcripts | Check `sessions/` for `.jsonl` files |
| Engram: "no API key configured" | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |
| Engram Observer returns `None` | No pending messages for that thread |
| Token counts seem wrong | Install tiktoken: `pip3 install tiktoken` |
