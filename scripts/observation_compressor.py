#!/usr/bin/env python3
"""Compress OpenClaw session transcripts into structured observations.

Inspired by claude-mem: extract tool calls and results from session JSONL,
generate LLM prompts for compression into structured observations, achieving
97%+ compression on verbose tool output.

Usage:
    python3 observation_compressor.py <transcript.jsonl> [--output observations.md]
    python3 observation_compressor.py <session_dir/> --all [--output-dir DIR]
    python3 observation_compressor.py <transcript.jsonl> --stats

Part of claw-compactor. License: MIT.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.tokens import estimate_tokens
from lib.tokenizer_optimizer import optimize_tokens
from lib.exceptions import FileNotFoundError_, MemCompressError

logger = logging.getLogger(__name__)

# Observation types for classification
OBSERVATION_TYPES = [
    "feature",
    "bugfix",
    "decision",
    "discovery",
    "config",
    "deployment",
    "data",
    "investigation",
]

# LLM prompt for compressing a session segment
COMPRESS_PROMPT = """You are a session observation extractor. Compress the following session transcript segment into structured observations.

Rules:
- Extract ONLY facts: what was done, what was the result, what was decided
- Remove all tool output verbosity -- just capture the key information
- Each observation should be self-contained and useful for future reference
- Use the XML format below
- Multiple observations per segment are fine
- Skip trivial operations (cd, ls with no interesting output, etc)

Transcript segment:
---
{segment}
---

Output observations in this format:
```xml
<observations>
  <observation>
    <type>{types_hint}</type>
    <title>Brief descriptive title</title>
    <facts>
      - Key fact 1
      - Key fact 2
    </facts>
    <narrative>One sentence summary of what happened.</narrative>
  </observation>
</observations>
```"""


def parse_session_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Parse an OpenClaw session .jsonl file.

    Each line is a JSON object with type, message, etc.
    Returns list of parsed message dicts.
    Raises FileNotFoundError_ if file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError_(f"Session file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    messages: List[Dict[str, Any]] = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Normalize: extract role from nested message if present
            if "message" in obj and isinstance(obj["message"], dict):
                msg = obj["message"]
                msg["_type"] = obj.get("type", "message")
                msg["_id"] = obj.get("id", "")
                msg["_timestamp"] = obj.get("timestamp", "")
                messages.append(msg)
            elif "role" in obj:
                # Flat message format (role/content at top level)
                messages.append(obj)
            elif "type" in obj:
                # Session start or metadata
                messages.append({"role": obj.get("type", "unknown"), "_type": obj["type"], **obj})
        except json.JSONDecodeError:
            logger.debug("Skipping malformed JSONL line: %s", line[:80])
            continue

    return messages


def extract_tool_interactions(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract tool call/result pairs from parsed messages.

    Supports OpenClaw's native JSONL format:
      - Tool calls:   role="assistant", content block type="toolCall",
                      fields: name (str), arguments (dict), id (str)
      - Tool results: role="toolResult" (top-level), fields: toolName, toolCallId,
                      content=[{type:"text", text:"..."}]

    Also handles legacy OpenAI-style tool_calls arrays for compatibility.

    Returns list of interaction dicts with tool_name, input_summary, output_summary.
    """
    interactions: List[Dict[str, Any]] = []
    # Index pending interactions by tool call id for result matching
    pending: Dict[str, Dict[str, Any]] = {}

    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")

        # --- OpenClaw native: assistant message with toolCall content blocks ---
        if role == "assistant" and isinstance(content, list):
            # Grab any assistant text from the same message (thinking narration)
            assistant_text = ""
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    assistant_text = b.get("text", "")[:200]
                    break

            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "toolCall"):
                    continue
                tool_name = block.get("name") or block.get("toolName") or "unknown"
                args = block.get("arguments") or block.get("input") or {}
                call_id = block.get("id", "")
                interaction = {
                    "tool_name": tool_name,
                    "input_summary": json.dumps(args)[:300] if isinstance(args, dict) else str(args)[:300],
                    "output_summary": "",
                    "output_size": 0,
                    "assistant_text": assistant_text,
                }
                interactions.append(interaction)
                if call_id:
                    pending[call_id] = interaction

        # --- OpenClaw native: toolResult message ---
        elif role == "toolResult":
            tool_call_id = msg.get("toolCallId", "")
            # Result text lives in content[0].text
            result_text = ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        result_text = block.get("text", "")
                        break
            elif isinstance(content, str):
                result_text = content

            # Match by toolCallId first, fall back to last pending
            target = pending.pop(tool_call_id, None)
            if target is None and interactions:
                # Fallback: attach to most recent interaction without a result
                for ix in reversed(interactions):
                    if not ix["output_summary"]:
                        target = ix
                        break
            if target is not None and not target["output_summary"]:
                target["output_summary"] = result_text[:500]
                target["output_size"] = len(result_text)

        # --- Legacy OpenAI-style tool_calls array ---
        elif role == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                interaction = {
                    "tool_name": func.get("name", "unknown"),
                    "input_summary": func.get("arguments", "")[:300],
                    "output_summary": "",
                    "output_size": 0,
                    "assistant_text": content[:200] if isinstance(content, str) else "",
                }
                interactions.append(interaction)
                call_id = tc.get("id", "")
                if call_id:
                    pending[call_id] = interaction

        # --- Legacy OpenAI-style role=tool result ---
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            result_text = ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        result_text = str(block.get("result") or block.get("text") or "")
                        break
            elif isinstance(content, str):
                result_text = content

            target = pending.pop(tool_call_id, None)
            if target is None and interactions:
                for ix in reversed(interactions):
                    if not ix["output_summary"]:
                        target = ix
                        break
            if target is not None and not target["output_summary"]:
                target["output_summary"] = result_text[:500]
                target["output_size"] = len(result_text)

    return interactions


def generate_observation_prompt(segment: List[Dict[str, Any]]) -> str:
    """Generate an LLM prompt for compressing a session segment."""
    types_hint = '|'.join(OBSERVATION_TYPES)
    lines = []
    for interaction in segment:
        lines.append(f"Tool: {interaction.get('tool_name', 'unknown')}")
        lines.append(f"Input: {interaction.get('input_summary', '')}")
        output_size = interaction.get('output_size', len(interaction.get('output_summary', '')))
        lines.append(f"Output ({output_size} chars): {interaction.get('output_summary', '')[:200]}")
        lines.append("")
    segment_text = '\n'.join(lines)
    return COMPRESS_PROMPT.format(segment=segment_text, types_hint=types_hint)


def rule_extract_observations(
    interactions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract observations using rule-based heuristics (no LLM needed).

    Groups interactions by tool and extracts key patterns.
    """
    if not interactions:
        return []

    observations: List[Dict[str, Any]] = []

    for interaction in interactions:
        tool = interaction["tool_name"]
        output = interaction.get("output_summary", "") or interaction.get("result", "") or ""
        assistant = interaction.get("assistant_text", "")

        # Classify
        obs_type = "discovery"
        if "error" in output.lower() or "fail" in output.lower():
            obs_type = "bugfix"
        elif tool in ("write", "edit"):
            obs_type = "feature"
        elif tool in ("exec",) and ("deploy" in output.lower() or "docker" in output.lower()):
            obs_type = "deployment"
        elif tool in ("exec",) and any(k in output.lower() for k in ("config", "setup", "install")):
            obs_type = "config"

        title = assistant[:80] if assistant else f"{tool} operation"
        facts = [f"Tool: {tool}"]
        if output:
            # Extract key facts from output
            output_lines = output.split('\n')
            for line in output_lines[:5]:
                line = line.strip()
                if line and len(line) > 5:
                    facts.append(line[:100])

        observations.append({
            "type": obs_type,
            "title": title,
            "facts": facts,
            "narrative": assistant[:200] if assistant else f"Ran {tool}",
        })

    return observations


def format_observations_xml(observations: List[Dict[str, Any]]) -> str:
    """Format observations as XML."""
    lines = ["<observations>"]
    for obs in observations:
        lines.append("  <observation>")
        lines.append(f"    <type>{obs['type']}</type>")
        lines.append(f"    <title>{obs.get('title', '') or obs.get('summary', '')}</title>")
        lines.append("    <facts>")
        for fact in obs.get("facts", []):
            lines.append(f"      - {fact}")
        lines.append("    </facts>")
        lines.append(f"    <narrative>{obs.get('narrative', '')}</narrative>")
        lines.append("  </observation>")
    lines.append("</observations>")
    return '\n'.join(lines)


def format_observations_md(observations: List[Dict[str, Any]]) -> str:
    """Format observations as markdown."""
    lines = ["# Session Observations", ""]
    for i, obs in enumerate(observations, 1):
        lines.append(f"## {i}. [{obs['type']}] {obs.get('title', '') or obs.get('summary', '')}")
        lines.append("")
        if obs.get("facts"):
            lines.append("**Facts:**")
            for fact in obs["facts"]:
                lines.append(f"- {fact}")
            lines.append("")
        if obs.get("narrative"):
            lines.append(f"**Result:** {obs['narrative']}")
            lines.append("")
    return '\n'.join(lines)


def compress_session(
    path: Path,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Compress a single session transcript.

    Returns dict with observation count, tokens before/after, etc.
    """
    messages = parse_session_jsonl(path)
    if not messages:
        return {
            "file": str(path),
            "messages": 0,
            "interactions": 0,
            "observations": 0,
            "tokens_before": 0,
            "tokens_after": 0,
        }

    interactions = extract_tool_interactions(messages)
    observations = rule_extract_observations(interactions)

    # Estimate tokens
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    tokens_before = estimate_tokens(raw_text)

    if observations:
        md = format_observations_md(observations)
        tokens_after = estimate_tokens(md)
    else:
        tokens_after = 0

    result: Dict[str, Any] = {
        "file": str(path),
        "messages": len(messages),
        "interactions": len(interactions),
        "observations": observations,
        "observation_count": len(observations),
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
    }

    if use_llm and interactions:
        result["llm_prompt"] = generate_observation_prompt(interactions)

    return result


def main():
    parser = argparse.ArgumentParser(description="Compress session transcripts")
    parser.add_argument("path", help="Session .jsonl file or directory")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--all", action="store_true", help="Process all sessions in directory")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    p = Path(args.path)
    if args.all and p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    else:
        files = [p]

    results = [compress_session(f) for f in files]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        total_before = sum(r["tokens_before"] for r in results)
        total_after = sum(r["tokens_after"] for r in results)
        total_obs = sum(r["observation_count"] for r in results)
        pct = ((total_before - total_after) / total_before * 100) if total_before else 0
        print(f"Processed {len(results)} session(s)")
        print(f"Observations: {total_obs}")
        print(f"Tokens: {total_before:,} -> {total_after:,} ({pct:.1f}% savings)")


if __name__ == "__main__":
    main()
