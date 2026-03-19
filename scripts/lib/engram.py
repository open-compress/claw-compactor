"""
engram.py — EngramEngine: LLM-driven Observational Memory for claw-compactor.

Architecture (Layer 6 — sits on top of the 5 deterministic layers):

    Layer 1 — Rule engine     (compress_memory.py)
    Layer 2 — Dictionary      (dictionary_compress.py)
    Layer 3 — Observation     (observation_compressor.py) ← rule-based
    Layer 4 — RLE patterns    (lib/rle.py)
    Layer 5 — CCP             (lib/tokenizer_optimizer.py)
    ──────────────────────────────────────────────────────
    Layer 6 — Engram (THIS)   ← LLM-driven, real-time

EngramEngine maintains three memory layers per thread:
    • pending.jsonl    — raw un-observed messages
    • observations.md  — Observer-compressed event log  (append-only)
    • reflections.md   — Reflector-distilled long-term context

Two LLM agents run automatically when token thresholds are exceeded:
    • Observer   : pending messages  → structured observation log
    • Reflector  : accumulated obs   → compressed long-term reflection

Zero required dependencies: Python 3.9+.
Optional: httpx (faster HTTP), tiktoken (exact token counts).

Part of claw-compactor / Engram layer. License: MIT.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from claw_compactor.tokens import estimate_tokens
from claw_compactor.engram_storage import EngramStorage
from claw_compactor.engram_prompts import (
    OBSERVER_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    OBSERVER_USER_TEMPLATE,
    REFLECTOR_USER_TEMPLATE,
)
from claw_compactor.engram_llm import EngramLLMClient
from claw_compactor.engram_utils import (
    now_utc,
    count_messages_tokens,
    messages_to_text,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_OBSERVER_THRESHOLD = 30_000   # tokens — pending messages before observe
DEFAULT_REFLECTOR_THRESHOLD = 40_000  # tokens — accumulated obs before reflect
DEFAULT_MODEL_ANTHROPIC = "claude-opus-4-5"
DEFAULT_MODEL_OPENAI = "gpt-4o"
DEFAULT_MAX_TOKENS = 4096

MAX_OBSERVER_INPUT_TOKENS = 80_000   # max tokens per Observer LLM call
MAX_REFLECTOR_INPUT_TOKENS = 80_000  # max tokens per Reflector LLM call


# ---------------------------------------------------------------------------
# EngramEngine
# ---------------------------------------------------------------------------

class EngramEngine:
    """
    Real-time, LLM-driven Observational Memory engine.

    Usage::

        engine = EngramEngine(workspace_path="/path/to/workspace")
        engine.add_message("thread-1", role="user", content="Hello!")
        engine.add_message("thread-1", role="assistant", content="Hi!")
        ctx_str = engine.build_system_context("thread-1")
        engine.observe("thread-1")
        engine.reflect("thread-1")

    Args:
        workspace_path:       Workspace root (data stored at {workspace}/memory/engram/).
        observer_threshold:   Token count of pending messages that triggers Observer.
        reflector_threshold:  Token count of accumulated observations that triggers Reflector.
        model:                LLM model identifier (auto-detected per provider).
        max_tokens:           Max tokens the LLM may produce per call.
        anthropic_api_key:    Anthropic API key (falls back to ANTHROPIC_API_KEY env).
        openai_api_key:       OpenAI API key (falls back to OPENAI_API_KEY env).
        openai_base_url:      OpenAI-compatible base URL (default: official OpenAI).
        config:               Raw dict to override any of the above.
    """

    def __init__(
        self,
        workspace_path: str | Path,
        observer_threshold: int = DEFAULT_OBSERVER_THRESHOLD,
        reflector_threshold: int = DEFAULT_REFLECTOR_THRESHOLD,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}

        self.observer_threshold = cfg.get("observer_threshold", observer_threshold)
        self.reflector_threshold = cfg.get("reflector_threshold", reflector_threshold)

        # API keys — explicit args > config dict > env vars
        _anthropic_key = (
            anthropic_api_key
            or cfg.get("anthropic_api_key")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        _openai_key = (
            openai_api_key
            or cfg.get("openai_api_key")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        _openai_base = (
            openai_base_url
            or cfg.get("openai_base_url")
            or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        )

        # Model selection (explicit arg > config > ENGRAM_MODEL env > provider default)
        _env_model = os.environ.get("ENGRAM_MODEL", "")
        _max_tokens = cfg.get("max_tokens", max_tokens)
        if model:
            _model = model
        elif cfg.get("model"):
            _model = cfg["model"]
        elif _env_model:
            _model = _env_model
        elif _anthropic_key:
            _model = cfg.get("anthropic_model", DEFAULT_MODEL_ANTHROPIC)
        else:
            _model = cfg.get("openai_model", DEFAULT_MODEL_OPENAI)

        self.llm = EngramLLMClient(
            model=_model,
            max_tokens=_max_tokens,
            anthropic_api_key=_anthropic_key,
            openai_api_key=_openai_key,
            openai_base_url=_openai_base,
        )
        self.storage = EngramStorage(Path(workspace_path))

        if not _anthropic_key and not _openai_key:
            logger.warning(
                "EngramEngine: no API key configured. "
                "Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable LLM compression."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        timestamp: Optional[str] = None,
        auto_observe: bool = True,
    ) -> Dict[str, Any]:
        """Add a message to the thread and auto-trigger observe/reflect if needed."""
        ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        message = {"role": role, "content": content, "timestamp": ts}
        self.storage.append_message(thread_id, message)

        if not auto_observe:
            return {
                "observed": False,
                "reflected": False,
                "pending_tokens": 0,
                "observation_tokens": 0,
                "error": None,
            }

        return self._check_thresholds(thread_id)

    def _check_thresholds(self, thread_id: str) -> Dict[str, Any]:
        """Check Observer and Reflector thresholds and trigger as needed."""
        status: Dict[str, Any] = {
            "observed": False,
            "reflected": False,
            "pending_tokens": 0,
            "observation_tokens": 0,
            "error": None,
        }

        pending = self.storage.read_pending(thread_id)
        pending_tokens = count_messages_tokens(pending)
        status["pending_tokens"] = pending_tokens

        if pending_tokens >= self.observer_threshold:
            logger.info(
                "Engram: Observer triggered (thread=%s, pending_tokens=%d >= %d)",
                thread_id, pending_tokens, self.observer_threshold,
            )
            try:
                self._run_observer(thread_id, pending)
                status["observed"] = True
            except Exception as exc:
                logger.error("Engram: Observer failed: %s", exc)
                status["error"] = str(exc)

        obs_text = self.storage.read_observations(thread_id)
        obs_tokens = estimate_tokens(obs_text)
        status["observation_tokens"] = obs_tokens

        if obs_tokens >= self.reflector_threshold:
            logger.info(
                "Engram: Reflector triggered (thread=%s, obs_tokens=%d >= %d)",
                thread_id, obs_tokens, self.reflector_threshold,
            )
            try:
                self._run_reflector(thread_id, obs_text)
                status["reflected"] = True
            except Exception as exc:
                logger.error("Engram: Reflector failed: %s", exc)
                if status["error"]:
                    status["error"] += "; " + str(exc)
                else:
                    status["error"] = str(exc)

        return status

    def batch_ingest(
        self,
        thread_id: str,
        messages: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> Dict[str, Any]:
        """Bulk-write messages then check thresholds once at the end."""
        for msg in messages:
            self.add_message(
                thread_id,
                msg["role"],
                msg["content"],
                msg.get("timestamp"),
                auto_observe=False,
            )
        return self._check_thresholds(thread_id)

    def observe(self, thread_id: str) -> Optional[str]:
        """Manually trigger the Observer for a thread regardless of thresholds."""
        pending = self.storage.read_pending(thread_id)
        if not pending:
            logger.info("Engram observe: no pending messages for thread=%s", thread_id)
            return None
        return self._run_observer(thread_id, pending)

    def reflect(self, thread_id: str) -> Optional[str]:
        """Manually trigger the Reflector for a thread regardless of thresholds."""
        obs_text = self.storage.read_observations(thread_id)
        if not obs_text.strip():
            logger.info("Engram reflect: no observations for thread=%s", thread_id)
            return None
        return self._run_reflector(thread_id, obs_text)

    def get_context(self, thread_id: str) -> Dict[str, Any]:
        """Return the full three-layer memory context for a thread."""
        observations = self.storage.read_observations(thread_id)
        reflection = self.storage.read_reflection(thread_id)
        recent_messages = self.storage.read_pending(thread_id)
        meta = self.storage.read_meta(thread_id)

        obs_tokens = estimate_tokens(observations)
        ref_tokens = estimate_tokens(reflection)
        pending_tokens = count_messages_tokens(recent_messages)

        return {
            "thread_id": thread_id,
            "observations": observations,
            "reflection": reflection,
            "recent_messages": recent_messages,
            "stats": {
                "observation_tokens": obs_tokens,
                "reflection_tokens": ref_tokens,
                "pending_tokens": pending_tokens,
                "total_tokens": obs_tokens + ref_tokens + pending_tokens,
                "pending_count": len(recent_messages),
            },
            "meta": meta,
        }

    def build_system_context(self, thread_id: str) -> str:
        """Build a compact, injectable system-context string for this thread."""
        ctx = self.get_context(thread_id)
        parts: List[str] = []

        if ctx["reflection"].strip():
            parts.append("## Long-Term Memory (Reflections)\n" + ctx["reflection"])

        if ctx["observations"].strip():
            obs_lines = ctx["observations"].splitlines()
            if len(obs_lines) > 200:
                obs_lines = obs_lines[-200:]
            parts.append("## Recent Observations\n" + "\n".join(obs_lines))

        if not parts:
            return ""

        total = ctx["stats"]["total_tokens"]
        parts.append(f"\n<!-- engram_tokens: {total} -->")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_observer(self, thread_id: str, messages: List[dict]) -> str:
        """Run Observer LLM, persist result, clear pending queue."""
        total_tokens = count_messages_tokens(messages)

        if total_tokens <= MAX_OBSERVER_INPUT_TOKENS:
            observation = self._llm_observe(messages)
            ts = now_utc()
            self.storage.append_observation(thread_id, observation, timestamp=ts)
            self.storage.clear_pending(thread_id)
            logger.debug(
                "Engram: Observer done (thread=%s, chars=%d)", thread_id, len(observation)
            )
            return observation

        # Batch path — split messages into chunks
        logger.info(
            "Engram: Observer batching (thread=%s, total_tokens=%d, max=%d)",
            thread_id, total_tokens, MAX_OBSERVER_INPUT_TOKENS,
        )

        all_observations: List[str] = []
        batch_start = 0

        while batch_start < len(messages):
            batch: List[dict] = []
            batch_tokens = 0
            next_start = batch_start

            for i in range(batch_start, len(messages)):
                msg = messages[i]
                msg_tokens = count_messages_tokens([msg])
                if batch_tokens + msg_tokens > MAX_OBSERVER_INPUT_TOKENS and batch:
                    break
                batch.append(msg)
                batch_tokens += msg_tokens
                next_start = i + 1

            if not batch:
                batch = [messages[batch_start]]
                next_start = batch_start + 1

            logger.info(
                "Engram: Observer batch %d (thread=%s, msgs=%d, tokens=%d)",
                len(all_observations) + 1, thread_id, len(batch), batch_tokens,
            )

            observation = self._llm_observe(batch)
            all_observations.append(observation)
            batch_start = next_start

        combined = "\n\n---\n\n".join(all_observations)
        ts = now_utc()
        self.storage.append_observation(thread_id, combined, timestamp=ts)
        self.storage.clear_pending(thread_id)

        logger.debug(
            "Engram: Observer done (thread=%s, batches=%d, chars=%d)",
            thread_id, len(all_observations), len(combined),
        )
        return combined

    def _run_reflector(self, thread_id: str, observations: str) -> str:
        """Run Reflector LLM, persist result (overwrites previous reflection)."""
        obs_tokens = estimate_tokens(observations)

        if obs_tokens > MAX_REFLECTOR_INPUT_TOKENS:
            lines = observations.splitlines()
            truncated: List[str] = []
            running_tokens = 0
            for line in reversed(lines):
                line_tokens = estimate_tokens(line)
                if running_tokens + line_tokens > MAX_REFLECTOR_INPUT_TOKENS:
                    break
                truncated.append(line)
                running_tokens += line_tokens
            observations = "\n".join(reversed(truncated))
            logger.info(
                "Engram: Reflector input truncated (thread=%s, %d -> %d tokens)",
                thread_id, obs_tokens, running_tokens,
            )

        reflection = self._llm_reflect(observations)
        ts = now_utc()
        self.storage.write_reflection(thread_id, reflection, timestamp=ts)
        logger.debug(
            "Engram: Reflector done (thread=%s, chars=%d)", thread_id, len(reflection)
        )
        return reflection

    def _llm_observe(self, messages: List[dict]) -> str:
        """Format messages and call the Observer LLM."""
        text = messages_to_text(messages)
        current_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_content = OBSERVER_USER_TEMPLATE.format(
            current_datetime=current_dt,
            messages_text=text,
        )
        return self.llm.call(OBSERVER_SYSTEM_PROMPT, user_content)

    def _llm_reflect(self, observations: str) -> str:
        """Format observations and call the Reflector LLM."""
        current_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_content = REFLECTOR_USER_TEMPLATE.format(
            current_datetime=current_dt,
            observations_text=observations,
        )
        return self.llm.call(REFLECTOR_SYSTEM_PROMPT, user_content)
