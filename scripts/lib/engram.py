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

import json
import logging
import os
import time
import urllib.request
import urllib.error
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional httpx import
# ---------------------------------------------------------------------------
try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_OBSERVER_THRESHOLD = 30_000   # tokens — pending messages before observe
DEFAULT_REFLECTOR_THRESHOLD = 40_000  # tokens — accumulated obs before reflect
DEFAULT_MODEL_ANTHROPIC = "claude-opus-4-5"
DEFAULT_MODEL_OPENAI = "gpt-4o"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

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
        # Add messages — auto-triggers observe/reflect when thresholds exceeded
        engine.add_message("thread-1", role="user", content="Hello!")
        engine.add_message("thread-1", role="assistant", content="Hi!")

        # Get context to inject into a system prompt
        ctx_str = engine.build_system_context("thread-1")

        # Force observe/reflect manually
        engine.observe("thread-1")
        engine.reflect("thread-1")

    Args:
        workspace_path:       Workspace root. Engram data is stored at
                              ``{workspace}/memory/engram/``.
        observer_threshold:   Token count of pending messages that triggers
                              the Observer (default 30 000).
        reflector_threshold:  Token count of accumulated observations that
                              triggers the Reflector (default 40 000).
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
        self.max_tokens = cfg.get("max_tokens", max_tokens)

        # API keys — explicit args > config dict > env vars
        self.anthropic_api_key = (
            anthropic_api_key
            or cfg.get("anthropic_api_key")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.openai_api_key = (
            openai_api_key
            or cfg.get("openai_api_key")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.openai_base_url = (
            openai_base_url
            or cfg.get("openai_base_url")
            or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        )

        # Model selection (explicit arg > config > ENGRAM_MODEL env > provider default)
        _env_model = os.environ.get("ENGRAM_MODEL", "")
        if model:
            self.model = model
        elif cfg.get("model"):
            self.model = cfg["model"]
        elif _env_model:
            self.model = _env_model
        elif self.anthropic_api_key:
            self.model = cfg.get("anthropic_model", DEFAULT_MODEL_ANTHROPIC)
        else:
            self.model = cfg.get("openai_model", DEFAULT_MODEL_OPENAI)

        self.storage = EngramStorage(Path(workspace_path))

        if not self.anthropic_api_key and not self.openai_api_key:
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
        """
        Add a message to the thread and auto-trigger observe/reflect if needed.

        Args:
            thread_id:    Conversation thread identifier.
            role:         Message role (``"user"`` / ``"assistant"`` / ``"system"``).
            content:      Message text content.
            timestamp:    Optional ``HH:MM`` or ISO timestamp string.
            auto_observe: If False, skip threshold check (only write to pending).
                          Use False for bulk ingestion; call _check_thresholds()
                          manually at the end. Defaults to True (backward-compatible).

        Returns:
            Status dict::

                {
                    "observed": bool,
                    "reflected": bool,
                    "pending_tokens": int,
                    "observation_tokens": int,
                    "error": str | None,
                }
        """
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
        """
        Check Observer and Reflector thresholds and trigger as needed.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Status dict with ``observed``, ``reflected``, ``pending_tokens``,
            ``observation_tokens``, and ``error`` keys.
        """
        status: Dict[str, Any] = {
            "observed": False,
            "reflected": False,
            "pending_tokens": 0,
            "observation_tokens": 0,
            "error": None,
        }

        # --- Check Observer threshold ---
        pending = self.storage.read_pending(thread_id)
        pending_tokens = _count_messages_tokens(pending)
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

        # --- Check Reflector threshold (after possible observation) ---
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
        """
        Bulk-write messages then check thresholds once at the end.

        More efficient than calling add_message() in a loop when ingesting
        large amounts of historical data, because threshold checks (which
        may trigger expensive LLM calls) are deferred until all messages
        have been written.

        Args:
            thread_id:  Conversation thread identifier.
            messages:   List of message dicts with keys ``role``, ``content``,
                        and optional ``timestamp``.
            batch_size: Unused parameter kept for API future-proofing.

        Returns:
            Status dict from the final ``_check_thresholds()`` call.
        """
        for msg in messages:
            self.add_message(
                thread_id,
                msg["role"],
                msg["content"],
                msg.get("timestamp"),
                auto_observe=False,
            )
        # Check thresholds once after all messages are written
        return self._check_thresholds(thread_id)

    def observe(self, thread_id: str) -> Optional[str]:
        """
        Manually trigger the Observer for a thread regardless of thresholds.

        Args:
            thread_id: Thread identifier.

        Returns:
            Observation text if pending messages exist, else None.
        """
        pending = self.storage.read_pending(thread_id)
        if not pending:
            logger.info("Engram observe: no pending messages for thread=%s", thread_id)
            return None
        return self._run_observer(thread_id, pending)

    def reflect(self, thread_id: str) -> Optional[str]:
        """
        Manually trigger the Reflector for a thread regardless of thresholds.

        Args:
            thread_id: Thread identifier.

        Returns:
            Reflection text if observations exist, else None.
        """
        obs_text = self.storage.read_observations(thread_id)
        if not obs_text.strip():
            logger.info("Engram reflect: no observations for thread=%s", thread_id)
            return None
        return self._run_reflector(thread_id, obs_text)

    def get_context(self, thread_id: str) -> Dict[str, Any]:
        """
        Return the full three-layer memory context for a thread.

        Returns:
            Context dict::

                {
                    "thread_id": str,
                    "observations": str,
                    "reflection": str,
                    "recent_messages": list[dict],
                    "stats": {
                        "observation_tokens": int,
                        "reflection_tokens": int,
                        "pending_tokens": int,
                        "total_tokens": int,
                        "pending_count": int,
                    },
                    "meta": dict,
                }
        """
        observations = self.storage.read_observations(thread_id)
        reflection = self.storage.read_reflection(thread_id)
        recent_messages = self.storage.read_pending(thread_id)
        meta = self.storage.read_meta(thread_id)

        obs_tokens = estimate_tokens(observations)
        ref_tokens = estimate_tokens(reflection)
        pending_tokens = _count_messages_tokens(recent_messages)

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
        """
        Build a compact, injectable system-context string for this thread.

        Includes (in priority order):
          1. Reflection (long-term context, if present)
          2. Recent observations (up to ~200 lines / ~8K tokens)
          3. Token budget summary comment

        Args:
            thread_id: Thread identifier.

        Returns:
            Formatted string ready to prepend to a system prompt. Empty string
            if there is no context at all.
        """
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
        """Run Observer LLM, persist result, clear pending queue.

        If messages exceed MAX_OBSERVER_INPUT_TOKENS, process in batches.
        """
        total_tokens = _count_messages_tokens(messages)

        if total_tokens <= MAX_OBSERVER_INPUT_TOKENS:
            # Normal path — single call
            observation = self._llm_observe(messages)
            ts = _now_utc()
            self.storage.append_observation(thread_id, observation, timestamp=ts)
            self.storage.clear_pending(thread_id)
            logger.debug(
                "Engram: Observer done (thread=%s, chars=%d)", thread_id, len(observation)
            )
            return observation

        # Batch path — split messages into chunks that fit within the token limit
        logger.info(
            "Engram: Observer batching (thread=%s, total_tokens=%d, max=%d)",
            thread_id, total_tokens, MAX_OBSERVER_INPUT_TOKENS,
        )

        all_observations: List[str] = []
        batch_start = 0

        while batch_start < len(messages):
            # Build a batch that fits within the token limit
            batch: List[dict] = []
            batch_tokens = 0
            next_start = batch_start

            for i in range(batch_start, len(messages)):
                msg = messages[i]
                msg_tokens = _count_messages_tokens([msg])
                if batch_tokens + msg_tokens > MAX_OBSERVER_INPUT_TOKENS and batch:
                    # This message would overflow; stop here
                    break
                batch.append(msg)
                batch_tokens += msg_tokens
                next_start = i + 1

            if not batch:
                # Single message exceeds limit — include it anyway to avoid infinite loop
                batch = [messages[batch_start]]
                next_start = batch_start + 1

            logger.info(
                "Engram: Observer batch %d (thread=%s, msgs=%d, tokens=%d)",
                len(all_observations) + 1, thread_id, len(batch), batch_tokens,
            )

            observation = self._llm_observe(batch)
            all_observations.append(observation)
            batch_start = next_start

        # Combine all batch observations
        combined = "\n\n---\n\n".join(all_observations)
        ts = _now_utc()
        self.storage.append_observation(thread_id, combined, timestamp=ts)
        self.storage.clear_pending(thread_id)

        logger.debug(
            "Engram: Observer done (thread=%s, batches=%d, chars=%d)",
            thread_id, len(all_observations), len(combined),
        )
        return combined

    def _run_reflector(self, thread_id: str, observations: str) -> str:
        """Run Reflector LLM, persist result (overwrites previous reflection).

        If observations exceed MAX_REFLECTOR_INPUT_TOKENS, truncate to the most
        recent content (tail) to stay within the LLM context window.
        """
        obs_tokens = estimate_tokens(observations)

        if obs_tokens > MAX_REFLECTOR_INPUT_TOKENS:
            # Keep the most recent observations (tail)
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
        ts = _now_utc()
        self.storage.write_reflection(thread_id, reflection, timestamp=ts)
        logger.debug(
            "Engram: Reflector done (thread=%s, chars=%d)", thread_id, len(reflection)
        )
        return reflection

    def _llm_observe(self, messages: List[dict]) -> str:
        """Format messages and call the Observer LLM."""
        messages_text = _messages_to_text(messages)
        current_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_content = OBSERVER_USER_TEMPLATE.format(
            current_datetime=current_dt,
            messages_text=messages_text,
        )
        return self._call_llm(OBSERVER_SYSTEM_PROMPT, user_content)

    def _llm_reflect(self, observations: str) -> str:
        """Format observations and call the Reflector LLM."""
        current_dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        user_content = REFLECTOR_USER_TEMPLATE.format(
            current_datetime=current_dt,
            observations_text=observations,
        )
        return self._call_llm(REFLECTOR_SYSTEM_PROMPT, user_content)

    def _call_llm(self, system: str, user: str) -> str:
        """
        Call LLM API. Prefers Anthropic if key available, else OpenAI-compatible.

        Args:
            system: System prompt.
            user:   User message content.

        Returns:
            Assistant response text.

        Raises:
            RuntimeError: If no API key is configured.
            Exception:    On HTTP or parsing errors.
        """
        if self.anthropic_api_key:
            return self._call_anthropic(system, user)
        if self.openai_api_key:
            return self._call_openai_compatible(system, user)
        raise RuntimeError(
            "EngramEngine: no API key configured. "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable."
        )

    def _call_anthropic(self, system: str, user: str) -> str:
        """Call the Anthropic Messages API."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": DEFAULT_ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = _http_post(url, headers, body)
        content = data.get("content", [])
        for block in content:
            if block.get("type") == "text":
                return block["text"]
        raise ValueError(f"Engram: no text content in Anthropic response: {data}")

    def _call_openai_compatible(self, system: str, user: str) -> str:
        """Call an OpenAI-compatible chat completions endpoint."""
        base = self.openai_base_url.rstrip("/")
        url = f"{base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = _http_post(url, headers, body)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Engram: unexpected OpenAI response structure: {data}"
            ) from exc


# ---------------------------------------------------------------------------
# HTTP helper (httpx preferred, urllib fallback)
# ---------------------------------------------------------------------------

# HTTP status codes that should not be retried (client errors)
_NO_RETRY_CODES = {400, 401, 403}
# HTTP status codes that are transient and worth retrying
_RETRY_CODES = {429, 500, 502, 503, 504}
# Exception types that indicate transient network issues
_RETRY_EXCEPTIONS = (ConnectionError, ConnectionResetError, TimeoutError,
                     urllib.error.URLError)


def _http_post(url: str, headers: dict, body: dict, max_retries: int = 3) -> dict:
    """
    POST JSON body to *url* and return parsed JSON response.

    Retries on transient HTTP errors (429, 500, 502, 503, 504) and network
    exceptions using exponential back-off: 2, 4, 8 seconds between attempts.
    Non-retriable errors (400, 401, 403) are raised immediately.

    Args:
        url:         Target URL.
        headers:     HTTP headers dict.
        body:        Request body (will be JSON-serialised).
        max_retries: Maximum number of retry attempts (default 3).

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: On non-retriable HTTP errors or after exhausting retries.
    """
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

    if _HTTPX_AVAILABLE and _httpx is not None:
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                with _httpx.Client(timeout=120.0) as client:
                    resp = client.post(url, headers=headers, content=payload)
                    if resp.status_code in _NO_RETRY_CODES:
                        raise RuntimeError(
                            f"Engram HTTP {resp.status_code} from {url}: {resp.text}"
                        )
                    if resp.status_code in _RETRY_CODES and attempt < max_retries:
                        delay = 2 ** (attempt + 1)
                        logger.warning(
                            "Engram HTTP %d, retry %d/%d in %ds…",
                            resp.status_code, attempt + 1, max_retries, delay,
                        )
                        time.sleep(delay)
                        last_exc = RuntimeError(
                            f"Engram HTTP {resp.status_code} from {url}"
                        )
                        continue
                    resp.raise_for_status()
                    return resp.json()
            except _RETRY_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = 2 ** (attempt + 1)
                    logger.warning(
                        "Engram network error (%s), retry %d/%d in %ds…",
                        exc, attempt + 1, max_retries, delay,
                    )
                    time.sleep(delay)
                else:
                    raise
        raise last_exc or RuntimeError(f"Engram: max retries exceeded for {url}")

    # Fallback: stdlib urllib
    last_exc2: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code in _NO_RETRY_CODES:
                body_text = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Engram HTTP {exc.code} from {url}: {body_text}"
                ) from exc
            if exc.code in _RETRY_CODES and attempt < max_retries:
                delay = 2 ** (attempt + 1)
                logger.warning(
                    "Engram HTTP %d, retry %d/%d in %ds…",
                    exc.code, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                last_exc2 = exc
                continue
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Engram HTTP {exc.code} from {url}: {body_text}"
            ) from exc
        except _RETRY_EXCEPTIONS as exc:
            last_exc2 = exc
            if attempt < max_retries:
                delay = 2 ** (attempt + 1)
                logger.warning(
                    "Engram network error (%s), retry %d/%d in %ds…",
                    exc, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
            else:
                raise
    raise last_exc2 or RuntimeError(f"Engram: max retries exceeded for {url}")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _count_messages_tokens(messages: List[dict]) -> int:
    """Estimate token count for a list of message dicts."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += estimate_tokens(block.get("text", ""))
                    total += estimate_tokens(str(block.get("input", "")))
        else:
            total += estimate_tokens(str(content))
        total += 4  # per-message overhead
    return total


def _messages_to_text(messages: List[dict]) -> str:
    """Serialise a list of message dicts into a human-readable text block."""
    lines: List[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        ts = msg.get("timestamp", "")
        ts_str = f" [{ts}]" if ts else ""
        content = msg.get("content", "")

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        parts.append(
                            f"[tool_call: {block.get('name')} "
                            f"input={json.dumps(block.get('input', {}), ensure_ascii=False)[:200]}]"
                        )
                    elif btype == "tool_result":
                        raw = block.get("content", "")
                        if isinstance(raw, list):
                            raw = " ".join(
                                b.get("text", "") for b in raw if isinstance(b, dict)
                            )
                        parts.append(f"[tool_result: {str(raw)[:500]}]")
                    else:
                        parts.append(str(block))
            content_str = "\n".join(parts)
        else:
            content_str = str(content)

        lines.append(f"[{i + 1}] {role}{ts_str}:\n{content_str}\n")

    return "\n".join(lines)
