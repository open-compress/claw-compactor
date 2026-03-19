"""
engram_http.py — HTTP POST helper with retry logic for Engram LLM calls.

Prefers httpx when available, falls back to stdlib urllib.
Part of claw-compactor / Engram layer. License: MIT.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

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

# HTTP status codes that should not be retried (client errors)
_NO_RETRY_CODES = {400, 401, 403}
# HTTP status codes that are transient and worth retrying
_RETRY_CODES = {429, 500, 502, 503, 504}
# Exception types that indicate transient network issues
_RETRY_EXCEPTIONS = (ConnectionError, ConnectionResetError, TimeoutError,
                     urllib.error.URLError)


def http_post(url: str, headers: dict, body: dict, max_retries: int = 3) -> dict:
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
        return _post_httpx(url, headers, payload, max_retries)

    return _post_urllib(url, headers, payload, max_retries)


def _post_httpx(url: str, headers: dict, payload: bytes, max_retries: int) -> dict:
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


def _post_urllib(url: str, headers: dict, payload: bytes, max_retries: int) -> dict:
    last_exc: Optional[Exception] = None
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
                last_exc = exc
                continue
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Engram HTTP {exc.code} from {url}: {body_text}"
            ) from exc
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
