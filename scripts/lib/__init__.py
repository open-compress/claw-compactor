"""claw-compactor shared library.

Core utilities for token estimation, markdown parsing, deduplication,
dictionary encoding, run-length encoding, and format optimization.

Layer 6 — Engram Observational Memory engine:
    EngramEngine      — main public API
    EngramStorage     — file-system storage backend
    OBSERVER_SYSTEM_PROMPT   — Observer LLM system prompt
    REFLECTOR_SYSTEM_PROMPT  — Reflector LLM system prompt

Part of claw-compactor. License: MIT.
"""

# Expose Engram classes at package level for convenient imports:
#   from scripts.lib import EngramEngine
#   from scripts.lib import EngramStorage
from claw_compactor.engram import EngramEngine  # noqa: F401
from claw_compactor.engram_storage import EngramStorage  # noqa: F401
from claw_compactor.engram_prompts import (  # noqa: F401
    OBSERVER_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
    OBSERVER_USER_TEMPLATE,
    REFLECTOR_USER_TEMPLATE,
)

__all__ = [
    "EngramEngine",
    "EngramStorage",
    "OBSERVER_SYSTEM_PROMPT",
    "REFLECTOR_SYSTEM_PROMPT",
    "OBSERVER_USER_TEMPLATE",
    "REFLECTOR_USER_TEMPLATE",
]
