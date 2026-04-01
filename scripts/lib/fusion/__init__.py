"""Fusion Pipeline — 14-stage LLM token compression framework.

Stages (execution order):
    QuantumLock(3)        KV-cache alignment for system prompts
    Cortex(5)             Content type + language auto-detection
    Photon(8)             Base64 image compression
    RLE(10)               Path/IP/enum shorthand encoding
    SemanticDedup(12)     SimHash near-duplicate block elimination
    Ionizer(15)           JSON array statistical sampling (reversible)
    LogCrunch(16)         Build/test log line folding
    SearchCrunch(17)      Search result deduplication
    DiffCrunch(18)        Git diff context folding
    StructuralCollapse(20) Import merging + repeated pattern collapse
    Neurosyntax(25)       AST-aware code compression (tree-sitter)
    Nexus(35)             ML token-level compression
    TokenOpt(40)          Tokenizer format optimization
    Abbrev(45)            Natural language abbreviation (text only)

Core abstractions:
    FusionContext    Immutable input snapshot flowing through the pipeline
    FusionResult     Immutable output from a single stage
    FusionStage      Abstract base: should_apply() + apply()
    FusionPipeline   Ordered chain with timing and metrics
    FusionEngine     Unified entry point (see engine.py)

Part of claw-compactor v7. License: MIT.
"""
from claw_compactor.fusion.base import FusionStage, FusionContext, FusionResult
from claw_compactor.fusion.pipeline import FusionPipeline, FusionPipelineResult

__all__ = [
    "FusionStage",
    "FusionPipeline",
    "FusionContext",
    "FusionResult",
    "FusionPipelineResult",
]

# v8: Conversation-level compaction (inspired by Claude Code architecture)
from claw_compactor.fusion.tool_result_budget import budget_tool_results
from claw_compactor.fusion.conversation_summarizer import summarize_conversation
from claw_compactor.fusion.tiered_compaction import (
    CompactionLevel,
    CircuitBreaker,
    FileAccessTracker,
    compact,
    determine_level,
)

__all__ += [
    "budget_tool_results",
    "summarize_conversation",
    "CompactionLevel",
    "CircuitBreaker",
    "FileAccessTracker",
    "compact",
    "determine_level",
]

# v8.1: Additional Claude Code-inspired features
from claw_compactor.fusion.llm_summarizer import LLMSummarizer
from claw_compactor.fusion.plan_reinjection import PlanTaskTracker
from claw_compactor.fusion.skill_reinjection import SkillSchemaTracker
from claw_compactor.fusion.compact_hooks import HookRegistry, HookPhase
from claw_compactor.fusion.content_stripper import strip_images_and_docs
from claw_compactor.fusion.cache_prefix import CachePrefixManager

__all__ += [
    "LLMSummarizer",
    "PlanTaskTracker",
    "SkillSchemaTracker",
    "HookRegistry",
    "HookPhase",
    "strip_images_and_docs",
    "CachePrefixManager",
]
