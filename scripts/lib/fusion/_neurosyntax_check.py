"""Check tree-sitter availability and warn if missing."""
import logging

logger = logging.getLogger("claw_compactor.neurosyntax")

try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning(
        "tree-sitter not installed. Neurosyntax will use regex fallback "
        "(~8%% compression vs ~25%% with tree-sitter). "
        "Install with: pip install tree-sitter tree-sitter-python"
    )
