"""Pretty-print per-stage compression statistics as a CLI table.

Usage::

    from claw_compactor.fusion.engine import FusionEngine
    from claw_compactor.fusion.stats_formatter import format_stats_table

    engine = FusionEngine()
    result, stats, markers, warnings = engine.compress(text)
    print(format_stats_table(stats))

Part of claw-compactor. License: MIT.
"""
from __future__ import annotations

from typing import Any


def format_stats_table(stats: dict[str, Any], *, show_skipped: bool = False) -> str:
    """Format per-stage compression stats as a human-readable table.

    Args:
        stats: The stats dict returned by ``FusionEngine.compress()``.
        show_skipped: If True, include stages that were skipped (0 reduction).

    Returns:
        Multi-line string with aligned columns.
    """
    per_stage = stats.get("per_stage", [])
    if not per_stage:
        return "(no per-stage data)"

    rows: list[tuple[str, int, int, int, float, float]] = []
    for s in per_stage:
        if s["skipped"] and not show_skipped:
            continue
        orig = s["original_tokens"]
        comp = s["compressed_tokens"]
        saved = orig - comp
        pct = (saved / orig * 100) if orig > 0 else 0.0
        rows.append((s["name"], orig, comp, saved, pct, s["timing_ms"]))

    if not rows:
        return "(all stages skipped)"

    # Column headers
    hdr = ("Stage", "Before", "After", "Saved", "%", "ms")
    # Compute column widths
    name_w = max(len(hdr[0]), max(len(r[0]) for r in rows))
    num_w = 8

    lines: list[str] = []
    # Header
    lines.append(
        f"{'Stage':<{name_w}}  {'Before':>{num_w}}  {'After':>{num_w}}"
        f"  {'Saved':>{num_w}}  {'%':>6}  {'ms':>8}"
    )
    lines.append("─" * (name_w + num_w * 3 + 6 + 8 + 10))

    for name, orig, comp, saved, pct, ms in rows:
        lines.append(
            f"{name:<{name_w}}  {orig:>{num_w},}  {comp:>{num_w},}"
            f"  {saved:>{num_w},}  {pct:>5.1f}%  {ms:>7.1f}"
        )

    # Totals
    lines.append("─" * (name_w + num_w * 3 + 6 + 8 + 10))
    total_orig = stats.get("original_tokens", 0)
    total_comp = stats.get("compressed_tokens", 0)
    total_saved = total_orig - total_comp
    total_pct = stats.get("reduction_pct", 0.0)
    total_ms = stats.get("total_timing_ms", 0.0)
    lines.append(
        f"{'Total':<{name_w}}  {total_orig:>{num_w},}  {total_comp:>{num_w},}"
        f"  {total_saved:>{num_w},}  {total_pct:>5.1f}%  {total_ms:>7.1f}"
    )

    return "\n".join(lines)
