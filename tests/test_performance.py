"""Performance benchmark — full pipeline on real workspace data.

Runs each compression technique in sequence, measures token savings,
and outputs a clear performance report.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from claw_compactor.tokens import estimate_tokens, using_tiktoken
from claw_compactor.dictionary import build_codebook, compress_text, decompress_text
from claw_compactor.tokenizer_optimizer import optimize_tokens
from claw_compactor.rle import compress as rle_compress
from compress_memory import rule_compress

WORKSPACE = Path("/home/user/workspace")


def _collect_md_files(workspace: Path) -> list:
    """Collect all .md files in workspace (excluding skills internals)."""
    files = []
    for f in sorted(workspace.rglob("*.md")):
        # Skip node_modules, .git, __pycache__, and deep skill code
        rel = str(f.relative_to(workspace))
        if any(skip in rel for skip in [
            "node_modules", ".git", "__pycache__", ".pytest_cache",
            "scripts/", "tests/", "references/"
        ]):
            continue
        files.append(f)
    return files


def _read_all(files: list) -> dict:
    """Read all files into {path: content} dict."""
    contents = {}
    for f in files:
        try:
            contents[str(f)] = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return contents


class TestPerformanceBenchmark:
    """Benchmark compression on real workspace data."""

    @pytest.fixture(scope="class")
    def workspace_data(self):
        """Load real workspace files."""
        if not WORKSPACE.exists():
            pytest.skip("Workspace not available")
        files = _collect_md_files(WORKSPACE)
        if not files:
            pytest.skip("No .md files found")
        return _read_all(files)

    def test_full_pipeline_benchmark(self, workspace_data):
        """Run full pipeline on real workspace, report savings per technique."""
        all_text = '\n\n'.join(workspace_data.values())

        # Stage 0: Baseline
        baseline_tokens = estimate_tokens(all_text)
        assert baseline_tokens > 0, "Empty workspace"

        # Stage 1: Rule engine compression
        rule_compressed = rule_compress(all_text)
        rule_tokens = estimate_tokens(rule_compressed)

        # Stage 2: Dictionary compression
        # Build codebook from rule-compressed texts for correct roundtrip
        rule_texts = [rule_compress(t) for t in workspace_data.values()]
        codebook = build_codebook(rule_texts, min_freq=3)
        dict_compressed = compress_text(rule_compressed, codebook)
        dict_tokens = estimate_tokens(dict_compressed)

        # Stage 3: Tokenizer optimization
        tok_optimized = optimize_tokens(dict_compressed, aggressive=True)
        tok_tokens = estimate_tokens(tok_optimized)

        # Stage 4: RLE
        rle_result = rle_compress(tok_optimized, [str(WORKSPACE)])
        rle_tokens = estimate_tokens(rle_result)

        # Build report
        stages = [
            ("Baseline", baseline_tokens, baseline_tokens),
            ("Rule Engine", baseline_tokens, rule_tokens),
            ("Dictionary Encode", rule_tokens, dict_tokens),
            ("Tokenizer Optimize", dict_tokens, tok_tokens),
            ("RLE Patterns", tok_tokens, rle_tokens),
        ]

        report_lines = [
            "",
            "=== Compression Performance Report ===",
            f"Files: {len(workspace_data)} | Tokenizer: {'tiktoken' if using_tiktoken() else 'heuristic'}",
            "",
            f"{'Technique':<22} | {'Before':>8} | {'After':>8} | {'Saved':>6} | {'%':>6}",
            "-" * 22 + "-+-" + "-" * 8 + "-+-" + "-" * 8 + "-+-" + "-" * 6 + "-+-" + "-" * 6,
        ]

        for name, before, after in stages[1:]:
            saved = before - after
            pct = (saved / before * 100) if before > 0 else 0
            report_lines.append(
                f"{name:<22} | {before:>8,} | {after:>8,} | {saved:>6,} | {pct:>5.1f}%"
            )

        total_saved = baseline_tokens - rle_tokens
        total_pct = (total_saved / baseline_tokens * 100) if baseline_tokens > 0 else 0
        report_lines.extend([
            "-" * 22 + "-+-" + "-" * 8 + "-+-" + "-" * 8 + "-+-" + "-" * 6 + "-+-" + "-" * 6,
            f"{'TOTAL':<22} | {baseline_tokens:>8,} | {rle_tokens:>8,} | {total_saved:>6,} | {total_pct:>5.1f}%",
            "",
        ])

        report = '\n'.join(report_lines)
        print(report)

        # Assertions
        assert rule_tokens <= baseline_tokens, "Rule engine should not increase tokens"
        assert total_pct > 5, f"Combined savings only {total_pct:.1f}%, expected > 5%"

        # Verify dictionary roundtrip on the actual data
        decompressed = decompress_text(dict_compressed, codebook)
        assert decompressed == rule_compressed, "Dictionary roundtrip failed on real data!"

    def test_per_file_breakdown(self, workspace_data):
        """Show per-file compression potential."""
        results = []
        for path, content in workspace_data.items():
            before = estimate_tokens(content)
            after = estimate_tokens(rule_compress(content))
            optimized = estimate_tokens(optimize_tokens(content, aggressive=True))
            results.append({
                "file": Path(path).name,
                "original": before,
                "rule": after,
                "optimized": optimized,
            })

        results.sort(key=lambda r: r["original"], reverse=True)

        print("\n=== Per-File Breakdown (top 10) ===")
        print(f"{'File':<35} | {'Original':>8} | {'Rule':>8} | {'Optimized':>8} | {'Saved':>6}")
        print("-" * 80)
        for r in results[:10]:
            saved = r["original"] - r["optimized"]
            pct = (saved / r["original"] * 100) if r["original"] > 0 else 0
            print(f"{r['file']:<35} | {r['original']:>8,} | {r['rule']:>8,} | {r['optimized']:>8,} | {pct:>5.1f}%")

    def test_dictionary_codebook_quality(self, workspace_data):
        """Verify codebook quality: entries should save more than they cost."""
        texts = list(workspace_data.values())
        cb = build_codebook(texts, min_freq=3)

        if not cb:
            pytest.skip("No codebook entries built")

        print(f"\n=== Codebook Quality ({len(cb)} entries) ===")
        print(f"{'Code':<6} | {'Phrase':<50} | {'Len':>4}")
        print("-" * 65)
        for code, phrase in sorted(cb.items()):
            print(f"{code:<6} | {phrase[:50]:<50} | {len(phrase):>4}")

        # Every entry should have phrase longer than code
        for code, phrase in cb.items():
            assert len(phrase) > len(code), f"Code {code} maps to shorter phrase '{phrase}'"
