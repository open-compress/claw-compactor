# Contributing to Claw Compactor

Thanks for your interest in contributing! Claw Compactor is an open-source project and we welcome contributions of all kinds.

## Getting Started

### Prerequisites

- Python 3.9+
- Git

### Setup

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/claw-compactor.git
cd claw-compactor

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode with all extras
pip install -e ".[dev,accurate]"

# Verify everything works
pytest tests/ -x -q
```

## Development Workflow

### 1. Find Something to Work On

- Check [open issues](https://github.com/open-compress/claw-compactor/issues) — look for `good first issue` or `help wanted` labels
- Have an idea? Open an issue first to discuss before investing time

### 2. Create a Branch

```bash
git checkout -b feat/your-feature-name
# or: fix/your-bug-fix, docs/your-doc-update
```

### 3. Make Your Changes

- Follow existing code style and patterns
- Keep changes focused — one feature or fix per PR
- Add tests for new functionality

### 4. Test Your Changes

Run the full test suite:

```bash
pytest tests/ -x -q
```

Run a specific test file:

```bash
pytest tests/test_pipeline.py -v
```

Run tests matching a keyword:

```bash
pytest tests/ -k "compression" -v
```

```bash
# Run the full test suite
pytest tests/ -x -q

# Run a specific test file
pytest tests/test_fusion_engine.py -v

# Run with coverage
pytest tests/ --cov=scripts --cov=claw_compactor --cov-report=term-missing
```

All PRs must pass CI on Python 3.9–3.12. The test suite has 1600+ tests — don't be alarmed, they run fast.

### 5. Submit a PR

1. Push your branch to your fork
2. Open a Pull Request against `main`
3. Fill in the PR template with a clear description
4. Link any related issues

## Code Guidelines

### Architecture

Claw Compactor is built around a 14-stage Fusion Pipeline. Each stage is a self-contained compressor inheriting from `FusionStage`. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

### Key Principles

- **Immutability** — `FusionContext` is frozen. Every stage produces a new `FusionResult`. Never mutate inputs.
- **Gate-before-compress** — Each stage has `should_apply()`. If a stage doesn't apply to the content type, it should be a no-op at zero cost.
- **Zero required dependencies** — The core pipeline runs without any external packages. Optional dependencies (tiktoken, tree-sitter) are runtime-detected.

### Adding a New Fusion Stage

1. Create a new file in `scripts/lib/fusion/stages/`
2. Inherit from `FusionStage`
3. Implement `should_apply()` and `apply()`
4. Register it in the stage registry
5. Add tests covering happy path, edge cases, and the gate condition

```python
from scripts.lib.fusion.base import FusionStage, FusionContext, FusionResult

class MyStage(FusionStage):
    name = "my_stage"
    order = 22  # controls execution order in the pipeline

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_type == "log"

    def apply(self, ctx: FusionContext) -> FusionResult:
        compressed = my_logic(ctx.content)
        return FusionResult(content=compressed, ...)
```

### Style

- Type hints on all public functions
- Docstrings for non-obvious logic
- Functions under 50 lines, files under 800 lines
- No deep nesting (4 levels max)

## Reporting Issues

Please include:

- **Python version** (`python --version`)
- **OS** (macOS, Linux, Windows)
- **Steps to reproduce** — minimal example preferred
- **Expected vs actual behavior**
- **Traceback** if applicable

## Community

- [Discord](https://discord.com/invite/clawd) — ask questions, discuss ideas
- [GitHub Discussions](https://github.com/open-compress/claw-compactor/discussions) — longer-form conversations

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
