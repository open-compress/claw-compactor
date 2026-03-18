"""Shared test fixtures for claw-compactor tests."""

import os
import sys
import pytest
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with sample memory files."""
    memory = tmp_path / "MEMORY.md"
    memory.write_text(
        "# Memory\n\n"
        "## Decisions\n"
        "- Use Python 3.10+\n"
        "- Deploy on AWS\n\n"
        "## Notes\n"
        "- Meeting with team on Monday\n"
        "- Review PR #42\n",
        encoding="utf-8",
    )
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "2025-01-15.md").write_text(
        "# 2025-01-15\n\n## Done\n- Fixed bug in parser\n- Deployed v2.1\n\n## Learned\n- tiktoken is fast\n",
        encoding="utf-8",
    )
    (mem_dir / "2025-01-16.md").write_text(
        "# 2025-01-16\n\n## Done\n- Code review for PR #43\n- Updated docs\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def empty_file(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("", encoding="utf-8")
    return f


@pytest.fixture
def unicode_file(tmp_path):
    f = tmp_path / "unicode.md"
    f.write_text(
        "# 记忆笔记\n\n## 决策 Decisions\n- 使用 Python 3.10+\n- Deploy on AWS 东京区域\n- 日本語テスト\n\n## 备注 Notes\n- emoji test: 🎉🔥💡\n- Ñoño señor café\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def large_file(tmp_path):
    f = tmp_path / "large.md"
    lines = ["# Large Memory File\n"]
    for i in range(2000):
        lines.append("## Section {}\n- Item {}: This is entry number {} with some filler content to make it realistic.\n- Detail: The value is approximately {:.2f}\n\n".format(i, i, i, i * 3.14))
    content = ''.join(lines)
    assert len(content) > 100000
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def broken_markdown(tmp_path):
    f = tmp_path / "broken.md"
    f.write_text("# Unclosed header\n##No space after hash\n### \n- \n- - nested dash\n```\nunclosed code block\nnormal text\n########## too many hashes\n", encoding="utf-8")
    return f


@pytest.fixture
def headers_only(tmp_path):
    f = tmp_path / "headers_only.md"
    f.write_text("# Title\n## Section 1\n## Section 2\n### Subsection\n", encoding="utf-8")
    return f


@pytest.fixture
def single_line(tmp_path):
    f = tmp_path / "single.md"
    f.write_text("Just one line.", encoding="utf-8")
    return f


@pytest.fixture
def duplicate_content(tmp_path):
    d = tmp_path / "dupes"
    d.mkdir()
    (d / "a.md").write_text("# Notes\n\n## Setup\n- Install Python 3.10\n- Run pip install requirements\n- Configure the database connection string\n- Set environment variables for production\n\n", encoding="utf-8")
    (d / "b.md").write_text("# Notes\n\n## Setup Instructions\n- Install Python 3.10\n- Run pip install requirements\n- Configure the database connection string\n- Set environment variables for production deployment\n\n## Other\n- Unique content here\n", encoding="utf-8")
    return d
