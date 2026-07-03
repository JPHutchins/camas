# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path

import pytest

from camas.main.pep723 import (
	camas_requirement_from,
	parse_camas_requirement,
	version_specifier,
	with_mcp_extra,
)

PEP723_BLOCK = """# /// script
# requires-python = ">=3.10"
# dependencies = ["camas>=0.1.8", "pytest"]
# ///"""


def test_parse_pinned_camas() -> None:
	assert parse_camas_requirement(PEP723_BLOCK) == "camas>=0.1.8"


def test_parse_camas_with_mcp_extra() -> None:
	source = """# /// script
# dependencies = ["camas[mcp]==0.1.18", "pytest"]
# ///"""
	assert parse_camas_requirement(source) == "camas[mcp]==0.1.18"


def test_parse_camas_with_multi_extras() -> None:
	source = """# /// script
# dependencies = ["camas[test,check]>=0.1"]
# ///"""
	assert parse_camas_requirement(source) == "camas[test,check]>=0.1"


def test_parse_bare_camas() -> None:
	source = """# /// script
# dependencies = ["camas"]
# ///"""
	assert parse_camas_requirement(source) == "camas"


def test_parse_no_camas_in_deps() -> None:
	source = """# /// script
# dependencies = ["other-pkg", "pytest"]
# ///"""
	assert parse_camas_requirement(source) is None


def test_parse_no_block() -> None:
	assert parse_camas_requirement("print('hello')") is None


def test_parse_no_dependencies_key() -> None:
	source = """# /// script
# requires-python = ">=3.10"
# ///"""
	assert parse_camas_requirement(source) is None


def test_parse_malformed_toml() -> None:
	source = """# /// script
# dependencies = [not valid toml
# ///"""
	assert parse_camas_requirement(source) is None


def test_parse_empty_block() -> None:
	source = """# /// script
# ///"""
	assert parse_camas_requirement(source) is None


def test_parse_rejects_camas_prefix_name() -> None:
	"""`camas-foo` is a different distribution — must not match."""
	source = """# /// script
# dependencies = ["camas-foo>=1.0"]
# ///"""
	assert parse_camas_requirement(source) is None


def test_parse_rejects_camas_suffix_name() -> None:
	"""`my-camas` is a different distribution — must not match."""
	source = """# /// script
# dependencies = ["my-camas>=1.0"]
# ///"""
	assert parse_camas_requirement(source) is None


def test_parse_case_insensitive_name() -> None:
	source = """# /// script
# dependencies = ["CAMAS>=0.1.8"]
# ///"""
	assert parse_camas_requirement(source) == "CAMAS>=0.1.8"


def test_with_mcp_extra_adds_to_no_extras() -> None:
	assert with_mcp_extra("camas>=0.1.8") == "camas[mcp]>=0.1.8"


def test_with_mcp_extra_preserves_existing_mcp() -> None:
	assert with_mcp_extra("camas[mcp]==1") == "camas[mcp]==1"


def test_with_mcp_extra_merges_mcp() -> None:
	assert with_mcp_extra("camas[test]==1") == "camas[test,mcp]==1"


def test_with_mcp_extra_bare_camas() -> None:
	assert with_mcp_extra("camas") == "camas[mcp]"


def test_with_mcp_extra_multiple_extras() -> None:
	assert with_mcp_extra("camas[check,test]>=0.1") == "camas[check,test,mcp]>=0.1"


def test_with_mcp_extra_rejects_non_camas() -> None:
	with pytest.raises(ValueError, match="Not a camas requirement"):
		with_mcp_extra("other-pkg>=1.0")


def test_version_specifier_pinned() -> None:
	assert version_specifier("camas[mcp]>=0.1.8") == ">=0.1.8"


def test_version_specifier_equals() -> None:
	assert version_specifier("camas[mcp]==0.1.18") == "==0.1.18"


def test_version_specifier_bare_camas_is_none() -> None:
	assert version_specifier("camas") is None


def test_version_specifier_extras_no_spec_is_none() -> None:
	assert version_specifier("camas[test]") is None


def test_version_specifier_rejects_non_camas() -> None:
	with pytest.raises(ValueError, match="Not a camas requirement"):
		version_specifier("other-pkg>=1.0")


def test_camas_requirement_from_file(tmp_path: Path) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text(PEP723_BLOCK)
	assert camas_requirement_from(tasks_py) == "camas>=0.1.8"


def test_camas_requirement_from_no_file() -> None:
	assert camas_requirement_from(Path("/nonexistent/tasks.py")) is None


def test_camas_requirement_from_no_block(tmp_path: Path) -> None:
	tasks_py = tmp_path / "tasks.py"
	tasks_py.write_text("print('hello')")
	assert camas_requirement_from(tasks_py) is None


def test_parse_bare_hash_line_in_block() -> None:
	"""A block line starting with ``#`` but not ``# `` (no space) still gets its content stripped."""
	source = """# /// script
#dependencies = ["camas>=0.1.8"]
# ///"""
	assert parse_camas_requirement(source) == "camas>=0.1.8"


def test_parse_block_with_non_hash_line() -> None:
	"""A block with a bare line (no leading ``#``) — the non-comment line is ignored and the
	hash-prefixed lines are still parsed."""
	source = """# /// script
#dependencies = ["camas>=0.1.8"]
bare line without hash
# ///"""
	assert parse_camas_requirement(source) == "camas>=0.1.8"
