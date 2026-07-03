# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""MCP version-mismatch warning tests: the ``camas_list``/``camas_docs`` output prepends a
warning when the running camas version does not satisfy the PEP 723 pin in ``tasks.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from camas.main.state import LoadOk
from camas.mcp.serve import check_version_pin

if TYPE_CHECKING:
	from pathlib import Path

	import pytest


def test_no_warning_when_no_pin(tmp_path: Path) -> None:
	"""No PEP 723 ``camas`` block means no warning."""
	(tmp_path / "tasks.py").write_text("from camas import Task\nlint = Task('echo hi')\n")
	from camas.mcp.serve import resolve_project

	state = resolve_project(tmp_path)
	assert isinstance(state, LoadOk)
	assert check_version_pin(state) is None


def test_no_warning_when_satisfied(tmp_path: Path) -> None:
	"""A satisfied ``==`` pin (running version matches) means no warning."""
	from importlib.metadata import version as _camas_version

	running = _camas_version("camas")
	(tmp_path / "tasks.py").write_text(
		f'# /// script\n# dependencies = ["camas=={running}"]\n# ///\n'
		"from camas import Task\nlint = Task('echo hi')\n"
	)
	from camas.mcp.serve import resolve_project

	state = resolve_project(tmp_path)
	assert isinstance(state, LoadOk)
	assert check_version_pin(state) is None


def test_mismatch_produces_warning(tmp_path: Path) -> None:
	"""A pin that does not match the running version produces a warning string."""
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas==999.0.0"]\n# ///\n'
		"from camas import Task\nlint = Task('echo hi')\n"
	)
	from camas.mcp.serve import resolve_project

	state = resolve_project(tmp_path)
	assert isinstance(state, LoadOk)
	assert check_version_pin(state) is not None
	assert "does not satisfy tasks.py pin" in (check_version_pin(state) or "")


def test_version_satisfies_equal() -> None:
	from camas.mcp.serve import version_satisfies

	assert version_satisfies("0.1.8", "==0.1.8") is True
	assert version_satisfies("0.1.9", "==0.1.8") is False
	assert version_satisfies("0.1.18", "==0.1.18") is True
	assert version_satisfies("1.0.0.dev1", "==1.0.0") is True
	assert version_satisfies("1.0.0rc2", "==1.0.0") is True


def test_version_satisfies_equal_zero_pads_segments() -> None:
	"""PEP 440 release-segment equivalence: ``1.0 == 1.0.0`` either direction."""
	from camas.mcp.serve import version_satisfies

	assert version_satisfies("1.0.0", "==1.0") is True
	assert version_satisfies("1.0", "==1.0.0") is True
	assert version_satisfies("1.1", ">=1.0.0") is True
	assert version_satisfies("1.0.0", ">=1.1") is False


def test_version_satisfies_greater_equal() -> None:
	from camas.mcp.serve import version_satisfies

	assert version_satisfies("0.2.0", ">=0.1.8") is True
	assert version_satisfies("0.1.8", ">=0.1.8") is True
	assert version_satisfies("0.1.7", ">=0.1.8") is False


def test_version_satisfies_greater() -> None:
	from camas.mcp.serve import version_satisfies

	assert version_satisfies("0.2.0", ">0.1.8") is True
	assert version_satisfies("0.1.8", ">0.1.8") is False


def test_version_satisfies_less() -> None:
	from camas.mcp.serve import version_satisfies

	assert version_satisfies("0.1.7", "<0.1.8") is True
	assert version_satisfies("0.1.8", "<0.1.8") is False
	assert version_satisfies("0.1.9", "<0.1.8") is False


def test_version_satisfies_unrecognized_skips() -> None:
	"""Unrecognized specifiers (``~=``, compound) return ``True`` (skip) to avoid false positives."""
	from camas.mcp.serve import version_satisfies

	assert version_satisfies("0.1.8", "~=0.1") is True  # unrecognized — skip
	assert version_satisfies("0.1.8", ">=0.1,<0.2") is True  # compound — skip (prefix checks >=)


def test_version_satisfies_unparseable_skips() -> None:
	"""A version string that fails to parse returns ``True`` (skip)."""
	from camas.mcp.serve import version_satisfies

	assert version_satisfies("0.1.8", "==not.a.version") is True  # can't parse pinned
	assert version_satisfies("not.a.version", "==0.1.8") is True  # can't parse running — skip


def test_check_version_pin_no_source() -> None:
	"""An EmptyState (no tasks.py) returns None — no warning."""
	from camas.main.state import EMPTY_STATE

	assert check_version_pin(EMPTY_STATE) is None


def test_version_satisfies_less_equal() -> None:
	from camas.mcp.serve import version_satisfies

	assert version_satisfies("0.1.8", "<=0.1.8") is True
	assert version_satisfies("0.1.9", "<=0.1.8") is False
	assert version_satisfies("0.1.7", "<=0.1.8") is True


def test_check_version_pin_bare_camas_no_spec(tmp_path: Path) -> None:
	"""A tasks.py with a bare ``camas`` dependency (no version specifier) produces no warning."""
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas"]\n# ///\n'
		"from camas import Task\nlint = Task('echo hi')\n"
	)
	from camas.mcp.serve import resolve_project

	state = resolve_project(tmp_path)
	assert isinstance(state, LoadOk)
	assert check_version_pin(state) is None


def test_check_version_pin_missing_distribution_returns_none(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""If camas isn't installed as a distribution metadata entry, degrade to no warning."""
	(tmp_path / "tasks.py").write_text(
		'# /// script\n# dependencies = ["camas==999.0.0"]\n# ///\n'
		"from camas import Task\nlint = Task('echo hi')\n"
	)
	from importlib.metadata import PackageNotFoundError

	from camas.mcp import serve as serve_mod
	from camas.mcp.serve import resolve_project

	state = resolve_project(tmp_path)
	assert isinstance(state, LoadOk)

	def _raise(name: str) -> str:
		raise PackageNotFoundError(name)

	monkeypatch.setattr(serve_mod, "version", _raise)
	assert check_version_pin(state) is None
