# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence
	from pathlib import Path


def test_entrypoint_mcp_branch_invokes_serve(monkeypatch: pytest.MonkeyPatch) -> None:
	from camas.main import main
	from camas.mcp import serve

	calls: list[list[str]] = []
	monkeypatch.setattr(serve, "serve_stdio", calls.append)
	monkeypatch.setattr("sys.argv", ["camas", "mcp", "--rich"])
	main()
	assert calls == [["--rich"]]


def test_entrypoint_mcp_branch_missing_extra_exits_with_feature_hint(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	from camas.main import main

	original_import = __import__

	def fake_import(
		name: str,
		globals: Mapping[str, object] | None = None,
		locals: Mapping[str, object] | None = None,
		fromlist: Sequence[str] = (),
		level: int = 0,
	) -> object:
		if name == "mcp":
			raise ModuleNotFoundError("No module named 'mcp'", name="mcp")
		return original_import(name, globals, locals, fromlist, level)

	for module in ("camas.mcp.serve", "mcp"):
		monkeypatch.delitem(sys.modules, module, raising=False)
	monkeypatch.setattr("sys.argv", ["camas", "mcp"])
	with patch("builtins.__import__", fake_import), pytest.raises(SystemExit) as exc:
		main()

	assert exc.value.code == 2
	err = capsys.readouterr().err
	assert "camas mcp: requires feature camas[mcp]" in err
	assert "pip" not in err
	assert "uv" not in err


def test_entrypoint_mcp_branch_diagnoses_non_mcp_missing_import(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""A broken dependency other than the optional ``mcp`` extra (e.g. a shadowed
	``pydantic_core`` under a leaked ``PYTHONPATH``) must not crash with a raw traceback —
	Claude Code swallows it and the agent silently loses every camas tool. It gets one
	actionable diagnostic on stderr and exit 2 instead."""
	from camas.main import main

	original_import = __import__

	def fake_import(
		name: str,
		globals: Mapping[str, object] | None = None,
		locals: Mapping[str, object] | None = None,
		fromlist: Sequence[str] = (),
		level: int = 0,
	) -> object:
		if name == "pydantic":
			raise ModuleNotFoundError("No module named 'pydantic'", name="pydantic")
		return original_import(name, globals, locals, fromlist, level)

	for module in ("camas.mcp.serve", "pydantic"):
		monkeypatch.delitem(sys.modules, module, raising=False)
	monkeypatch.setattr("sys.argv", ["camas", "mcp"])
	with patch("builtins.__import__", fake_import), pytest.raises(SystemExit) as exc:
		main()

	assert exc.value.code == 2
	err = capsys.readouterr().err
	assert "pydantic" in err
	assert sys.executable in err
	assert "not a missing camas[mcp] extra" in err


def test_entrypoint_mcp_gate_branch_invokes_gate_cli(monkeypatch: pytest.MonkeyPatch) -> None:
	from camas.main import main
	from camas.mcp import serve

	calls: list[list[str]] = []

	def fake_gate_cli(argv: list[str]) -> int:
		calls.append(argv)
		return 0

	monkeypatch.setattr(serve, "gate_cli", fake_gate_cli)
	monkeypatch.setattr("sys.argv", ["camas", "mcp", "gate", "--paths", "a.py"])
	with pytest.raises(SystemExit) as exc:
		main()
	assert exc.value.code == 0
	assert calls == [["--paths", "a.py"]]


def test_entrypoint_mcp_fix_branch_invokes_fix_cli(monkeypatch: pytest.MonkeyPatch) -> None:
	from camas.main import dispatch, main

	calls: list[list[str]] = []

	def fake_fix_cli(argv: list[str]) -> int:
		calls.append(argv)
		return 0

	monkeypatch.setattr(dispatch, "fix_cli", fake_fix_cli)
	monkeypatch.setattr("sys.argv", ["camas", "mcp", "fix", "--paths", "a.py"])
	with pytest.raises(SystemExit) as exc:
		main()
	assert exc.value.code == 0
	assert calls == [["--paths", "a.py"]]


def test_entrypoint_mcp_gate_branch_missing_extra_exits_with_feature_hint(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	from camas.main import main

	original_import = __import__

	def fake_import(
		name: str,
		globals: Mapping[str, object] | None = None,
		locals: Mapping[str, object] | None = None,
		fromlist: Sequence[str] = (),
		level: int = 0,
	) -> object:
		if name == "mcp":
			raise ModuleNotFoundError("No module named 'mcp'", name="mcp")
		return original_import(name, globals, locals, fromlist, level)

	for module in ("camas.mcp.serve", "mcp"):
		monkeypatch.delitem(sys.modules, module, raising=False)
	monkeypatch.setattr("sys.argv", ["camas", "mcp", "gate"])
	with patch("builtins.__import__", fake_import), pytest.raises(SystemExit) as exc:
		main()

	assert exc.value.code == 2
	assert "camas mcp gate: requires feature camas[mcp]" in capsys.readouterr().err


def test_entrypoint_mcp_gate_branch_diagnoses_non_mcp_missing_import(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""Same diagnostic-and-exit-2 behavior as the plain ``camas mcp`` branch, for ``gate``."""
	from camas.main import main

	original_import = __import__

	def fake_import(
		name: str,
		globals: Mapping[str, object] | None = None,
		locals: Mapping[str, object] | None = None,
		fromlist: Sequence[str] = (),
		level: int = 0,
	) -> object:
		if name == "pydantic":
			raise ModuleNotFoundError("No module named 'pydantic'", name="pydantic")
		return original_import(name, globals, locals, fromlist, level)

	for module in ("camas.mcp.serve", "pydantic"):
		monkeypatch.delitem(sys.modules, module, raising=False)
	monkeypatch.setattr("sys.argv", ["camas", "mcp", "gate"])
	with patch("builtins.__import__", fake_import), pytest.raises(SystemExit) as exc:
		main()

	assert exc.value.code == 2
	err = capsys.readouterr().err
	assert "pydantic" in err
	assert "not a missing camas[mcp] extra" in err


def test_camas_list_does_not_import_mcp_stack(tmp_path: Path) -> None:
	"""The MCP server's heavy deps must never load on the ``camas <task>`` hot path."""
	from textwrap import dedent

	script = dedent("""
		import sys
		sys.argv = ['camas', '--list']
		from camas.main import main
		try:
			main()
		except SystemExit:
			pass
		heavy = {'mcp', 'pydantic', 'pydantic_core', 'starlette', 'uvicorn', 'anyio'}
		leaked = sorted(m for m in sys.modules if m.split('.')[0] in heavy)
		assert not leaked, leaked
	""")
	result = subprocess.run(
		[sys.executable, "-c", script],
		cwd=tmp_path,
		capture_output=True,
		text=True,
		check=False,
	)
	assert result.returncode == 0, result.stderr


def test_import_failure_diagnostic_names_module_and_executable() -> None:
	from camas.mcp.cli import import_failure_diagnostic

	exc = ModuleNotFoundError("No module named 'pydantic_core'", name="pydantic_core")
	text = import_failure_diagnostic(exc, "/usr/bin/python3.14", None)
	assert "pydantic_core" in text
	assert "/usr/bin/python3.14" in text
	assert "PYTHONPATH" not in text


def test_import_failure_diagnostic_falls_back_to_str_without_name() -> None:
	from camas.mcp.cli import import_failure_diagnostic

	text = import_failure_diagnostic(ImportError("cannot import name 'X' from 'Y'"), "py", None)
	assert "cannot import name 'X' from 'Y'" in text


def test_import_failure_diagnostic_flags_leaked_pythonpath() -> None:
	from camas.mcp.cli import import_failure_diagnostic

	exc = ModuleNotFoundError("No module named 'x'", name="x")
	text = import_failure_diagnostic(exc, "py", "/nix/store/leaked-closure")
	assert "/nix/store/leaked-closure" in text
	assert "unset PYTHONPATH" in text
	assert "mkShell" in text


def test_report_import_failure_prints_feature_hint_for_missing_mcp(
	capsys: pytest.CaptureFixture[str],
) -> None:
	from camas.mcp.cli import report_import_failure

	with pytest.raises(SystemExit) as exc:
		report_import_failure(
			ModuleNotFoundError("No module named 'mcp'", name="mcp"),
			feature_hint="camas mcp: requires feature camas[mcp]",
		)
	assert exc.value.code == 2
	assert capsys.readouterr().err == "camas mcp: requires feature camas[mcp]\n"


def test_report_import_failure_prints_full_diagnostic_for_other_import_errors(
	capsys: pytest.CaptureFixture[str],
) -> None:
	from camas.mcp.cli import report_import_failure

	with pytest.raises(SystemExit) as exc:
		report_import_failure(
			ModuleNotFoundError("No module named 'pydantic_core'", name="pydantic_core"),
			feature_hint="camas mcp: requires feature camas[mcp]",
		)
	assert exc.value.code == 2
	err = capsys.readouterr().err
	assert "pydantic_core" in err
	assert "requires feature camas[mcp]" not in err


def test_broken_dependency_import_prints_diagnostic_and_exits_2(tmp_path: Path) -> None:
	"""End-to-end reproduction of the field report (issue #169): a nix ``mkShell`` leaking
	``PYTHONPATH`` shadows ``pydantic_core`` with a broken module of the same name. Before this
	fix, ``camas mcp`` died with a raw traceback that Claude Code swallows, so the agent lost
	every camas tool with no visible error."""
	shadow = tmp_path / "shadow"
	shadow.mkdir()
	(shadow / "pydantic_core.py").write_text(
		"raise ModuleNotFoundError("
		"\"No module named 'pydantic_core._pydantic_core'\", name='pydantic_core._pydantic_core')\n"
	)
	inherited = os.environ.get("PYTHONPATH")
	shadowed = str(shadow) if inherited is None else f"{shadow}{os.pathsep}{inherited}"
	result = subprocess.run(
		[sys.executable, "-m", "camas", "mcp"],
		capture_output=True,
		text=True,
		env={**os.environ, "PYTHONPATH": shadowed},
		check=False,
	)
	assert result.returncode == 2
	assert "pydantic_core._pydantic_core" in result.stderr
	assert sys.executable in result.stderr
	assert "unset PYTHONPATH" in result.stderr
	assert "Traceback" not in result.stderr
