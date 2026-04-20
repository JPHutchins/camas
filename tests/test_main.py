# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from camas.main import build_parser, main, parse_effects, parse_expression


def test_help(tmp_path: Path) -> None:
	result = subprocess.run(
		[sys.executable, "-m", "camas", "--help"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
	)
	assert result.returncode == 0
	assert "expression" in result.stdout


def test_version(tmp_path: Path) -> None:
	result = subprocess.run(
		[sys.executable, "-m", "camas", "--version"],
		cwd=tmp_path,
		capture_output=True,
		text=True,
	)
	assert result.returncode == 0
	assert "camas" in result.stdout


def test_no_args() -> None:
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas"]):
			main()


def test_parse_effects_rejects_invalid_syntax() -> None:
	with pytest.raises(ValueError, match="invalid syntax"):
		parse_effects("Summary(unbalanced(")


def test_parse_effects_rejects_non_tuple() -> None:
	with pytest.raises(ValueError, match="must be a tuple"):
		parse_effects("Summary()")


def test_parse_effects_rejects_unknown_effect() -> None:
	with pytest.raises(ValueError, match="unsupported syntax"):
		parse_effects("(Bogus(),)")


def test_parse_effects_rejects_non_effect_value() -> None:
	with pytest.raises(ValueError, match="expected an Effect"):
		parse_effects("(SummaryOptions(),)")


def test_dispatch_rejects_bad_effects(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	monkeypatch.chdir(tmp_path)
	with pytest.raises(SystemExit, match="2"):
		with patch("sys.argv", ["camas", "--effects=nope(", 'Task("echo hi")']):
			main()
	assert "--effects" in capsys.readouterr().err


def test_parser_has_expression_arg() -> None:
	parser = build_parser()
	args = parser.parse_args(['Task("echo hi")'])
	assert args.expression == 'Task("echo hi")'
	assert args.dry_run is False


def test_parser_dry_run_flag() -> None:
	parser = build_parser()
	args = parser.parse_args(["--dry-run", 'Task("echo hi")'])
	assert args.dry_run is True


def test_dry_run_prints_tree(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="0"):
		with patch(
			"sys.argv",
			["camas", "--dry-run", 'Parallel(tasks=(Task("echo a"), Task("echo b")))'],
		):
			main()
	captured = capsys.readouterr()
	assert "echo a" in captured.out
	assert "echo b" in captured.out


def test_dry_run_matrix(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="0"):
		with patch(
			"sys.argv",
			[
				"camas",
				"--dry-run",
				'Parallel(tasks=(Task("test {PY}"),), matrix={"PY": ("3.12", "3.13")})',
			],
		):
			main()
	captured = capsys.readouterr()
	assert "3.12" in captured.out
	assert "3.13" in captured.out


def test_successful_execution() -> None:
	with pytest.raises(SystemExit, match="0"):
		with patch("sys.argv", ["camas", 'Task(("python", "-c", "pass"), name="ok")']):
			main()


def test_failed_execution() -> None:
	with pytest.raises(SystemExit, match="1"):
		with patch(
			"sys.argv", ["camas", 'Task(("python", "-c", "raise SystemExit(1)"), name="fail")']
		):
			main()


def test_invalid_expression() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression("not valid +++")


def test_unknown_type() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Foo(tasks=(Task("a"),))')


def test_bare_string() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('"just a string"')


@pytest.mark.parametrize(
	"expr",
	[
		'os.system("ls")',
		'__import__("os")',
	],
)
def test_rejects_unsafe(expr: str) -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression(expr)


def test_dict_with_non_str_key() -> None:
	with pytest.raises(SystemExit, match="2"):
		parse_expression('Task("x", env={1: "val"})')
