# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from camas.main.parser import build_parser, resolve_jobs

if TYPE_CHECKING:
	from collections.abc import Mapping


def test_parser_has_expression_arg() -> None:
	parser = build_parser()
	args = parser.parse_args(['Task("echo hi")'])
	assert args.expression == 'Task("echo hi")'
	assert args.dry_run is False


def test_parser_dry_run_flag() -> None:
	parser = build_parser()
	args = parser.parse_args(["--dry-run", 'Task("echo hi")'])
	assert args.dry_run is True


def test_parser_jobs_flag() -> None:
	assert build_parser().parse_args(["--jobs", "4", "x"]).jobs == 4


def test_parser_jobs_defaults_none() -> None:
	assert build_parser().parse_args(["x"]).jobs is None


@pytest.mark.parametrize("bad", ["0", "-2", "abc"])
def test_parser_jobs_rejects_invalid(bad: str) -> None:
	with pytest.raises(SystemExit):
		build_parser().parse_args(["--jobs", bad, "x"])


def test_resolve_jobs_cli_wins(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CAMAS_JOBS", "2")
	assert resolve_jobs(8) == 8


def test_resolve_jobs_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CAMAS_JOBS", "3")
	assert resolve_jobs(None) == 3


def test_resolve_jobs_unset(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("CAMAS_JOBS", raising=False)
	assert resolve_jobs(None) is None


def test_resolve_jobs_bad_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CAMAS_JOBS", "nope")
	with pytest.raises(ValueError, match="CAMAS_JOBS expects"):
		resolve_jobs(None)


def test_resolve_jobs_nonpositive_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("CAMAS_JOBS", "0")
	with pytest.raises(ValueError, match=">= 1"):
		resolve_jobs(None)


def test_build_parser_format_help_no_tasks_no_effects(monkeypatch: pytest.MonkeyPatch) -> None:
	from typing import Any

	from camas.main import effects as effects_mod

	def empty_discover() -> tuple[Mapping[str, Any], tuple[tuple[str, Any], ...]]:
		return {}, ()

	# Patch ``discover_effects`` (the ``functools.cache``-wrapped Python
	# callable) rather than ``available_effects`` so the patch survives
	# under mypyc compilation, where compiled-to-compiled calls bypass
	# the module dict.
	monkeypatch.setattr(effects_mod, "discover_effects", empty_discover)
	parser = build_parser()
	out = parser.format_help()
	assert "Available tasks" not in out
	assert "Available Effects" not in out
	assert "Try:" in out
