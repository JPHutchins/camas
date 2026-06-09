# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING

from camas.main.parser import build_parser

if TYPE_CHECKING:
	from collections.abc import Mapping

	import pytest


def test_parser_has_expression_arg() -> None:
	parser = build_parser()
	args = parser.parse_args(['Task("echo hi")'])
	assert args.expression == 'Task("echo hi")'
	assert args.dry_run is False


def test_parser_dry_run_flag() -> None:
	parser = build_parser()
	args = parser.parse_args(["--dry-run", 'Task("echo hi")'])
	assert args.dry_run is True


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
