# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
import inspect
import pkgutil
import sys
import types
from dataclasses import dataclass, field
from pkgutil import ModuleInfo
from typing import TYPE_CHECKING

import pytest

from camas import Config
from camas.effect.status import Status
from camas.effect.summary import Summary
from camas.effect.termtree import Termtree
from camas.main.effects import (
	available_effects,
	default_effect_names,
	discover_effects,
	eval_value,
	format_effect_call,
	format_effects_expr,
	parse_effects,
	reachable_classes,
	resolve_default_effects,
	resolve_effects,
	running_under_agent,
	signature_fields,
)
from camas.main.mypyc import MISSING

if TYPE_CHECKING:
	from collections.abc import Iterator
	from pathlib import Path


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
		parse_effects("(42,)")


def test_discover_effects_returns_builtin_set() -> None:
	constructors, effects = discover_effects()
	names = {n for n, _ in effects}
	assert names == {"Summary", "Termtree", "GitHubChecks", "Status", "Timings"}
	assert "Auto" in constructors
	assert "Fixed" in constructors


def test_parse_effects_constructs_github_checks_with_kwarg() -> None:
	out = parse_effects('(GitHubChecks(name_prefix="pfx/"),)')
	assert len(out) == 1
	from camas.effect.github_checks import GitHubChecks

	assert isinstance(out[0], GitHubChecks)


def test_available_effects_merges_scope_into_builtins() -> None:
	"""User scope_effects appears alongside built-ins in both maps."""
	from camas.effect.summary import Summary

	scope = {"AliasedSummary": Summary}
	constructors, effects = available_effects(scope)
	assert "AliasedSummary" in constructors
	assert constructors["AliasedSummary"] is Summary
	assert ("AliasedSummary", Summary) in effects
	# Built-ins remain present.
	assert "Summary" in constructors
	assert any(name == "Summary" for name, _ in effects)


def test_available_effects_empty_scope_is_passthrough() -> None:
	"""Empty scope returns the same shape as discover_effects with no copying."""
	cons_a, eff_a = available_effects({})
	cons_b, eff_b = discover_effects()
	assert cons_a is cons_b
	assert eff_a is eff_b


def test_parse_effects_uses_scope_effects() -> None:
	"""A name only present in scope_effects is callable from --effects."""
	from camas.effect.summary import Summary

	out = parse_effects("(Aliased(),)", {"Aliased": Summary})
	assert len(out) == 1
	assert isinstance(out[0], Summary)


def test_eval_value_constant_string() -> None:
	node = ast.parse('"hello"', mode="eval").body
	assert eval_value(node, {}) == "hello"


def test_eval_value_bare_name() -> None:
	from camas.effect.summary import Summary

	node = ast.parse("Summary", mode="eval").body
	assert eval_value(node, {"Summary": Summary}) is Summary


def test_isinstance_effect_branch_in_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Cover the plugin path: a module-scope Effect *instance* (not class)."""
	import camas.effect as effect_pkg
	from camas.effect.summary import Summary

	plugin = types.ModuleType("camas.effect._test_plugin")
	setattr(plugin, "my_effect", Summary())  # noqa: B010
	monkeypatch.setitem(sys.modules, "camas.effect._test_plugin", plugin)

	real_iter = pkgutil.iter_modules
	finders = [info.module_finder for info in real_iter(list(effect_pkg.__path__))]

	def fake_iter(path: list[str]) -> Iterator[ModuleInfo]:
		yield from real_iter(path)
		yield ModuleInfo(finders[0], "_test_plugin", False)

	monkeypatch.setattr(pkgutil, "iter_modules", fake_iter)
	discover_effects.cache_clear()
	try:
		_, effects = discover_effects()
	finally:
		discover_effects.cache_clear()
	names = {n for n, _ in effects}
	assert "my_effect" in names


def test_reachable_classes_seen_short_circuit() -> None:
	from camas.effect.summary import Summary

	seen = {Summary}
	out = reachable_classes(Summary, seen)
	assert out is seen


def test_signature_fields_namedtuple_with_defaults() -> None:
	from camas.effect.summary import Summary

	fields = signature_fields(Summary)
	names = [n for n, _, _, _ in fields]
	assert names == ["term_width", "show_passing"]


def test_signature_fields_namedtuple_without_defaults() -> None:
	from camas.effect.summary import Fixed

	fields = signature_fields(Fixed)
	assert [(n, d is MISSING) for n, _, _, d in fields] == [("columns", True)]


def test_signature_fields_no_init() -> None:
	"""builtins.object has no inspectable params for our purposes."""

	class Empty: ...

	assert signature_fields(Empty) == []


def test_signature_fields_via_inspect_signature() -> None:
	"""Ordinary class (not NamedTuple, not dataclass) → inspect.signature path."""

	class C:
		def __init__(self, a: int, b: str = "x") -> None: ...

	fields = signature_fields(C)
	names = [n for n, _, _, _ in fields]
	assert names == ["a", "b"]


def test_signature_fields_var_args() -> None:
	"""``*args``/``**kwargs`` are reported with their kinds."""

	class Tricky:
		def __init__(self, *args: int, **kwargs: int) -> None: ...

	fields = signature_fields(Tricky)
	assert any(name in {"args", "kwargs"} for name, _, _, _ in fields)


def test_signature_fields_dataclass() -> None:
	@dataclass
	class D:
		x: int
		y: str = "hi"
		z: tuple[int, ...] = field(default=(), init=False)

	fields = signature_fields(D)
	names = [n for n, _, _, _ in fields]
	assert names == ["x", "y"]


def test_signature_fields_signature_raises(monkeypatch: pytest.MonkeyPatch) -> None:
	"""inspect.signature raises ValueError for some C-implemented callables."""

	class Plain: ...

	def boom(_obj: object) -> object:
		raise ValueError("no signature")

	monkeypatch.setattr(inspect, "signature", boom)
	assert signature_fields(Plain) == []


def test_format_effect_call_omits_default_args() -> None:
	assert format_effect_call(Termtree()) == "Termtree()"
	assert format_effect_call(Termtree(show_passing=True)) == "Termtree(show_passing=True)"


def test_format_effects_expr_singleton_keeps_trailing_comma() -> None:
	assert format_effects_expr(()) == "()"
	assert format_effects_expr((Termtree(),)) == "(Termtree(),)"
	assert format_effects_expr((Termtree(), Summary())) == "(Termtree(), Summary())"


@pytest.mark.parametrize(
	("github", "expected"),
	[(False, (Termtree(),)), (True, (Status(output_mode="github"),))],
)
def test_resolve_default_effects_builtin_when_unset(
	github: bool, expected: tuple[object, ...]
) -> None:
	"""No override → the engine's environment default."""
	out = resolve_default_effects(Config(), github=github)
	assert [type(e).__name__ for e in out] == [type(e).__name__ for e in expected]


def test_resolve_default_effects_honors_override() -> None:
	override = (Summary(),)
	assert resolve_default_effects(Config(default_effects=override), github=False) is override


def test_resolve_default_effects_empty_override_is_no_effects() -> None:
	"""An explicit ``()`` is honored, not treated as unset."""
	assert resolve_default_effects(Config(default_effects=()), github=False) == ()


def test_resolve_effects_parses_expression_over_default() -> None:
	out = resolve_effects("(Summary(),)", Config(), github=False)
	assert [type(e).__name__ for e in out] == ["Summary"]


def test_resolve_effects_none_uses_config_default() -> None:
	override = (Summary(),)
	assert resolve_effects(None, Config(default_effects=override), github=False) is override


def test_default_adds_timings_with_camas_locally(tmp_path: Path) -> None:
	(tmp_path / ".camas").mkdir()
	out = resolve_default_effects(Config(), github=False, base=tmp_path)
	assert [type(e).__name__ for e in out] == ["Termtree", "Timings"]


def test_default_omits_timings_under_github(tmp_path: Path) -> None:
	(tmp_path / ".camas").mkdir()
	out = resolve_default_effects(Config(), github=True, base=tmp_path)
	assert [type(e).__name__ for e in out] == ["Status"]


def test_default_omits_timings_without_camas(tmp_path: Path) -> None:
	out = resolve_default_effects(Config(), github=False, base=tmp_path)
	assert [type(e).__name__ for e in out] == ["Termtree"]


def test_default_omits_timings_without_base() -> None:
	assert [type(e).__name__ for e in resolve_default_effects(Config(), github=False)] == [
		"Termtree"
	]


def test_resolve_effects_threads_base_to_default(tmp_path: Path) -> None:
	(tmp_path / ".camas").mkdir()
	out = resolve_effects(None, Config(), github=False, base=tmp_path)
	assert any(type(e).__name__ == "Timings" for e in out)


@pytest.mark.parametrize(
	("github", "expected"), [(False, frozenset({"Termtree"})), (True, frozenset({"Status"}))]
)
def test_default_effect_names_tracks_environment(github: bool, expected: frozenset[str]) -> None:
	assert default_effect_names(Config(), github=github) == expected


def test_agent_default_is_status_not_termtree() -> None:
	out = resolve_default_effects(Config(), github=False, agent=True)
	assert [type(e).__name__ for e in out] == ["Status"]


def test_agent_default_adds_timings_with_camas(tmp_path: Path) -> None:
	(tmp_path / ".camas").mkdir()
	out = resolve_default_effects(Config(), github=False, agent=True, base=tmp_path)
	assert [type(e).__name__ for e in out] == ["Status", "Timings"]


def test_github_default_ignores_agent() -> None:
	out = resolve_default_effects(Config(), github=True, agent=True)
	assert [type(e).__name__ for e in out] == ["Status"]


def test_default_effect_names_marks_status_for_agent() -> None:
	assert default_effect_names(Config(), github=False, agent=True) == frozenset({"Status"})


def test_running_under_agent_detects_claudecode(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("CAMAS_AGENT", raising=False)
	monkeypatch.setenv("CLAUDECODE", "1")
	assert running_under_agent() is True


def test_running_under_agent_via_camas_agent(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("CLAUDECODE", raising=False)
	monkeypatch.setenv("CAMAS_AGENT", "1")
	assert running_under_agent() is True


def test_running_under_agent_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.delenv("CLAUDECODE", raising=False)
	monkeypatch.delenv("CAMAS_AGENT", raising=False)
	assert running_under_agent() is False
