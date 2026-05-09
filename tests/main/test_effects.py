# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
import inspect
import pkgutil
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pkgutil import ModuleInfo

import pytest

from camas.main.effects import (
	discover_effects,
	eval_value,
	parse_effects,
	reachable_classes,
	signature_fields,
)
from camas.main.mypyc import MISSING


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


def test_discover_effects_returns_summary_termtree() -> None:
	constructors, effects = discover_effects()
	names = {n for n, _ in effects}
	assert names == {"Summary", "Termtree"}
	assert "SummaryOptions" in constructors
	assert "Auto" in constructors
	assert "Fixed" in constructors


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
	setattr(plugin, "my_effect", Summary())
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
	from camas.effect.summary import SummaryOptions

	fields = signature_fields(SummaryOptions)
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
