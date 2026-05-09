# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
import functools
from collections.abc import Mapping
from typing import Any

from ..core.effect import Effect
from .expression import format_syntax_error
from .mypyc import MISSING, signature_fields_from_source


@functools.cache
def discover_effects() -> tuple[Mapping[str, Any], tuple[tuple[str, Any], ...]]:
	"""Walk every public binding in ``camas.effect.*`` and return
	``(constructors, effects)``.

	``constructors`` is every concrete class transitively referenced through
	Effect constructor signatures — what the ``--effects`` mini-language is
	allowed to call. ``effects`` is the (name, class) pairs that satisfy the
	``Effect`` protocol.

	Imports are scoped so ``camas <task>`` doesn't pay for the inspection
	unless something asks for the listing. Result is cached.
	"""
	import importlib
	import inspect
	import pkgutil

	import camas.effect as effect_pkg

	constructors: dict[str, Any] = {}
	effects: list[tuple[str, Any]] = []
	for _, modname, _ in pkgutil.iter_modules(effect_pkg.__path__):
		mod = importlib.import_module(f"{effect_pkg.__name__}.{modname}")
		for name, obj in vars(mod).items():
			if name.startswith("_") or obj is Effect:
				continue
			if inspect.isclass(obj) and obj.__module__ == mod.__name__:
				constructors[name] = obj
				if issubclass(obj, Effect):
					effects.append((name, obj))  # pyright: ignore[reportUnknownArgumentType]
			elif isinstance(obj, Effect):
				constructors[name] = obj
				effects.append((name, obj))  # pyright: ignore[reportUnknownArgumentType]

	for _, obj in effects:
		if inspect.isclass(obj):
			for ref in reachable_classes(obj):
				constructors.setdefault(ref.__name__, ref)
	return constructors, tuple(effects)


def reachable_classes(cls: Any, seen: set[Any] | None = None) -> set[Any]:
	"""Collect every class transitively reachable through ``cls``'s init signature."""
	seen = seen if seen is not None else set()
	if cls in seen:
		return seen
	seen.add(cls)
	for _, _, annotation, _ in signature_fields(cls):
		for leaf in flatten_annotation(annotation):
			if isinstance(leaf, type) and getattr(leaf, "__module__", "").startswith(
				"camas.effect"
			):
				reachable_classes(leaf, seen)
	return seen


def flatten_annotation(annotation: Any) -> Any:
	"""Yield every leaf type from a possibly-generic/union annotation."""
	from typing import get_args

	args = get_args(annotation)
	if not args:
		yield annotation
		return
	for a in args:
		yield from flatten_annotation(a)


def signature_fields(cls: Any) -> list[tuple[str, Any, Any, Any]]:
	"""Return ``[(name, kind, annotation, default), ...]`` for ``cls``'s constructor.

	Handles NamedTuple, dataclass, and ordinary classes uniformly. ``kind`` is
	``inspect.Parameter.kind`` so callers can render ``*args``/``**kwargs``.
	``default`` is ``MISSING`` when the parameter is required.

	Falls back to AST-parsing the module's ``.py`` source when ``inspect``
	yields no annotations — the case under mypyc compilation, which erases
	function ``__annotations__`` and mangles class-level ones. The ``.py``
	is shipped alongside the ``.so`` by ``mypycify``, so the fallback works
	on installed wheels.
	"""
	import dataclasses
	import inspect
	from typing import get_type_hints

	if hasattr(cls, "_fields"):
		try:
			hints = get_type_hints(cls)
		except (NameError, TypeError):
			hints = {}
		defaults = getattr(cls, "_field_defaults", {})
		fields: list[tuple[str, Any, Any, Any]] = [
			(
				n,
				inspect.Parameter.POSITIONAL_OR_KEYWORD,
				hints.get(n, Any),
				defaults.get(n, MISSING),
			)
			for n in cls._fields
		]
		if any(annot is Any or annot is type for _, _, annot, _ in fields):
			from_src = signature_fields_from_source(cls)
			if from_src is not None:
				return from_src
		return fields
	if dataclasses.is_dataclass(cls):
		return [
			(
				f.name,
				inspect.Parameter.POSITIONAL_OR_KEYWORD,
				f.type,
				f.default if f.default is not dataclasses.MISSING else MISSING,
			)
			for f in dataclasses.fields(cls)
			if f.init
		]
	try:
		sig = inspect.signature(cls)
	except (ValueError, TypeError):
		return []
	from_inspect: list[tuple[str, Any, Any, Any]] = [
		(
			p.name,
			p.kind,
			p.annotation if p.annotation is not inspect.Parameter.empty else Any,
			p.default if p.default is not inspect.Parameter.empty else MISSING,
		)
		for p in sig.parameters.values()
	]
	if from_inspect and all(annot is Any for _, _, annot, _ in from_inspect):
		from_src = signature_fields_from_source(cls)
		if from_src is not None:
			return from_src
	return from_inspect


def parse_effects(expr: str) -> tuple[Effect[Any], ...]:
	"""Parse a ``--effects`` expression into a tuple of Effect instances.

	>>> [type(e).__name__ for e in parse_effects("(Summary(),)")]
	['Summary']
	>>> [type(e).__name__ for e in parse_effects("(Termtree(), Summary())")]
	['Termtree', 'Summary']
	>>> parse_effects("(Summary(options=SummaryOptions(term_width=Fixed(80))),)")[0].options.term_width
	Fixed(columns=80)
	>>> parse_effects("()")
	()
	"""
	try:
		tree = ast.parse(expr, mode="eval")
	except SyntaxError as e:
		raise ValueError(format_syntax_error(expr, e)) from e
	if not isinstance(tree.body, ast.Tuple):
		raise ValueError(f"--effects must be a tuple, got {type(tree.body).__name__}")
	constructors, _ = discover_effects()
	return tuple(expect_effect(eval_value(elt, constructors)) for elt in tree.body.elts)


def eval_value(node: ast.expr, constructors: Mapping[str, Any]) -> Any:
	"""Evaluate a node in the --effects mini-language. Returns Any since the
	concrete type depends on the AST shape; ``expect_effect`` validates at the
	tuple level."""
	match node:
		case ast.Constant(value=val):
			return val
		case ast.Name(id=name) if name in constructors:
			return constructors[name]
		case ast.Call(func=ast.Name(id=name), args=args, keywords=keywords) if name in constructors:
			return constructors[name](
				*(eval_value(a, constructors) for a in args),
				**{
					kw.arg: eval_value(kw.value, constructors)
					for kw in keywords
					if kw.arg is not None
				},
			)
		case _:
			raise ValueError(
				f"unsupported syntax (known: {', '.join(sorted(constructors))}): {ast.dump(node)}"
			)


def expect_effect(value: Any) -> Effect[Any]:
	if not isinstance(value, Effect):
		raise ValueError(f"expected an Effect instance, got {type(value).__name__}")
	return value  # pyright: ignore[reportUnknownVariableType,reportReturnType]
