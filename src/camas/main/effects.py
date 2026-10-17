# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING, Any

from ..core.timings import (
	Observed,  # noqa: TC001  # runtime name get_type_hints resolves; TYPE_CHECKING-only would NameError
)
from ..v0.effect import Effect
from .expression import format_syntax_error
from .mypyc import MISSING, signature_fields_from_source

if TYPE_CHECKING:
	from collections.abc import Mapping
	from pathlib import Path

	from ..v0.config import Config


@functools.cache
def discover_effects() -> tuple[Mapping[str, Any], tuple[tuple[str, Any], ...]]:
	import importlib
	import inspect
	import pkgutil

	import camas.effect as effect_pkg

	constructors: dict[str, Any] = {}
	effects: list[tuple[str, Any]] = []
	for _, modname, _ in pkgutil.iter_modules(effect_pkg.__path__):
		if modname.startswith("_"):
			continue
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
	from typing import get_args

	args = get_args(annotation)
	if not args:
		yield annotation
		return
	for a in args:
		yield from flatten_annotation(a)


def signature_fields(cls: Any) -> list[tuple[str, Any, Any, Any]]:
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
		sig = inspect.signature(cls, eval_str=True)
	except (ValueError, TypeError, NameError):
		try:
			sig = inspect.signature(cls)
		except (ValueError, TypeError):
			from_src = signature_fields_from_source(cls)
			return from_src if from_src is not None else []
	from_inspect: list[tuple[str, Any, Any, Any]] = [
		(
			p.name,
			p.kind,
			p.annotation if p.annotation is not inspect.Parameter.empty else Any,
			p.default if p.default is not inspect.Parameter.empty else MISSING,
		)
		for p in sig.parameters.values()
	]
	if not from_inspect or all(annot is Any for _, _, annot, _ in from_inspect):
		from_src = signature_fields_from_source(cls)
		if from_src is not None:
			return from_src
	return from_inspect


def available_effects(
	scope_effects: Mapping[str, type[Effect[Any]]] = {},
) -> tuple[Mapping[str, Any], tuple[tuple[str, Any], ...]]:
	builtin_constructors, builtin_effects = discover_effects()
	if not scope_effects:
		return builtin_constructors, builtin_effects
	return (
		{**builtin_constructors, **scope_effects},
		builtin_effects + tuple(scope_effects.items()),
	)


def parse_effects(
	expr: str, scope_effects: Mapping[str, type[Effect[Any]]] = {}
) -> tuple[Effect[Any], ...]:
	"""
	Raises:
		ValueError: on syntax errors, non-tuple expressions, unknown names,
			or elements that are not Effect instances.

	>>> [type(e).__name__ for e in parse_effects("(Summary(),)")]
	['Summary']
	>>> parse_effects("(Summary(term_width=Fixed(80)),)")[0]._term_width
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
	constructors, _ = available_effects(scope_effects)
	return tuple(expect_effect(eval_value(elt, constructors)) for elt in tree.body.elts)


def eval_value(node: ast.expr, constructors: Mapping[str, Any]) -> Any:
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


def running_under_agent() -> bool:
	import os

	return os.environ.get("CLAUDECODE") == "1" or bool(os.environ.get("CAMAS_AGENT"))


def resolve_default_effects(
	config: Config,
	*,
	github: bool,
	agent: bool = False,
	base: Path | None = None,
) -> tuple[Effect[Any], ...]:
	configured = config.effects(github=github)
	if configured is not None:
		return configured
	from ..effect.status import Status
	from ..effect.termtree import Termtree

	if github:
		return (Status(output_mode="github"),)
	renderer: Effect[Any] = Status() if agent else Termtree()
	camas = config.camas_path(base) if base is not None else None
	if camas is not None and camas.is_dir():
		from ..effect.timings import Timings

		return (renderer, Timings(camas_dir=camas))
	return (renderer,)


def default_effect_names(config: Config, *, github: bool, agent: bool = False) -> frozenset[str]:
	return frozenset(
		type(e).__name__ for e in resolve_default_effects(config, github=github, agent=agent)
	)


def resolve_effects(
	expr: str | None,
	config: Config,
	*,
	github: bool,
	agent: bool = False,
	scope_effects: Mapping[str, type[Effect[Any]]] = {},
	base: Path | None = None,
) -> tuple[Effect[Any], ...]:
	if expr is None:
		return resolve_default_effects(config, github=github, agent=agent, base=base)
	return parse_effects(expr, scope_effects)


def keyed_to_run(effects: tuple[Effect[Any], ...], observed: Observed) -> tuple[Effect[Any], ...]:
	from ..effect.timings import Timings

	if not any(isinstance(e, Timings) for e in effects):
		return effects
	return tuple(
		e.for_run(observed.scope, observed.identities) if isinstance(e, Timings) else e
		for e in effects
	)


def format_effect_call(effect: Effect[Any]) -> str:
	"""
	>>> from camas.effect.status import Status
	>>> from camas.effect.termtree import Termtree
	>>> format_effect_call(Termtree())
	'Termtree()'
	>>> format_effect_call(Status(output_mode="github"))
	"Status(output_mode='github')"
	"""
	cls = type(effect)
	args = ", ".join(
		f"{name}={value!r}"
		for name, _kind, _annotation, default in signature_fields(cls)
		if (value := getattr(effect, f"_{name}", MISSING)) is not MISSING and value != default
	)
	return f"{cls.__name__}({args})"


def format_effects_expr(effects: tuple[Effect[Any], ...]) -> str:
	"""
	>>> from camas.effect.status import Status
	>>> from camas.effect.termtree import Termtree
	>>> format_effects_expr(())
	'()'
	>>> format_effects_expr((Termtree(),))
	'(Termtree(),)'
	>>> format_effects_expr((Termtree(), Status(output_mode="github")))
	"(Termtree(), Status(output_mode='github'))"
	"""
	calls = [format_effect_call(e) for e in effects]
	if len(calls) == 1:
		return f"({calls[0]},)"
	return f"({', '.join(calls)})"
