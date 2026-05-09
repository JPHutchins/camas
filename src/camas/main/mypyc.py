# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins
"""AST-based fallback for class signature inspection under mypyc compilation.

mypyc compiles modules to ``.so`` and erases function ``__annotations__`` /
mangles class-level ones, so ``inspect`` and ``typing.get_type_hints`` lose
type information. ``mypycify`` ships the original ``.py`` next to the
compiled artifact, and these helpers parse it directly to recover the
signatures the help formatter needs.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

MISSING: Final = object()


def signature_fields_from_source(cls: Any) -> list[tuple[str, Any, Any, Any]] | None:
	"""AST-parse the ``.py`` source defining ``cls`` and return the same shape
	as ``effects.signature_fields``. Returns ``None`` when the source isn't reachable.

	mypyc preserves the original ``.py`` next to the compiled ``.so``; we
	locate it via the module's ``__file__`` directory plus the class's
	module name. Annotations and defaults are evaluated in the module's
	namespace so type references resolve to real classes (enabling the
	signature renderer's recursion into nested types).
	"""
	import inspect
	import sys as _sys

	mod = _sys.modules.get(cls.__module__)
	mod_file = getattr(mod, "__file__", None) if mod is not None else None
	if mod is None or not isinstance(mod_file, str):
		return None
	module_dir = Path(mod_file).parent
	py_path = module_dir / f"{cls.__module__.rsplit('.', 1)[-1]}.py"
	if not py_path.is_file():
		return None
	try:
		tree = ast.parse(py_path.read_text(encoding="utf-8"))
	except (OSError, SyntaxError):
		return None
	class_node = next(
		(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__),
		None,
	)
	if class_node is None:
		return None
	ns = vars(mod)
	if hasattr(cls, "_fields"):
		return [
			(
				stmt.target.id,
				inspect.Parameter.POSITIONAL_OR_KEYWORD,
				resolve_in_namespace(stmt.annotation, ns),
				resolve_in_namespace(stmt.value, ns) if stmt.value is not None else MISSING,
			)
			for stmt in class_node.body
			if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
		]
	init_node = next(
		(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
		None,
	)
	if init_node is None:
		return None
	args = init_node.args.args[1:]  # drop ``self``
	defaults = init_node.args.defaults
	required = len(args) - len(defaults)
	return [
		(
			arg.arg,
			inspect.Parameter.POSITIONAL_OR_KEYWORD,
			resolve_in_namespace(arg.annotation, ns) if arg.annotation else Any,
			MISSING if i < required else resolve_in_namespace(defaults[i - required], ns),
		)
		for i, arg in enumerate(args)
	]


def resolve_in_namespace(node: ast.expr, ns: Mapping[str, Any]) -> Any:
	"""Evaluate an AST expression in ``ns`` (the source module's globals).
	Falls back to the unparsed string when evaluation fails — typically a
	forward reference to a name not yet bound when the source is read."""
	src = ast.unparse(node)
	try:
		return eval(src, dict(ns))  # noqa: S307 — evaluating our own annotations
	except Exception:  # noqa: BLE001
		return src
