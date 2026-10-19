# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final, NamedTuple

SOURCE: Final = Path("src")


class Unresolved(NamedTuple):
	message: str


class Module(NamedTuple):
	path: Path
	source: str
	tree: ast.Module


class Target(NamedTuple):
	module: Module
	name: str


def resolve(dotted: str) -> Target | Unresolved:
	module, _, name = dotted.rpartition(".")
	found = module_at(module)
	if isinstance(found, Unresolved):
		return Unresolved(f"{found.message} for {dotted!r}")
	return Target(found, name)


def module_at(dotted: str) -> Module | Unresolved:
	base = SOURCE.joinpath(*dotted.split("."))
	found = next(
		(
			path
			for path in (base.with_suffix(".py"), base / "__init__.py")
			if dotted and path.is_file()
		),
		None,
	)
	if found is None:
		return Unresolved(f"no module {dotted!r} under {SOURCE.as_posix()}/")
	source = found.read_text(encoding="utf-8")
	return Module(found, source, ast.parse(source))


def constants(module: Module) -> dict[str, str]:
	return {
		target.id: segment(module.source, statement.value)
		for statement in module.tree.body
		if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value is not None
		for target in (
			statement.targets if isinstance(statement, ast.Assign) else (statement.target,)
		)
		if isinstance(target, ast.Name)
	}


def segment(source: str, node: ast.AST) -> str:
	return ast.get_source_segment(source, node) or ""


def string_constant(statement: ast.stmt | None) -> str | None:
	if (
		isinstance(statement, ast.Expr)
		and isinstance(statement.value, ast.Constant)
		and isinstance(statement.value.value, str)
	):
		return statement.value.value
	return None


def type_definitions(module: Module) -> dict[str, ast.ClassDef]:
	return {node.name: node for node in module.tree.body if isinstance(node, ast.ClassDef)}
