# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
import doctest
import itertools
import shlex
import textwrap
from typing import NamedTuple

from .source import Unresolved, module_at


class Example(NamedTuple):
	tasks_py: str
	bindings: tuple[str, ...]
	invocations: tuple[tuple[str, ...], ...]


def example_section(dotted: str, binding: str) -> tuple[str, ...] | Unresolved:
	module = module_at(dotted)
	if isinstance(module, Unresolved):
		return module
	found = tuple(
		example
		for example in examples(ast.get_docstring(module.tree) or "")
		if binding in example.bindings
	)
	match found:
		case (chosen,):
			return (
				"```python",
				*chosen.tasks_py.rstrip("\n").split("\n"),
				"```",
				*(
					(
						"",
						"```console",
						*(f"$ camas {shlex.join(args)}" for args in chosen.invocations),
						"```",
					)
					if chosen.invocations
					else ()
				),
			)
		case ():
			return Unresolved(
				f"no example in {module.path.as_posix()}'s docstring binds {binding!r}"
			)
		case _:
			return Unresolved(
				f"{len(found)} examples in {module.path.as_posix()}'s docstring bind {binding!r}"
			)


def examples(docstring: str) -> tuple[Example, ...]:
	parsed = doctest.DocTestParser().parse(docstring)
	runs = itertools.groupby(
		parsed, key=lambda piece: isinstance(piece, doctest.Example) or not str(piece).strip()
	)
	sources = tuple(
		"".join(piece.source for piece in group if isinstance(piece, doctest.Example))
		for is_run, group in runs
		if is_run
	)
	return tuple(example for source in sources if (example := lifted(source)) is not None)


def lifted(source: str) -> Example | None:
	calls = tuple(
		(node.func.id, node)
		for node in ast.walk(ast.parse(source))
		if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
	)
	written = next(
		(
			call.args[0].value
			for name, call in calls
			if name == "dedent"
			and call.args
			and isinstance(call.args[0], ast.Constant)
			and isinstance(call.args[0].value, str)
		),
		None,
	)
	if written is None:
		return None
	tasks_py = textwrap.dedent(written)
	return Example(
		tasks_py,
		tuple(
			bound
			for statement in ast.parse(tasks_py).body
			for bound in (
				tuple(target.id for target in statement.targets if isinstance(target, ast.Name))
				if isinstance(statement, ast.Assign)
				else (statement.name,)
				if isinstance(statement, ast.ClassDef)
				else ()
			)
		),
		tuple(
			tuple(
				str(argument.value) for argument in call.args if isinstance(argument, ast.Constant)
			)
			for name, call in sorted(calls, key=lambda pair: (pair[1].lineno, pair[1].col_offset))
			if name == "camas"
		),
	)
