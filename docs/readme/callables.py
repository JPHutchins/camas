# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast

from .docstring import Examples, Text, split
from .markdown import details
from .signature import signature
from .source import Unresolved, class_definitions, resolve, segment


def function_section(dotted: str) -> tuple[str, ...] | Unresolved:
	target = resolve(dotted)
	if isinstance(target, Unresolved):
		return target
	function = next(
		(
			node
			for node in target.module.tree.body
			if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
			and node.name == target.name
		),
		None,
	)
	if function is None:
		return Unresolved(f"no function {target.name!r} in {target.module.path.as_posix()}")
	parts = split(ast.get_docstring(function) or "")
	examples = tuple(part for part in parts if isinstance(part, Examples))
	return (
		"```python",
		signature(function, target.module),
		"```",
		*(line for part in parts if isinstance(part, Text) for line in ("", *part.lines)),
		*(
			(
				"",
				f"<details><summary><code>{target.name}</code> examples</summary>",
				"",
				*(line for part in examples for line in ("```pycon", *part.lines, "```", "")),
				"</details>",
			)
			if examples
			else ()
		),
	)


def protocol_section(dotted: str) -> tuple[str, ...] | Unresolved:
	target = resolve(dotted)
	if isinstance(target, Unresolved):
		return target
	found = class_definitions(target.module)
	if target.name not in found:
		return Unresolved(f"no class {target.name!r} in {target.module.path.as_posix()}")
	cls = found[target.name]
	bases = ", ".join(segment(target.module.source, base) for base in cls.bases)
	methods = tuple(
		node
		for node in cls.body
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
		and not node.name.startswith("_")
	)
	return (
		"```python",
		f"class {cls.name}({bases}):" if bases else f"class {cls.name}:",
		*(f"    {signature(method, target.module)}: ..." for method in methods),
		"```",
		*details(
			f"<code>{cls.name}</code> methods",
			tuple((method.name, doc) for method in methods if (doc := ast.get_docstring(method))),
		),
	)
