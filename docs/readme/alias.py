# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
import inspect

from .source import Unresolved, resolve, segment, string_constant


def alias_section(dotted: str) -> tuple[str, ...] | Unresolved:
	target = resolve(dotted)
	if isinstance(target, Unresolved):
		return target
	body = target.module.tree.body
	found = next(
		(
			(value, following)
			for statement, following in zip(body, (*body[1:], None), strict=True)
			if isinstance(statement, ast.AnnAssign)
			and isinstance(statement.target, ast.Name)
			and statement.target.id == target.name
			and isinstance(statement.annotation, ast.Name)
			and statement.annotation.id == "TypeAlias"
			and (value := statement.value) is not None
		),
		None,
	)
	if found is None:
		return Unresolved(f"no alias {target.name!r} in {target.module.path.as_posix()}")
	value, following = found
	source = target.module.source
	shown = (
		value.value
		if isinstance(value, ast.Constant) and isinstance(value.value, str)
		else segment(source, value)
	)
	doc = string_constant(following)
	return (
		"```python",
		f"{target.name}: TypeAlias = {shown}",
		"```",
		*(("", *inspect.cleandoc(doc).split("\n")) if doc else ()),
	)
