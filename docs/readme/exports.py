# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast

from .source import Unresolved, module_at


def exports_section(dotted: str) -> tuple[str, ...] | Unresolved:
	module = module_at(dotted)
	if isinstance(module, Unresolved):
		return module
	statements = (
		nested
		for statement in module.tree.body
		for nested in (statement.body if isinstance(statement, ast.If) else (statement,))
	)
	names = tuple(
		alias.name
		for statement in statements
		if isinstance(statement, ast.ImportFrom)
		for alias in statement.names
		if alias.asname == alias.name
	)
	if not names:
		return Unresolved(f"no `import X as X` re-exports in {module.path.as_posix()}")
	return ("```python", f"from {dotted} import (", *(f"    {name}," for name in names), ")", "```")
