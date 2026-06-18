# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``camas_docs`` tool: camas's authoring tutorial, read live from the installed source."""

from __future__ import annotations

import ast
from pathlib import Path

from . import wire


def camas_source_dir() -> Path:
	"""The installed camas package directory, resolved at call time (mypyc-safe)."""
	import camas

	return Path(camas.__file__).parent


def to_docs_response() -> wire.DocsResponse:
	"""The ``camas_docs`` payload: the source path and the ``__init__.py`` tutorial."""
	source = camas_source_dir()
	init = (source / "__init__.py").read_text(encoding="utf-8")
	tutorial = ast.get_docstring(ast.parse(init)) or ""
	return wire.DocsResponse(source=str(source), tutorial=tutorial)
