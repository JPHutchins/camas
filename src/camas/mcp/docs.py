# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``camas_docs`` tool: camas's authoring tutorial, read live from the installed source."""

from __future__ import annotations

import ast

from ..paths import camas_package_dir
from . import wire


def to_docs_response() -> wire.DocsResponse:
	"""The ``camas_docs`` payload: the source path and the ``__init__.py`` tutorial."""
	source = camas_package_dir()
	init = (source / "__init__.py").read_text(encoding="utf-8")
	tutorial = ast.get_docstring(ast.parse(init)) or ""
	return wire.DocsResponse(source=str(source), tutorial=tutorial)
