# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast

from ..paths import camas_package_dir
from . import wire


def to_docs_response() -> wire.DocsResponse:
	source = camas_package_dir()
	init = (source / "__init__.py").read_text(encoding="utf-8")
	tutorial = ast.get_docstring(ast.parse(init)) or ""
	return wire.DocsResponse(source=str(source), tutorial=tutorial)
