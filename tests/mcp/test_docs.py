# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
from pathlib import Path

from camas.mcp.docs import camas_source_dir, to_docs_response


def test_camas_source_dir_points_at_installed_package() -> None:
	source = camas_source_dir()
	assert source.name == "camas"
	assert (source / "__init__.py").is_file()


def test_to_docs_response_serves_init_docstring() -> None:
	resp = to_docs_response()
	assert resp.source == str(camas_source_dir())
	for token in ("Task(", "Sequential", "Parallel", "Config"):
		assert token in resp.tutorial
	init = (Path(resp.source) / "__init__.py").read_text(encoding="utf-8")
	assert resp.tutorial == ast.get_docstring(ast.parse(init))


def test_docs_states_github_default_is_declared() -> None:
	resp = to_docs_response()
	assert "github_task" in resp.tutorial
	assert "never infers" in resp.tutorial
