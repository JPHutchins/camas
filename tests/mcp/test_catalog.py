# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from camas import Config, Parallel, Sequential, Task
from camas.core.timings import TaskTiming
from camas.mcp.catalog import to_list_response


def test_catalog_lists_sorted_with_markers() -> None:
	lint = Task("ruff check .", name="lint", help="Lint sources")
	test = Task("pytest", name="test")
	ci = Sequential(lint, test, name="ci")
	resp = to_list_response(
		{"ci": ci, "lint": lint, "test": test}, Config(default_task=ci, github_task=lint)
	)
	assert [t.name for t in resp.tasks] == ["ci", "lint", "test"]
	assert (resp.default, resp.github_default) == ("ci", "lint")
	by_name = {t.name: t for t in resp.tasks}
	assert by_name["ci"].is_default is True
	assert by_name["lint"].is_github_default is True
	assert by_name["lint"].help == "Lint sources"
	assert by_name["test"].help is None


def test_command_preview_is_fully_typed_expression() -> None:
	d = Task("d's actual command", name="d")
	e = Task("e's actual command", name="e")
	a = Parallel(d, e, name="a")
	resp = to_list_response({"ci": Sequential(a, Parallel(Task("b"), Task("c")), name="ci")}, None)
	assert (
		resp.tasks[0].command_preview
		== 'Sequential(Parallel(Task("d\'s actual command", name="d"), Task("e\'s actual command", name="e"), name="a"), Parallel(Task("b"), Task("c")), name="ci")'
	)


def test_no_config_marks_nothing() -> None:
	resp = to_list_response({"t": Task("echo hi", name="t")}, None)
	assert (resp.default, resp.github_default) == (None, None)
	assert resp.tasks[0].is_default is False
	assert resp.tasks[0].is_github_default is False


def test_config_without_defaults_yields_no_markers() -> None:
	resp = to_list_response({"t": Task("echo hi", name="t")}, Config())
	assert (resp.default, resp.github_default) == (None, None)


def test_matrix_axes_reported_as_lists() -> None:
	node = Parallel(Task("test {PY}"), matrix={"PY": ("3.13", "3.14")}, name="m")
	resp = to_list_response({"m": node}, None)
	assert resp.tasks[0].matrix_axes == {"PY": ["3.13", "3.14"]}
	assert resp.tasks[0].command_preview == (
		'Parallel(Task("test 3.13", name="test 3.13 [PY=3.13]"), '
		'Task("test 3.14", name="test 3.14 [PY=3.14]"), name="m")'
	)


def test_estimates_composed_onto_matching_task() -> None:
	ci = Sequential(Task("ruff", name="fmt"), Task("pytest", name="run"), name="ci")
	cache = {"fmt": TaskTiming(0.5, 1), "run": TaskTiming(2.0, 3)}
	resp = to_list_response({"ci": ci, "lint": Task("ruff", name="lint")}, None, cache)
	by_name = {t.name: t for t in resp.tasks}
	assert by_name["ci"].estimated_s == 2.5
	assert by_name["ci"].samples == 1
	assert by_name["ci"].slowest_leaf == "run"
	assert by_name["ci"].slowest_s == 2.0
	assert by_name["lint"].estimated_s is None
