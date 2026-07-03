# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING

from camas import Parallel, Sequential, Task
from camas.core.matrix import expand_matrix
from camas.core.scope import scope_to_changed, to_changed, with_default_paths

if TYPE_CHECKING:
	from pathlib import Path


def _cmd(node: object) -> str | tuple[str, ...]:
	assert isinstance(node, Task)
	return node.cmd


def test_prefix_string_full_run_uses_prefix() -> None:
	fmt = Task("ruff format {paths}", mutates=True, paths=".")
	assert with_default_paths(fmt) == Task("ruff format .", mutates=True, paths=".")


def test_prefix_string_scoped_run_injects_changed_files() -> None:
	fmt = Task("ruff format {paths}", name="fmt", paths=".")
	assert scope_to_changed(fmt, ("src/a.py", "src/b.py")) == Task(
		"ruff format src/a.py src/b.py", name="fmt", paths="."
	)


def test_prefix_string_filters_to_its_subtree() -> None:
	tsc = Task("tsc {paths}", name="tsc", paths="frontend")
	assert scope_to_changed(tsc, ("frontend/app.ts", "backend/api.py")) == Task(
		"tsc frontend/app.ts", name="tsc", paths="frontend"
	)


def test_prefix_string_skips_when_nothing_under_it() -> None:
	tsc = Task("tsc {paths}", paths="frontend")
	assert scope_to_changed(tsc, ("backend/api.py",)) is None


def test_none_paths_leaf_is_untouched_and_always_runs() -> None:
	mypy = Task("mypy .", name="mypy")
	assert scope_to_changed(mypy, ("anything.py",)) == mypy
	assert with_default_paths(mypy) == mypy


def test_placeholderless_command_always_runs_paths_is_noop() -> None:
	"""A command with no ``{paths}`` can't narrow, so it always runs and its ``paths`` is a
	no-op — camas errs on correctness: the tool might touch the edited files."""
	mypy = Task("mypy .", name="mypy", paths="src")
	assert scope_to_changed(mypy, ("src/app.py",)) == mypy
	assert scope_to_changed(mypy, ("docs/readme.md",)) == mypy


def test_placeholder_without_paths_defaults_to_whole_project() -> None:
	"""A ``{paths}`` leaf with no scope (own or inherited) defaults to the whole project, so it
	narrows to the changed files on a scoped run and to ``.`` on a full run."""
	assert _cmd(scope_to_changed(Task("ruff {paths}"), ("a.py",))) == "ruff a.py"
	assert _cmd(with_default_paths(Task("ruff {paths}"))) == "ruff ."


def test_group_paths_inherited_by_placeholder_child() -> None:
	"""A group's ``paths`` is the default scope of a ``{paths}`` child lacking its own — baked into
	leaves by ``expand_matrix`` like ``cwd`` — so the child scopes to the changed files and runs."""
	tree = expand_matrix(Parallel(Task("ruff format {paths}", name="fmt"), paths="."))
	scoped = scope_to_changed(tree, ("src/a.py", "src/b.py"))
	assert isinstance(scoped, Parallel)
	assert _cmd(scoped.tasks[0]) == "ruff format src/a.py src/b.py"
	full = with_default_paths(tree)
	assert isinstance(full, Parallel)
	assert _cmd(full.tasks[0]) == "ruff format ."


def test_group_paths_do_not_prune_placeholderless_child() -> None:
	"""A ``{paths}``-less child under a group ``paths`` still runs — inherited ``paths`` only fills
	a ``{paths}`` placeholder, it never prunes a command that can't narrow."""
	tree = expand_matrix(Parallel(Task("mypy .", name="mypy"), paths="src"))
	scoped = scope_to_changed(tree, ("docs/x.md",))
	assert isinstance(scoped, Parallel)
	assert _cmd(scoped.tasks[0]) == "mypy ."


def test_callable_paths_full_control() -> None:
	def py_only(changed: tuple[str, ...]) -> tuple[str, ...]:
		return (".",) if not changed else tuple(c for c in changed if c.endswith(".py"))

	lint = Task("ruff check {paths}", name="lint", paths=py_only)
	assert _cmd(scope_to_changed(lint, ("a.py", "b.rs"))) == "ruff check a.py"
	assert scope_to_changed(lint, ("b.rs",)) is None
	assert _cmd(with_default_paths(lint)) == "ruff check ."


def test_tuple_command_splices_path_tokens() -> None:
	fmt = Task(("ruff", "format", "{paths}"), name="fmt", paths=".")
	assert _cmd(scope_to_changed(fmt, ("a.py", "b.py"))) == ("ruff", "format", "a.py", "b.py")


def test_full_run_default_never_prunes() -> None:
	tree = Parallel(Task("ruff {paths}", paths="src"), Task("mypy ."))
	assert with_default_paths(tree) == Parallel(Task("ruff src", paths="src"), Task("mypy ."))


def test_group_pruned_when_all_leaves_skip() -> None:
	tree = Parallel(Task("a {paths}", paths="a"), Task("b {paths}", paths="b"))
	assert scope_to_changed(tree, ("c/x.py",)) is None


def test_nested_structure_preserved_through_pruning() -> None:
	lint = Task("ruff check {paths}", name="lint", paths=".")
	tsc = Task("tsc {paths}", name="tsc", paths="frontend")
	mypy = Task("mypy .", name="mypy")
	tree = Sequential(Parallel(lint, tsc), mypy, name="ci")
	assert scope_to_changed(tree, ("src/app.py",)) == Sequential(
		Parallel(Task("ruff check src/app.py", name="lint", paths=".")),
		mypy,
		name="ci",
	)


def test_backslash_changed_paths_are_normalized() -> None:
	fmt = Task("ruff format {paths}", name="fmt", paths=".")
	assert _cmd(scope_to_changed(fmt, ("src\\a.py",))) == "ruff format src/a.py"


def test_paths_with_spaces_are_shell_quoted_in_string_command() -> None:
	fmt = Task("ruff format {paths}", name="fmt", paths=".")
	assert _cmd(scope_to_changed(fmt, ("a b.py",))) == "ruff format 'a b.py'"


def test_to_changed_drops_blank_entries(tmp_path: Path) -> None:
	"""#132: a blank ``--paths`` entry (or a hook delivering an empty path) contributes nothing,
	rather than resolving to ``.`` and widening a scoped run to the whole tree."""
	assert to_changed(["", "  "], tmp_path) == ()
	assert to_changed(["", "a.py"], tmp_path) == ("a.py",)


def test_to_changed_splits_comma_separated_entries(tmp_path: Path) -> None:
	"""#132: one comma-joined entry scopes identically to the same paths passed separately, so
	``--paths a,b`` matches across every entrypoint that routes through ``to_changed``."""
	assert to_changed(["a.py,b.py"], tmp_path) == ("a.py", "b.py")
	assert to_changed(["a.py,b.py"], tmp_path) == to_changed(["a.py", "b.py"], tmp_path)
