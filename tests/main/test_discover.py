# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

import pytest

from camas import Config, Parallel, Sequential, Task
from camas.main.discover import (
	ComposeChildError,
	check_segment,
	compose_from,
	composed_view,
	discover_children,
	is_pruned_dir,
	load_py_state,
	rebase_cwd,
	rebase_paths,
	rebase_str_prefix,
	rebase_tree,
	rebase_when,
	state_from_scope,
	wrap_pathscope,
	wrap_when,
)
from camas.main.state import LoadErr, LoadOk
from camas.main.tasks import load_own, name_scope_config


def _write(path: Path, source: str) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(source)
	return path


# --- discover_children ---


def test_discover_children_nearest_wins_no_descent_past_a_found_tasks_py(tmp_path: Path) -> None:
	_write(tmp_path / "child" / "tasks.py", "")
	_write(tmp_path / "child" / "deeper" / "tasks.py", "")
	assert discover_children(tmp_path) == (tmp_path / "child",)


def test_discover_children_gap_dirs_collapse(tmp_path: Path) -> None:
	_write(tmp_path / "gap" / "nested" / "tasks.py", "")
	assert discover_children(tmp_path) == (tmp_path / "gap" / "nested",)


def test_discover_children_sorted_deterministic(tmp_path: Path) -> None:
	_write(tmp_path / "zeta" / "tasks.py", "")
	_write(tmp_path / "alpha" / "tasks.py", "")
	_write(tmp_path / "mid" / "tasks.py", "")
	assert discover_children(tmp_path) == (
		tmp_path / "alpha",
		tmp_path / "mid",
		tmp_path / "zeta",
	)


@pytest.mark.parametrize("pruned", ["node_modules", ".venv", "venv", "__pycache__", ".hidden"])
def test_discover_children_prunes_ignored_and_hidden_dirs(tmp_path: Path, pruned: str) -> None:
	_write(tmp_path / pruned / "tasks.py", "")
	assert discover_children(tmp_path) == ()


def test_discover_children_planted_node_modules_tasks_py_not_found(tmp_path: Path) -> None:
	_write(tmp_path / "node_modules" / "some-pkg" / "tasks.py", "")
	assert discover_children(tmp_path) == ()


@pytest.mark.skipif(
	sys.platform == "win32", reason="symlink creation requires elevated privileges on Windows"
)
def test_discover_children_skips_symlinked_dir(tmp_path: Path) -> None:
	real = _write(tmp_path / "real" / "tasks.py", "").parent
	link = tmp_path / "link"
	link.symlink_to(real, target_is_directory=True)
	assert discover_children(tmp_path) == (real,)


def test_is_pruned_dir_pure() -> None:
	assert is_pruned_dir(".git")
	assert is_pruned_dir("node_modules")
	assert not is_pruned_dir("src")


# --- rebase_cwd / rebase_str_prefix ---


def test_rebase_cwd_none_root_anchors_to_rel() -> None:
	assert rebase_cwd(None, PurePosixPath("services/api"), is_root=True) == Path("services/api")


def test_rebase_cwd_none_nested_stays_none() -> None:
	assert rebase_cwd(None, PurePosixPath("services/api"), is_root=False) is None


def test_rebase_cwd_relative_nests_under_rel() -> None:
	assert rebase_cwd(Path("rust"), PurePosixPath("services/api"), is_root=True) == Path(
		"services/api/rust"
	)
	assert rebase_cwd(Path("rust"), PurePosixPath("services/api"), is_root=False) == Path(
		"services/api/rust"
	)


def test_rebase_cwd_absolute_unchanged() -> None:
	absolute = Path.cwd()
	assert rebase_cwd(absolute, PurePosixPath("services/api"), is_root=True) == absolute


def test_rebase_str_prefix_dot_collapses_to_rel() -> None:
	assert rebase_str_prefix(".", PurePosixPath("services/api")) == "services/api"


def test_rebase_str_prefix_nests_under_rel() -> None:
	assert rebase_str_prefix("src", PurePosixPath("services/api")) == "services/api/src"


# --- rebase_paths / wrap_pathscope ---


def test_rebase_paths_none_stays_none() -> None:
	assert rebase_paths(None, PurePosixPath("api")) is None


def test_rebase_paths_str_dot() -> None:
	assert rebase_paths(".", PurePosixPath("api")) == "api"


def test_rebase_paths_str_prefix() -> None:
	assert rebase_paths("src", PurePosixPath("api")) == "api/src"


def test_rebase_paths_callable_wraps() -> None:
	wrapped = rebase_paths(lambda c: c or (".",), PurePosixPath("api"))
	assert not isinstance(wrapped, str)
	assert wrapped is not None
	assert wrapped(()) == ("api",)


def test_wrap_pathscope_full_run_reprefixes_default() -> None:
	scoped = wrap_pathscope(lambda c: c or (".",), PurePosixPath("api"))
	assert scoped(()) == ("api",)


def test_wrap_pathscope_scoped_strips_and_reprefixes() -> None:
	scoped = wrap_pathscope(lambda c: c, PurePosixPath("api"))
	assert scoped(("api/src/a.py",)) == ("api/src/a.py",)


def test_wrap_pathscope_scoped_filters_non_matching_entries() -> None:
	scoped = wrap_pathscope(lambda c: c, PurePosixPath("api"))
	assert scoped(("other/b.py",)) == ()


def test_wrap_pathscope_scoped_all_filtered_still_calls_inner_with_empty() -> None:
	"""A changed set that is non-empty but entirely outside ``rel`` takes a different branch
	than a genuinely empty (full-run) changed set, though both end up calling ``inner(())``."""
	scoped = wrap_pathscope(lambda c: c or ("fallback",), PurePosixPath("api"))
	assert scoped(("other/x.py",)) == ("api/fallback",)


def test_wrap_pathscope_mixed_changed_keeps_only_matches() -> None:
	scoped = wrap_pathscope(lambda c: c, PurePosixPath("api"))
	assert scoped(("api/a.py", "other/b.py", "api/sub/c.py")) == (
		"api/a.py",
		"api/sub/c.py",
	)


# --- rebase_when / wrap_when ---


def test_rebase_when_none_stays_none() -> None:
	assert rebase_when(None, PurePosixPath("api")) is None


def test_rebase_when_str() -> None:
	assert rebase_when("src", PurePosixPath("api")) == "api/src"


def test_rebase_when_tuple() -> None:
	assert rebase_when(("src", "include"), PurePosixPath("api")) == ("api/src", "api/include")


def test_rebase_when_callable_wraps() -> None:
	wrapped = rebase_when(bool, PurePosixPath("api"))
	assert not isinstance(wrapped, (str, tuple))
	assert wrapped is not None
	assert wrapped(()) is False


def test_wrap_when_filters_to_rel_before_calling_inner() -> None:
	wrapped = wrap_when(lambda c: "x" in c, PurePosixPath("api"))
	assert wrapped(("api/x", "other/y")) is True
	assert wrapped(("other/y",)) is False


# --- rebase_tree ---


def test_rebase_tree_leaf_paths_str_and_root_cwd() -> None:
	rebased = rebase_tree(
		Task("ruff {paths}", paths="."), PurePosixPath("services/api"), is_root=True
	)
	assert rebased == Task("ruff {paths}", cwd=Path("services/api"), paths="services/api")


def test_rebase_tree_leaf_nested_relative_cwd() -> None:
	rebased = rebase_tree(
		Task("cargo build", cwd="rust"), PurePosixPath("services/api"), is_root=False
	)
	assert rebased == Task("cargo build", cwd=Path("services/api/rust"))


def test_rebase_tree_group_anchors_root_children_stay_unanchored() -> None:
	rebased = rebase_tree(
		Sequential(Task("cargo build"), name="ci"), PurePosixPath("services/api"), is_root=True
	)
	assert rebased == Sequential(Task("cargo build"), cwd=Path("services/api"), name="ci")
	assert isinstance(rebased, Sequential)
	assert rebased.tasks[0].cwd is None


def test_rebase_tree_preserves_help_matrix_env_mutates_agent_format() -> None:
	from camas.v0.task import AgentFormat

	leaf = Task(
		"ruff check {paths}",
		help="lint",
		mutates=True,
		paths="src",
		when=("src", "lib"),
		agent_format=AgentFormat("--sarif", "sarif"),
	)
	group = Parallel(leaf, name="checks", matrix={"X": ("1", "2")}, env={"K": "v"})
	rebased = rebase_tree(group, PurePosixPath("api"), is_root=True)
	assert isinstance(rebased, Parallel)
	assert rebased.name == "checks"
	assert rebased.matrix == {"X": ("1", "2")}
	assert rebased.env == {"K": "v"}
	inner = rebased.tasks[0]
	assert isinstance(inner, Task)
	assert inner.help == "lint"
	assert inner.mutates is True
	assert inner.paths == "api/src"
	assert inner.when == ("api/src", "api/lib")
	assert inner.agent_format == AgentFormat("--sarif", "sarif")


def test_rebase_tree_callable_paths_and_when_are_wrapped() -> None:
	leaf = Task(
		"ruff {paths}",
		paths=lambda c: c or (".",),
		when=bool,
	)
	rebased = rebase_tree(leaf, PurePosixPath("api"), is_root=True)
	assert isinstance(rebased, Task)
	assert not isinstance(rebased.paths, str)
	assert rebased.paths is not None
	assert rebased.paths(()) == ("api",)
	assert not isinstance(rebased.when, (str, tuple))
	assert rebased.when is not None
	assert rebased.when(("api/x",)) is True


# --- check_segment ---


def test_check_segment_accepts_plain_name() -> None:
	check_segment("api", Path("tasks.py"))


def test_check_segment_rejects_mcp() -> None:
	with pytest.raises(ValueError, match="reserved"):
		check_segment("mcp", Path("services/mcp/tasks.py"))


def test_check_segment_rejects_dotted_name() -> None:
	with pytest.raises(ValueError, match="namespace delimiter"):
		check_segment("a.b", Path("services/a.b/tasks.py"))


# --- ComposeChildError ---


def test_compose_child_error_carries_source_and_cause() -> None:
	cause = RuntimeError("boom")
	source = Path("child/tasks.py")
	err = ComposeChildError(source, cause)
	assert err.source == source
	assert err.cause is cause
	assert "boom" in str(err)


# --- compose_from / composed_view ---


def test_compose_from_dotted_key(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(tmp_path / "api" / "tasks.py", "from camas import Task\nbuild = Task('build api')\n")
	loaded = compose_from(tmp_path, load_own(tmp_path / "tasks.py"))
	assert loaded.tasks["api.build"] == Task("build api", name="build", cwd=Path("api"))


def test_compose_from_bare_namespace_default_key(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(
		tmp_path / "api" / "tasks.py",
		"from camas import Config, Task\n"
		"build = Task('build api')\n"
		"_ = Config(default_task=build)\n",
	)
	loaded = compose_from(tmp_path, load_own(tmp_path / "tasks.py"))
	assert loaded.tasks["api"] == Task("build api", name="build", cwd=Path("api"))
	assert loaded.tasks["api.build"] == Task("build api", name="build", cwd=Path("api"))


def test_compose_from_recurses_into_grandchildren(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(tmp_path / "libs" / "tasks.py", "")
	_write(
		tmp_path / "libs" / "search" / "tasks.py",
		"from camas import Task\nlint = Task('lint search')\n",
	)
	loaded = compose_from(tmp_path, load_own(tmp_path / "tasks.py"))
	assert loaded.tasks["libs.search.lint"] == Task(
		"lint search", name="lint", cwd=Path("libs/search")
	)


def test_compose_from_discover_false_skips_children(tmp_path: Path) -> None:
	_write(
		tmp_path / "tasks.py",
		"from camas import Config\n_ = Config(discover=False)\n",
	)
	_write(tmp_path / "api" / "tasks.py", "from camas import Task\nbuild = Task('build api')\n")
	loaded = compose_from(tmp_path, load_own(tmp_path / "tasks.py"))
	assert loaded.tasks == {}


def test_compose_from_discoverable_false_prunes_whole_subtree(tmp_path: Path) -> None:
	_write(
		tmp_path / "api" / "tasks.py",
		"from camas import Config, Task\n"
		"build = Task('build api')\n"
		"_ = Config(default_task=build, discoverable=False)\n",
	)
	_write(
		tmp_path / "api" / "sub" / "tasks.py",
		"from camas import Task\nlint = Task('lint sub')\n",
	)
	_write(tmp_path / "tasks.py", "")
	loaded = compose_from(tmp_path, load_own(tmp_path / "tasks.py"))
	assert loaded.tasks == {}


def test_compose_from_empty_child_contributes_nothing(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(tmp_path / "api" / "tasks.py", "")
	loaded = compose_from(tmp_path, load_own(tmp_path / "tasks.py"))
	assert loaded.tasks == {}


def test_compose_from_config_only_child_bare_key_only(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(
		tmp_path / "api" / "tasks.py",
		"from camas import Config, Task\n_ = Config(default_task=Task('inline'))\n",
	)
	loaded = compose_from(tmp_path, load_own(tmp_path / "tasks.py"))
	assert set(loaded.tasks) == {"api"}
	assert loaded.tasks["api"] == Task("inline", cwd=Path("api"))


def test_compose_from_config_without_default_contributes_nothing(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(
		tmp_path / "api" / "tasks.py",
		"from camas import Config\n_ = Config(camas_dir='.other')\n",
	)
	loaded = compose_from(tmp_path, load_own(tmp_path / "tasks.py"))
	assert loaded.tasks == {}


def test_compose_from_effects_only_child_not_composed(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(
		tmp_path / "api" / "tasks.py",
		"from camas.v0.effect import Effect\n"
		"class MyEffect:\n"
		"    async def setup(self, task): ...\n"
		"    async def on_event(self, event, states, ctx): ...\n"
		"    async def teardown(self, ctxs): ...\n",
	)
	loaded = compose_from(tmp_path, load_own(tmp_path / "tasks.py"))
	assert loaded.tasks == {}
	assert loaded.scope_effects == {}


def test_compose_from_bare_key_collision_names_both_files(tmp_path: Path) -> None:
	root = _write(tmp_path / "tasks.py", "from camas import Task\napi = Task('root api')\n")
	child = _write(
		tmp_path / "api" / "tasks.py",
		"from camas import Config, Task\n"
		"build = Task('child build')\n"
		"_ = Config(default_task=build)\n",
	)
	with pytest.raises(ValueError, match="defined in both") as exc:
		compose_from(tmp_path, load_own(root))
	message = str(exc.value)
	assert str(root) in message
	assert str(child) in message


def test_compose_from_dotted_key_collision_names_both_files(tmp_path: Path) -> None:
	root = _write(
		tmp_path / "tasks.py",
		"from camas import Task\nglobals()['api.build'] = Task('root dotted')\n",
	)
	child = _write(
		tmp_path / "api" / "tasks.py", "from camas import Task\nbuild = Task('child build')\n"
	)
	with pytest.raises(ValueError, match="defined in both") as exc:
		compose_from(tmp_path, load_own(root))
	message = str(exc.value)
	assert str(root) in message
	assert str(child) in message


def test_compose_from_rejects_mcp_child_dir(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(tmp_path / "mcp" / "tasks.py", "from camas import Task\nx = Task('x')\n")
	with pytest.raises(ValueError, match="reserved"):
		compose_from(tmp_path, load_own(tmp_path / "tasks.py"))


def test_compose_from_rejects_dotted_child_dir_name(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(tmp_path / "a.b" / "tasks.py", "from camas import Task\nx = Task('x')\n")
	with pytest.raises(ValueError, match="namespace delimiter"):
		compose_from(tmp_path, load_own(tmp_path / "tasks.py"))


def test_composed_view_matches_compose_from(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(tmp_path / "api" / "tasks.py", "from camas import Task\nbuild = Task('build api')\n")
	assert composed_view(tmp_path / "tasks.py") == compose_from(
		tmp_path, load_own(tmp_path / "tasks.py")
	)


# --- load_py_state ---


def test_load_py_state_composes_successfully(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	_write(tmp_path / "api" / "tasks.py", "from camas import Task\nbuild = Task('build api')\n")
	state = load_py_state(tmp_path / "tasks.py")
	assert isinstance(state, LoadOk)
	assert "api.build" in state.tasks


def test_load_py_state_wraps_child_error_with_child_source(tmp_path: Path) -> None:
	_write(tmp_path / "tasks.py", "")
	broken = _write(tmp_path / "api" / "tasks.py", "raise RuntimeError('boom child')\n")
	state = load_py_state(tmp_path / "tasks.py")
	assert isinstance(state, LoadErr)
	assert state.source == broken
	assert isinstance(state.exception, RuntimeError)
	assert "boom child" in str(state.exception)


def test_load_py_state_own_error_uses_own_source(tmp_path: Path) -> None:
	broken = _write(tmp_path / "tasks.py", "raise RuntimeError('boom own')\n")
	state = load_py_state(broken)
	assert isinstance(state, LoadErr)
	assert state.source == broken
	assert "boom own" in str(state.exception)


# --- state_from_scope ---


def test_state_from_scope_composes_descendants(tmp_path: Path) -> None:
	root = tmp_path / "tasks.py"
	_write(tmp_path / "api" / "tasks.py", "from camas import Task\nbuild = Task('build api')\n")
	scope: dict[str, object] = {"__file__": str(root), "lint": Task("ruff .")}
	state = state_from_scope(scope)
	assert isinstance(state, LoadOk)
	assert state.tasks["lint"] == Task("ruff .", name="lint")
	assert state.tasks["api.build"] == Task("build api", name="build", cwd=Path("api"))


def test_state_from_scope_source_none_returns_own(tmp_path: Path) -> None:
	scope: dict[str, object] = {"lint": Task("ruff .")}
	state = state_from_scope(scope)
	assert isinstance(state, LoadOk)
	assert state.source is None
	assert state.tasks["lint"] == Task("ruff .", name="lint")


def test_state_from_scope_source_none_reserved_name_errors() -> None:
	scope: dict[str, object] = {"mcp": Task("echo hi")}
	state = state_from_scope(scope)
	assert isinstance(state, LoadErr)
	assert state.source == Path("tasks.py")
	assert "reserved" in str(state.exception)


def test_state_from_scope_discover_false_skips_composition(tmp_path: Path) -> None:
	root = tmp_path / "tasks.py"
	_write(tmp_path / "api" / "tasks.py", "from camas import Task\nbuild = Task('build api')\n")
	scope: dict[str, object] = {
		"__file__": str(root),
		"lint": Task("ruff ."),
		"_": Config(discover=False),
	}
	state = state_from_scope(scope)
	assert isinstance(state, LoadOk)
	assert set(state.tasks) == {"lint"}


def test_state_from_scope_wraps_child_error_with_child_source(tmp_path: Path) -> None:
	root = tmp_path / "tasks.py"
	broken = _write(tmp_path / "api" / "tasks.py", "raise RuntimeError('boom scope child')\n")
	scope: dict[str, object] = {"__file__": str(root)}
	state = state_from_scope(scope)
	assert isinstance(state, LoadErr)
	assert state.source == broken
	assert "boom scope child" in str(state.exception)


def test_state_from_scope_generic_error_uses_source(tmp_path: Path) -> None:
	root = tmp_path / "tasks.py"
	_write(
		tmp_path / "api" / "tasks.py",
		"from camas import Config, Task\n"
		"build = Task('build api')\n"
		"_ = Config(default_task=build)\n",
	)
	scope: dict[str, object] = {
		"__file__": str(root),
		"api": Task("root api", name="api"),
	}
	state = state_from_scope(scope)
	assert isinstance(state, LoadErr)
	assert state.source == root
	assert "defined in both" in str(state.exception)


# --- name_scope_config carries discover/discoverable (regression) ---


def test_name_scope_config_preserves_discover_and_discoverable_defaults() -> None:
	resolved = name_scope_config({"_": Config()})
	assert resolved is not None
	assert resolved.discover is True
	assert resolved.discoverable is True


def test_name_scope_config_preserves_discover_false() -> None:
	resolved = name_scope_config({"_": Config(discover=False)})
	assert resolved is not None
	assert resolved.discover is False
	assert resolved.discoverable is True


def test_name_scope_config_preserves_discoverable_false() -> None:
	resolved = name_scope_config({"_": Config(discoverable=False)})
	assert resolved is not None
	assert resolved.discover is True
	assert resolved.discoverable is False
