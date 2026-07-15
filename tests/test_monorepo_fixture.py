# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

from camas import Parallel
from camas.main.compose import load_scope

FIXTURE: Final = Path(__file__).parent / "fixtures" / "monorepo"
BADCHILD: Final = Path(__file__).parent / "fixtures" / "monorepo-badchild"

SHOW_OUTPUT: Final = "--effects=(Summary(show_passing=True),)"


def _camas(
	*args: str, cwd: Path = FIXTURE, github: bool = False
) -> subprocess.CompletedProcess[str]:
	env = {
		k: v
		for k, v in os.environ.items()
		if k not in ("CLAUDECODE", "CAMAS_AGENT", "GITHUB_ACTIONS")
	}
	env["NO_COLOR"] = "1"
	if github:
		env["GITHUB_ACTIONS"] = "true"
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=cwd,
		capture_output=True,
		text=True,
		encoding="utf-8",
		env=env,
		stdin=subprocess.DEVNULL,
		check=False,
	)


def test_list_shows_composed_namespaces() -> None:
	r = _camas("--list")
	assert r.returncode == 0
	for name in (
		"hello",
		"libs",
		"libs.build",
		"libs.fix",
		"libs.check",
		"libs.search",
		"libs.search.lint",
		"libs.search.test",
		"api",
		"api.deploy",
		"api.fix",
		"api.check",
		"web",
		"web.build",
		"web.ship",
		"web.fix",
		"web.check",
	):
		assert name in r.stdout, r.stdout


def test_bare_default_composes_each_child_default() -> None:
	r = _camas(SHOW_OUTPUT)
	assert r.returncode == 0
	assert f"libs-build {FIXTURE / 'libs'}" in r.stdout, r.stdout
	assert f"api-deploy {FIXTURE / 'services' / 'api'}" in r.stdout, r.stdout
	assert "web-build ." in r.stdout, r.stdout


def test_named_root_task_runs_at_root() -> None:
	r = _camas(SHOW_OUTPUT, "hello")
	assert r.returncode == 0
	assert f"hello {FIXTURE}" in r.stdout, r.stdout


def test_bare_namespace_runs_child_default() -> None:
	r = _camas(SHOW_OUTPUT, "libs")
	assert r.returncode == 0
	assert f"libs-build {FIXTURE / 'libs'}" in r.stdout, r.stdout


def test_nested_bare_namespace_runs_grandchild_default() -> None:
	r = _camas(SHOW_OUTPUT, "libs.search")
	assert r.returncode == 0
	assert f"search-lint {FIXTURE / 'libs' / 'search' / 'src'}" in r.stdout, r.stdout


def test_composed_leaf_anchors_cwd_two_levels_down() -> None:
	r = _camas(SHOW_OUTPUT, "libs.search.lint")
	assert r.returncode == 0
	assert f"search-lint {FIXTURE / 'libs' / 'search' / 'src'}" in r.stdout, r.stdout


def test_composed_leaf_without_cwd_runs_in_child_dir() -> None:
	r = _camas(SHOW_OUTPUT, "libs.search.test")
	assert r.returncode == 0
	assert f"search-test {FIXTURE / 'libs' / 'search'}" in r.stdout, r.stdout


def test_gap_dir_named_by_explicit_import() -> None:
	r = _camas(SHOW_OUTPUT, "api.deploy")
	assert r.returncode == 0
	assert f"api-deploy {FIXTURE / 'services' / 'api'}" in r.stdout, r.stdout


def test_dry_run_shows_composed_cwd() -> None:
	r = _camas("--dry-run", "libs.search.lint")
	assert r.returncode == 0
	assert f"(cwd: {Path('libs/search/src')})" in r.stdout, r.stdout


def test_context_local_runs_default() -> None:
	r = _camas(SHOW_OUTPUT, "web")
	assert r.returncode == 0
	assert f"web-build {FIXTURE / 'web'}" not in r.stdout, r.stdout
	assert "web-build" in r.stdout, r.stdout


def test_context_github_runs_github_task() -> None:
	r = _camas(SHOW_OUTPUT, "web", github=True)
	assert r.returncode == 0
	assert f"web-ship {FIXTURE / 'web'}" in r.stdout, r.stdout


def test_github_composite_runs_each_child_github_default() -> None:
	r = _camas(SHOW_OUTPUT, github=True)
	assert r.returncode == 0
	assert f"libs-build {FIXTURE / 'libs'}" in r.stdout, r.stdout
	assert f"api-deploy {FIXTURE / 'services' / 'api'}" in r.stdout, r.stdout
	assert f"web-ship {FIXTURE / 'web'}" in r.stdout, r.stdout


def test_rendered_labels_carry_the_project_namespace() -> None:
	r = _camas(SHOW_OUTPUT)
	assert r.returncode == 0
	for label in ("libs.build", "api.deploy", "web.build"):
		assert label in r.stdout, r.stdout


def test_github_status_lines_carry_the_project_namespace() -> None:
	r = _camas(github=True)
	assert r.returncode == 0
	for label in ("[libs.build]", "[api.deploy]", "[web.ship]"):
		assert label in r.stdout, r.stdout


def test_fix_composite_runs_each_child_fix() -> None:
	r = _camas("mcp", "fix", "--dry-run")
	assert r.returncode == 0
	for cwd, printed in (("libs", "libs-fix"), ("services/api", "api-fix"), ("web", "web-fix")):
		assert f"'{printed}'" in r.stdout, r.stdout
		assert f"(cwd: {Path(cwd)})" in r.stdout, r.stdout


def test_each_config_field_composes_the_childs_matching_field() -> None:
	config = load_scope(FIXTURE / "tasks.py").config
	assert config is not None
	assert config.agent is not None
	fields = (
		(config.default_task, ["libs.build", "api.deploy", "web.build"]),
		(config.github_task, ["libs.build", "api.deploy", "web.ship"]),
		(config.agent.fix, ["libs.fix", "api.fix", "web.fix"]),
		(config.agent.check, ["libs.check", "api.check", "web.check"]),
	)
	for node, expected in fields:
		assert isinstance(node, Parallel)
		assert [leaf.name for leaf in node.tasks] == expected


def test_expression_composes_dotted_refs() -> None:
	r = _camas(SHOW_OUTPUT, "{libs.search.lint, api.deploy}")
	assert r.returncode == 0
	assert f"search-lint {FIXTURE / 'libs' / 'search' / 'src'}" in r.stdout, r.stdout
	assert f"api-deploy {FIXTURE / 'services' / 'api'}" in r.stdout, r.stdout


def test_paths_scoping_rebases_across_the_boundary() -> None:
	r = _camas(SHOW_OUTPUT, "--paths", "web/app.py", "web.build")
	assert r.returncode == 0
	assert "web-build app.py" in r.stdout, r.stdout


def test_paths_outside_namespace_runs_nothing() -> None:
	r = _camas("--paths", "services/api/x.py", "web.build")
	assert r.returncode == 0
	assert "nothing to run" in r.stdout, r.stdout


def test_scoped_run_gates_composed_child_by_its_directory() -> None:
	r = _camas(SHOW_OUTPUT, "--paths", "libs/x.py")
	assert r.returncode == 0
	assert "libs-build" in r.stdout, r.stdout
	assert "api-deploy" not in r.stdout, r.stdout
	assert "web-build" not in r.stdout, r.stdout


def test_scoped_run_gates_other_composed_child_by_its_directory() -> None:
	r = _camas(SHOW_OUTPUT, "--paths", "services/api/x.py")
	assert r.returncode == 0
	assert "api-deploy" in r.stdout, r.stdout
	assert "libs-build" not in r.stdout, r.stdout
	assert "web-build" not in r.stdout, r.stdout


def test_run_from_child_subdir_gets_local_view() -> None:
	r = _camas(SHOW_OUTPUT, "lint", cwd=FIXTURE / "libs" / "search" / "src")
	assert r.returncode == 0
	assert f"search-lint {FIXTURE / 'libs' / 'search' / 'src'}" in r.stdout, r.stdout


def test_child_dir_list_is_the_local_view() -> None:
	r = _camas("--list", cwd=FIXTURE / "libs" / "search")
	assert r.returncode == 0
	assert "lint" in r.stdout, r.stdout
	assert "test" in r.stdout, r.stdout
	assert "libs.search" not in r.stdout, r.stdout
	assert "hello" not in r.stdout, r.stdout


def test_run_from_gap_dir_finds_root() -> None:
	r = _camas(SHOW_OUTPUT, "api.deploy", cwd=FIXTURE / "services")
	assert r.returncode == 0
	assert f"api-deploy {FIXTURE / 'services' / 'api'}" in r.stdout, r.stdout


def test_unknown_dotted_name_gets_clean_error() -> None:
	r = _camas("libs.search.nope")
	assert r.returncode == 2
	assert "no task named 'libs.search.nope'" in r.stderr, r.stderr


def test_broken_child_poisons_load_with_child_attribution() -> None:
	r = _camas("ok", cwd=BADCHILD)
	assert r.returncode != 0
	output = r.stdout + r.stderr
	assert str(BADCHILD / "broken" / "tasks.py") in output, output
	assert "boom in broken child" in output, output


def test_broken_child_list_still_works() -> None:
	r = _camas("--list", cwd=BADCHILD)
	assert r.returncode == 0
	assert "broken" in r.stdout, r.stdout
