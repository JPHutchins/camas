# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

FIXTURE: Final = Path(__file__).parent / "fixtures" / "monorepo"
COLLISION: Final = Path(__file__).parent / "fixtures" / "monorepo-collision"
BADCHILD: Final = Path(__file__).parent / "fixtures" / "monorepo-badchild"

SHOW_OUTPUT: Final = "--effects=(Summary(show_passing=True),)"


def _camas(*args: str, cwd: Path = FIXTURE) -> subprocess.CompletedProcess[str]:
	return subprocess.run(
		[sys.executable, "-m", "camas", *args],
		cwd=cwd,
		capture_output=True,
		text=True,
		encoding="utf-8",
		env={**os.environ, "NO_COLOR": "1"},
		check=False,
	)


def test_list_shows_composed_namespaces() -> None:
	r = _camas("--list")
	assert r.returncode == 0
	for name in (
		"hello",
		"libs",
		"libs.build",
		"libs.search",
		"libs.search.lint",
		"libs.search.test",
		"libs.history.lint",
		"libs.my-lib.check",
		"api",
		"api.deploy",
		"queue.work",
		"tools.info",
		"pkgs.tools.check",
	):
		assert name in r.stdout, r.stdout


def test_colliding_mount_extends_with_parent_segment() -> None:
	r = _camas(SHOW_OUTPUT, "pkgs.tools.check")
	assert r.returncode == 0
	assert f"pkgs-tools-check {FIXTURE / 'pkgs' / 'tools'}" in r.stdout, r.stdout


def test_list_hides_private_undiscovered_and_foreign() -> None:
	r = _camas("--list")
	assert r.returncode == 0
	for absent in ("internal", "tools.child", "never-discovered", "legacy"):
		assert absent not in r.stdout, r.stdout


def test_dry_run_shows_composed_cwd() -> None:
	r = _camas("--dry-run", "libs.search.test")
	assert r.returncode == 0
	assert f"(cwd: {Path('libs/search/src')})" in r.stdout, r.stdout


def test_composed_leaf_runs_in_its_own_directory() -> None:
	r = _camas(SHOW_OUTPUT, "libs.search.lint")
	assert r.returncode == 0
	assert f"search-lint {FIXTURE / 'libs' / 'search'}" in r.stdout, r.stdout


def test_bare_default_runs_at_root() -> None:
	r = _camas(SHOW_OUTPUT)
	assert r.returncode == 0
	assert f"hello {FIXTURE}" in r.stdout, r.stdout


def test_bare_namespace_runs_child_default() -> None:
	r = _camas(SHOW_OUTPUT, "api")
	assert r.returncode == 0
	assert f"api-deploy {FIXTURE / 'services' / 'api'}" in r.stdout, r.stdout


def test_nested_bare_namespace_runs_child_default() -> None:
	r = _camas(SHOW_OUTPUT, "libs.search")
	assert r.returncode == 0
	assert f"search-lint {FIXTURE / 'libs' / 'search'}" in r.stdout, r.stdout


def test_hyphenated_namespace_dispatches_verbatim() -> None:
	r = _camas(SHOW_OUTPUT, "libs.my-lib.check")
	assert r.returncode == 0
	assert f"my-lib-check {FIXTURE / 'libs' / 'my-lib'}" in r.stdout, r.stdout


def test_expression_composes_dotted_refs() -> None:
	r = _camas(SHOW_OUTPUT, "{libs.search.lint, api.deploy}")
	assert r.returncode == 0
	assert f"search-lint {FIXTURE / 'libs' / 'search'}" in r.stdout, r.stdout
	assert f"api-deploy {FIXTURE / 'services' / 'api'}" in r.stdout, r.stdout


def test_queue_discovered_through_foreign_tasks_py() -> None:
	r = _camas(SHOW_OUTPUT, "queue.work")
	assert r.returncode == 0
	assert f"queue-work {FIXTURE / 'legacy' / 'queue'}" in r.stdout, r.stdout


def test_run_from_gap_dir_finds_root_and_anchors_cwd() -> None:
	r = _camas(SHOW_OUTPUT, "libs.search.lint", cwd=FIXTURE / "services")
	assert r.returncode == 0
	assert f"search-lint {FIXTURE / 'libs' / 'search'}" in r.stdout, r.stdout


def test_run_from_child_subdir_gets_local_view_and_anchored_cwd() -> None:
	r = _camas(SHOW_OUTPUT, "lint", cwd=FIXTURE / "libs" / "search" / "src")
	assert r.returncode == 0
	assert f"search-lint {FIXTURE / 'libs' / 'search'}" in r.stdout, r.stdout


def test_child_dir_list_is_the_local_view() -> None:
	r = _camas("--list", cwd=FIXTURE / "libs" / "search")
	assert r.returncode == 0
	assert "lint" in r.stdout, r.stdout
	assert "test" in r.stdout, r.stdout
	assert "libs.search" not in r.stdout, r.stdout
	assert "hello" not in r.stdout, r.stdout


def test_paths_scoping_rebases_across_the_boundary() -> None:
	r = _camas(SHOW_OUTPUT, "--paths", "libs/history/somefile.py", "libs.history.lint")
	assert r.returncode == 0
	assert "history-lint somefile.py" in r.stdout, r.stdout


def test_paths_outside_namespace_runs_nothing() -> None:
	r = _camas("--paths", "services/api/x.py", "libs.history.lint")
	assert r.returncode == 0
	assert "nothing to run" in r.stdout, r.stdout


def test_unknown_dotted_name_gets_clean_error() -> None:
	r = _camas("libs.search.nope")
	assert r.returncode == 2
	assert "no task named 'libs.search.nope'" in r.stderr, r.stderr


def test_collision_names_both_files() -> None:
	r = _camas("search", cwd=COLLISION)
	assert r.returncode != 0
	output = r.stdout + r.stderr
	assert "defined in both" in output, output
	assert str(COLLISION / "tasks.py") in output, output
	assert str(COLLISION / "search" / "tasks.py") in output, output


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
