"""camas task definitions for this project."""

from pathlib import Path

from camas import Config, Parallel, Sequential, Task

format = Task("uv run ruff format .")
format_check = Task("uv run ruff format --check .")
lint = Task("uv run ruff check .")
lint_fix = Task("uv run ruff check --fix .")
fix = Sequential(lint_fix, format)
mypy = Task("uv run mypy .")
ty = Task("uv run ty check")
zuban = Task("uv run zuban check src tests --exclude tests.fixtures")
pyrefly = Task("uv run pyrefly check")
pyright = Task("uv run pyright src tests")

typecheck = Parallel(mypy, pyright, ty, zuban, pyrefly)
test = Task("uv run pytest --doctest-modules -v -m 'not slow'")
coverage = Task(
	"uv run pytest --doctest-modules -m 'not slow' --cov --cov-report=term-missing --cov-report=xml"
)

nix = Task("nix flake check --all-systems --print-build-logs")

all = Sequential(fix, Parallel(typecheck, coverage))
check = Parallel(format_check, lint, typecheck, test)

# Per-cell filesystem isolation. This is a rare workaround, NOT something camas
# requires — the matrix / cwd / env primitives below are general; we just
# compose them here to dodge one flaky type-checker.
#
# `check` runs five type-checkers, and the matrix fans it across six Python
# versions in parallel: ~6 `uv sync`s and ~30 checkers pound ONE working tree at
# once. pyrefly globs `**/*.py*` over the whole tree and intermittently reads a
# file that a sibling cell's `uv sync` is mid-way through deleting under its
# `.venv`, dying with "No such file or directory (os error 2)" and flaking CI.
# That race is pyrefly's, not ours; the fix is simply to stop sharing the tree.
#
# So each cell runs in a throwaway `git worktree` — a sibling dir holding only
# tracked files, with its own `.venv` — and no two cells ever touch the same
# path. `rm -rf` then `add --force` is idempotent and safe to run concurrently
# across cells (no `git worktree prune`, which would race a sibling mid-rebuild).
# Worktrees check out HEAD, so commit locally before running the matrix if you
# want it to cover your working changes.
#
# The dir name has no leading dot on purpose: pyrefly skips hidden directories,
# so a `.`-prefixed worktree silently matches zero files and "passes" vacuously.
_worktree = "../camas-matrix/{PY}"

matrix = Sequential(
	Task(f"rm -rf {_worktree}"),
	Task(f"git worktree add --force --detach {_worktree} HEAD"),
	Sequential(
		Task("uv sync"),
		check,
		cwd=_worktree,
		env={"UV_PYTHON": "{PY}"},
	),
	matrix={
		"PY": tuple(
			stripped
			for line in (Path(__file__).parent / ".python-version").read_text().splitlines()
			if (stripped := line.strip()) and not stripped.startswith("#")
		)
	},
)

_ = Config(default_task=all, github_task=check)
