"""camas task definitions for this project."""

from pathlib import Path

from camas import Config, Parallel, Sequential, Task

format = Task("uv run ruff format {paths}", mutates=True, paths=".")
format_check = Task("uv run ruff format --check {paths}", paths=".")
lint = Task("uv run ruff check {paths}", paths=".")
lint_fix = Task("uv run ruff check --fix {paths}", mutates=True, paths=".")
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

matrix = Sequential(
	Task("uv sync"),
	check,
	env={"UV_PROJECT_ENVIRONMENT": ".venv-{PY}", "UV_PYTHON": "{PY}"},
	matrix={
		"PY": tuple(
			stripped
			for line in (Path(__file__).parent / ".python-version").read_text().splitlines()
			if (stripped := line.strip()) and not stripped.startswith("#")
		)
	},
)

_ = Config(default_task=all, github_task=check)
