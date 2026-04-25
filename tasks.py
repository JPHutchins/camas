"""camas task definitions for this project."""

from camas import Parallel, Sequential, Task

format = Task("uv run ruff format .")
format_check = Task("uv run ruff format --check .")
lint = Task("uv run ruff check .")
lint_fix = Task("uv run ruff check --fix .")
fix = Sequential(lint_fix, format)
mypy = Task("uv run mypy .")
ty = Task("uv run ty check")
zuban = Task("uv run zuban check --exclude tests.fixtures")
pyrefly = Task("uv run pyrefly check")
pyright = Task("uv run pyright src tests")

typecheck = Parallel(mypy, pyright, ty, zuban, pyrefly)
test = Task("uv run pytest --doctest-modules -v -m 'not slow'")
coverage = Task(
	"uv run pytest --doctest-modules -m 'not slow' --cov --cov-report=term-missing --cov-report=xml"
)

all = Sequential(fix, Parallel(typecheck, test))
check = Parallel(format_check, lint, typecheck, test)

matrix = Sequential(
	Task("uv sync"),
	check,
	env={"UV_PROJECT_ENVIRONMENT": ".venv-{PY}", "UV_PYTHON": "{PY}"},
	matrix={"PY": ("3.10", "3.11", "3.12", "3.13", "3.14", "3.15")},
)
