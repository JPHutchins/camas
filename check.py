"""camas task definitions for this project.

Run directly with ``python check.py <task>``. Every module-level
``Task``/``Sequential``/``Parallel`` becomes an invokable task by its binding
name. Private names (leading ``_``) are skipped.
"""

from camas import Parallel, Sequential, Task
from camas.main import run_cli

format = Task("uv run ruff format .")
format_check = Task("uv run ruff format --check .")
lint = Task("uv run ruff check .")
mypy = Task("uv run mypy .")
ty = Task("uv run ty check")
zuban = Task("uv run zuban check")
pyrefly = Task("uv run pyrefly check")
pyright = Task("uv run pyright src tests")

typecheck = Parallel(tasks=(mypy, pyright, ty, zuban, pyrefly))
test = Task("uv run pytest --doctest-modules -v -m 'not slow'")

all = Sequential(
	tasks=(
		format,
		Parallel(tasks=(lint, typecheck, test)),
	)
)

check = Sequential(
	tasks=(
		format_check,
		Parallel(tasks=(lint, typecheck, test)),
	)
)

venvs = Parallel(
	tasks=(
		Sequential(
			tasks=(
				Task("uv venv --python {PY} .venv-{PY}"),
				Task("uv sync", env={"UV_PROJECT_ENVIRONMENT": ".venv-{PY}"}),
			)
		),
	),
	matrix={"PY": ("3.12", "3.13", "3.14")},
)

matrix = Parallel(
	tasks=(check,),
	matrix={"UV_PROJECT_ENVIRONMENT": (".venv-3.12", ".venv-3.13", ".venv-3.14")},
)


if __name__ == "__main__":
	run_cli(globals())
