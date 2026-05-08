"""tasks.py is regular Python — read .python-version to drive a matrix."""

from pathlib import Path

from camas import Sequential, Task


def read_python_versions(path: Path) -> tuple[str, ...]:
	return tuple(
		stripped
		for line in path.read_text().splitlines()
		if (stripped := line.strip()) and not stripped.startswith("#")
	)


PYTHON_VERSIONS = read_python_versions(Path(__file__).parent / ".python-version")

check = Sequential(
	Task("uv sync"),
	Task("uv run pytest"),
	env={"UV_PROJECT_ENVIRONMENT": ".venv-{PY}", "UV_PYTHON": "{PY}"},
	matrix={"PY": PYTHON_VERSIONS},
)
