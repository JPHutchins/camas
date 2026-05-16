from pathlib import Path

from camas import Parallel, Task

# Type error: matrix values must be tuples, not bare strings.
build = Parallel(
	Task("build --target={TARGET}"),
	matrix={"TARGET": "x86_64"},
)

# Eval error: file doesn't exist here.
PY_VERSIONS = (Path(__file__).parent / ".python-version").read_text().splitlines()

test = Parallel(
	*(Task(f"pytest --python={py}") for py in PY_VERSIONS),
)
