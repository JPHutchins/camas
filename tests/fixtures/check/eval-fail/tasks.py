from pathlib import Path

from camas import Parallel, Task

PY_VERSIONS = (Path(__file__).parent / ".python-version").read_text().splitlines()

test = Parallel(
	*(Task(f"pytest --python={py}") for py in PY_VERSIONS),
)
