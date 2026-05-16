from pathlib import Path

from camas import Parallel, Task

# Read a sibling config file at module load — pattern mirrors the
# python-version-matrix fixture, but the file isn't present here so eval
# fails with FileNotFoundError. Type checkers don't model filesystem state,
# so this is eval-fail-only.
PY_VERSIONS = (Path(__file__).parent / ".python-version").read_text().splitlines()

test = Parallel(
	*(Task(f"pytest --python={py}") for py in PY_VERSIONS),
)
