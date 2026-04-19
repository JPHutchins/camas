# Camas

Generic parallel/sequential task runner with a live tree-style TUI.

[![CI](https://github.com/JPHutchins/camas/actions/workflows/ci.yaml/badge.svg)](https://github.com/JPHutchins/camas/actions/workflows/ci.yaml)

## Install

```
uv add camas
```

or

```
pip install camas
```

## Quick start

Camas takes a typed Python expression describing a tree of `Task`, `Sequential`,
and `Parallel` nodes and runs it, streaming progress into a live tree in your
terminal.

Run two checks in parallel:
```
python -m camas 'Parallel(tasks=(Task("ruff check ."), Task("mypy .")))'
```

A typical CI pipeline — format-check, then checks in parallel, then tests:
```
python -m camas 'Sequential(tasks=(
    Task("ruff format . --check"),
    Parallel(tasks=(Task("mypy ."), Task("pyright ."))),
    Task("pytest"),
))'
```

Run one task across a matrix of Python versions in parallel:
```
python -m camas 'Parallel(
    tasks=(Task("pytest --python {PY}"),),
    matrix={"PY": ("3.12", "3.13", "3.14")},
)'
```

Use `--dry-run` to print the task tree without executing it.

## Library

The same building blocks are available as a library:

```python
import asyncio
from camas import Parallel, Sequential, Task, run
from camas.effect.termtree import Termtree, TermtreeOptions

task = Sequential(tasks=(
    Task("ruff format . --check"),
    Parallel(tasks=(Task("mypy ."), Task("pyright ."))),
    Task("pytest"),
))

result = asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),)))
raise SystemExit(result.returncode)
```

`Task`, `Sequential`, and `Parallel` are immutable `NamedTuple` values. A
`Sequential` short-circuits on the first non-zero exit; a `Parallel` runs its
children concurrently. Both accept an optional `matrix` mapping for variable
expansion in task commands, names, and environments.

## License

[MIT](LICENSE)
