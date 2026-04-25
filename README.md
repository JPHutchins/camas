# Camas

Parallel and sequential task-tree runner. [![CI](https://github.com/JPHutchins/camas/actions/workflows/ci.yaml/badge.svg)](https://github.com/JPHutchins/camas/actions/workflows/ci.yaml)

![demo](https://raw.githubusercontent.com/JPHutchins/camas/gh-storage/demos/demo-latest.gif)

That demo can be defined in a strongly typed Python* file as:

```python
from pathlib import Path
from camas import Parallel, Sequential, Task

src_tauri = Path("src-tauri")
python_sdk = Path("python-sdk")
node = Path("node_modules/.bin")

frontend = Sequential((
  Task(f"{node}/prettier --write ."),
  Parallel((
    Task(f"{node}/eslint src/"),
    Task(f"{node}/tsc --noEmit"),
    Task(f"{node}/vitest run"),
),),),)

backend = Sequential((
  Task("cargo fmt --all", cwd=src_tauri),
  Parallel((
    Task("cargo clippy --all-targets --locked -- -D warnings", cwd=src_tauri),
    Task("cargo test --all-targets --locked", cwd=src_tauri),
),),),)

sdk = Sequential((
  Task("uv run ruff check --fix .", cwd=python_sdk),
  Task("uv run ruff format .", cwd=python_sdk),
  Parallel((
    Task("uv run mypy .", cwd=python_sdk),
    Task("uv run pytest", cwd=python_sdk),
),),),)

all = Parallel((frontend, backend, sdk))
```

> *formatted tightly for browser readability

It can be defined in `pyproject.toml` or provided as a command-line argument.

## Install

```
pipx install camas
```

Or use it as a development dependency in your Python project:

```toml
[dependency-groups]
dev = [
# ... other dependencies
"camas",
]
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
