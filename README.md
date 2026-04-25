# Camas

Parallel and sequential task-tree runner. [![CI](https://github.com/JPHutchins/camas/actions/workflows/ci.yaml/badge.svg)](https://github.com/JPHutchins/camas/actions/workflows/ci.yaml)

![demo](https://raw.githubusercontent.com/JPHutchins/camas/gh-storage/demos/demo-latest.gif)

That demo can be defined in a strongly typed Python file as:

```python
from pathlib import Path
from camas import Parallel, Sequential, Task

src_tauri = Path("src-tauri")
python_sdk = Path("python-sdk")
node = Path("node_modules/.bin")

frontend = Sequential(
  f"{node}/prettier --write .",
  Parallel(
    f"{node}/eslint src/",
    f"{node}/tsc --noEmit",
    f"{node}/vitest run",
  ),
)

backend = Sequential(
  Task("cargo fmt --all", cwd=src_tauri),
  Parallel(
    Task("cargo clippy --all-targets --locked -- -D warnings", cwd=src_tauri),
    Task("cargo test --all-targets --locked", cwd=src_tauri),
  ),
)

sdk = Sequential(
  Task("uv run ruff check --fix .", cwd=python_sdk),
  Task("uv run ruff format .", cwd=python_sdk),
  Parallel(
    Task("uv run mypy .", cwd=python_sdk),
    Task("uv run pytest", cwd=python_sdk),
  ),
)

all = Parallel(frontend, backend, sdk)
```

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
python -m camas 'Parallel("ruff check .", "mypy .")'
```

A typical CI pipeline — format-check, then checks in parallel, then tests:
```
python -m camas 'Sequential(
    "ruff format . --check",
    Parallel("mypy .", "pyright ."),
    "pytest",
)'
```

Run one task across a matrix of Python versions in parallel:
```
python -m camas 'Parallel("pytest --python {PY}", matrix={"PY": ("3.12", "3.13", "3.14")})'
```

### Coercion: bare strings, tuple/set literals

Two kinds of implicit coercion, with different scopes:

**`str` → `Task`** works everywhere — class API, CLI expressions, TOML values:

```python
# tasks.py — variadic *tasks accepts str positional args
fix = Sequential("ruff check --fix .", "ruff format .")
```

**Tuple → `Sequential`, set → `Parallel`** is parser-side only — CLI
expressions and `[tool.camas.tasks]` values. The same CI pipeline can be
written without constructors at all:

```
python -m camas '("ruff format . --check", {"mypy .", "pyright ."}, "pytest")'
```

```toml
[tool.camas.tasks]
mypy = "mypy ."
pyright = "pyright ."
typecheck = "{mypy, pyright}"             # set literal → Parallel of refs
fix      = "(ruff_fix, format)"           # tuple literal → Sequential of refs
ci       = "(format_check, typecheck, test)"
```

We deliberately don't surface tuple/set coercion in Python source: type
checkers can't see implicit conversions, so `set[Task]` would show up as
unhashable / non-overlapping in your IDE. In `tasks.py`, use `Sequential(...)`
/ `Parallel(...)` directly — short aliases (`S = Sequential`, `P = Parallel`)
get you the same compactness without the inconsistency.

Use `--dry-run` to print the task tree without executing it.

## Library

The same building blocks are available as a library:

```python
import asyncio
from camas import Parallel, Sequential, run
from camas.effect.termtree import Termtree, TermtreeOptions

task = Sequential(
    "ruff format . --check",
    Parallel("mypy .", "pyright ."),
    "pytest",
)

result = asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),)))
raise SystemExit(result.returncode)
```

`Task` is an immutable `NamedTuple`; `Sequential` and `Parallel` are immutable
frozen dataclasses. A `Sequential` short-circuits on the first non-zero exit; a
`Parallel` runs its children concurrently. Both accept children variadically
(`*tasks`) — bare strings are coerced to anonymous `Task`s — and an optional
`matrix` mapping for variable expansion in task commands, names, and
environments.

## License

[MIT](LICENSE)
