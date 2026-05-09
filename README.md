# Camas

Parallel and sequential task-tree runner with matrix expansion and effects plugins.

![demo](https://raw.githubusercontent.com/JPHutchins/camas/gh-storage/demos/demo-latest.gif)

## Example

```python
from camas import Parallel, Sequential

ci = Sequential(
  Parallel(
    "ruff format . --check",
    "npx prettier ."
  ),
  Parallel(
    "ruff check .",
    "mypy .",
    "npx eslint src/",
    "pytest",
    "npx tsc --noEmit"
  ),
)
```

The animated tree above is from a live test fixture — [see the walkthrough](#walkthrough).

## Install

```
pipx install camas
```

## Why camas?

camas is **not a build system**. camas is for the specific job of running structured trees of shell commands.

| | Python task runners† | just | Task | Mage | camas |
|---|---|---|---|---|---|
| Project scope | **Python projects only** | Any | Any | **Go projects only** | Any |
| Definition language | pyproject.toml TOML or `@task` decorator | justfile DSL | YAML | Go | Python (typed AST) |
| Inline anonymous parallel groups | No | No | No (must be a named task) | No | **Yes** |
| Parallel execution | poe yes; Invoke / taskipy no | `[parallel]` attr | `deps:` (parallel) | `mg.Deps(...)` | **`Parallel(...)`** |
| Matrix expansion | No | No | `for:` + `parallel:true` | Go loops | **`matrix=`** |
| CLI matrix override (e.g. `--PY 3.13`) | No | No | No | No | **Yes** |
| Live tree output | No | No | `prefixed` / `group` modes | No | **`Termtree` (default)** |
| Pluggable output renderers | No | No | 3 built-ins | No | **`--effects`** |
| Type checking on task definitions | Partial | No | Editor schema | Yes (Go) | **mypy / pyright** |
| First release / status | 2013–2020, stable | 2016, stable | 2017, stable | 2017, stable | 2026, alpha |
| Ecosystem | Moderate (Python) | Large | Large | Moderate | None yet |

†[poethepoet](https://poethepoet.natn.io), [Invoke](https://www.pyinvoke.org), [taskipy](https://github.com/taskipy/taskipy) — pyproject.toml-bound runners that assume a Python project.

| If you need... | Reach for |
|---|---|
| Reproducible, hermetic builds | [Nix](https://nixos.org) |
| Incremental file-based builds (skip when inputs unchanged) | [Task](https://taskfile.dev) |
| Simple project command menu | [just](https://just.systems) |
| Parameterized tasks (`--env=staging`, prompts, vars) | [Task](https://taskfile.dev) |
| Go project, build logic in Go | [Mage](https://magefile.org) |
| Inline parallel/sequential trees with a live view | **camas** |
| Pluggable output effects (live tree locally, summary in CI) | **camas** |
| Matrix runs across versions/platforms, overridable from the CLI | **camas** |

## CI integration

The renderer is swappable, not the tree. Run the same `tasks.py` locally with the live `Termtree` and in CI with the post-run `Summary` — one flag changes the output, the pipeline definition is unchanged.

```yaml
- run: uv run camas check --effects='(Summary(SummaryOptions(Fixed(90))),)'
```

The project's own [.github/workflows/ci.yaml](https://github.com/JPHutchins/camas/blob/main/.github/workflows/ci.yaml) is a working example.

## Effects plugins

Define an `Effect` in your `tasks.py` and it's discovered automatically — usable by name from `--effects` and listed under `camas --effects`. See [examples/effect-plugin/](https://github.com/JPHutchins/camas/tree/main/examples/effect-plugin) for a typed `Tail` effect that streams per-task output as it arrives.

## Reference

- **[examples/](https://github.com/JPHutchins/camas/tree/main/examples)** — full project layouts under test coverage. The canonical reference for how to structure `tasks.py`, use `[tool.camas.tasks]` in `pyproject.toml`, drive a matrix from `.python-version`, or scope a 2-axis matrix from the CLI.
- **[src/camas/](https://github.com/JPHutchins/camas/tree/main/src/camas)** — typed Python with thorough docstrings. `camas --help` and `camas <task> --help` link back here.
- **`camas`** with no args lists tasks; `camas <task> --help` shows the expanded tree, matrix axes, and override flags.

## Walkthrough

The animated tree at the top is generated from this `tasks.py`:

<details>
<summary><code>examples/tauri-app/tasks.py</code> — Rust + TypeScript + Python in one tree</summary>

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

fix = Parallel(
    f"{node}/prettier --write .",
    Task("cargo fmt --all", cwd=src_tauri),
    Sequential(
        Task("uv run ruff check --fix .", cwd=python_sdk),
        Task("uv run ruff format .", cwd=python_sdk),
    ),
)

build = Parallel(
    Task("npm run tauri build {FLAG}"),
    matrix={"FLAG": ("-- --debug", "")},
    help="Debug and release builds (FLAG='-- --debug' debug, FLAG='' release)",
)
```

</details>

```
$ cd examples/tauri-app
```

**List the tasks** —

```
$ camas --list
```

<details>
<summary>output</summary>

```
Available tasks from .../tauri-app/tasks.py:
  all       frontend | backend | sdk
  backend   cargo fmt --all, cargo clippy --all-targets --locked -- -D warnings | cargo test --all-targets --locked
  build     Debug and release builds (FLAG='-- --debug' debug, FLAG='' release)  [matrix: FLAG×2 (-- --debug..)]
  fix       node_modules/.bin/prettier --write . | cargo fmt --all | (uv run ruff check --fix ., uv run ruff format .)
  frontend  node_modules/.bin/prettier --write ., node_modules/.bin/eslint src/ | node_modules/.bin/tsc --noEmit | node_modules/.bin/vitest run
  sdk       uv run ruff check --fix ., uv run ruff format ., uv run mypy . | uv run pytest
```

</details>

**Preview what `all` would run** —

```
$ camas --dry-run all
```

<details>
<summary>output</summary>

```
all ∥
┃ frontend →
┃ ├─ node_modules/.bin/prettier --write .
┃ └─ node_modules/.bin/eslint src/ | node_modules/.bin/tsc --noEmit | node_modules/.bin/vitest run
┃   ┃ node_modules/.bin/eslint src/
┃   ┃ node_modules/.bin/tsc --noEmit
┃   ┃ node_modules/.bin/vitest run
┃ backend →
┃ ├─ cargo fmt --all  (cwd: src-tauri)
┃ └─ cargo clippy --all-targets --locked -- -D warnings | cargo test --all-targets --locked
┃   ┃ cargo clippy --all-targets --locked -- -D warnings  (cwd: src-tauri)
┃   ┃ cargo test --all-targets --locked  (cwd: src-tauri)
┃ sdk →
┃ ├─ uv run ruff check --fix .  (cwd: python-sdk)
┃ ├─ uv run ruff format .  (cwd: python-sdk)
┃ └─ uv run mypy . | uv run pytest
┃   ┃ uv run mypy .  (cwd: python-sdk)
┃   ┃ uv run pytest  (cwd: python-sdk)
```

</details>

Tree-symbol key: `∥` ends a Parallel header, `→` ends a Sequential. Children hang off `┃` (parallel siblings, run concurrently) or `├─` / `└─` (sequential steps, short-circuit on first failure).

**Discover a task's matrix axes and override flags** —

```
$ camas build --help
```

<details>
<summary>output</summary>

```
usage: camas build [-h] [--dry-run] [--effects EFFECTS] [--FLAG VAL[,VAL...]]

Debug and release builds (FLAG='-- --debug' debug, FLAG='' release)

runs the 'build' task:
build ∥
┃ npm run tauri build -- --debug [FLAG=-- --debug]: npm run tauri build -- --debug  FLAG=-- --debug
┃ npm run tauri build  [FLAG=]: npm run tauri build   FLAG=

Matrix axes (override with --AXIS VAL[,VAL...]):
  --FLAG  -- --debug,
```

</details>

**Pin the matrix from the CLI** —

```
$ camas build --dry-run --FLAG '-- --debug'
```

<details>
<summary>output</summary>

```
build ∥
┃ npm run tauri build -- --debug [FLAG=-- --debug]: npm run tauri build -- --debug  FLAG=-- --debug
```

</details>

