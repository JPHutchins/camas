<img align="right" width="150" src="https://raw.githubusercontent.com/JPHutchins/camas/main/resources/artwork/camas-451x1023.png" alt="camas-sketch-by-jph">

# Camas

A task runner with parallel execution, matrix expansion, and pluggable output effects.

- **For developers:** live tree view that updates in place as tasks stream
- **For CI/CD:** one definition drives both local runs and CI
- **For LLMs:** _coming soon — structured JSON stream + MCP_

<br clear="all">

&nbsp;

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

> [!TIP]
> Add extras with PEP 621, e.g. `camas[github_checks]`

pipx:
```
pipx install camas
```

uv:
```
uv tool install camas
```

Nix:
```
nix run github:JPHutchins/camas                     # default
nix run github:JPHutchins/camas#with-github-checks  # adds httpx for the GitHubChecks effect
nix run github:JPHutchins/camas#with-check          # adds ty for `camas --check`
nix run github:JPHutchins/camas#all                 # both extras
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

The renderer is swappable, not the tree. Run the same `tasks.py` locally with the live `Termtree` and in CI with `Status` — one flag changes the output, the pipeline definition is unchanged.

**On GitHub Actions, no flag is needed.** Camas detects `GITHUB_ACTIONS=true` and defaults `--effects` to `(Status(StatusOptions(output_mode="github")),)` — collapsed workflow groups, ISO timestamps with millisecond precision, ANSI colors preserved.

```yaml
- run: uv run camas check
```

On other CI providers (or to opt into a specific mode), spell it out:

```yaml
- run: uv run camas check --effects='(Status(StatusOptions(output_mode="errors")),)'
```

See the [`OutputMode`](https://github.com/JPHutchins/camas/blob/main/src/camas/effect/status.py) literal and `block_for` doctests for the per-mode behavior. The [`status-modes-demo` job](https://github.com/JPHutchins/camas/blob/main/.github/workflows/ci.yaml) renders one CI run per mode for visual comparison.

For per-leaf visibility in the PR Checks panel, add the `GitHubChecks` effect alongside `Status` (opt-in extra: `camas[github_checks]`). Each leaf task becomes its own check run, so reviewers see `lint` / `mypy` / `pytest` pass-or-fail individually instead of one monolithic log.

```yaml
- run: |
    uv run --extra github_checks camas matrix --effects='(
      Status(StatusOptions(output_mode="github")),
      GitHubChecks(GitHubChecksOptions(
        sha="${{ github.event.pull_request.head.sha || github.sha }}",
      )),
    )'
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

The job needs `permissions: checks: write`. Defaults read `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, and `GITHUB_SHA` from the Actions env. The `pull_request.head.sha` override attaches checks to the PR head rather than the synthetic merge commit. See the [`github-checks-demo` job](https://github.com/JPHutchins/camas/blob/main/.github/workflows/ci.yaml) for a working example.

**When to use it.** Two things it gives you:

- **SSOT between local dev and CI.** Your `camas matrix` terminal view and the PR Checks panel show the same per-leaf shape (`lint [PY=3.10]`, `mypy [PY=3.14]`, …). The matrix definition lives once in `tasks.py`; CI doesn't re-encode it in YAML.
- **One runner instead of N.** Camas-side parallel matrix on a single runner produces the same per-`(cell, leaf)` granularity that a GHA matrix gets across N runners — at 1/N the minutes and camas gives you efficient task parallelization pushing that runner to 100% utilization as much as possible. Worth it on paid-runner budgets.

The downside is that fork PRs get a read-only `GITHUB_TOKEN` from GitHub and can't write checks — usually a non-issue for Enterprise teams where contributors have push to the org repo.

**When not to bother.** OSS gets free runners — just GHA-matrix them in parallel for faster wall-clock time. The cost is small: you give up either SSOT (matrix definition moves into YAML) or per-leaf-UI granularity (one PR check entry per runner instead of per leaf) — pick one. What *is* a deal-breaker for OSS is fork PRs: external-contributor PRs return 403 from the Checks API, so per-leaf entries silently don't appear. The `github-checks-demo` job is marked `continue-on-error` so it doesn't block CI when that happens, but `GitHubChecks` isn't a workable OSS solution.

## Effects plugins

Define an `Effect` in your `tasks.py` and it's discovered automatically — usable by name from `--effects` and listed under `camas --effects`. See [examples/effect-plugin/](https://github.com/JPHutchins/camas/tree/main/tests/fixtures/effect-plugin) for a typed `Tail` effect that streams per-task output as it arrives.

## Versioning

The public API lives behind [`camas.v0`](https://github.com/JPHutchins/camas/blob/main/src/camas/v0/__init__.py), versioned the way semver versions the package: `v0` pairs with camas 0.x and is exactly as loose as semver says 0.x is. The surface prefers to grow — new names, fields appended with defaults — but breaking changes remain possible until 1.0, made deliberately and noted in releases. At 1.0 the contract hardens: a stable-era namespace never removes or changes an exported name, a breaking change forces the next `camas.vN`, and old namespaces keep shipping, so a `tasks.py` or effect plugin written against one keeps working across upgrades.

The top-level surface (`from camas import Task, Sequential, Parallel, Effect`) re-exports the definers from the latest version namespace — best effort across major generations, fine for a `tasks.py` that lives next to its dev environment. The full plugin contract — `TaskNode`, the `TaskEvent` stream, `LeafState`, `Completion` — lives only in `camas.v0`. Everything else (`camas.core.*`, `camas.main.*`) is internal: importable today, but with no stability promise.

To pin a minimum camas *feature* level, use your dependency declaration (`camas>=0.x` in `pyproject.toml`, or PEP 723 inline metadata) — the import path covers API shape; the package pin covers feature availability.

## Reference

- **[examples/](https://github.com/JPHutchins/camas/tree/main/tests/fixtures)** — full project layouts under test coverage. The canonical reference for how to structure `tasks.py`, use `[tool.camas.tasks]` in `pyproject.toml`, drive a matrix from `.python-version`, or scope a 2-axis matrix from the CLI.
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

