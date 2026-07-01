<img align="right" width="150" src="https://raw.githubusercontent.com/JPHutchins/camas/main/resources/artwork/camas-451x1023.png" alt="camas-sketch-by-jph">

# Camas

A task runner with parallel execution, matrix expansion, MCP, and pluggable output effects.

- **For developers:** live tree view that updates in place as tasks stream
- **For CI/CD:** one definition drives both local runs and CI
- **For LLMs:** a closed edit→validate→run loop over structured MCP

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

Then scaffold a starter `tasks.py` in your project root:

```
camas --init
```

The starter demonstrates leaf tasks, `Sequential`/`Parallel` composition, a matrix, a [`Config`](#config) default task, and the optional [PEP 723 standalone block](#standalone-taskspy-pep-723) — cross-platform placeholder commands ready to be swapped for your real ones.

`--init` also creates a gitignored `.camas/` directory beside `tasks.py`. Camas writes run logs and a per-leaf timing cache there, so `camas --list` can annotate tasks with an estimated duration; delete the directory to opt out. Rename or relocate it with `Config(camas_dir=...)`.

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

**On GitHub Actions, no flag is needed.** Camas detects `GITHUB_ACTIONS=true` and defaults `--effects` to `(Status(output_mode="github"),)` — collapsed workflow groups, ISO timestamps with millisecond precision, ANSI colors preserved.

**Under an AI coding agent, no flag is needed either.** Camas detects agents (the `CLAUDECODE` env var, or set `CAMAS_AGENT=1`) and defaults to the line-oriented `Status` renderer instead of the live `Termtree`, whose cursor-redraw frames bloat captured output. Prefer the `camas mcp` server over the CLI from an agent; a `Config` `default_effects` always overrides the detection.

```yaml
- run: uv run camas check
```

On other CI providers (or to opt into a specific mode), spell it out:

```yaml
- run: uv run camas check --effects='(Status(output_mode="errors"),)'
```

See the [`OutputMode`](https://github.com/JPHutchins/camas/blob/main/src/camas/effect/status.py) literal and `block_for` doctests for the per-mode behavior. The [`status-modes-demo` job](https://github.com/JPHutchins/camas/blob/main/.github/workflows/ci.yaml) renders one CI run per mode for visual comparison.

For per-leaf visibility in the PR Checks panel, add the `GitHubChecks` effect alongside `Status` (opt-in extra: `camas[github_checks]`). Each leaf task becomes its own check run, so reviewers see `lint` / `mypy` / `pytest` pass-or-fail individually instead of one monolithic log.

```yaml
- run: |
    uv run --extra github_checks camas matrix --effects='(
      Status(output_mode="github"),
      GitHubChecks(sha="${{ github.event.pull_request.head.sha || github.sha }}"),
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

## Config

Bind a `Config` in `tasks.py` and bare `camas` (no arguments) runs its `default_task`:

```python
from camas import Config, Sequential, Task

lint = Task("ruff check .")
test = Task("pytest")
ci = Sequential(lint, test, name="ci")

_ = Config(default_task=ci)
```

Now `camas` runs `ci`, `camas --dry-run` previews it, and the default's matrix axes stay overridable (`camas --PY 3.13`). With no `Config` (or no `default_task`), bare `camas` prints the full help — task listing, effects, hints — and exits non-zero.

`github_task` is the CI counterpart, falling back to `default_task` when unset; it runs under `GITHUB_ACTIONS=true`. Paired with the [automatic `Status` effect](#ci-integration), one bare `camas` does the right thing in both places:

```python
_ = Config(
  default_task=ci,                            # bare `camas` locally
  github_task=Sequential(ci, "pytest --cov"), # bare `camas` under GitHub Actions
)
```

`Config` is discovered by type, so the binding's name never matters — `_` by convention. Defining two is an error.

Because `github_task` reproduces CI, running it before a push catches a CI failure locally. If it is a named task, a one-line git hook guards every push:

```sh
echo 'exec camas check' > .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

`git push --no-verify` still bypasses it deliberately. An LLM agent needs no hook — `camas_list` reports the CI task as `github_default`, so it runs `camas_run` with that name before pushing.

## Time budget (`--under`)

Once camas has timed a task's leaves (the `.camas/` cache, surfaced by `camas --list`), `camas --under=<duration>` runs only the leaves whose estimate fits a wall-clock budget — the fast inner-loop subset, picked for you instead of by hand.

```
camas --under=1s            # budget the Config default task
camas --under=500ms check   # budget a named task or expression
```

It runs the **mutating** leaves first, in sequence, then the read-only rest in parallel — so formatters never race a checker over the same files. Mark a leaf that writes the workspace with `mutates=True`:

```python
fix = Task("ruff check --fix .", mutates=True)
fmt = Task("ruff format .", mutates=True)
lint = Task("ruff check .")
```

```
$ camas --under=1s --dry-run
Time budget 1.00s — running 6 leaf(s) (0 unmeasured), excluded 2 over budget.
  over budget: pyright ~4.57s, coverage ~20.98s
fix → fmt → (mypy | ty | zuban | pyrefly)
```

Durations are `1s`, `500ms`, `2m`, `1h`, or a bare number of seconds. Only leaves *measured* to exceed the budget are excluded; a leaf with no recorded timing yet runs anyway (and is thereby measured) — skipping it would keep it forever untimed. The budget is per-leaf: a measured leaf runs when its own estimate fits, so the parallel group's wall-clock stays near the budget.

**For agents**, `camas_run` exposes the same budget as its `under` argument (omit `task` to budget the project default), and the response's `budget` field reports what was selected and excluded — a tight, time-boxed validate loop over structured MCP.

## Path scoping (`--paths`)

`camas <task> --paths <path>…` scopes a run to changed paths instead of the whole tree. A leaf opts in by writing the `{paths}` placeholder in its command and declaring its scope with `paths=` — a directory prefix, or a `(changed) -> paths` callable:

```python
py = Task("ruff format {paths}", mutates=True, paths="src")
web = Task("prettier --write {paths}", mutates=True, paths="web")
_ = Config(agent=Claude(fix=Sequential(py, web)))
```

`--paths` works on any task — `camas check --paths src/a.py`. Without it, every `{paths}` resolves to its full-run default (`ruff format src`); with it, each leaf runs only over the changed files it covers, and a leaf that covers none is dropped (`--paths` is repeatable, or comma-separated).

For the Claude Code plugin, you **register** the auto-fix node — whatever you named it — to `Config.agent.fix`; the FileChanged hook runs *that* node over the just-changed file, zero model tokens. Install the plugin (`/plugin marketplace add JPHutchins/camas`), run `camas mcp init --hooks` (which writes the hook with the launcher it resolves for your project), or wire it by hand:

```jsonc
// .claude/settings.json
{ "hooks": { "FileChanged": [
  { "hooks": [{ "type": "command", "command": "uvx 'camas[mcp]' mcp fix --paths ${file_path}" }] } ] } }
```

`camas mcp fix` runs the registered `Config.agent.fix` node (not a task named `fix` — that's just `camas fix`, your own task); with no fix registered it is a clean no-op, so the hook is harmless without it. The `uvx 'camas[mcp]'` launcher needs only `uv` on PATH (no global camas install); with `camas` already installed, bare `camas mcp fix …` works too.

## Effects plugins

Define an `Effect` in your `tasks.py` and it's discovered automatically — usable by name from `--effects` and listed under `camas --effects`. See [examples/effect-plugin/](https://github.com/JPHutchins/camas/tree/main/tests/fixtures/effect-plugin) for a typed `Tail` effect that streams per-task output as it arrives.

## Versioning

The public API is published through **versioned namespaces** — [`camas.v0`](https://github.com/JPHutchins/camas/tree/main/src/camas/v0) today, `camas.v1` and beyond later. Import from a generation to pin the API shape: a name a generation exports is never removed or changed within that generation. The scheme tracks semver — `v0` pairs with camas 0.x and is as loose as semver says 0.x is (the surface prefers to grow; breaking changes stay possible until 1.0, made deliberately and noted in releases). At 1.0 a generation freezes: a breaking change forces the next `camas.vN`, and published generations keep shipping, so a `tasks.py` or effect plugin pinned to one keeps working across upgrades.

The top-level `camas` namespace is the unversioned alias for the latest generation: `from camas import Task, Sequential, Parallel, Effect, Config` re-exports that generation's five headline definers and is kept **1:1** with its package surface. The rest of the public API for a generation — `TaskNode`, the `TaskEvent` stream, `LeafState`, `Completion` — lives in that generation's submodules, e.g. `from camas.v0.task_event import TaskEvent`. Everything under `camas.core` / `camas.main` is internal — it consumes whatever generations are installed and carries no stability promise.

To pin a minimum camas *feature* level, use your dependency declaration (`camas>=0.x` in `pyproject.toml`, or PEP 723 inline metadata) — the import path covers API shape; the package pin covers feature availability.

## Standalone `tasks.py` (PEP 723)

For a non-Python project that wants a single-file task runner — no `pyproject.toml`, no venv to manage — give `tasks.py` a [PEP 723](https://peps.python.org/pep-0723/) header and a `run_cli(globals())` entry point, then run it with any PEP 723-aware tool:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["camas>=0.1.8"]
# ///
"""Build tasks for my project."""

from camas import Parallel, Task, run_cli

lint = Task("ruff check .")
test = Task("pytest")
check = Parallel(lint, test)

if __name__ == "__main__":
    run_cli(globals())
```

```bash
uv run tasks.py check     # uv reads the header, builds an ephemeral env, runs
uv run tasks.py --list    # every camas flag still works
pipx run tasks.py test
```

`run_cli(globals())` introspects the module for `Task` / `Sequential` / `Parallel` and `Effect` bindings — exactly what camas does when it auto-discovers a `tasks.py` — so the standalone file behaves identically to a discovered one: `camas <task> --help`, `--dry-run`, matrix overrides, and `--check` all work and cite the file. `run_cli` is part of the stable surface (`from camas import run_cli`, or `from camas.v0 import run_cli` to pin the generation); it's imported lazily, so a plain `from camas import Task` doesn't pull it in.

The header owns version pinning (`dependencies = ["camas>=0.1.8"]`) and the interpreter floor (`requires-python`); it's inert to `camas --check` and to auto-discovery, which read the module the same way with or without it.

## Reference

- **[examples/](https://github.com/JPHutchins/camas/tree/main/tests/fixtures)** — full project layouts under test coverage. The canonical reference for how to structure `tasks.py`, use `[tool.camas.tasks]` in `pyproject.toml`, drive a matrix from `.python-version`, or scope a 2-axis matrix from the CLI.
- **[src/camas/](https://github.com/JPHutchins/camas/tree/main/src/camas)** — typed Python with thorough docstrings. `camas --help` and `camas <task> --help` link back here.
- **`camas`** with no args runs the `Config` default task (or prints the full help when none is defined); `camas <task> --help` shows the expanded tree, matrix axes, and override flags.

## Walkthrough

The animated tree at the top is generated from this `tasks.py`:

<details>
<summary><code>examples/tauri-app/tasks.py</code> — Rust + TypeScript + Python in one tree</summary>

```python
from pathlib import Path

from camas import Config, Parallel, Sequential, Task

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

_ = Config(default_task=all)
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

