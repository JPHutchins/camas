<!-- Generated from docs/README.in.md by `python -m docs.readme`; edit that file. -->
<!-- DRAFT for JP: each bullet list outlines README.md.old's prose for rewriting; ⚠️ bullets are proofreading notes; ✍️ bullets mark where your prose on each type goes, right above its generated section. Code examples and tables are kept verbatim, in their original places. -->
<img align="right" width="150" src="https://raw.githubusercontent.com/JPHutchins/camas/main/resources/artwork/camas-451x1023.png" alt="camas-sketch-by-jph">

# Camas

- A task runner with parallel execution, matrix expansion, MCP, and pluggable output effects.
- **For developers:** live tree view that updates in place as tasks stream
- **For CI/CD:** one definition drives both local runs and CI
- **For LLMs:** a closed edit→validate→run loop over structured MCP

<br clear="all">

&nbsp;

![demo](https://raw.githubusercontent.com/JPHutchins/camas/gh-storage/demos/demo-latest.gif)

## Example

- The example runs two formatters in parallel, then five lints and tests in parallel.

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

- The animated tree above comes from a live test fixture; link to the [walkthrough](#walkthrough).
- ⚠️ Your WIP removed the operators paragraph (`|`, `+`, `-`, `>`, and `.extend`/`.remove`). They're public API, so cover them somewhere or cut them on purpose. See the Operators bullets under Reference.

## Install

pipx:
```
pipx install camas
```

uv:
```
uv tool install camas
```

<details><summary>Optional Features (PEP 621):</summary>

- `github_checks`: httpx, for the `GitHubChecks` effect
- `ctrf`: msgspec, for the `Ctrf` effect
- `check`: ty, for `camas --check`
- `mcp`: the MCP server (`camas mcp`); includes `check`
- `all`: every extra above
- ⚠️ This list is copied by hand from `pyproject.toml`'s `[project.optional-dependencies]`. An `extras` directive could generate it.

</details>

Nix:
```
nix run github:JPHutchins/camas
```

<details><summary>Nix Optional Features</summary>

- `#default`: no extras
- `#with-github-checks`, `#with-ctrf`, `#with-check`, `#with-mcp`: one package per extra (`with-<extra>`, `_` → `-`)
- `#all`: every extra
- ⚠️ This list is copied by hand from `flake.nix`.

</details>

- Or manage it with your dependency tooling of choice.
- Python floor (was a NOTE):
  - camas requires Python ≥ 3.10.
  - In a project that still supports older interpreters (`requires-python = ">=3.8"`), `uv add camas` / `uv lock` fail to resolve, because camas can't install for the sub-3.10 part of the range.
  - Fix 1: guard the dependency with an environment marker, `"camas[mcp]==0.1.26 ; python_version >= '3.10'"`.
  - Fix 2: keep camas out of the project environment and launch it with uvx (`camas mcp init --launcher uvx`).
  - Either way camas runs from its own ≥3.10 interpreter and shells out, so it still drives the sub-floor cells.
- Then scaffold a starter `tasks.py` in your project root:

```
camas --init
```

<details><summary><code>camas --init</code></summary>

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["camas"]
# ///
"""Project tasks — run with `camas`.

Every task below is a cross-platform placeholder (`python -c ...`);
replace the placeholders with your real commands:

    lint = Task("ruff check .")
    test = Task("cargo test", cwd=Path("rust"))
    build = Task("npm run build")

The `# /// script` block above is optional PEP 723 inline metadata. With
it (plus the `__main__` block at the bottom), PEP 723-aware tools can run
this file standalone — `uv run tasks.py`, `pipx run tasks.py` — building
a throwaway env with camas installed; handy for non-Python projects. It is
also where camas gets version-pinned (`dependencies = ["camas>=X.Y"]`).
Both are inert under plain `camas`; delete them if you don't want the
standalone path.
"""

from camas import Claude, Config, Parallel, Sequential, Task, run_cli

# A leaf is any shell command, shlex-split (so quote the -c payload). A bare
# string inside Sequential/Parallel coerces to an anonymous Task; tuple form
# Task(("python", "-c", "...")) skips shlex when quoting gets hairy.
hello = Task("python -c \"print('hello from camas')\"")

# matrix= clones the subtree per axis value, interpolating {NAME} into
# cmd/env/cwd; env= is scoped to the subtree. Pin an axis from the CLI:
# camas greet --NAME Grace
greet = Parallel(
    Task("python -c \"import os; print(os.environ['GREETING'] + ', {NAME}!')\""),
    matrix={"NAME": ("Ada", "Grace")},
    env={"GREETING": "hello"},
    help="say hello to everyone at once",
)

# Sequential runs in order, short-circuiting on the first failure; Parallel
# runs concurrently. They nest freely. The binding name is the task name
# (this defines `ci`); pass name= only to rename or to name a nested,
# anonymous group like the Parallel below.
ci = Sequential(
    hello,
    Parallel(greet, "python --version"),
)

# Your deterministic, behavior-preserving auto-fixers, with {paths} so a run scopes to the
# changed files. Register the node (named anything) to Config.agent.fix below; the camas_gate
# tool settles it over the changed files before checking, and the Claude Code Stop hook runs it at
# every turn end, for free. Replace the placeholder with your real fixers, e.g.:
#   autofix = Task("ruff check --fix {paths}", mutates=True, paths=".")
#   autofix = Sequential(Task("ruff check --fix {paths}", mutates=True), Task("ruff format {paths}", mutates=True), paths=".")
autofix = Task('python -c "" {paths}', name="autofix", mutates=True, paths=".")

# Config is discovered by type, under any binding name (here `_`): bare `camas` runs
# default_task — or github_task under GitHub Actions, falling back to default_task when unset.
# agent= wires the Claude Code plugin: agent.fix is the registered autofix node above; the gate
# checks default_task (override with Claude(fix=..., check=...)). A checking
# leaf can add agent_format=("--output-format sarif", "sarif") for machine-readable gate output.
_ = Config(default_task=ci, agent=Claude(fix=autofix))

# The PEP 723 standalone flow (see the docstring): running this file directly
# (`uv run tasks.py <task>`) dispatches through camas. Inert when the `camas`
# command discovers this file itself.
if __name__ == "__main__":
    run_cli(globals())
```

</details>

- The starter:
  - demonstrates leaf tasks, `Sequential`/`Parallel` composition, a matrix, a [`Config`](#config) default task, and the optional [PEP 723 standalone block](#standalone-taskspy-pep-723)
  - uses cross-platform placeholder commands, ready to swap for your real ones
- `--init` also creates a gitignored `.camas/` directory beside `tasks.py`:
  - it holds run logs and a per-leaf timing cache, so `camas --list` can show each task's estimated duration
  - delete it to opt out; rename or relocate it with `Config(camas_dir=...)`
- Venvs inside `.camas/` (was a TIP):
  - point a multi-version matrix's per-version environments into `.camas/` (`env={"UV_PROJECT_ENVIRONMENT": ".camas/.venv-{PY}", "UV_PYTHON": "{PY}"}`), which is already gitignored
  - in the project root, a later `.`-scanning tool (a formatter, a linter that isn't gitignore-aware) descends into `.venv-3.13/` and friends
  - a formatter's default `.venv` exclude often won't match a `.venv-3.13` suffix, so the breakage depends on the tool and only appears *after* a matrix run
- `camas --init --verbose` writes a kitchen-sink `tasks.py`:
  - it covers every `Task`/`Sequential`/`Parallel`/`Config` option: path scoping, matrix expansion, `agent_format` structured output, `Config(agent=Claude(fix=..., check=..., default=...))`, and a commented recipe for sourcing a matrix axis from your project's SSOT
  - each option is worked and explained in place
  - the MCP `camas_init` tool writes this template by default (`verbose=false` for the short one)

## Why camas?

- camas is **not a build system**; it's for running structured trees of shell commands.
- ⚠️ The comparison tables are hand-maintained claims about other tools; check them when you rewrite.

| | Python task runners† | just | Task | Mage | camas |
|---|---|---|---|---|---|
| Project scope | **Python projects only** | Any | Any | **Go projects only** | Any |
| Definition language | pyproject.toml TOML or `@task` decorator | justfile DSL | YAML | Go | Python (typed AST) |
| Inline anonymous parallel groups | No | No | No (must be a named task) | No | **Yes** |
| Parallel execution | poe yes; Invoke / taskipy no | `[parallel]` attr | `deps:` (parallel) | `mg.Deps(...)` | **`Parallel(...)`** |
| Matrix expansion | No | No | `for:` + `parallel:true` | Go loops | **`matrix=`** |
| Coupled (non-cross-product) fan-out | No | No | No | Go loops | **`variants=`** |
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

- The renderer is swappable, not the tree: the same `tasks.py` runs locally with the live `Termtree` and in CI with `Status`. One flag changes the output; the pipeline definition stays the same.
- **On GitHub Actions, no flag is needed:**
  - camas detects `GITHUB_ACTIONS=true` and defaults `--effects` to `(Status(output_mode="github"),)`
  - the result is collapsed workflow groups, ISO timestamps with millisecond precision, and preserved ANSI colors
- **Under an AI coding agent, no flag is needed either:**
  - camas detects agents (the `CLAUDECODE` env var, or `CAMAS_AGENT=1`) and defaults to the line-oriented `Status` instead of the live `Termtree`, whose cursor-redraw frames bloat captured output
  - from an agent, prefer the `camas mcp` server over the CLI
  - a `Config` `default_effects` always overrides the detection

```yaml
- run: uv run camas check
```

- On other CI providers, or to opt into a specific mode, spell out the effect:

```yaml
- run: uv run camas check --effects='(Status(output_mode="errors"),)'
```

- Per-mode behavior: the `OutputMode` literal (below) and the `block_for` doctests in [`status.py`](https://github.com/JPHutchins/camas/blob/main/src/camas/effect/status.py). The [`status-modes-demo` job](https://github.com/JPHutchins/camas/blob/main/.github/workflows/ci.yaml) renders one CI run per mode for comparison.

```python
OutputMode: TypeAlias = Literal["quiet", "all", "errors", "stream", "github"]
```

- `GitHubChecks` gives per-leaf visibility in the PR Checks panel:
  - add it alongside `Status`; it's in the opt-in extra `camas[github_checks]`
  - each leaf becomes its own check run, so reviewers see `lint` / `mypy` / `pytest` pass or fail individually instead of one monolithic log

```yaml
- run: |
    uv run --extra github_checks camas matrix --effects='(
      Status(output_mode="github"),
      GitHubChecks(sha="${{ github.event.pull_request.head.sha || github.sha }}"),
    )'
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- What the example needs:
  - the job needs `permissions: checks: write`
  - the defaults read `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, and `GITHUB_SHA` from the Actions environment
  - the `pull_request.head.sha` override attaches checks to the PR head rather than the synthetic merge commit
  - the [`github-checks-demo` job](https://github.com/JPHutchins/camas/blob/main/.github/workflows/ci.yaml) is a working example
  - ⚠️ `github_checks.py`'s public module docstring states the extra, the permission, the env defaults, and the head-SHA override. A `module` directive could generate this list.

- ✍️ `GitHubChecks`: what it's for and how it's used

Per-leaf check runs on the GitHub Checks API.

| `GitHubChecks` | Type | Default | |
|---|---|---|---|
| `token` | `str \| None` | `None` |  |
| `repository` | `str \| None` | `None` |  |
| `sha` | `str \| None` | `None` |  |
| `name_prefix` | `str` | `""` |  |
| `tail_bytes` | `int` | `8192` |  |
| `fail_on_api_error` | `bool` | `False` |  |

- **When to use it:**
  - SSOT between local dev and CI: the `camas matrix` terminal view and the PR Checks panel show the same per-leaf shape (`lint [PY=3.10]`, `mypy [PY=3.14]`, …). The matrix is defined once, in `tasks.py`; CI doesn't re-encode it in YAML.
  - One runner instead of N: camas's parallel matrix on one runner gives the same per-`(cell, leaf)` granularity a GHA matrix gets across N runners, at 1/N the minutes, and camas's parallelism keeps that runner close to 100% utilization. It's worth it on paid-runner budgets.
  - Downside: fork PRs get a read-only `GITHUB_TOKEN` and can't write checks. That's usually a non-issue for Enterprise teams, whose contributors can push to the org repo.
- **When not to bother:**
  - OSS gets free runners, so GHA-matrix them in parallel for faster wall-clock time
  - the cost is giving up either SSOT (the matrix moves into YAML) or per-leaf UI granularity (one check per runner instead of per leaf); pick one
  - fork PRs are the deal-breaker for OSS: external contributors' PRs get 403 from the Checks API, so the per-leaf entries silently don't appear
  - the `github-checks-demo` job is `continue-on-error` for that reason; `GitHubChecks` isn't a workable OSS solution

### GitHub Actions matrix (`--github-matrix`)

- It fans out across N runners *without* giving up SSOT:
  - `camas <task> --github-matrix` emits the task's fan-out as the JSON that GHA's `strategy.matrix` consumes
  - a `discover` job sources it from `tasks.py`, and downstream jobs read it with `fromJSON`, so the fan-out lives in one place
  - this repo dogfoods it: see the [`discover` and `check` jobs](https://github.com/JPHutchins/camas/blob/main/.github/workflows/ci.yaml), which fan out over `.python-version`
- There are three shapes, one per way a `tasks.py` fans out:

| `tasks.py` | emitted | job runs |
|---|---|---|
| `matrix={"PY": (...)}` — a cross product | `{"PY": ["3.13", "3.14"]}` | `camas <task>` with the cell's axes pinned (flags, or job `env`) |
| `variants=({...}, {...})` — coupled bundles | `{"include": [{...}, {...}]}` | same, pinning the cell's keys |
| a `Parallel` of named tasks, no axes | `{"task": ["build", "lint", "test"]}` | `camas ${{ matrix.task }}` |

- The `task` shape is the common CI shape: N independent jobs, no version matrix. Its values are the *binding* names `camas <task>` dispatches on, so an emitted job can't 404 in CI.
- Either object-of-arrays form composes with a YAML-side axis such as `os`, which a shell command couldn't change from inside a job anyway:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest]
    task: ${{ fromJSON(needs.discover.outputs.matrix).task }}
```

- `--github-matrix=axes|variants|tasks` pins the shape. Pin it in your `discover` step, so a later `tasks.py` refactor fails there instead of quietly handing the workflow a differently shaped object.
- camas checks the emitted object against the task's real run-set:
  - each job must run exactly its own cell's leaves, and the jobs together must cover every leaf exactly once
  - a fan-out with no faithful projection (independent fan-outs in one tree, a plain leaf beside matrixed siblings) is rejected rather than silently widened or narrowed
- Python below camas's floor (was a NOTE):
  - camas needs ≥3.10, but a library might test 3.8+; the sub-floor cells can't `uv run camas`, because camas isn't installable there
  - naively switching to `uvx` doesn't help: `setup-uv`'s `python-version` (and any job-level `UV_PYTHON`) also pins `uvx camas` onto the sub-floor interpreter
  - decouple camas's interpreter from the cell's by scoping `UV_PYTHON` to the sync step only:

```yaml
- uses: astral-sh/setup-uv@...             # no python-version -> no job-wide UV_PYTHON
- run: uv sync
  env:
    UV_PYTHON: ${{ matrix.python-version }}    # step-scoped: builds the cell's .venv
- run: uvx camas==0.1.26 <task>            # camas gets its own >=3.10 interpreter;
                                            # its leaves' `uv run` reuse the synced .venv
```

- The `discover` job still emits the matrix from `tasks.py`, so SSOT holds; only the per-cell launcher changes.

### Machine-readable report (`Ctrf`)

- For a CI artifact or an AI code review's input, add the `Ctrf` effect (opt-in extra: `camas[ctrf]`).
- It writes the run as a [CTRF](https://ctrf.io) JSON report: each leaf is a test with status, duration, output, command, and exit code.
- `path=` writes a file; the default is stdout.

```sh
uv run --extra ctrf camas check --effects='(Status(output_mode="errors"), Ctrf(path="ctrf-report.json"))'
```

- ✍️ `Ctrf`: what it's for and how it's used

A CTRF JSON test report.

| `Ctrf` | Type | Default | |
|---|---|---|---|
| `path` | `str \| None` | `None` |  |
| `tail_bytes` | `int` | `8192` |  |

## Config

- Bind a `Config` in `tasks.py`, and bare `camas` (no arguments) runs its `default_task`:

```python
from camas import Config, Sequential, Task

lint = Task("ruff check .")
test = Task("pytest")
ci = Sequential(lint, test, name="ci")

_ = Config(default_task=ci)
```

- Then `camas` runs `ci`, `camas --dry-run` previews it, and the default task's matrix axes stay overridable (`camas --PY 3.13`).
- With no `Config` (or no `default_task`), bare `camas` prints the full help (task listing, effects, hints) and exits non-zero.
- `github_task` is the CI counterpart:
  - it runs under `GITHUB_ACTIONS=true`, falling back to `default_task` when unset
  - paired with the [automatic `Status` effect](#ci-integration), one bare `camas` does the right thing in both places

```python
_ = Config(
  default_task=ci,                            # bare `camas` locally
  github_task=Sequential(ci, "pytest --cov"), # bare `camas` under GitHub Actions
)
```

- `Config` is discovered by type, so the binding's name never matters (`_` by convention); defining two is an error.
- Running `github_task` before a push catches a CI failure locally:
  - ⚠️ This paragraph was under "Leaf color" in the old README; it's about `github_task`, so I moved it here.
  - if it's a named task, a one-line git hook guards every push:

```sh
echo 'exec camas check' > .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

- `git push --no-verify` still bypasses the hook deliberately.
- An LLM agent needs no hook: when a `github_task` is declared, `camas_list` reports it as `github_default`, so the agent runs `camas_run` with that name before pushing. The field is `null` when no `github_task` is set; camas never infers it from your CI workflow files.

- ✍️ `Config`: what it's for and how it's used

Project configuration for a `tasks.py`.

| `Config` | Type | Default | |
|---|---|---|---|
| `default_task` | `TaskNode \| None` | `None` | The task a bare `camas` runs, unless `github_task` applies. |
| `github_task` | `TaskNode \| None` | `None` | The bare task under GitHub Actions; `None` defers to `default_task`. |
| `default_effects` | `tuple[Effect[Any], ...] \| None` | `None` | `None` keeps the engine's default; `()` runs none. |
| `default_github_effects` | `tuple[Effect[Any], ...] \| None` | `None` | `None` keeps the engine's default; `()` runs none. |
| `camas_dir` | `str` | `".camas"` | Subdirectory for run logs and timings. |
| `agent` | `Agent \| None` | `None` | The nodes the agent gate and MCP tools run. |
| `leaf_color` | `bool` | `True` | Whether camas forces color on in leaf subprocesses. |

- ✍️ `Claude`: what it's for and how it's used

The Claude Code agent integration.

| `Claude` | Type | Default | |
|---|---|---|---|
| `fix` | `TaskNode` | required | The deterministic, behavior-preserving autofix node. |
| `check` | `TaskNode \| None` | `None` | The node the gate checks; `None` defers to the project's bare task. |
| `default` | `TaskNode \| None` | `None` | The task a no-task `camas_run` runs; `None` defers to the check node, then the GitHub task, then the default task. |

```python
Agent: TypeAlias = Claude
```

The supported coding-agent integrations.

### Leaf color (`leaf_color`)

- camas runs every leaf on a pipe, so a tool that probes for a terminal drops its color, and the tree and any ANSI-rendering CI log lose it.
- So camas forces color on in the leaf environment (`FORCE_COLOR=1`, `CLICOLOR_FORCE=1`). That's why `camas lint` stays colored where a bare `lint | cat` wouldn't.
- That environment is inherited, so it also reaches commands a leaf spawns itself, including one whose output the leaf *captures and asserts on*:
  - a test checking that a CLI prints `Passes: 1, failures: 1` passes by hand and fails under camas, because the string arrives wrapped in ANSI
  - when a leaf's output is data rather than display, turn the forcing off, project-wide or per leaf:

```python
_ = Config(leaf_color=False) # project-wide
check = Task("cargo test -p mycli", env={"NO_COLOR": "1"}) # or just this leaf
```

- `NO_COLOR` wins over both, per [no-color.org](https://no-color.org).
- `leaf_color=False` only stops camas from forcing; a leaf that asks for color itself (`env={"FORCE_COLOR": "1"}`, or an exported one) still gets it.

## Time budget (`--under`)

- Once camas has timed a task's leaves (the `.camas/` cache, surfaced by `camas --list`), `camas --under=<duration>` runs only the leaves whose estimate fits a wall-clock budget: the fast inner-loop subset, picked for you instead of by hand.

```
camas --under=1s            # budget the Config default task
camas --under=500ms check   # budget a named task or expression
```

- It runs the **mutating** leaves first, in sequence, then the read-only rest in parallel, so formatters never race a checker over the same files. Mark a leaf that writes the workspace with `mutates=True`:

```python
fix = Task("ruff check --fix .", mutates=True)
fmt = Task("ruff format .", mutates=True)
lint = Task("ruff check .")
```

- ⚠️ This `--dry-run` output was captured by hand from camas's own tasks; a `cli` directive could generate it (open question 5).

```
$ camas --under=1s --dry-run
Time budget 1.00s — running 6 leaf(s) (0 unmeasured), excluded 2 over budget.
  over budget: pyright ~4.57s, coverage ~20.98s
fix → fmt → (mypy | ty | zuban | pyrefly)
```

- Durations are `1s`, `500ms`, `2m`, `1h`, or a bare number of seconds.
- Only leaves *measured* over the budget are excluded. A leaf with no recorded timing yet runs anyway, and is measured; skipping it would keep it untimed forever.
- The budget is per leaf: a measured leaf runs when its own estimate fits, so the parallel group's wall-clock time stays near the budget.
- **For agents:** `camas_run` takes the same budget as its `under` argument (omit `task` to budget the project default), and the response's `budget` field reports what was selected and excluded. That makes a tight, time-boxed validate loop over structured MCP.

## Path scoping (`--paths`)

- `camas <task> --paths <path>…` scopes a run to changed paths instead of the whole tree.
- A command opts in by writing the `{paths}` placeholder and declaring its scope with `paths=`: a directory prefix, or a `(changed) -> paths` callable.
- A `Sequential`/`Parallel` may carry `paths=` too. It's the default scope for descendant `{paths}` commands that set none, the same way `env`/`cwd` propagate into a group's leaves:

```python
py = Task("ruff format {paths}", mutates=True, paths="src")
web = Task("prettier --write {paths}", mutates=True, paths="web")
# the group's paths="." is the default for both children (neither sets its own):
autofix = Sequential(Task("ruff check --fix {paths}", mutates=True), Task("ruff format {paths}", mutates=True), paths=".")
_ = Config(agent=Claude(fix=Sequential(py, web, autofix)))
```

- `--paths` works on any task (`camas check --paths src/a.py`):
  - without it, every `{paths}` resolves to its full-run default (`ruff format src`)
  - with it, each `{paths}` command runs only over the changed files it covers, and a command covering none is dropped
  - it's repeatable, or comma-separated
- A command **without** `{paths}` can't be narrowed, so its `paths=` is a no-op and it always runs, unless it declares `when=`. camas errs on correctness: a tool it can't narrow might be affected by the edit.
- `when=` scopes a command that can't take `{paths}` (`cargo build`, `nix flake check`, `ctest`):
  - it takes a directory-prefix string or `Path` (coerced to its POSIX prefix), a tuple of those, or a `(changed) -> bool` callable
  - on a scoped run, a leaf whose `when=` doesn't match the changed set is dropped; a full run never consults it
  - like `paths=`, a group's `when=` is the default for descendant leaves that set none
  - a leaf with a `cwd` but no `when=` gates on its `cwd` directory (the monorepo file-tree default); set `when="."` to opt back into always-run

```python
build = Task("cargo build", when="src")                # runs only when src/ changed
flake = Task("nix flake check", when=("flake.nix", "nix"))
tests = Task("cargo test", cwd="code-gen")             # when= defaults to "code-gen"
```

- A `paths=` callable is called with `()` on a full run:
  - one that filters the changed set would return `()` and strip the command's arguments entirely (a formatter reading stdin on no args hangs)
  - `by_suffix(suffixes, default=...)` is the safe factory: it filters the changed files by suffix on a scoped run and returns `default` on a full run (`by_glob` is the shell-glob variant)

```python
tidy = Task("clang-tidy {paths}", paths=by_suffix((".c", ".h"), default=("src",)))
```

```python
PathScope: TypeAlias = Callable[[tuple[str, ...]], tuple[str, ...]]
```

Maps the changed paths to the arguments at `{paths}`; called with `()` on a full run,
and returning `()` on a scoped run skips the leaf.

```python
WhenPredicate: TypeAlias = Callable[[tuple[str, ...]], bool]
```

Whether a leaf runs for the changed paths; never called on a full run.

```python
def by_suffix(suffixes: tuple[str, ...], default: tuple[str, ...] = (".",)) -> PathScope
```

A `PathScope` for files ending in one of `suffixes`.

<details><summary><code>by_suffix</code> examples</summary>

```pycon
>>> f = by_suffix((".c", ".h"), default=("src", "include"))
>>> f(())
('src', 'include')
>>> f(("a.c", "b.py", "c.h"))
('a.c', 'c.h')
>>> f(("b.py",))
()
```

</details>

```python
def by_glob(patterns: tuple[str, ...], default: tuple[str, ...] = (".",)) -> PathScope
```

A `PathScope` for files matching one of the shell-glob `patterns`.

<details><summary><code>by_glob</code> examples</summary>

```pycon
>>> f = by_glob(("homeassistant/**/*.py", "homeassistant/**/*.pyi"), default=("homeassistant",))
>>> f(())
('homeassistant',)
>>> f(("homeassistant/core.py", "homeassistant/x.pyi", "tests/t.py", "README.md"))
('homeassistant/core.py', 'homeassistant/x.pyi')
>>> f(("homeassistant/components/hue/light.py",))
('homeassistant/components/hue/light.py',)
>>> f(("README.md",))
()
```

</details>

- `camas --check` (and the MCP `camas_check`) flags both authoring mistakes as advisory warnings: a leaf whose own `paths=` can never apply (no `{paths}` token; use `when=` instead), and a `{paths}` callable that goes empty on a full run.

## Claude Code

- ⚠️ This material was under "Path scoping" in the old README. It's a topic of its own, so it gets its own section here.
- You **register** the autofix node, whatever you named it, as `Config.agent.fix`:
  - the `camas_gate` MCP tool settles *that* node over the changed files before it checks
  - a `Stop` hook runs it at every turn end: deterministically, at zero model tokens, and never between an agent's edits, so a half-finished edit (an import not yet used) survives until the agent gates
  - the headless `camas mcp gate` stays read-only, so CI checks exactly what was committed
- `camas mcp init --claude` writes the full Claude Code setup in one command: `.mcp.json`, the Stop autofix and async nudge hooks, the tiered camas-fixer agents, and the `gate` skill.
- It resolves the launcher for your project (`uv run camas`/`uv run tasks.py`, `uvx`, or a PATH `camas`) and pins it:
  - to `tasks.py`'s PEP 723 block when present, else to the running camas release
  - a dev/local build is left unpinned, since it isn't published to pin against
  - re-run `camas mcp init --claude` after bumping either, to keep the pin current
- The launcher it picks is the one that can still see your project's tools:
  - a `uv.lock`/PEP 723 project gets `uv run`
  - a `camas` in the project's environment (a virtual environment, or a flake devShell out of `/nix/store`) is preferred over `uvx`, which is isolated from both and can't run `ruff`/`mypy`/`treefmt`/`go` at all
  - a global `camas` isn't preferred: it shares nothing with the project, and the pinned `uvx` entry is more portable
  - detection reads where `camas` itself resolves, not `VIRTUAL_ENV`/`IN_NIX_SHELL`, which only report that a shell was entered
  - when init picks the environment's `camas`, the committed entry needs your team in that same environment (activate the venv, or enter the devShell, before starting the client); init says so
  - `--launcher uv|uvx|camas` forces a strategy
- `camas mcp init` alone writes a bare `.mcp.json` for any MCP client, with the same launcher and pin resolution and no Claude Code files.
- Both warn when git excludes a file they just wrote:
  - a repo whose `.gitignore` covers `.claude/` can only commit `.mcp.json`, so the autofix/gate loop stays per-developer (each teammate re-runs `camas mcp init --claude` in their own clone) until that rule is opened up
  - the warning prints the `.gitignore` lines that would open it up
- Or wire it by hand:

```jsonc
// .claude/settings.json
{ "hooks": { "Stop": [
  { "hooks": [{ "type": "command", "command": "camas mcp fix" }] } ] } }
```

- `camas mcp fix`:
  - runs the registered `Config.agent.fix` node, not a task named `fix` (that's just `camas fix`, your own task)
  - reads the changed files from `--paths` or a `PostToolBatch` event on stdin; a `Stop` event names none, so the fix covers the whole tree
  - with no fix registered it's a clean no-op, so the hook is harmless without one
  - it's synchronous: it holds the turn until it has run, so it can't race the agent's next edit
- The second `Stop` hook, `camas mcp gate --under 5s --nudge` (`async: true, asyncRewake: true`):
  - a headless, time-boxed check that first waits for the sibling fix to mark the prompt settled (Claude Code runs a `Stop` event's hooks in parallel)
  - silent when green; otherwise it wakes the main agent with a reminder to delegate to the camas-fixer ladder (see the `gate` skill), without ever blocking the turn
  - self-limiting: at most one wake per prompt (it honors the Stop event's `stop_hook_active` and tracks the `prompt_id` it last nudged)
  - a configuration state (no check node registered, a `tasks.py` load error, a missing `camas[mcp]` extra) exits `0` silently instead of waking the agent over something a rewake can't fix

## Drift gate (`Clean`)

- `Clean` keeps committed generated code up to date: it runs the generator, then fails if the working tree changed.

```python
from camas import Clean, Task

openapi = Clean(Task("make update-openapi", mutates=True))
```

- `openapi` is a `Sequential` of three leaves: a check that the tree is clean, the generator, and the same check again.
  - a dirty tree fails the first check and skips the generator
  - a failed second check prints the drift: the `git diff` of the changed tracked files, staged or not, then the `git status --porcelain` lines, untracked files included
- `check=` replaces the whole-tree default with any task or command string whose exit 0 means clean:
  - `check="git diff --exit-code -- schema/"` scopes the gate to one directory
  - but `git diff` doesn't see untracked files, so a generator that adds files needs a check that does
  - a check can't carry `{paths}`
- `before=False` drops the first check; a tree that starts dirty then fails as drift.
- The default check reads git's view of the whole tree, where ignored paths don't count, so nothing else may write while a gate runs:
  - a formatter sequenced before a gate fails the gate's first check whenever it changes a file
  - two gates under one `Parallel` see each other's writes
  - run the gates first, in sequence:

```python
from camas import Clean, Sequential, Task

generators = Sequential(
  Clean(Task("go generate ./...", mutates=True)),
  Clean(Task("make update-openapi", mutates=True)),
)
```

```python
def Clean(mutator: Task | str, *, check: Task | str = GIT_PORCELAIN(), before: bool = True, name: str | None = None) -> Sequential
```

A drift gate around `mutator`: fail when the working tree is not clean.

Raises:

- `ValueError`: on a `mutator` without `mutates=True` or a `check` with `{paths}`.

<details><summary><code>Clean</code> examples</summary>

```pycon
>>> [t.name for t in Clean(Task("make gen", mutates=True)).tasks]
['make gen-before', None, 'make gen-after']
>>> Clean(Task("make gen", mutates=True), before=False).tasks[0].cmd
'make gen'
```

</details>

```python
def GIT_PORCELAIN() -> Task
```

A fresh copy of the default `Clean` drift check.

## When `Parallel` doesn't help

- `Parallel` runs in the time of its slowest leaf only when the machine has room for every leaf at once.
- It pays for independent work (distinct tools, or a long emulator or integration run overlapped with quick static checks) and across CI runners, each with its own cores and build directory.
- Otherwise, keep the work in one leaf or in sequence:
  - **Leaves that each fill the CPU:** multicore tools on one machine share the same cores, so `golangci-lint` beside another Go linter, or `cargo clippy` beside `cargo test` on a warm cache, gains little over running them in sequence. `--jobs N` caps a fan-out that oversubscribes the machine; it can't make one leaf faster.
  - **Leaves behind one build lock:** concurrent `cargo` commands over one `target/` wait on cargo's build-directory lock. Keep the batched invocation (`cargo hack`, `cargo-batch`, an `xtask`) as a single leaf.
  - **A tool with its own scheduler:** three `nix build .#attr` leaves evaluate the flake three times; one `nix build .#a .#b .#c` evaluates it once and schedules the builds itself.
  - **Leaves that write the tree:** `mutates=True` orders leaves only under [`--under`](#time-budget---under). In a normal run, mutating leaves in one `Parallel` race each other and their read-only siblings, and a [`Clean`](#drift-gate-clean) gate fails on writes it didn't make. Run them first, in sequence:

```python
from camas import Parallel, Sequential, Task

dev = Sequential(
  Task("ruff format .", mutates=True),
  Task("ruff check --fix .", mutates=True),
  Parallel("mypy .", "pytest"),
)
```

- To fan out across machines, let camas split the outer axes (target, chip, toolchain version) over CI runners with [`--github-matrix`](#github-actions-matrix---github-matrix), and keep each tool's batched invocation inside one leaf.

## Monorepos

- A `tasks.py` composes others with `Project`. Binding one imports a child `tasks.py` as a task node (a self-contained child project) and mounts the child's own tasks under the binding name:

```python
from camas import Claude, Config, Parallel, Project

libs = Project("libs")            # camas libs, camas libs.search.lint, ...
api  = Project("services/api")    # name the handle whatever you like

_ = Config(
    default_task=Parallel(libs, api, name="all"),      # each child's default_task, in parallel
    github_task=Parallel(libs, api, name="ci"),        # each child's github_task
    agent=Claude(
        fix=Parallel(libs, api, name="fix"),           # each child's fix node
        check=Parallel(libs, api, name="check"),       # each child's check node
    ),
)
```

- Name each composite, or bind it to a variable, so `camas_list` can report it as the default. An anonymous inline `Parallel`/`Sequential` in a `Config` task field shows up as `null`, and `camas --check` flags it.
- A reference composes the child's **matching field**:
  - the same bare `libs` grabs the child's `default_task` in `default_task`, its `github_task` in `github_task`, its fix node in `agent.fix`, and its check node in `agent.check`
  - so a `Parallel` of references in any slot is that slot composed across the monorepo, each child contributing its own
- A binding name resolves by context: `camas libs` runs whatever a bare `camas` runs in that directory (its default locally, its `github_task` under CI, its agent default under an agent).
- `camas libs.search.lint` reaches a task the child exposes (because `libs/tasks.py` itself did `search = Project("search")`), and expressions compose across namespaces (`camas '{libs.search.lint, api.deploy}'`).
- Nodes stay anchored where they were authored: a leaf's `cwd` and its `paths=`/`when=` scopes are relative to its own `tasks.py`, rebased across the boundary wherever `camas` is invoked from.
- Because a leaf's `when=` defaults to its `cwd`, a scoped run (the gate, or `--paths`) runs only the children whose directory changed, with no per-leaf `when=` anywhere in a child's `tasks.py`.
- Children are referenced by path relative to the importing file, and live within its directory.
- The [monorepo fixture](https://github.com/JPHutchins/camas/tree/main/tests/fixtures/monorepo) exercises the permutations.

```python
def Project(path: str) -> TaskNode
```

Another `tasks.py`, by path relative to this one, as a task node.

## Effects

- Define an `Effect` in your `tasks.py` and it's discovered automatically: usable by name from `--effects`, and listed under `camas --effects`.
- The [effect-plugin fixture](https://github.com/JPHutchins/camas/tree/main/tests/fixtures/effect-plugin) has a typed `Tail` effect that streams each task's output as it arrives.
- ⚠️ The old link text said `examples/effect-plugin/`, but it points at `tests/fixtures/effect-plugin`.
- An effect author reads the event stream and the per-leaf states below.

```python
class Effect(Protocol[T]):
    async def setup(self, task: TaskNode) -> T: ...
    async def on_event(self, event: TaskEvent, states: Sequence[LeafState], ctx: T) -> T: ...
    async def teardown(self, ctxs: tuple[T, ...]) -> None: ...
```

<details><summary><code>Effect</code> methods</summary>

- `setup`: Return the context every leaf starts with.
- `on_event`: Return the event's leaf's next context; calls for different leaves may overlap.
- `teardown`: Receive every leaf's final context, in leaf-index order.

</details>

```python
TaskEvent: TypeAlias = StartedEvent | OutputEvent | CompletedEvent
```

An event a run emits.

- ✍️ `StartedEvent`: what it's for and how it's used

A leaf started.

| `StartedEvent` | Type | Default | |
|---|---|---|---|
| `task` | `Task` | required |  |
| `leaf_index` | `int` | required | The leaf's index into the `states` that `on_event` receives. |
| `timestamp` | `datetime` | required | When the event happened. |

<details><summary><code>StartedEvent</code> examples</summary>

```pycon
>>> from datetime import datetime
>>> from camas.v0 import Task
>>> StartedEvent(Task("hi"), 0, datetime(2026, 1, 1, 12, 0, 0))
StartedEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, timestamp=datetime.datetime(2026, 1, 1, 12, 0))
```

</details>

- ✍️ `OutputEvent`: what it's for and how it's used

A leaf printed a line.

| `OutputEvent` | Type | Default | |
|---|---|---|---|
| `task` | `Task` | required |  |
| `leaf_index` | `int` | required | The leaf's index into the `states` that `on_event` receives. |
| `line` | `bytes` | required |  |
| `timestamp` | `datetime` | required | When the event happened. |

<details><summary><code>OutputEvent</code> examples</summary>

```pycon
>>> from datetime import datetime
>>> from camas.v0 import Task
>>> OutputEvent(Task("hi"), 0, b"hello", datetime(2026, 1, 1, 12, 0, 1))
OutputEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, line=b'hello', timestamp=datetime.datetime(2026, 1, 1, 12, 0, 1))
```

</details>

- ✍️ `CompletedEvent`: what it's for and how it's used

A leaf completed.

| `CompletedEvent` | Type | Default | |
|---|---|---|---|
| `task` | `Task` | required |  |
| `leaf_index` | `int` | required | The leaf's index into the `states` that `on_event` receives. |
| `completion` | `Completion` | required |  |
| `timestamp` | `datetime` | required | When the event happened. |

<details><summary><code>CompletedEvent</code> examples</summary>

```pycon
>>> from datetime import datetime
>>> from camas.v0 import Task
>>> from camas.v0.completion import Finished, Skipped
>>> CompletedEvent(Task("hi"), 0, Finished(0, 1.0, (b"done",)), datetime(2026, 1, 1, 12, 0, 2))
CompletedEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, completion=Finished(returncode=0, elapsed=1.0, output=(b'done',)), timestamp=datetime.datetime(2026, 1, 1, 12, 0, 2))
>>> CompletedEvent(Task("hi"), 0, Skipped(1), datetime(2026, 1, 1, 12, 0, 0))
CompletedEvent(task=Task(cmd='hi', name=None, env={}, cwd=None), leaf_index=0, completion=Skipped(returncode=1, blocked_by=None), timestamp=datetime.datetime(2026, 1, 1, 12, 0))
```

</details>

```python
LeafState: TypeAlias = Waiting | Running | Interrupting | Completed
```

Where a leaf is in its run.

- ✍️ `Waiting`: what it's for and how it's used

A leaf not yet started.

| `Waiting` | Type | Default | |
|---|---|---|---|
| `task` | `Task` | required |  |

<details><summary><code>Waiting</code> examples</summary>

```pycon
>>> from camas.v0 import Task
>>> Waiting(Task("echo hi"))
Waiting(task=Task(cmd='echo hi', name=None, env={}, cwd=None))
```

</details>

- ✍️ `Running`: what it's for and how it's used

A running leaf.

| `Running` | Type | Default | |
|---|---|---|---|
| `task` | `Task` | required |  |
| `start_time` | `datetime` | required | When the leaf started. |
| `last_line` | `bytes` | required | The leaf's latest output line; empty before its first. |

<details><summary><code>Running</code> examples</summary>

```pycon
>>> from datetime import datetime
>>> from camas.v0 import Task
>>> Running(Task("echo hi"), datetime(2026, 1, 1, 12, 0, 0), b"output")
Running(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=datetime.datetime(2026, 1, 1, 12, 0), last_line=b'output')
```

</details>

- ✍️ `Interrupting`: what it's for and how it's used

A signalled leaf that has not exited yet.

| `Interrupting` | Type | Default | |
|---|---|---|---|
| `task` | `Task` | required |  |
| `start_time` | `datetime` | required | When the leaf started. |
| `last_line` | `bytes` | required | The leaf's latest output line; empty before its first. |
| `presses` | `int` | required | Ctrl-C presses this leaf has absorbed. |

<details><summary><code>Interrupting</code> examples</summary>

```pycon
>>> from datetime import datetime
>>> from camas.v0 import Task
>>> Interrupting(Task("echo hi"), datetime(2026, 1, 1, 12, 0, 0), b"output", 1)
Interrupting(task=Task(cmd='echo hi', name=None, env={}, cwd=None), start_time=datetime.datetime(2026, 1, 1, 12, 0), last_line=b'output', presses=1)
```

</details>

- ✍️ `Completed`: what it's for and how it's used

A leaf that is done.

| `Completed` | Type | Default | |
|---|---|---|---|
| `task` | `Task` | required |  |
| `completion` | `Completion` | required |  |

<details><summary><code>Completed</code> examples</summary>

```pycon
>>> from camas.v0 import Task
>>> from camas.v0.completion import Finished, Skipped
>>> Completed(Task("echo hi"), Finished(0, 0.5, ()))
Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Finished(returncode=0, elapsed=0.5, output=()))
>>> Completed(Task("echo hi"), Skipped(1))
Completed(task=Task(cmd='echo hi', name=None, env={}, cwd=None), completion=Skipped(returncode=1, blocked_by=None))
```

</details>

```python
Completion: TypeAlias = Finished | Skipped | Stopped | Errored
```

How a leaf's run ended.

- ✍️ `Finished`: what it's for and how it's used

A leaf that ran to exit.

| `Finished` | Type | Default | |
|---|---|---|---|
| `returncode` | `int` | required |  |
| `elapsed` | `float` | required | Seconds the leaf ran. |
| `output` | `Sequence[bytes]` | required | The lines the leaf printed, in order. |

<details><summary><code>Finished</code> examples</summary>

```pycon
>>> Finished(0, 1.234, (b"all clean",))
Finished(returncode=0, elapsed=1.234, output=(b'all clean',))
```

</details>

- ✍️ `Skipped`: what it's for and how it's used

A leaf skipped because of an earlier leaf's outcome.

| `Skipped` | Type | Default | |
|---|---|---|---|
| `returncode` | `int` | required | The blocking leaf's return code. |
| `blocked_by` | `str \| None` | `None` | The blocking leaf's label, when known. |

<details><summary><code>Skipped</code> examples</summary>

```pycon
>>> Skipped(1)
Skipped(returncode=1, blocked_by=None)
>>> Skipped(1, "lint").blocked_by
'lint'
```

</details>

- ✍️ `Stopped`: what it's for and how it's used

A leaf of a run or pipe the runner stopped.

| `Stopped` | Type | Default | |
|---|---|---|---|
| `returncode` | `int` | required |  |
| `elapsed` | `float` | required | Seconds the leaf ran. |
| `output` | `Sequence[bytes]` | required | The lines the leaf printed, in order. |

<details><summary><code>Stopped</code> examples</summary>

```pycon
>>> Stopped(130, 0.5, (b"caught SIGINT",))
Stopped(returncode=130, elapsed=0.5, output=(b'caught SIGINT',))
```

</details>

- ✍️ `Errored`: what it's for and how it's used

A leaf whose command could not start.

| `Errored` | Type | Default | |
|---|---|---|---|
| `returncode` | `int` | required |  |
| `message` | `str` | required | Why the command could not start. |

<details><summary><code>Errored</code> examples</summary>

```pycon
>>> Errored(127, "no such file or directory: does-not-exist")
Errored(returncode=127, message='no such file or directory: does-not-exist')
```

</details>

### Built-in effects

- `Termtree` (the default off-agent and off-CI): a live task tree in the terminal
- `Status`: line-oriented status output (the default under GitHub Actions and agents)
- `Summary`: the final task tree, rendered once after the run
- `Ctrf`: a CTRF JSON test report (see above)
- `GitHubChecks`: per-leaf check runs on the GitHub Checks API (see above)
- `Timings`: each leaf's observed duration, recorded for later runs
- ⚠️ These one-line summaries are the classes' docstrings, which `camas --effects` prints. A directive could generate them; the tables below cover only the constructors. `Status`'s format-string defaults render as long f-strings.

- ✍️ `Termtree`: what it's for and how it's used

A live task tree in the terminal.

| `Termtree` | Type | Default | |
|---|---|---|---|
| `frame_interval_ms` | `float` | `16.667` |  |
| `show_passing` | `bool` | `False` |  |
| `output_ctrl_c` | `bool` | `False` |  |

- ✍️ `Status`: what it's for and how it's used

Line-oriented status output.

| `Status` | Type | Default | |
|---|---|---|---|
| `output_mode` | `OutputMode` | `"errors"` |  |
| `started_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{CAMAS_VIOLET}▶ [{{name}}] started{RESET}"` |  |
| `finished_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{GREEN}✓ [{{name}}] success{RESET} ({{elapsed:.3f}}s)"` |  |
| `failed_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{RED}✗ [{{name}}] error{RESET} exit={{rc}} ({{elapsed:.3f}}s)"` |  |
| `skipped_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{GREY}⏭ [{{name}}] skipped{RESET} (prior rc={{rc}})"` |  |
| `stopped_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{YELLOW}■ [{{name}}] stopped{RESET} exit={{rc}} ({{elapsed:.3f}}s)"` |  |
| `errored_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{RED}⚠ [{{name}}] errored{RESET} {{msg}}"` |  |
| `output_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}] · [{{name}}]{RESET} {{line}}"` |  |

- ✍️ `Summary`: what it's for and how it's used

The final task tree, rendered once after the run.

| `Summary` | Type | Default | |
|---|---|---|---|
| `term_width` | `TermWidth` | `Auto()` |  |
| `show_passing` | `bool` | `False` |  |

```python
TermWidth: TypeAlias = Auto | Fixed
```

The terminal width `Summary` renders to.

- ✍️ `Fixed`: what it's for and how it's used

A fixed terminal width.

| `Fixed` | Type | Default | |
|---|---|---|---|
| `columns` | `int` | required |  |

<details><summary><code>Fixed</code> examples</summary>

```pycon
>>> Fixed(200)
Fixed(columns=200)
```

</details>

- ✍️ `Timings`: what it's for and how it's used

Each leaf's observed duration, recorded for later runs.

| `Timings` | Type | Default | |
|---|---|---|---|
| `camas_dir` | `Path` | required |  |
| `scope` | `int` | `0` |  |
| `identities` | `tuple[timings.CacheKey, ...] \| None` | `None` |  |

## Versioning

- The public API is published through **versioned namespaces**: [`camas.v0`](https://github.com/JPHutchins/camas/tree/main/src/camas/v0) today, `camas.v1` and beyond later.
- Import from a generation to pin the API shape: a name a generation exports is never removed or changed within that generation.
- The scheme tracks semver:
  - `v0` pairs with camas 0.x and is as loose as semver says 0.x is: the surface prefers to grow, and breaking changes stay possible until 1.0, made deliberately and noted in releases
  - at 1.0 a generation freezes: a breaking change forces the next `camas.vN`, and published generations keep shipping, so a `tasks.py` or effect plugin pinned to one keeps working across upgrades
- The top-level `camas` namespace is the unversioned alias for the latest generation, kept **1:1** with its package surface:
  - ⚠️ The old text said `camas` re-exports "five headline definers" (`Task, Sequential, Parallel, Effect, Config`). It re-exports 14 names plus `run_cli`; the generated lists below are the source of truth.

```python
from camas import (
    GIT_PORCELAIN,
    Agent,
    AgentFormat,
    Claude,
    Clean,
    Config,
    Effect,
    Parallel,
    Pipe,
    Project,
    Sequential,
    Task,
    by_glob,
    by_suffix,
    run_cli,
)
```

```python
from camas.v0 import (
    Agent,
    Claude,
    Config,
    Effect,
    GIT_PORCELAIN,
    AgentFormat,
    Clean,
    Parallel,
    Pipe,
    Project,
    Sequential,
    Task,
    by_glob,
    by_suffix,
    run_cli,
)
```

- The rest of a generation's public API (`TaskNode`, the `TaskEvent` stream, `LeafState`, `Completion`) lives in that generation's submodules, e.g. `from camas.v0.task_event import TaskEvent`.
- Everything under `camas.core` / `camas.main` is internal: it consumes whatever generations are installed and carries no stability promise.
- To pin a minimum camas *feature* level, use your dependency declaration (`camas>=0.x` in `pyproject.toml`, or PEP 723 inline metadata). The import path covers API shape; the package pin covers feature availability.

## Standalone `tasks.py` (PEP 723)

- For a non-Python project that wants a single-file task runner (no `pyproject.toml`, no venv to manage): give `tasks.py` a [PEP 723](https://peps.python.org/pep-0723/) header and a `run_cli(globals())` entry point, then run it with any PEP 723-aware tool.
- ⚠️ `tests/fixtures/pep723/tasks.py` is a tested version of this example; a `file` directive could replace the hand-written one.

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

- `run_cli(globals())` introspects the module for `Task` / `Sequential` / `Parallel` and `Effect` bindings, exactly as camas does when it auto-discovers a `tasks.py`, so the standalone file behaves identically to a discovered one: `camas <task> --help`, `--dry-run`, matrix overrides, and `--check` all work and cite the file.
- `run_cli` is part of the stable surface (`from camas import run_cli`, or `from camas.v0 import run_cli` to pin the generation). It's imported lazily, so a plain `from camas import Task` doesn't pull it in.
  - ⚠️ `camas.v0` doesn't re-export `run_cli` (see the generated lists under Versioning), so `from camas.v0 import run_cli` fails.
- The header owns version pinning (`dependencies = ["camas>=0.1.8"]`) and the interpreter floor (`requires-python`). It's inert to `camas --check` and to auto-discovery, which read the module the same way with or without it.

```python
def run_cli(scope: Mapping[str, object]) -> None
```

Run the camas CLI against a `tasks.py`'s `globals()`, for `uv run tasks.py`.

## Reference

- [Examples](https://github.com/JPHutchins/camas/tree/main/tests/fixtures): full project layouts under test coverage, and the canonical reference for structuring `tasks.py`, using `[tool.camas.tasks]` in `pyproject.toml`, driving a matrix from `.python-version`, or scoping a 2-axis matrix from the CLI.
  - ⚠️ The old link text said `examples/`, but it points at `tests/fixtures`.
- [src/camas/](https://github.com/JPHutchins/camas/tree/main/src/camas): typed Python with thorough docstrings; `camas --help` and `camas <task> --help` link back here.
- `camas` with no arguments runs the `Config` default task, or prints the full help when none is defined; `camas <task> --help` shows the expanded tree, matrix axes, and override flags.
- Operators:
  - ⚠️ These are missing from the draft (see Example). `|` extends a plain `Parallel`, `+` a plain `Sequential`, `-` removes a direct child, and `>` builds a `Pipe`; a left operand with fields, of a subclass, or of another kind nests whole. `+` binds tighter than `|`. `>` chains as a comparison, so parenthesize. A composed pipe is always truthy.

### `Task`

- ✍️ `Task`: what it's for and how it's used

A leaf that runs one command.

| `Task` | Type | Default | |
|---|---|---|---|
| `cmd` | `str \| tuple[str, ...]` | required | Argv tokens, or a string split by POSIX shell rules on every OS. |
| `name` | `str \| None` | `None` | Renames a module-level binding, or names a nested node. |
| `env` | `Mapping[str, str]` | `MappingProxyType({})` | Variables set on top of the inherited environment. |
| `cwd` | `str \| Path \| None` → `Path \| None` | `None` | The directory the command runs in. |
| `help` | `str \| None` | `None` | A description of the task. |
| `mutates` | `bool` | `False` | The leaf writes the workspace. |
| `paths` | `str \| PathScope \| None` | `None` | The scope a `{paths}` command narrows to on a scoped run. |
| `when` | `str \| Path \| tuple[str \| Path, ...] \| WhenPredicate \| None` → `str \| tuple[str, ...] \| WhenPredicate \| None` | `None` | Which changes keep the leaf in a scoped run. |
| `agent_format` | `AgentFormat \| tuple[str, OutputKind] \| None` → `AgentFormat \| None` | `None` | The structured-output variant of the command that gate runs use. |

<details><summary><code>Task</code> examples</summary>

```pycon
>>> Task("echo hi")
Task(cmd='echo hi', name=None, env={}, cwd=None)
>>> Task(("ruff", "check", "."), name="lint")
Task(cmd=('ruff', 'check', '.'), name='lint', env={}, cwd=None)
>>> Task("cargo test", cwd=Path("src-tauri")).cwd == Path("src-tauri")
True
>>> Task("cargo test", cwd="src-tauri").cwd == Path("src-tauri")
True
>>> Task("ruff check .", help="Lint all sources").help
'Lint all sources'
>>> Task("ruff format .", mutates=True)
Task(cmd='ruff format .', name=None, env={}, cwd=None, mutates=True)
>>> Task("ruff format {paths}", mutates=True, paths=".")
Task(cmd='ruff format {paths}', name=None, env={}, cwd=None, mutates=True, paths='.')
>>> Task("cargo build", when=("src", "include"))
Task(cmd='cargo build', name=None, env={}, cwd=None, when=('src', 'include'))
>>> Task("cargo build", when=Path("src")).when
'src'
>>> Task("cargo build", when=(Path("src"), "include")).when
('src', 'include')
>>> Task("ruff check .", agent_format=AgentFormat("--output-format sarif", "sarif"))
Task(cmd='ruff check .', name=None, env={}, cwd=None, agent_format=AgentFormat(args='--output-format sarif', kind='sarif', limit=8000))
>>> Task("ruff check .", agent_format=("--output-format sarif", "sarif")).agent_format
AgentFormat(args='--output-format sarif', kind='sarif', limit=8000)
>>> hash(Task("a")) == hash(Task("a"))
True
>>> {Task("a", env={"K": "v"}), Task("a", env={"K": "v"})} == {Task("a", env={"K": "v"})}
True
```

</details>

- ✍️ `AgentFormat`: what it's for and how it's used

A leaf's agent-only structured-output variant.

| `AgentFormat` | Type | Default | |
|---|---|---|---|
| `args` | `str` | required | Appended to the command on gate runs; a literal `{report}` names the file the gate reads. |
| `kind` | `OutputKind` | required | The format of the diagnostics `args` produces. |
| `limit` | `int` | `8_000` | Characters of a structured payload or report file the gate inlines. |

<details><summary><code>AgentFormat</code> examples</summary>

```pycon
>>> AgentFormat("--output-format sarif", "sarif")
AgentFormat(args='--output-format sarif', kind='sarif', limit=8000)
>>> AgentFormat("--junitxml {report}", "junit").args
'--junitxml {report}'
```

</details>

```python
OutputKind: TypeAlias = Literal["sarif", "rdjson", "lsp", "junit", "tap", "raw"]
```

The format a leaf's diagnostics are in.

### `Sequential` and `Parallel`

- ✍️ `Sequential`: what it's for and how it's used

A group of tasks that run one after another, short-circuiting on failure.

| `Sequential` | Type | Default | |
|---|---|---|---|
| `*tasks` | `TaskNode \| str` → `tuple[TaskNode, ...]` |  | The child nodes; a `str` becomes a `Task`. |
| `name` | `str \| None` | `None` | Renames a module-level binding, or names a nested node. |
| `matrix` | `dict[str, tuple[str, ...]] \| None` | `None` | Axis values the group expands over, as a cross product. |
| `variants` | `tuple[dict[str, str], ...] \| None` | `None` | Coupled bundles of axis values, not crossed with each other. |
| `env` | `dict[str, str] \| None` → `dict[str, str]` | `None` | Variables the group's leaves inherit. |
| `cwd` | `str \| Path \| None` → `Path \| None` | `None` | A default working directory for the group's leaves. |
| `help` | `str \| None` | `None` | A description of the task. |
| `paths` | `str \| PathScope \| None` | `None` | A default `{paths}` scope for the group's leaves. |
| `when` | `str \| Path \| tuple[str \| Path, ...] \| WhenPredicate \| None` → `str \| tuple[str, ...] \| WhenPredicate \| None` | `None` | A default `when` for the group's leaves. |

<details><summary><code>Sequential</code> examples</summary>

```pycon
>>> Sequential("build", "test", name="ci").tasks
(Task(cmd='build', name=None, env={}, cwd=None), Task(cmd='test', name=None, env={}, cwd=None))
```

</details>

### `Pipe`

- ✍️ `Pipe`: what it's for and how it's used

Leaf stages piped stdout to stdin, each its own argv, with no shell.

| `Pipe` | Type | Default | |
|---|---|---|---|
| `*tasks` | `TaskNode \| str` → `tuple[TaskNode, ...]` |  | The child nodes; a `str` becomes a `Task`. |
| `agent_only` | `bool` | `False` | Pipe on agent runs only; a human run keeps the first stage alone. |
| `name` | `str \| None` | `None` | Renames a module-level binding, or names a nested node. |
| `matrix` | `dict[str, tuple[str, ...]] \| None` | `None` | Axis values the group expands over, as a cross product. |
| `variants` | `tuple[dict[str, str], ...] \| None` | `None` | Coupled bundles of axis values, not crossed with each other. |
| `env` | `dict[str, str] \| None` → `dict[str, str]` | `None` | Variables the group's leaves inherit. |
| `cwd` | `str \| Path \| None` → `Path \| None` | `None` | A default working directory for the group's leaves. |
| `help` | `str \| None` | `None` | A description of the task. |
| `paths` | `str \| PathScope \| None` | `None` | A default `{paths}` scope for the group's leaves. |
| `when` | `str \| Path \| tuple[str \| Path, ...] \| WhenPredicate \| None` → `str \| tuple[str, ...] \| WhenPredicate \| None` | `None` | A default `when` for the group's leaves. |

<details><summary><code>Pipe</code> examples</summary>

```pycon
>>> Pipe("cargo clippy --message-format=json", "clippy-sarif").agent_only
False
>>> Pipe("a", "b", agent_only=True).tasks[0].cmd
'a'
>>> Pipe()
Traceback (most recent call last):
    ...
ValueError: Pipe needs at least one stage
>>> Pipe("a", "b")
Pipe(tasks=(Task(cmd='a', name=None, env={}, cwd=None), Task(cmd='b', name=None, env={}, cwd=None)), name=None, matrix=None, env={}, cwd=None)
>>> Pipe("a", Sequential("b"))
Traceback (most recent call last):
    ...
ValueError: Pipe stages must be Tasks — a nested group would mean several commands sharing one stream
```

</details>

### Types

```python
TaskNode: TypeAlias = Task | Sequential | Parallel | Pipe
```

Any node of a task tree.

```python
Nodes: TypeAlias = TaskNode | str | Sequence[TaskNode]
```

One node or a sequence of nodes; a lone `str` is a `Task`.

## Walkthrough

- The animated tree at the top is generated from this `tasks.py`: Rust, TypeScript, and Python in one tree.

<details>
<summary><code>tests/fixtures/tauri-app/tasks.py</code></summary>

```python
"""camas task definitions for the tauri-app fixture."""

from pathlib import Path

from camas import Config, Parallel, Sequential, Task

src_tauri = Path("src-tauri")
python_sdk = Path("python-sdk")
node = Path("node_modules/.bin")

frontend = Task(f"{node}/prettier --write .") + (
    Task(f"{node}/eslint src/") | Task(f"{node}/tsc --noEmit") | Task(f"{node}/vitest run")
)

backend = Task("cargo fmt --all", cwd=src_tauri) + (
    Task("cargo clippy --all-targets --locked -- -D warnings", cwd=src_tauri)
    | Task("cargo test --all-targets --locked", cwd=src_tauri)
)

sdk = (
    Task("uv run ruff check --fix .", cwd=python_sdk)
    + Task("uv run ruff format .", cwd=python_sdk)
    + (Task("uv run mypy .", cwd=python_sdk) | Task("uv run pytest", cwd=python_sdk))
)

all = frontend | backend | sdk

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

- ⚠️ The command outputs below are hand-captured and already stale: `camas --list` now prints `all (default)` and wider columns. A `cli` directive could generate them from the fixture (open question 5). The `cd examples/tauri-app` path is stale too; it's `tests/fixtures/tauri-app`.

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

- Tree-symbol key: `∥` ends a `Parallel` header and `→` ends a `Sequential`. Children hang off `┃` (parallel siblings, run concurrently) or `├─` / `└─` (sequential steps, which stop at the first failure).

**Discover a task's matrix axes and override flags** —

```
$ camas build --help
```

<details>
<summary>output</summary>

```
usage: camas build [-h] [--dry-run | --github-matrix [SHAPE]] [--effects EFFECTS] [--FLAG VAL[,VAL...]]

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

## From `__init__.py` (to distribute in the restructure)

- ⚠️ `__init__.py`'s docstring prose, outlined so the README covers it before that file shrinks to its doctests. Points the draft above already covers are left out: `Project`, `Clean`, the `Config` defaults, the pre-push hook, `when=`/`by_suffix`, and agent detection on the CLI.
- **Running tasks** (Getting started):
  - `camas <name>` runs a task; `camas --list` enumerates them; `camas --help` shows everything (tasks, effects, hints); `camas --effects` lists the output renderers; `camas --tree` prints every task's full expansion
  - `--dry-run` prints the tree without running it
  - a trailing `-- ARG ...` forwards arguments to the dispatched task's command (`camas test -- -v`)
  - shell forms the examples stand for: `camas hello`, `camas ci --dry-run`, `camas test -- -v`, `camas greet --NAME=Ada`, `camas build --effects='(Loud(),)'`
- **Defining tasks:**
  - a leaf is *any* shell command; the doctests use `python -c` only because every CI environment has Python. Real leaves look like `Task("ruff check .")`, `Task("cargo test", cwd=Path("rust"))`, `Task("npm test")`
  - a module-level `Task`/`Sequential`/`Parallel` binding takes its **variable name** as the task name (`lint = Task("ruff check .")` defines `lint`), so `name=` is only for renaming or naming a nested, anonymous group; in the `[tool.camas.tasks]` TOML form the table key is the name
  - `Sequential` runs in order and stops at the first failure: later leaves are reported `Skipped`, not run, and camas exits non-zero; `Parallel` runs concurrently
  - bare strings inside the constructors become anonymous `Task`s, split by shell rules (`shlex`), so prefer simple forms; use the tuple form for a complex command to skip `shlex` entirely
  - `cwd` and `env` are scoped per leaf
  - `help="..."` overrides what `camas --list` prints for a task; rarely needed (the command or tree usually documents itself), but there for cryptic commands
  - model independent, read-only work as `Parallel` (wall-clock `max`, not `sum`): a `&&` chain in a previous runner was *sequencing*, not a dependency, so reach for `Sequential` only for real ordering (a mutating step, or one that consumes a prior step's output)
- **Matrix and variants:**
  - `--AXIS VAL` overrides a matrix axis from the CLI, and `<task> --help` documents each task's override flags
  - an override takes comma-separated values (`--NAME1=Clara,Dolores`), which can introduce values the project didn't list
  - `variants=` is a coupled fan-out (values that only make sense together, like a target and its toolchain): it runs the listed bundles, not their product
  - pinning one of a bundle's keys from the CLI *filters* to the bundles that bind it; replacing the value would fabricate a pairing the project never declared
  - source matrix axis values from the project's single source of truth (e.g. read `.python-version`) instead of hardcoding a list that drifts; `tasks.py` is real Python
- **Optimizations (`--under`):**
  - the budgeted subset is what a pre-commit / git hook wants
  - a cold cache (a fresh clone) runs the whole tree under a budget until each leaf has been measured once
  - a pipe kept whole for its untimed siblings runs its over-budget stages too
  - `by_glob` is a prefix-plus-suffix, `**`-aware shell-glob scope, for porting a pre-commit `files:` filter
- **Continuous integration:**
  - with `github_task` set, a CI step is just `camas` (or `uv run camas`), so the workflow doesn't go stale when `github_task` changes
- **AI coding agents:**
  - prefer the MCP server: `camas mcp [--help | --plain]`; `camas mcp init --claude` (`.mcp.json` + hooks + fixer ladder + gate skill); `camas mcp init` (`.mcp.json` only, for any MCP client)
  - the MCP server re-executes `tasks.py` on every tool call, so a command computed from the filesystem at the top level of `tasks.py` (a glob, a file list) stays fresh; values computed in a separately imported module are import-cached and can go stale; prefer `{paths}`/`when=` over snapshotting file lists where freshness matters
  - a detected agent on the CLI gets a one-line stderr banner steering it to the MCP tools; `CAMAS_NO_MCP_HINT=1` silences it
  - `Claude.fix` is declared, not inferred from `mutates=`, because a mutating leaf may be a compiler or codegen rather than a fixer; scope it with `{paths}` on a leaf or a whole group
  - `Claude.check` is scoped by `--paths` and time-boxed by `--under`; the plugin hands it to a tiered ladder of background camas-fixer subagents (lint/format tiers, then a test/coverage tier) that drive the changed scope to green off the main agent's context, rather than blocking on a hook
  - `agent_format=("--output-format sarif", "sarif")` makes the gate collect machine-readable diagnostics, appended only on gate runs, never human runs; the kinds are `sarif`, `rdjson`, `lsp`, `junit`, `tap`, `raw`
  - `AgentFormat(limit=...)` (default 8000 characters): a structured payload over it is never dumped or tailed, since a truncated SARIF/JUnit document is invalid, so the gate points at the full file or log instead
  - path mode, for tools that write diagnostics to a *file* (`pytest --junitxml`, `pytest-json-report`): put the literal `{report}` in `args` (`agent_format=("--junitxml {report}", "junit")`), and the gate substitutes an allocated path and reads that file
- **Output and effects:**
  - `Summary(show_passing=True)` prints every passing task's output after the run
  - a custom `Effect` defined inline in `tasks.py` is discovered automatically: `camas --help` lists it under *Available Effects* beside the built-ins (so an agent reading `--help` sees the plugin), and `camas --effects` with no value prints the effects-only listing
  - `--effects` takes a *tuple* of any length, so several effects run in one go: `--effects=(Summary(), Loud())` gives both the post-run report and the streaming lines
- **Reference:**
  - [tests/](https://github.com/JPHutchins/camas/blob/main/tests) for advanced, granular library usage
