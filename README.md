<!-- Generated from docs/README.in.md by `python -m docs.readme`; edit that file. -->
<!-- DRAFT for JP: each bullet list outlines README.md.old's prose for rewriting; ⚠️ bullets are proofreading notes; ✍️ bullets mark where your prose on each type goes, right above its generated section; ✚ bullets mark topics with no prose yet. Code examples and tables are kept verbatim. The `example` sections show the doctests' commands as written, including the `--effects=(Summary(show_passing=True),)` they use to check output. -->
<img align="right" width="150" src="https://raw.githubusercontent.com/JPHutchins/camas/main/resources/artwork/camas-451x1023.png" alt="camas-sketch-by-jph">

# Camas

A task runner with parallel execution, matrix expansion, MCP, and pluggable output effects.

- **For developers:** live tree view that updates in place as tasks stream
- **For CI/CD:** one definition drives both local runs and CI
- **For agents:** a closed edit→validate→run loop over structured MCP

<br clear="all">

&nbsp;

![demo](https://raw.githubusercontent.com/JPHutchins/camas/gh-storage/demos/demo-latest.gif)

<hr>
<details><summary>Table Of Contents</summary>

- [Example tasks file](#example-tasks-file)
- [Getting started](#getting-started)
  - [Install](#install)
  - [As a Development Environment Dependency](#as-a-development-environment-dependency)
  - [Your first `tasks.py`](#your-first-taskspy)
  - [Running tasks](#running-tasks)
- [Defining tasks](#defining-tasks)
  - [Tasks and groups](#tasks-and-groups)
  - [Pipes](#pipes)
  - [Operators](#operators)
  - [Matrices and variants](#matrices-and-variants)
  - [The default task](#the-default-task)
- [Syntax and type safety](#syntax-and-type-safety)
  - [Typed definitions](#typed-definitions)
  - [Checking definitions (`camas --check`)](#checking-definitions-camas---check)
  - [Typed CLI expressions](#typed-cli-expressions)
- [Optimizations](#optimizations)
  - [Path scoping](#path-scoping)
  - [Time budget](#time-budget)
  - [When `Parallel` doesn't help](#when-parallel-doesnt-help)
- [Continuous integration](#continuous-integration)
  - [GitHub Actions](#github-actions)
  - [GitHub Actions matrix (`--github-matrix`)](#github-actions-matrix---github-matrix)
  - [Drift gate (`Clean`)](#drift-gate-clean)
  - [Pre-push hook](#pre-push-hook)
- [AI coding agents](#ai-coding-agents)
  - [The MCP server](#the-mcp-server)
  - [Claude Code setup](#claude-code-setup)
  - [Fix and gate hooks](#fix-and-gate-hooks)
  - [Agents on the CLI](#agents-on-the-cli)
  - [Structured diagnostics](#structured-diagnostics)
- [Project layout](#project-layout)
  - [Monorepos](#monorepos)
  - [Standalone `tasks.py` (PEP 723)](#standalone-taskspy-pep-723)
  - [Python projects](#python-projects)
- [Output and effects](#output-and-effects)
  - [Choosing output](#choosing-output)
  - [Built-in effects](#built-in-effects)
  - [Per-leaf checks (`GitHubChecks`)](#per-leaf-checks-githubchecks)
  - [Reports (`Ctrf`)](#reports-ctrf)
  - [Color in leaf output (`leaf_color`)](#color-in-leaf-output-leaf_color)
  - [Writing an effect](#writing-an-effect)
- [Versioning](#versioning)
- [Reference](#reference)
  - [Task definitions](#task-definitions)
  - [Configuration](#configuration)
  - [Functions](#functions)
  - [Effect API](#effect-api)
  - [Built-in effect constructors](#built-in-effect-constructors)
  - [Exports](#exports)
  - [CLI](#cli)
  - [Environment variables](#environment-variables)
  - [MCP tools](#mcp-tools)
- [Walkthrough](#walkthrough)

</details>
<hr>

### Example tasks file

This example runs two formatters in parallel, then five lints and tests in parallel.

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

> The animated tree above comes from a live test fixture; link to the [walkthrough](#walkthrough).

## Getting started

### Install

#### pipx
```
pipx install camas
```

#### uv
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

#### Nix
```
nix run github:JPHutchins/camas
```

<details><summary>Nix Optional Features</summary>

- `#default`: no extras
- `#with-github-checks`, `#with-ctrf`, `#with-check`, `#with-mcp`: one package per extra (`with-<extra>`, `_` → `-`)
- `#all`: every extra
- ⚠️ This list is copied by hand from `flake.nix`.

</details>

### As a Development Environment Dependency

Or manage camas with your dependency tooling of choice.

### Your first `tasks.py`

> Tasks definitions are Python files (`tasks.py`), but camas supports all programming languages and development environments, as well as Python.

Scaffold a starter `tasks.py` in your project root:

```
camas --init
```

<hr>
<details><summary>tasks.py generated by <code>`camas --init`</code></summary>

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

from camas import Claude, Config, Parallel, Task, run_cli

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

# `+` chains tasks into a Sequential, which runs them in order and stops at the
# first failure; `|` gathers them into a Parallel, which runs them concurrently.
# They nest freely (`+` binds tighter than `|`, so parenthesize), and
# Sequential(...) / Parallel(...) spell the same trees out. The binding name is
# the task name (this defines `ci`); pass name= only to rename, or to name a
# nested, anonymous group.
ci = hello + (greet | "python --version")

# Your deterministic, behavior-preserving auto-fixers, with {paths} so a run scopes to the
# changed files. Register the node (named anything) to Config.agent.fix below; the camas_gate
# tool settles it over the changed files before checking, and the Claude Code Stop hook runs it at
# every turn end, for free. Replace the placeholder with your real fixers, e.g.:
#   fix = Task("ruff check --fix {paths}", mutates=True, paths=".")
#   fix = Sequential(Task("ruff check --fix {paths}", mutates=True), Task("ruff format {paths}", mutates=True), paths=".")
fix = Task('python -c "" {paths}', mutates=True, paths=".")

# Config is discovered by type, under any binding name (here `_`): bare `camas` runs
# default_task — or github_task under GitHub Actions, falling back to default_task when unset.
# agent= wires the Claude Code plugin: agent.fix is the registered fix node above; the gate
# checks default_task (override with Claude(fix=..., check=...)). A checking
# leaf can add agent_format=("--output-format sarif", "sarif") for machine-readable gate output.
_ = Config(default_task=ci, agent=Claude(fix=fix))

# The PEP 723 standalone flow (see the docstring): running this file directly
# (`uv run tasks.py <task>`) dispatches through camas. Inert when the `camas`
# command discovers this file itself.
if __name__ == "__main__":
    run_cli(globals())
```

</details>
<hr>

The starter definition demonstrates [`Task`s](#task), composition of tasks into [`Sequential`s](#sequential) and [`Parallel`s](#parallel), [task-node-matrices](#matrices-and-variants), and camas [`Config`](#config). To guarantee that it works out of the box, it uses a [PEP 723 standalone block](#standalone-taskspy-pep-723) and tasks executing `python -c`, but the tasks themselves can be any command and arguments that the shell can execute.

The `--init` option also creates a gitignored `.camas/` directory. It is optional, but it enables advanced capabilities like [logging](#config), [task tree timing](#time-budget), and [expanded agent capabilities](#ai-coding-agents).

> `camas --init --verbose` provides a more exhaustive set of examples and is the default when the camas MCP uses its `camas_init` tool.

### Running tasks

Typical usage is to run `camas`, which will run the default task for the environment. For example, as a programmer at your desk, running `camas` will run the `Config.default_task` whereas an agent or continuous integration environment may have [their own defaults](#the-default-task).

> `camas --help` is the best place to find information about builtin options and your custom tasks.

Specifying a task node by name as the first argument, `camas <name>`, runs the tree from that task node.

Task nodes also provide help: `camas <name> --help`.

Arguments can be passed through to a `Task` be placing them after `--`: `camas test -- -v`.

`camas` controls the realtime output -- to terminal, network, filesystem, etc. -- through a concurrent "effects" system. The default effect for a programmer at their terminal is the builtin `TermTree` (TODO: link), and the effects can be selected at runtime using the `--effects` (TODO: link) CLI option.

```python
from camas import Task
hello = Task(("python", "-c", "print('hello world')"))
```

```console
$ camas '--effects=(Summary(show_passing=True),)' hello
$ camas --help
```

> The [walkthrough](#walkthrough) shows usage and example output.

## Defining tasks

### Tasks and groups

A task is *any* shell command and its arguments. For example, `Task("ruff check .")`, `Task("cargo test", cwd=Path("rust"))`, `Task("npm test")` are shell commands now defined as a task node in camas. Unless overridden by [`name=`](#task), the task is named and referenced by its **variable name**.

```python
from camas import Task

# Create the "lint" task, to be run with `camas lint`
lint = Task("ruff check .")
```
> Bare strings like `"ruff check ."` are [split by shell rules](https://docs.python.org/3/library/shlex.html#shlex.split) and, if this causes problems for complex commands, can be replaced by explicit shell splitting: [`("ruff", "check", ".")`](#task).

Tasks may be composed into [`Sequential`s](#sequential) and [`Parallel`s](#parallel). A `Sequential` runs a series of tasks one after the other, stopping at the first failure, while a `Parallel` runs them at the same time without influencing one another. Within these groups, tasks can be defined by bare strings for convenience, though note that these "anonymous" tasks cannot be referred to by name since they were not given a variable binding at file scope.

Internally, camas refers to the executed shell command -- the tip of the task tree, the leaf node -- as a _leaf_. [Task options](#task), like `cwd` (current working directory) and `env` (environment variable map), are scoped to the leaf task node on which they are defined, and inherited by child nodes until overridden by greater specificity.

Importantly, a task that mutates the filesystem _should_ be marked with `mutates=True` so that camas can validate task tree construction to avoid conflicts (mutating nodes should usually be run in series, not parallel).

See [`Task`](#task), [`Sequential`](#sequential), and [`Parallel`](#parallel),

<details><summary>Worked examples</summary>

```python
from pathlib import Path
from camas import Parallel, Sequential, Task

lint = Task(("python", "-c", "print('lint ok')"))
test = Task(("python", "-c", "print('test ok')"), cwd=Path("subdir"))
say = Task(
    ("python", "-c", "import os; print('VAR=' + os.environ['VAR'])"),
    env={"VAR": "scoped"},
)
echo = Task(("python", "-c", "import sys; print(*sys.argv[1:])"))

ci = Sequential(
    lint,
    Parallel(test, "python --version"),
    say,
    name="ci",
)
```

```console
$ camas --dry-run ci
$ camas --tree
$ camas '--effects=(Summary(show_passing=True),)' ci
$ camas echo -- one two three
```

```python
from camas import Sequential, Task
pipeline = Sequential(
    Task(("python", "-c", "raise SystemExit(2)"), name="boom"),
    Task(("python", "-c", "print('never reached')"), name="follow_up"),
    name="pipeline",
)
```

```console
$ camas '--effects=(Summary(),)' pipeline
```

</details>

### Pipes

A `Pipe` wraps a task leaf so that its' stdout is routed to the next task's stdin. See [`Pipe`](#pipe) and `Pipe(agent_only=True)` at [Structured diagnostics](#structured-diagnostics).

<details><summary>Worked examples</summary>

```python
from camas import Pipe, Task
upper = ("python", "-c", "import sys; print(sys.stdin.read().upper(), end='')")
shout = Pipe(Task(("python", "-c", "print('hello pipe')")), Task(upper), name="shout")
whisper = Task(("python", "-c", "print('quiet')")) > Task(upper)
```

```console
$ camas '--effects=(Summary(show_passing=True),)' shout
$ camas '--effects=(Summary(show_passing=True),)' whisper
```

</details>

### Operators

`camas` uses operator overloading to provide a more expressive syntax for composing task trees while maintaining static type safety.

| Operation | In | Out |
| --       | --   | --      |
| `a \| b`     | `Task \| Parallel \| Sequential` | `Parallel(a, b)` |
| `a + b`      | `Task \| Parallel \| Sequential` | `Sequential(a, b)` |
| `a - b`      | `Parallel \| Sequential` | `a` with `b` removed |
| `a > b`      | `Task` | `Pipe(a, b)` |

#### Special Rules

The `|` operator on a `Parallel` _extends_ the `Parallel` rather than _wrapping_ it. For example:

<!--TODO: get under doctests w/ fixtures-->
```python
(a | b) | c == Parallel(a, b, c)
(a | b) | c != Parallel(Parallel(a, b), c)
```

Likewise, the `+` operator on a `Sequential` returns a new sequence with the right side of the operator _added_ to the end:

```python
(a + b) + c == Sequential(a, b, c)
(a + b) + c != Sequential(Sequential(a, b), c)
```

`-` returns a new group with the right-hand-side (RHS) removed from the group on the left-hand-side (LHS).

```python
(a + b + c) - c == Sequential(a, b)
```

<details><summary>Worked example</summary> 

```python
from camas import Task
fmt = Task(("python", "-c", "print('formatted')"), mutates=True)
lint = Task(("python", "-c", "print('linted')"))
test = Task(("python", "-c", "print('tested')"))
check = lint | test
ci = fmt + check
quick = check - test
```

```console
$ camas --dry-run ci
$ camas --dry-run quick
$ camas '--effects=(Summary(show_passing=True),)' ci
```

</details>

### Matrices and variants

As part of its' promise to eliminate the need for ill-suited configuration file languages like YAML, TOML, and JSON, `camas` has builtin syntax for expressing common matrix permutations and variant lists. Of course, a `tasks.py` file can run, and even import, arbitrary Python, so it is always possible to fall back on procedural code to meet your unique requirements.

A `matrix` is typed as a map of _axis name_ string to _variant_ strings: `dict[str, tuple[str, ...]]`. For example, the `sliced` shell command takes a variable `FRUIT`, marked as a variable for string substitution with the surrounding curly braces, `{FRUIT}`, and the matrix map supplies the `FRUIT` variants of `(orange, apple, pear)`.

```python
sliced = Sequential(
  Task("slice {FRUIT}"),
  matrix={
    "FRUIT": ("orange", "apple", "pear")
  }
)
```
The `sliced` node will be run three times, in parallel:
```
slice orange
slice apple
slice pear
```

> Note: A matrix always runs its cells in parallel. The group type -- `Sequential` or `Parallel` -- only decides how the trees are run inside each cell. Generating your own sequential matrix can be done with Python syntax:
> 
> ```python
> sliced = Sequential(*(f"slice {fruit}" for fruit in fruits))
> ```

Considering that `matrix` is a map, it can map as many names as needed:
```python
paired = Sequential(
  Task("pair {FRUIT} {CHEESE}"),
  matrix={
    "FRUIT": ("orange", "apple", "pear"),
    "CHEESE": ("gruyere", "roquefort"),
  }
)
```
The `paired` node will be run six times (MxN, 3x2), in parallel:
```
pair orange gruyere
pair orange roquefort
pair apple gruyere
pair apple roquefort
pair pear gruyere
pair pear roquefort
```

> Note: matrix axis values are also environment variables. Every leaf in a cell gets `FRUIT=orange` in its env, overriding any same-named env entry, so a script can read `$FRUIT` with no `{FRUIT}` in the command.

In addition to substituting the corresponding name in shell strings, the matrix map will also substitute strings in the environment variable map and other Task arguments like `cwd`, `help`, `paths`, and `when`.

```python
firmware = Sequential(
  Task("cmake -S=. -B=build/{TOOLCHAIN}-{CONFIG} -DCMAKE_BUILD_TYPE={CONFIG}"),
  Task("cmake --build build/{TOOLCHAIN}-{CONFIG}"),
  env={"CMAKE_TOOLCHAIN_FILE": "cmake/{TOOLCHAIN}.cmake"},
  matrix={
    "TOOLCHAIN": ("gcc", "arm-none-eabi"), 
    "CONFIG": ("Debug", "Release")
  },
)
```
The `firmware` node will be run 4 times in parallel (2x2), with the `CMAKE_TOOLCHAIN_FILE` environment variable matching the value of `TOOLCHAIN`.

<details><summary>Firmware node matrix expanded</summary>

```
CMAKE_TOOLCHAIN_FILE=cmake/gcc.cmake \
  cmake -S=. -B=build/gcc-Debug -DCMAKE_BUILD_TYPE=Debug \
  && cmake --build build/gcc-Debug
CMAKE_TOOLCHAIN_FILE=cmake/gcc.cmake \
  cmake -S=. -B=build/gcc-Release -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build/gcc-Release
CMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake \
  cmake -S=. -B=build/arm-none-eabi-Debug -DCMAKE_BUILD_TYPE=Debug \
  && cmake --build build/arm-none-eabi-Debug
CMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake \
  cmake -S=. -B=build/arm-none-eabi-Release -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build/arm-none-eabi-Release
```

</details>

Matrix axis values can be overridden at the CLI by passing `--name=value(s)`, e.g. `--FRUIT=apple`, or passing multiple values for the axis, even ones not named in the tasks file: `--FRUIT=mango,banana`.

A matrix can be used to require an axis argument from the CLI by specifying the axis as empty in the tasks file:
```python
required = Sequential(
  "release {VERSION}",
  matrix={"VERSION": ()},
)
```

> While camas is not intended as a tool for writing CLIs, this string interpolation is a powerful tool for composing common routines that require runtime arguments.

A complement to the `matrix=` field is `variants=`. Like a matrix, it allows expressing string substitutions, but instead of doing so along the product, it is used to list tightly coupled bundles of options.

<details><summary>Worked examples</summary>

```python
from camas import Parallel, Task
greet = Parallel(
    Task(("python", "-c", "print('hello, {NAME}')")),
    matrix={"NAME": ("Grace", "Ada", "Wendy", "Alyssa")},
    name="greet",
)
```

```console
$ camas '--effects=(Summary(),)' greet
$ camas '--effects=(Summary(),)' greet --NAME=Ada
$ camas greet --help
```

```python
from camas import Parallel, Task
meet = Parallel(
    Task(("python", "-c", "print('hi {NAME1}, meet {NAME2}')")),
    matrix={
        "NAME1": ("Grace", "Ada"),
        "NAME2": ("Wendy", "Alyssa"),
    },
    name="meet",
)
```

```console
$ camas '--effects=(Summary(show_passing=True),)' meet
$ camas '--effects=(Summary(show_passing=True),)' meet --NAME1=Clara,Dolores --NAME2=Grace,Ada
$ camas meet --help
```

```python
from camas import Parallel, Task
port = Parallel(
    Task(("python", "-c", "print('build {TARGET} with {TOOL}')")),
    variants=(
        {"TARGET": "wasm", "TOOL": "emcc"},
        {"TARGET": "native", "TOOL": "clang"},
    ),
    name="port",
)
```

```console
$ camas '--effects=(Summary(show_passing=True),)' port
$ camas '--effects=(Summary(show_passing=True),)' port --TARGET=wasm
$ camas port --github-matrix
```

</details>

<details><summary>Reference</summary>

- [CLI](#cli): `--matrix KEY=VAL[,VAL...]`, and `--jobs` for a matrix too wide for the machine
- [Checking definitions (`camas --check`)](#checking-definitions-camas---check): catches a `variants=()` that expands to nothing
- [GitHub Actions matrix (`--github-matrix`)](#github-actions-matrix---github-matrix): split matrix cells across CI runners
- [Python projects](#python-projects): one environment per Python version in a version matrix
- [python-version-matrix](https://github.com/JPHutchins/camas/tree/main/tests/fixtures/python-version-matrix): a matrix read from `.python-version`
- [playwright-matrix-override](https://github.com/JPHutchins/camas/tree/main/tests/fixtures/playwright-matrix-override): a 2-axis browser × viewport matrix, scoped from the CLI
- [#236](https://github.com/JPHutchins/camas/issues/236): the embedded backend ↔ platform ↔ target case that motivated `variants=`
- [GitHub Actions: running variations of jobs](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/run-job-variations): GHA's `matrix`, and `include:`, the counterpart of `variants=`

</details>


### The default task

`camas` provides the [`Config`](#config) type for configuring the tasks file. To define a `Config`, bind an instance in your tasks.py file (the name doesn't matter, and `_ = Config()` is convention)

When that `Config` specifies a `default_task`, then that is the task that a bare `camas` runs for the programmer at their terminal.

```python
from camas import Config, Sequential, Task

lint = Task("ruff check .")
test = Task("pytest")
ci = Sequential(lint, test, name="ci")

_ = Config(default_task=ci)
```

The config can also be used to define context-aware defaults for continuous integration and coding agent environments. See [GitHub Actions](#github-actions) and [agent](#agent) for more information.

## Syntax and type safety

### Typed definitions

Your tasks file, `tasks.py`, is a statically typed python file checkable by mypy, pyright, ty, and other checkers and their LSPs.

### Checking definitions (`camas --check`)

`camas` can do a basic self-check of your tasks file with `camas --check`, ensuring your task trees are loadable and checking for a few common mistakes and misconfigurations.

If your runtime has [`ty`](https://docs.astral.sh/ty/) available, or if you've added the `camas[check]` extra, `--check` is enhanced by a `ty` type check that can identify additional static typing problems in your tasks file.

### Typed CLI expressions

Arguments to the CLI are well-defined statically typed expressions parsed by `camas` as an Abstract Syntax Tree. This means the making definitions at the CLI or in code is 1:1. For example, a traditional CLI may support an effects option like this:

```
--effects=status --status-output-mode=errors
```

The problem with this is that `--status-output-mode` is only valid when `--effects` includes `status` -- there is no way to avoid invalid configurations structurally. camas solves this be taking the statically typed expression instead. `--effects` takes a `tuple[Effect, ...]`:

```
--effects='(Status(output_mode="errors"),)'
```

> Note: the outer single quote is required to allow encoding spaces inside the expression.

## Optimizations

### Path scoping

Task nodes that provide `{paths}` interpolation along with their default `paths=` argument can override the value of `{paths}` at runtime using the `--paths` argument. While a programmer can utilize this, it is a sophisticated feature more intended for agents, particularly the included "fixer" Claude Code agents included with the camas MCP. Agents use this option to speed up their static analysis iteration loops by only checking the file trees that they are working on rather than the entire tree.

For shell commands that cannot take file arguments, a broader `when=` alternative may be utilized. When `--paths` is provided, these nodes will only run when the files with the `when` argument have been changed.

```python
check = Parallel(
  Task("ruff check {paths}", paths="src"),
  Task("prettier --check {paths}", paths="web"),
  Task("cargo build", when="src"),
  Task("nix flake check"),
)
```

The argument to `paths` and `when` is not just a list of file tree roots; it can also be callable in order to scope by predicate. camas provides builtin's `by_suffix` and `by_glob` for common `{paths}` use cases:

```python
tidy = Task(
  "clang-tidy {paths}",
  paths=by_suffix(
    (".c", ".h"),
    default=("src",),
  )
)
```

<details><summary>Reference</summary>

- [`Task`](#task): the `paths=` and `when=` fields
- [`Sequential`](#sequential) and [`Parallel`](#parallel): group-level `paths=` and `when=`, the defaults for their leaves
- [`PathScope`](#pathscope) and [`WhenPredicate`](#whenpredicate): the callable forms
- [`by_suffix`](#by_suffix) and [`by_glob`](#by_glob)
- [Monorepos](#monorepos): a leaf's `cwd` doubles as its `when=`
- [Checking definitions (`camas --check`)](#checking-definitions-camas---check): warnings for scopes that can never match
- [Time budget (`--under`)](#time-budget): the other way to shorten a run, and it combines with `--paths`
- [Fix and gate hooks](#fix-and-gate-hooks): how the fixer agents use `--paths`
- [MCP tools](#mcp-tools): the `paths` parameter on `camas_gate`, `camas_fix`, and `camas_run`
- [CLI](#cli): `--paths PATH[,PATH...]`

</details>


### Time budget

The `--under=<DURATION>` option runs only the task leaves that are expected to fit within the given duration. Any node, including the default node, may be provided, and the duration accepts standard units (defaults to seconds).

```
# Skip any leaf estimated to
# take longer than 5 seconds
camas --under=5s

# The same, for a named task
camas --under=500ms check
```

This option is intended for developer facing features like git-hooks or an agent's inner static analysis loop.

> Note: duration calculation requires the existence of the `.camas` cache folder, provided by `camas mcp init [options]`

<details><summary>Reference</summary>

- [`Task`](#task): the `mutates=` field
- [`Timings`](#timings): the effect that records the estimates; a `Config` that sets its own effects must include it
- [The default task](#the-default-task): what `--under` budgets when no task is named
- [Path scoping](#path-scoping): the other way to run less, and it combines with `--under`
- [Pre-push hook](#pre-push-hook): a budgeted run before every push
- [Fix and gate hooks](#fix-and-gate-hooks): how the agents use it
- [MCP tools](#mcp-tools): the `under` parameter on `camas_run` and `camas_gate`, and the `budget` report
- [CLI](#cli): `--under DURATION`

</details>

### When `Parallel` doesn't help

camas's parallel execution is a powerful speed-up for most development tasks. However, it's not a silver bullet, particularly when the tasks that are parallelized already use many cores or contend for resources. For example, cargo takes a lock on `target/`. That alone can be worked around by providing each leaf its own `CARGO_TARGET_DIR`, but even then, cargo and rustc are optimized to use available CPU cores, so parallel jobs are unlikely to speed things up and can even slow down the tasks due to added contention.

camas is a great way to make sure that your CI runners are constantly pinned to 100% CPU usage, but a wide fan-out can also oversubscribe the machine's CPU, memory, or disk, so it can be capped with `--jobs=N` (env: `CAMAS_JOBS`), or even spread across CI runners with `--github-matrix`.

<details><summary>Reference</summary>

- [`Task`](#task): the `mutates=` field
- [Time budget](#time-budget): the only mode where `mutates=True` reorders leaves
- [Drift gate (`Clean`)](#drift-gate-clean): why stray writes fail CI
- [GitHub Actions matrix (`--github-matrix`)](#github-actions-matrix---github-matrix): fan out across runners
- [CLI](#cli) and [Environment variables](#environment-variables): `--jobs N` and `CAMAS_JOBS`

</details>

## Continuous integration

### GitHub Actions

camas detects being run in the GitHub Actions environment and switches its effect `Status(output_mode="github")`, whose logs fold into collapsible groups by task leaf.

```yaml
- run: uvx camas check
```

`Config`'s `github_task` option sets what a bare `camas` runs in GitHub Actions; without it specified, it falls back to `default_task` (the local development default task node).

```yaml
- run: uvx camas
```

Tour tasks file is the Single Source of Truth (SSOT) for continuous integration and local development. Your checks can evolve over time without any need to modify your GitHub Actions file(s) or worry about desyncronization.

<details><summary>Reference</summary>

- [`Config`](#config): `github_task`, `default_task`, and `default_github_effects`
- [The default task](#the-default-task): what a bare `camas` runs locally
- [Choosing output](#choosing-output): the defaults locally, in CI, and under agents, and how `Config` overrides them
- [`Status`](#status) and [`OutputMode`](#outputmode): every output mode
- [Pre-push hook](#pre-push-hook): run `github_task` before you push
- [`status-modes-demo`](https://github.com/JPHutchins/camas/blob/main/.github/workflows/ci.yaml): one CI run per output mode, for comparison

</details>

### GitHub Actions matrix (`--github-matrix`)

camas provides the SSOT for your GitHub matrix fan-out as well. A developer on local can run the matrix parallelized across there cores while in CI the jobs will parallelize across runners. Again, this provides powerful Python syntax to express complex configurations, avoiding the fight with YAML.

You may can freely matrix across compiler versions, runtime environments, build configurations -- there are no limits -- with minimal YAML surface area.

#### Recipe

1. Pin the shape in a discover job that emits the matrix JSON using camas' `--github-matrix` option
2. Access the matrix using `fromJSON`

```yaml
jobs:
  discover:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.emit.outputs.matrix }}
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v10.3.3
      - id: emit
        run: echo "matrix=$(uvx camas check --github-matrix=tasks)" >> "$GITHUB_OUTPUT"

  check:
    needs: discover
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        task: ${{ fromJSON(needs.discover.outputs.matrix).task }}
      fail-fast: false
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v10.3.3
      - run: uvx camas ${{ matrix.task }}
```
The matrix and variants options meet different job distribution requirements. camas does some validation of the fan-out, refusing shapes that cannot be split exactly or those that would repeat a leaf, rather than willingly emitting a defective matrix.

| `tasks.py` | emitted | each job runs |
|---|---|---|
| `matrix={"PY": (...)}` | `{"PY": ["3.13", "3.14"]}` | `camas <task> --PY ${{ matrix.PY }}` |
| `variants=({...}, {...})` | `{"include": [{...}, {...}]}` | the same, one flag per key |
| a `Parallel` of named tasks | `{"task": ["build", "lint", "test"]}` | `camas ${{ matrix.task }}` |

Running your task trees through camas reduces the number of nodes presented in the GitHub Actions UX down to the count of root nodes that you choose to run -- potentially just one. If you'd like to get back that granular UX while still benefitting from camas SSOT, take a look at [Per-leaf checks (`Git.HubChecks`)](#per-leaf-checks-githubchecks)

<details><summary>Reference</summary>

- [Matrices and variants](#matrices-and-variants): the `matrix=` and `variants=` that become `axes` and `variants`
- [`Parallel`](#parallel): the `tasks` shape, one job per child
- [GitHub Actions](#github-actions): `github_task`, for a step that's just `camas`
- [Per-leaf checks (`GitHubChecks`)](#per-leaf-checks-githubchecks): the alternative, one runner with a check per leaf
- [Python projects](#python-projects): matrix cells below camas's own Python floor
- [CLI](#cli) and [MCP tools](#mcp-tools): `--github-matrix[=SHAPE]` and `camas_github_matrix`
- [camas's own CI](https://github.com/JPHutchins/camas/blob/main/.github/workflows/ci.yaml): the `discover`, `check`, and `per-task` jobs
- [GitHub Actions: running variations of jobs](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/run-job-variations)

</details>

### Drift gate (`Clean`)

`Clean` (TODO: link) is a wrapper around a task that mutates files that are under VCS and fails if the work tree changes after the wrapped task was run.

```python
openapi = Clean(Task("make update-openapi", mutates=True))
```

The clean check runs three operations in sequence:
1. check that the tree is clean
2. run the wrapped task node
3. check that the tree is clean

camas provides git-centric defaults -- using git porcelain for the working tree checks -- but the checks themselves can be provided to work with other VCS.

<details><summary>Reference</summary>

- [`Clean`](#clean) and [`GIT_PORCELAIN`](#git_porcelain): the full signature and the default check
- [`Task`](#task): the `mutates=` field
- [`Sequential`](#sequential): what `Clean` returns
- [When `Parallel` doesn't help](#when-parallel-doesnt-help): mutating leaves and shared state

</details>

### Pre-push hook

```sh
echo 'exec camas check' > .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

<!-- TODO: https://github.com/JPHutchins/camas/issues/357 -->

> An LLM agent needs no hook: when a `github_task` is declared, `camas_list` reports it as `github_default`, so the agent runs `camas_run` with that name before pushing. The field is `null` when no `github_task` is set; camas never infers it from your CI workflow files.

## AI coding agents

### The MCP server

> Enable the MCP by adding the `mcp` extra, e.g. `camas[mcp]`.

To initialize the camas mcp, use `camas mcp init`, which will write the `.mcp.json`. `camas mcp init --claude` will setup Claude Code hooks and skills.

The MCP client starts the MCP server with the `camas mcp` entrypoint and refreshes your tasks file to keep it up to date as the agent adds and edit tasks for its internal validation loop.

<details><summary>Reference</summary>

- [MCP tools](#mcp-tools): every tool and its description, generated from the server
- [`Claude`](#claude) and [`Config`](#config): the `agent` slots
- [Install](#install): the `mcp` extra
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Claude Code: MCP](https://code.claude.com/docs/en/mcp)

</details>

### Claude Code setup

camas enhances Claude Code by enabling agent-driven static analysis loops. `camas mcp init --claude` writes the whole setup: `.mcp.json`, two `Stop` hooks, three camas-fixer subagents, and the `gate` skill.

Your agent's defaults can differ from your own by configuring them: `_ = Config(agent=Claude(fix=..., check=..., default=...))`

<details><summary>Reference</summary>

- [Fix and gate hooks](#fix-and-gate-hooks): what the two hooks do
- [Standalone `tasks.py` (PEP 723)](#standalone-taskspy-pep-723): the header the pin follows
- The files init writes: the [`gate` skill](https://github.com/JPHutchins/camas/blob/main/src/camas/main/claude_gate_skill.md) and the camas-fixer subagents ([lint, Haiku](https://github.com/JPHutchins/camas/blob/main/src/camas/main/claude_agent_lint_haiku.md); [lint, Sonnet](https://github.com/JPHutchins/camas/blob/main/src/camas/main/claude_agent_lint_sonnet.md); [tests](https://github.com/JPHutchins/camas/blob/main/src/camas/main/claude_agent_test_fixer.md))
- Claude Code: [hooks](https://code.claude.com/docs/en/hooks), [subagents](https://code.claude.com/docs/en/sub-agents), [skills](https://code.claude.com/docs/en/skills)

</details>

### Fix and gate hooks

Think of the fix hook as "format and fix on save" for your agent. This applies your deterministic formatters and lint fixers so that your agent never needs to think about it.

camas also initializes an async "nudge". It runs `camas mcp gate --under 5s nudge` which may reveal static analysis reports that could not be resolved automatically. The agent may choose to deploy fixer subagents scoped `--paths` slices of the file tree, or reason about the reports itself.

In this way, the agent is supercharged by a tight static analysis loop as it works, rather than needing to fix everything at the end, or worse, after CI fails!

<details><summary>Reference</summary>

- [`Claude`](#claude): `fix`, `check`, and `default`
- [Path scoping](#path-scoping): how the fix and the gate narrow to the changed files
- [Time budget](#time-budget): the `--under` behind the nudge
- [MCP tools](#mcp-tools): `camas_gate` and `camas_fix`
- [Claude Code: hooks](https://code.claude.com/docs/en/hooks): `Stop`, `async`, and `asyncRewake`

</details>

### Agents on the CLI

camas detects an agent (`CLAUDECODE=1` or `CAMAS_AGENT=1`) and prints line-oriented `Status` output instead of the user-facing live tree default. It's preferred to enable the MCP for optimized formatting and well-typed schemas rather than using the CLI.

<details><summary>Reference</summary>

- [Choosing output](#choosing-output): the defaults locally, in CI, and under agents
- [`Status`](#status) and [`Termtree`](#termtree)
- [Environment variables](#environment-variables)

</details>

### Structured diagnostics

Static analysis tooling that has structured output can be enabled for agents alone. This means that the programmer can continue to work with a normal UX while the agent benefits from token-saving structured report formats.

```python
lint = Task(
  "ruff check .",
  agent_format=("--output-format sarif", "sarif"),
)
semgrep = Task(
  "semgrep scan --config auto",
  agent_format=("--sarif", "sarif"),
)
pytest = Task(
  "pytest",
  agent_format=("--junitxml {report}", "junit"),
)
vitest = Task(
  "npx vitest run",
  agent_format=("--reporter=junit --outputFile={report}", "junit"),
)
```

> Note: for tools that write to a file, a literal `{report}` becomes a path the gate allocates and then reads

A tool with no structured format of its own can pipe into a converter just for the agent:

```python
clippy = Pipe(
  Task("cargo clippy", agent_format=("--message-format=json", "raw")),
  Task("clippy-sarif", agent_format=("", "sarif")),
  agent_only=True,
)
```

Every other run, human or agent, runs `cargo clippy` alone.

<details><summary>Reference</summary>

- [`AgentFormat`](#agentformat) and [`OutputKind`](#outputkind): the fields and the supported kinds
- [Pipes](#pipes) and [`Pipe`](#pipe): `agent_only`
- Formats: [SARIF 2.1.0](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html), [rdjson](https://github.com/reviewdog/reviewdog/tree/master/proto/rdf), [LSP diagnostics](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#diagnostic), [TAP](https://testanything.org/)

</details>

## Project layout

### Monorepos

In order to compose subprojects that contain their own tasks file, camas provides [`Project`](#project). Binding a `Project` imports the child `tasks.py` as a task node and mounts the child's tasks under the binding name.

```python
from camas import Claude, Config, Parallel, Project

libs = Project("libs")
# camas libs, camas libs.search.lint, ...

api  = Project("services/api")
# name the handle whatever you like

_ = Config(
  default_task=Parallel(libs, api, name="all"),
  # each child's default_task, in parallel
  
  github_task=Parallel(libs, api, name="ci"),
  # each child's github_task
  
  agent=Claude(
    fix=Parallel(libs, api, name="fix"),
    # each child's fix node
    
    check=Parallel(libs, api, name="check"),
    # each child's check node
  ),
)
```

A reference composes the child's _matching fields_. For example, instead of needing to specify `default_task=libs.default_task`, `libs`'s default task is selected automatically since it was added to the parent's default tasks. This allows for composing all of the subprojects into a single parallel.

When running a subproject from the root, e.g. `camas libs`, it is environment aware, just like the root camas tasks file. From a terminal, it will run `libs`'s default task, in CI it would run `libs`'s GitHub task, and for an agent it will follow `libs`'s agent configuration.

A subproject's `cwd`, `paths=`, and `when=`, stay relative to its own tasks file. This way, a scoped run (via `--path`, or the agent gate) runs only the subprojects whose files changed.

<details><summary>Reference</summary>

- [`Project`](#project)
- [`Config`](#config) and [`Claude`](#claude): the slots a reference composes
- [The default task](#the-default-task): what a bare `camas` runs in each context
- [Path scoping](#path-scoping): a leaf's `cwd` as its `when=` fallback
- [Fix and gate hooks](#fix-and-gate-hooks): the gate's scoped runs
- [CLI](#cli): the `( )` and `{ }` task expressions
- [Monorepo fixture](https://github.com/JPHutchins/camas/tree/main/tests/fixtures/monorepo): every permutation, including the nested `libs.search`

</details>

### Standalone `tasks.py` (PEP 723)

A project without Python packaging can still use camas as a single-file task runner: give `tasks.py` a [PEP 723](https://peps.python.org/pep-0723/) header and a `run_cli(globals())` entry point.

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["camas"]
# ///
from camas import Sequential, Task, run_cli

lint = Task("cargo clippy")
test = Task("cargo test")
check = Sequential(lint, test)

if __name__ == "__main__":
  run_cli(globals())
```

Any PEP 723 runner installs camas and runs the file, with every camas option available:

```sh
uv run tasks.py check
pipx run tasks.py --list
```

<details><summary>Reference</summary>

- `camas` reads the file the same way with or without the header, so the rest of this README applies unchanged.
- Pin camas in `dependencies`. The MCP server needs `"camas[mcp]"`; `camas mcp init` then launches it through the file, as `uv run tasks.py mcp`.
- `camas --check` warns when the header lists camas but nothing calls `run_cli`, since `uv run tasks.py` would then exit without running anything.
- [`run_cli`](#run_cli)
- [The MCP server](#the-mcp-server) and [Claude Code setup](#claude-code-setup): the `mcp` extra and the launcher `camas mcp init` writes
- [Checking definitions (`camas --check`)](#checking-definitions-camas---check)
- [PEP 723](https://peps.python.org/pep-0723/), [uv: running scripts](https://docs.astral.sh/uv/guides/scripts/), [pipx](https://pipx.pypa.io/stable/)

</details>

### Python projects

There are a few special considerations when camas is used to manage the tasks of a Python project.

<details><summary>Supporting Python older than 3.10</summary>

camas requires Python 3.10 or newer, so in a project whose `requires-python` reaches lower (say `">=3.8"`), a plain `uv add camas` fails to resolve. Either guard the dependency with an environment marker:

```sh
uv add --dev "camas ; python_version >= '3.10'"
```

or keep camas out of the project environment and run it with `uvx camas`; `camas mcp init --launcher uvx` sets up the MCP server the same way. Either way camas runs on its own interpreter and still drives tasks on the older versions.

</details>

<details><summary>A Python version matrix</summary>

`tasks.py` is plain Python, so a version matrix can come straight from `.python-version`. Each cell gets its own environment with `UV_PROJECT_ENVIRONMENT` and its own interpreter with `UV_PYTHON`:

```python
from pathlib import Path

from camas import Sequential, Task

PYTHONS = tuple(Path(__file__).with_name(".python-version").read_text().split())

matrix = Sequential(
  Task("uv sync"),
  Task("uv run pytest"),
  env={"UV_PROJECT_ENVIRONMENT": ".camas/.venv-{PY}", "UV_PYTHON": "{PY}"},
  matrix={"PY": PYTHONS},
)
```

Keep the environments in `.camas/`, which is already gitignored. In the project root, a later formatter or linter that scans `.` descends into `.venv-3.13/`, and a default `.venv` exclude often doesn't match the suffix.

</details>

<details><summary>CI cells older than 3.10</summary>

A cell testing 3.8 can't install camas, and `uvx camas` fails as well if the job pins uv's Python: `setup-uv`'s `python-version` input, or any job-level `UV_PYTHON`, applies to `uvx` too. Leave both unset, and let the matrix above set `UV_PYTHON` on the leaves only:

```yaml
- uses: astral-sh/setup-uv@v10.3.3
- run: uvx camas matrix --PY ${{ matrix.PY }}
```

camas runs on its own interpreter, and each leaf runs on the cell's.

</details>

<details><summary>Tasks in <code>pyproject.toml</code></summary>

A simple project, or one migrating from tools like [taskipy](https://github.com/taskipy/taskipy), [poethepoet](https://github.com/nat-n/poethepoet), [PDM scripts](https://pdm-project.org/en/latest/usage/scripts/), or [Hatch scripts](https://hatch.pypa.io/latest/config/environment/overview/#scripts), can define its tasks in `pyproject.toml` instead of `tasks.py`. Each key is a task name, and each value is a command or an expression over other task names, with `( )` for a `Sequential` and `{ }` for a `Parallel`:

```toml
[tool.camas.tasks]
lint = "ruff check ."
test = "pytest"
check = "{lint, test}"
```

There's no `Config` in this form, so a bare `camas` prints help instead of running a default task. A `tasks.py` in the same directory takes precedence.

</details>

<details><summary>Reference</summary>

- [Matrices and variants](#matrices-and-variants) and [GitHub Actions matrix (`--github-matrix`)](#github-actions-matrix---github-matrix): the `PY` axis and its CI fan-out
- [Claude Code setup](#claude-code-setup): `--launcher uvx`
- [The default task](#the-default-task): what a `Config` adds over the TOML form
- Fixtures: [python-version-matrix](https://github.com/JPHutchins/camas/tree/main/tests/fixtures/python-version-matrix), [python-project](https://github.com/JPHutchins/camas/tree/main/tests/fixtures/python-project) (the TOML form), and [camas's own `tasks.py`](https://github.com/JPHutchins/camas/blob/main/tasks.py)
- [Environment markers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/#environment-markers), uv's [`UV_PROJECT_ENVIRONMENT`](https://docs.astral.sh/uv/reference/environment/#uv_project_environment) and [`UV_PYTHON`](https://docs.astral.sh/uv/reference/environment/#uv_python), [setup-uv](https://github.com/astral-sh/setup-uv)

</details>

## Output and effects

### Choosing output

- ✚ One table of the defaults: the live `Termtree` locally, `Status(output_mode="github")` under GitHub Actions, `Status` under an agent; `Config(default_effects=...)` (and `default_github_effects`) overrides all of them.
- `--effects` takes a *tuple* of any length, so several effects run in one go: `--effects=(Summary(), Loud())` gives both the post-run report and the streaming lines
- `Summary(show_passing=True)` prints every passing task's output after the run

### Built-in effects

- `Termtree` (the default for programmers at their terminal): a live task tree in the terminal
- `Status`: line-oriented status output (the default under GitHub Actions and agents using the CLI)
- `Summary`: the final task tree, rendered once after the run
- `Ctrf`: a CTRF JSON test report (see [Reports](#reports-ctrf))
- `GitHubChecks`: per-leaf check runs on the GitHub Checks API (see [Per-leaf checks](#per-leaf-checks-githubchecks))
- `Timings`: each leaf's observed duration, recorded for later runs

### Per-leaf checks (`GitHubChecks`)

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
- **When to use it:**
  - SSOT between local dev and CI: the `camas matrix` terminal view and the PR Checks panel show the same per-leaf shape (`lint [PY=3.10]`, `mypy [PY=3.14]`, …). The matrix is defined once, in `tasks.py`; CI doesn't re-encode it in YAML.
  - One runner instead of N: camas's parallel matrix on one runner gives the same per-`(cell, leaf)` granularity a GHA matrix gets across N runners, at 1/N the minutes, and camas's parallelism keeps that runner close to 100% utilization. It's worth it on paid-runner budgets.
  - Downside: fork PRs get a read-only `GITHUB_TOKEN` and can't write checks. That's usually a non-issue for Enterprise teams, whose contributors can push to the org repo.
- **When not to bother:**
  - OSS gets free runners, so GHA-matrix them in parallel for faster wall-clock time
  - the cost is giving up either SSOT (the matrix moves into YAML) or per-leaf UI granularity (one check per runner instead of per leaf); pick one
  - fork PRs are the deal-breaker for OSS: external contributors' PRs get 403 from the Checks API, so the per-leaf entries silently don't appear
  - the `github-checks-demo` job is `continue-on-error` for that reason; `GitHubChecks` isn't a workable OSS solution
- Full constructor: [`GitHubChecks`](#githubchecks).

### Reports (`Ctrf`)

- For a CI artifact or an AI code review's input, add the `Ctrf` effect (opt-in extra: `camas[ctrf]`).
- It writes the run as a [CTRF](https://ctrf.io) JSON report: each leaf is a test with status, duration, output, command, and exit code.
- `path=` writes a file; the default is stdout.

```sh
uv run --extra ctrf camas check --effects='(Status(output_mode="errors"), Ctrf(path="ctrf-report.json"))'
```

- Full constructor: [`Ctrf`](#ctrf).

### Color in leaf output (`leaf_color`)

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

### Writing an effect

- Define an `Effect` in your `tasks.py` and it's discovered automatically: usable by name from `--effects`, and listed under `camas --effects`.
- a custom `Effect` defined inline in `tasks.py` is discovered automatically: `camas --help` lists it under *Available Effects* beside the built-ins (so an agent reading `--help` sees the plugin), and `camas --effects` with no value prints the effects-only listing
- The [effect-plugin fixture](https://github.com/JPHutchins/camas/tree/main/tests/fixtures/effect-plugin) has a typed `Tail` effect that streams each task's output as it arrives.
- ⚠️ The old link text said `examples/effect-plugin/`, but it points at `tests/fixtures/effect-plugin`. The example below is `__init__.py`'s inline `Loud` effect; the fixture could also be embedded with a `file` directive.
- An effect author reads the event stream and the per-leaf states: see the [Effect API](#effect-api).

```python
from collections.abc import Sequence
from camas.v0.effect import Effect
from camas.v0.leaf_state import LeafState
from camas.v0.task import Parallel, Task, TaskNode
from camas.v0.task_event import OutputEvent, TaskEvent

class Loud(Effect[None]):
    async def setup(self, task: TaskNode) -> None: ...
    async def on_event(
        self, event: TaskEvent, states: Sequence[LeafState], ctx: None,
    ) -> None:
        if isinstance(event, OutputEvent):
            print(f"!! {event.task.name}: {event.line.decode().strip()}")
    async def teardown(self, ctxs: tuple[None, ...]) -> None: ...

build = Parallel(
    Task(("python", "-c", "print('built {STAGE}')")),
    matrix={"STAGE": ("compile", "link")},
    name="build",
)
```

```console
$ camas --help
$ camas --list
$ camas --effects
$ camas build --help
$ camas '--effects=(Loud(),)' build
$ camas '--effects=(Summary(), Loud())' build
```

## Versioning

- The public API is published through **versioned namespaces**: [`camas.v0`](https://github.com/JPHutchins/camas/tree/main/src/camas/v0) today, `camas.v1` and beyond later.
- Import from a generation to pin the API shape: a name a generation exports is never removed or changed within that generation.
- The scheme tracks semver:
  - `v0` pairs with camas 0.x and is as loose as semver says 0.x is: the surface prefers to grow, and breaking changes stay possible until 1.0, made deliberately and noted in releases
  - at 1.0 a generation freezes: a breaking change forces the next `camas.vN`, and published generations keep shipping, so a `tasks.py` or effect plugin pinned to one keeps working across upgrades
- The top-level `camas` namespace is the unversioned alias for the latest generation, kept **1:1** with its package surface (see [Exports](#exports)):
  - ⚠️ The old text said `camas` re-exports "five headline definers" (`Task, Sequential, Parallel, Effect, Config`). It re-exports 14 names plus `run_cli`; the generated lists under Exports are the source of truth.
- The rest of a generation's public API (`TaskNode`, the `TaskEvent` stream, `LeafState`, `Completion`) lives in that generation's submodules, e.g. `from camas.v0.task_event import TaskEvent`.
- Everything under `camas.core` / `camas.main` is internal: it consumes whatever generations are installed and carries no stability promise.
- To pin a minimum camas *feature* level, use your dependency declaration (`camas>=0.x` in `pyproject.toml`, or PEP 723 inline metadata). The import path covers API shape; the package pin covers feature availability.

## Reference

- [Examples](https://github.com/JPHutchins/camas/tree/main/tests/fixtures): full project layouts under test coverage, and the canonical reference for structuring `tasks.py`, using `[tool.camas.tasks]` in `pyproject.toml`, driving a matrix from `.python-version`, or scoping a 2-axis matrix from the CLI.
  - ⚠️ The old link text said `examples/`, but it points at `tests/fixtures`.
- [src/camas/](https://github.com/JPHutchins/camas/tree/main/src/camas): typed Python with thorough docstrings; `camas --help` and `camas <task> --help` link back here.
- [tests/](https://github.com/JPHutchins/camas/blob/main/tests) for advanced, granular library usage

### Task definitions

#### `Task`

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

#### `AgentFormat`

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

#### `OutputKind`

```python
OutputKind: TypeAlias = Literal["sarif", "rdjson", "lsp", "junit", "tap", "raw"]
```

The format a leaf's diagnostics are in.

#### `Sequential`

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

#### `Parallel`

- ✍️ `Parallel`: what it's for and how it's used

A group of tasks that run concurrently.

| `Parallel` | Type | Default | |
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

<details><summary><code>Parallel</code> examples</summary>

```pycon
>>> Parallel("lint", "typecheck").tasks
(Task(cmd='lint', name=None, env={}, cwd=None), Task(cmd='typecheck', name=None, env={}, cwd=None))
```

</details>

#### `Pipe`

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

#### `TaskNode`

```python
TaskNode: TypeAlias = Task | Sequential | Parallel | Pipe
```

Any node of a task tree.

#### `Nodes`

```python
Nodes: TypeAlias = TaskNode | str | Sequence[TaskNode]
```

One node or a sequence of nodes; a lone `str` is a `Task`.

#### `PathScope`

```python
PathScope: TypeAlias = Callable[[tuple[str, ...]], tuple[str, ...]]
```

Maps the changed paths to the arguments at `{paths}`; called with `()` on a full run,
and returning `()` on a scoped run skips the leaf.

#### `WhenPredicate`

```python
WhenPredicate: TypeAlias = Callable[[tuple[str, ...]], bool]
```

Whether a leaf runs for the changed paths; never called on a full run.

### Configuration

#### `Config`

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

#### `Claude`

- ✍️ `Claude`: what it's for and how it's used

The Claude Code agent integration.

| `Claude` | Type | Default | |
|---|---|---|---|
| `fix` | `TaskNode` | required | The deterministic, behavior-preserving fix node. |
| `check` | `TaskNode \| None` | `None` | The node the gate checks; `None` defers to the project's bare task. |
| `default` | `TaskNode \| None` | `None` | The task a no-task `camas_run` runs; `None` defers to the check node, then the GitHub task, then the default task. |

#### `Agent`

```python
Agent: TypeAlias = Claude
```

The supported coding-agent integrations.

### Functions

#### `Clean`

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

#### `GIT_PORCELAIN`

```python
def GIT_PORCELAIN() -> Task
```

A fresh copy of the default `Clean` drift check.

#### `Project`

```python
def Project(path: str) -> TaskNode
```

Another `tasks.py`, by path relative to this one, as a task node.

#### `by_suffix`

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

#### `by_glob`

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

#### `run_cli`

```python
def run_cli(scope: Mapping[str, object]) -> None
```

Run the camas CLI against a `tasks.py`'s `globals()`, for `uv run tasks.py`.

### Effect API

#### `Effect`

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

#### `TaskEvent`

```python
TaskEvent: TypeAlias = StartedEvent | OutputEvent | CompletedEvent
```

An event a run emits.

#### `StartedEvent`

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

#### `OutputEvent`

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

#### `CompletedEvent`

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

#### `LeafState`

```python
LeafState: TypeAlias = Waiting | Running | Interrupting | Completed
```

Where a leaf is in its run.

#### `Waiting`

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

#### `Running`

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

#### `Interrupting`

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

#### `Completed`

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

#### `Completion`

```python
Completion: TypeAlias = Finished | Skipped | Stopped | Errored
```

How a leaf's run ended.

#### `Finished`

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

#### `Skipped`

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

#### `Stopped`

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

#### `Errored`

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

### Built-in effect constructors

#### `Termtree`

- ✍️ `Termtree`: what it's for and how it's used

A live task tree in the terminal.

| `Termtree` | Type | Default | |
|---|---|---|---|
| `frame_interval_ms` | `float` | `16.667` | Milliseconds between redraws. |
| `show_passing` | `bool` | `False` | Print passing leaves' output after the run. |
| `output_ctrl_c` | `bool` | `False` | Print stopped leaves' output after the run. |

#### `Status`

- ✍️ `Status`: what it's for and how it's used

Line-oriented status output.

Its line formats are `str.format` templates; an empty one prints no line.

| `Status` | Type | Default | |
|---|---|---|---|
| `output_mode` | `OutputMode` | `"errors"` | Which leaf output to print, and when. |
| `started_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{CAMAS_VIOLET}▶ [{{name}}] started{RESET}"` | The line when a leaf starts, over `name`, `cmd`, `timestamp`, and `ms`. |
| `finished_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{GREEN}✓ [{{name}}] success{RESET} ({{elapsed:.3f}}s)"` | The line when a leaf exits 0, adding `rc` and `elapsed`. |
| `failed_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{RED}✗ [{{name}}] error{RESET} exit={{rc}} ({{elapsed:.3f}}s)"` | The line when a leaf exits non-zero, adding `rc` and `elapsed`. |
| `skipped_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{GREY}⏭ [{{name}}] skipped{RESET} (prior rc={{rc}})"` | The line for a skipped leaf, adding `rc`, the blocking leaf's. |
| `stopped_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{YELLOW}■ [{{name}}] stopped{RESET} exit={{rc}} ({{elapsed:.3f}}s)"` | The line for a stopped leaf, adding `rc` and `elapsed`. |
| `errored_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} " f"{RED}⚠ [{{name}}] errored{RESET} {{msg}}"` | The line for a leaf whose command can't start, adding `rc` and `msg`. |
| `output_fmt` | `str` | `f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}] · [{{name}}]{RESET} {{line}}"` | Each output line in `stream` mode, adding `line`. |

#### `OutputMode`

```python
OutputMode: TypeAlias = Literal["quiet", "all", "errors", "stream", "github"]
```

What `Status` prints of a leaf's output: `errors` a leaf's that exits non-zero, `all` every
leaf's, and `github` every leaf's in a collapsed workflow group, each once the leaf ends;
`stream` prints each line as it arrives instead, and `quiet` none.

#### `Summary`

- ✍️ `Summary`: what it's for and how it's used

The final task tree, rendered once after the run.

| `Summary` | Type | Default | |
|---|---|---|---|
| `term_width` | `TermWidth` | `Auto()` | `Auto()` measures the terminal; `Fixed(columns)` sets the width. |
| `show_passing` | `bool` | `False` | Print passing leaves' output after the tree. |

#### `TermWidth`

```python
TermWidth: TypeAlias = Auto | Fixed
```

The terminal width `Summary` renders to.

#### `Fixed`

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

#### `Ctrf`

- ✍️ `Ctrf`: what it's for and how it's used

A CTRF JSON test report.

| `Ctrf` | Type | Default | |
|---|---|---|---|
| `path` | `str \| None` | `None` | The file to write the report to; `None` writes it to stdout. |
| `tail_bytes` | `int` | `8192` | Bytes of each leaf's output the report keeps, counted from the end. |

#### `GitHubChecks`

- ✍️ `GitHubChecks`: what it's for and how it's used

Per-leaf check runs on the GitHub Checks API.

| `GitHubChecks` | Type | Default | |
|---|---|---|---|
| `token` | `str \| None` | `None` | `None` reads `GITHUB_TOKEN`. |
| `repository` | `str \| None` | `None` | `owner/repo`; `None` reads `GITHUB_REPOSITORY`. |
| `sha` | `str \| None` | `None` | The commit the check runs attach to; `None` reads `GITHUB_SHA`. |
| `name_prefix` | `str` | `""` | Prepended to each leaf's label to name its check run. |
| `tail_bytes` | `int` | `8192` | Bytes of each leaf's output its check run keeps, counted from the end. |
| `fail_on_api_error` | `bool` | `False` | Raise on a Checks API failure instead of logging it. |

#### `Timings`

- ✍️ `Timings`: what it's for and how it's used

Each leaf's observed duration, recorded for later runs.

| `Timings` | Type | Default | |
|---|---|---|---|
| `camas_dir` | `Path` | required | Where the timing cache is. |
| `scope` | `int` | `0` | The run's scope, recorded with each duration. |
| `identities` | `tuple[timings.CacheKey, ...] \| None` | `None` | One cache key per leaf, in leaf order, in place of the keys derived from labels. |

### Exports

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

### CLI

<details><summary><code>camas --help</code></summary>

```console
$ camas --help
usage: camas [-h] [--version] [--dry-run | --github-matrix [SHAPE]] [--list] [--tree] [--check]
             [--init] [--verbose] [--effects [EFFECTS]] [--matrix KEY=VAL[,VAL...]] [--jobs N]
             [--under DURATION] [--paths PATH[,PATH...]]
             [expression | mcp]

A task runner with parallel execution, matrix expansion, MCP, and pluggable output effects.

positional arguments:
  expression | mcp      name of a defined task, a camas expression, or 'mcp' (trailing '-- ARGS'
                        append to a Task's command)

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --dry-run             print the task tree without executing
  --github-matrix [SHAPE]
                        emit the task's fan-out as GitHub Actions strategy.matrix JSON and exit
                        (consume via fromJSON; ignores --paths/--under/--jobs/--effects). SHAPE is
                        auto (default), axes (a matrix= cross product as object-of-arrays),
                        variants (coupled cells as include: entries), or tasks (one job per child
                        of a Parallel, dispatched by binding name) — pin it in the CI discover job
                        so a tasks.py refactor that changes the emitted shape fails loudly instead
                        of silently reshaping the matrix
  --list                list all defined tasks and exit
  --tree                print every defined task's expanded tree and exit
  --check               check the task definition (ty)
  --init                write a commented starter tasks.py into the current directory and exit
  --verbose             with --init, scaffold the kitchen-sink template — every
                        Task/Sequential/Pipe/Parallel/Config option worked and explained — instead
                        of the minimal starter
  --effects [EFFECTS]   tuple of Effect instances; pass with no value to list available Effects
                        (default: (Termtree(),); (Status(output_mode='github'),) under
                        GITHUB_ACTIONS=true)
  --matrix KEY=VAL[,VAL...]
                        override a matrix axis (repeatable; e.g. --matrix PY=3.13)
  --jobs N              cap concurrently running leaves at N (also: CAMAS_JOBS); a pipe counts as
                        one leaf, its stages running together. A throttle, not a speedup — the
                        tree never runs wider than its own fan-out, so --jobs can only slow a run
                        down; reach for it only when a wide matrix oversubscribes the machine
                        (CPU/RAM/disk)
  --under DURATION      run only the task's leaves whose timed estimate fits DURATION (e.g. 1s,
                        500ms, 2m), mutating leaves (formatters) first then the read-only rest in
                        parallel; untimed leaves run (and are thereby measured), only leaves
                        measured over budget are skipped — except a pipe kept whole for its
                        untimed siblings, which runs its over-budget stages too — so a cold cache
                        runs the whole tree. Operates on the named task, or the Config default
                        when none is given
  --paths PATH[,PATH...]
                        scope the run to these changed paths (repeatable): inject them into each
                        leaf's {paths} and drop leaves that match none (a leaf's when= predicate
                        also prunes on no match), e.g. camas check --paths src/a.py. The
                        camas_gate MCP tool scopes the agent fix node and the checks this way

Available Effects:
  Ctrf  — A CTRF JSON test report.
    Ctrf(path: str | None = None, tail_bytes: int = 8192)

  GitHubChecks  — Per-leaf check runs on the GitHub Checks API.
    GitHubChecks(token: str | None = None, repository: str | None = None, sha: str | None = None, name_prefix: str = '', tail_bytes: int = 8192, fail_on_api_error: bool = False)

  Status  — Line-oriented status output.
    Status(output_mode: Literal['quiet', 'all', 'errors', 'stream', 'github'] = 'errors', started_fmt: str = '\x1b[90m[{timestamp:%Y-%m-%d %H:%M:%S}.{ms:03d}]\x1b[0m \x1b[38;5;135m▶ [{name}] started\x1b[0m', finished_fmt: str = '\x1b[90m[{timestamp:%Y-%m-%d %H:%M:%S}.{ms:03d}]\x1b[0m \x1b[32m✓ [{name}] success\x1b[0m ({elapsed:.3f}s)', failed_fmt: str = '\x1b[90m[{timestamp:%Y-%m-%d %H:%M:%S}.{ms:03d}]\x1b[0m \x1b[31m✗ [{name}] error\x1b[0m exit={rc} ({elapsed:.3f}s)', skipped_fmt: str = '\x1b[90m[{timestamp:%Y-%m-%d %H:%M:%S}.{ms:03d}]\x1b[0m \x1b[90m⏭ [{name}] skipped\x1b[0m (prior rc={rc})', stopped_fmt: str = '\x1b[90m[{timestamp:%Y-%m-%d %H:%M:%S}.{ms:03d}]\x1b[0m \x1b[33m■ [{name}] stopped\x1b[0m exit={rc} ({elapsed:.3f}s)', errored_fmt: str = '\x1b[90m[{timestamp:%Y-%m-%d %H:%M:%S}.{ms:03d}]\x1b[0m \x1b[31m⚠ [{name}] errored\x1b[0m {msg}', output_fmt: str = '\x1b[90m[{timestamp:%Y-%m-%d %H:%M:%S}.{ms:03d}] · [{name}]\x1b[0m {line}')

  Summary  — The final task tree, rendered once after the run.
    Summary(term_width: Auto | Fixed = Auto(), show_passing: bool = False)
      Auto()
      Fixed(columns: int)

  Termtree (default)  — A live task tree in the terminal.
    Termtree(frame_interval_ms: float = 16.667, show_passing: bool = False, output_ctrl_c: bool = False)

  Timings  — Each leaf's observed duration, recorded for later runs.
    Timings(camas_dir: Path, scope: int = 0, identities: tuple[CacheKey, ...] | None = None)

Try:
  camas '("echo Hello", "echo world!")'                      # ( ) → Sequential
  camas '{"echo Hello", "echo world!"}'                      # { } → Parallel
  camas --effects '(Summary(term_width=Fixed(60)),)' <task>  # post-run summary instead of live tree

Reference:
  Examples:                 https://github.com/JPHutchins/camas/tree/main/examples
  PyPI:                     https://pypi.org/project/camas/
```

</details>

<details><summary><code>camas mcp --help</code></summary>

```console
$ camas mcp --help
camas mcp — serve this project's tasks to AI agents over the Model Context Protocol.

Usage:
  camas mcp [--plain]        run the MCP stdio server (rich output by default)
  camas mcp init [--claude]  write this project's .mcp.json entry for the camas server;
    [--launcher uv|uvx|camas]  with --claude also configure Claude Code: .mcp.json +
                              the Stop hooks (fix + async nudge) + the tiered
                              camas-fixer agents + gate skill; --launcher forces
                              the launch strategy instead of auto-detecting
  camas mcp fix [--paths P]… run the registered agent fix node (Config.agent.fix) over the
                              changed paths (--paths, else a piped PostToolBatch event; a Stop
                              event names none, so the whole tree) — the Stop fix hook;
                              no-op if unregistered
  camas mcp gate [task]     run the gate once, headless and read-only — print the verdict as
    [--paths P]… [--under D]  JSON, exit 0 (continue) / 2 (block); scope to --paths or a piped
    [--jobs N] [--nudge]      PostToolBatch event; a Stop event gates the whole tree (CI +
                              benchmark); --under takes a duration (5, 1.5s, 500ms, 2m, 1h);
                              --nudge waits for the Stop fix, then emits the async Stop-hook
                              nudge text instead of the JSON verdict — at most once per prompt,
                              and exit 0 (no rewake) when no check node or camas[mcp] is missing

Options:
  --rich        accepted for back-compat; rich output is the default
  --plain       disable rich output (opt out of the 2025-11-25 tool fields)
  -h, --help    show this help
```

</details>

### Environment variables

- ✚ `CLAUDECODE` and `CAMAS_AGENT` (agent detection), `CAMAS_NO_MCP_HINT` (the agent banner), `CAMAS_JOBS` (`--jobs`), `GITHUB_ACTIONS` (CI detection), `NO_COLOR`; the leaf environment's `FORCE_COLOR` and `CLICOLOR_FORCE`.

### MCP tools

#### `camas_list`

List THIS project's camas-defined tasks: each task's name, help text, its
command expression, and any matrix axes it expands across (versions, targets,
toolchains, platforms — whatever the project declares). Matrix tasks show the
unexpanded template by default (their axis values are listed separately, which
keeps discovery compact); pass expand_matrix=true to inline every resolved leaf
recursively. Two tasks are flagged: the default (what the developer runs while
working) and the github default — the exact task this project's CI runs,
declared via Config(github_task=...) (camas is the single definition shared by
local and CI), so running it locally before you commit or push reproduces CI
exactly; github_default is null when the project declares no github_task (camas
never infers it from CI workflow files). run_default names the task a no-task
camas_run resolves (agent default, else check, else the github or default
task). Tasks whose leaves have been timed also
carry an estimated duration and their slowest leaf, so you can pick a quick task
for the inner loop and the thorough one before committing. Use this first to
discover valid task names for camas_run; it is the source of truth. Read-only;
runs nothing.

#### `camas_run`

Run THIS project's defined tasks exactly as composed by the project — honoring
its concurrency, matrix expansion across whatever axes it declares, and
per-task cwd/env. Use this INSTEAD of invoking the underlying commands by hand:
it runs the project-sanctioned command set, in the right order, in parallel
where declared, and returns a structured per-task pass/fail report. A failed
leaf's output is included inline (the last N lines) — read its log path only
when you see the truncation marker, or pass verbosity='full' to inline every
leaf's full output. For a tight inner loop, pass args=[…] to append flags to a
single-leaf task (camas's -- passthrough, e.g. args=['tests/test_x.py::test_y',
'-x'] to run and fail-fast on one test); composite tasks reject args, so target
a leaf. For a time-boxed inner loop, pass under=<seconds> to run only the leaves
whose recorded estimate fits — mutating leaves (formatters) first, then the
read-only rest in parallel (a pipe kept whole for its untimed siblings runs its
over-budget stages too); omit task to budget the project default. For an ad-hoc
scoped run of any task, pass paths=[…] (changed files, like the CLI --paths): each
{paths} command is narrowed to the files it covers, and it combines with under
(scope first, then budget). Compact failures-first summary by default; dry_run=true
previews the fully-resolved plan without executing. Omit task entirely to run the
project's configured default. A non-zero result means a task failed — a normal
result, not a tool error.

#### `camas_check`

Validate THIS project's tasks definition: re-read tasks.py from disk, evaluate
it (catching import and runtime errors), then run a static type checker (ty or
mypy) over it. Call this right after you write or edit tasks.py to confirm it
loads and type-checks before running anything — the tight authoring loop.
Returns a structured verdict (ok / load error / type error / no checker
available / no tasks file) carrying the exact diagnostics to fix. A broken file
is a normal result, not a tool error. On success the task catalog is refreshed,
so a newly-added task becomes runnable via camas_run without restarting the
server.

#### `camas_docs`

How to author or edit THIS project's tasks.py. Returns camas's own installed
source path and where its README is, with every heading's line number, served
live from the package so it never drifts: the Task / Sequential / Parallel / Pipe
/ Config API, matrix expansion, per-leaf cwd and env, and custom output Effects,
with worked examples and the full API reference. Read the sections you need
before writing tasks.py, then validate your work with camas_check. The cited
source path is the API source of truth — read it for exact signatures.

#### `camas_gate`

The SA-delegation gate: first run the project's registered deterministic fix
(Config.agent.fix — formatters, --fix linters) over the files just changed, then scope
THIS project's checks to them, run them, and return a binary verdict, so a residual is
never one the fix settles. Re-read a file before editing it after a gate: the fix
may have rewritten it. residual_class is
'green' (decision 'continue') when the checks pass, or 'needs_reasoning' (decision
'block') when a check still fails — then diagnostics carries the failing leaves. Pass
paths=[…] (the changed files) to scope; omit to gate the whole check node, which also
runs the fix over the whole tree. under=<seconds>
time-boxes the checks: leaves measured to exceed it are skipped (except the over-budget
stages of a pipe kept whole for its untimed siblings), untimed leaves run.

#### `camas_fix`

Run the project's registered deterministic fix node (`Config.agent.fix`)
scoped to the changed paths — mutating, behavior-preserving formatters and
`--fix` linters. It does not reason; it only fixes. Pass paths=[…] (the changed
files) to scope; omit to run the whole fix node un-scoped. Each `{paths}` command
is injected with the files it covers; a command without `{paths}` always runs
unless its `when=` excludes the changed set.
A no-op (exit 0, no leaves) when no fix node is registered (`Config.agent.fix` is
`None`). `jobs` caps concurrently running leaves (a pipe counts as one leaf, its stages running together).

#### `camas_init`

Scaffold a commented starter tasks.py in THIS project's root when it has none — the
MCP mirror of `camas --init`, for driving camas purely over the MCP. Defaults
(verbose=true) to the kitchen-sink template: every Task/Sequential/Parallel/Pipe/Config
option worked and explained in place — path scoping ({paths}/when=), matrix
expansion, agent_format structured output, and Config(agent=Claude(fix=...,
check=..., default=...)) — with cross-platform placeholder commands, so you can read
it once and see the whole authoring surface. Pass verbose=false for the same short
template `camas --init` (no --verbose) writes instead. Either way this creates the
.camas/ run-log and timing directory beside it, and returns the path plus the file's
content so you can edit from there. Refuses to overwrite an existing tasks.py
(returns status 'exists', file untouched). The tasks it creates are immediately
runnable via camas_run — no restart. Then read camas_docs and validate edits with
camas_check.

#### `camas_github_matrix`

Emit a task's fan-out as GitHub Actions strategy.matrix JSON — the MCP mirror of
`camas --github-matrix`, for authoring or updating a CI workflow. Three shapes, from
how tasks.py fans out: 'axes' projects a matrix= cross product as object-of-arrays,
'variants' enumerates coupled variants= cells as GHA include: entries, and 'tasks'
emits one job per child of a Parallel ({"task": [...]}, run as
`camas ${{ matrix.task }}`) — the common CI shape with no version matrix at all.
Omit shape to derive it; the response echoes the shape that was emitted plus a
job_command template, and pinning that shape in the workflow makes a later tasks.py
refactor fail loudly instead of silently reshaping the matrix. Pass task=<name>;
omit to use the project default. Pin axes with matrix_overrides (like camas_run),
e.g. {"PY": ["3.13"]} — a variants= key filters to the bundles binding it. Every
emission is verified against the real run-set: each job must run exactly its own
cell's leaves, and the jobs together every leaf exactly once, so a fan-out with no
faithful projection (independent fan-outs in one tree, a plain leaf beside matrixed
siblings) is a tool error, not a run failure. Read-only; runs nothing.

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
node = "node_modules/.bin"

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

```
$ cd tests/fixtures/tauri-app
```

**List the tasks** —

<details>
<summary><code>camas --list</code></summary>

```console
$ camas --list
Available tasks from .../tauri-app/tasks.py:
  all (default)  frontend | backend | sdk
  backend        cargo fmt --all, cargo clippy --all-targets --locked -- -D warnings | cargo test --all-targets --locked
  build          Debug and release builds (FLAG='-- --debug' debug, FLAG='' release)  [matrix: FLAG×2 (-- --debug..)]
  fix            node_modules/.bin/prettier --write . | cargo fmt --all | (uv run ruff check --fix ., uv run ruff format .)
  frontend       node_modules/.bin/prettier --write ., node_modules/.bin/eslint src/ | node_modules/.bin/tsc --noEmit | node_modules/.bin/vitest run
  sdk            uv run ruff check --fix ., uv run ruff format ., uv run mypy . | uv run pytest
```

</details>

**Preview what `all` would run** —

<details>
<summary><code>camas --dry-run all</code></summary>

```console
$ camas --dry-run all
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

<details>
<summary><code>camas build --help</code></summary>

```console
$ camas build --help
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

<details>
<summary><code>camas build --dry-run --FLAG '-- --debug'</code></summary>

```console
$ camas build --dry-run --FLAG '-- --debug'
build ∥
┃ npm run tauri build -- --debug [FLAG=-- --debug]: npm run tauri build -- --debug  FLAG=-- --debug
```

</details>
