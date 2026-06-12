# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins
r"""camas — parallel and sequential task-tree runner.

Define your task tree in ``tasks.py``. ``camas <name>`` runs a task by
name, ``camas --list`` enumerates them, ``camas --help`` shows
everything (tasks + effects + hints), ``camas --effects`` lists the
available output renderers, ``camas --tree`` prints every task's full
expansion, and ``--AXIS VAL`` overrides a matrix axis from the CLI.

A leaf is *any* shell command. The doctests below all run
``python -c ...`` snippets only because every CI environment has
Python; in your real ``tasks.py``, write the actual command::

    Task("ruff check .")
    Task("cargo test", cwd=Path("rust"))
    Task("npm test")

``Task``/``Sequential``/``Parallel`` also accept a ``help="..."``
kwarg that overrides what ``camas --list`` prints for that task —
rarely needed (the cmd or tree usually self-documents; this codebase
uses ``help=`` only in a handful of places), but available for
cryptic commands.

These four are the unversioned alias for the latest API generation; to
pin a generation, import from its namespace (``camas.v0``). See the
README's Versioning section.

**LLM agents:** prefer ``--effects='(Summary(),)'`` when invoking
``camas`` from a tool. ``Summary`` produces one compact post-run
report; the default ``Termtree`` is a live, redrawing animation that
bloats stdout with hundreds of frame redraws when captured. Humans
keep ``Termtree`` (it's the default).

The doctests below invoke ``camas`` as a subprocess so they can verify
behavior end-to-end. In real usage these are just shell commands::

    $ camas hello                              # one task
    $ camas ci --dry-run                       # preview without running
    $ camas test -- -v                         # forward "-v" to the task's cmd
    $ camas greet --NAME=Ada                   # pin a matrix axis
    $ camas build --effects='(Loud(),)'        # use a custom effect

Each ``subprocess.run([sys.executable, "-m", "camas", ...])`` below is
the equivalent of one of those.

Imports and a small factory used by every block: ``make_camas(tmp)``
returns a closure-over-``tmp`` that invokes the CLI from that tempdir.
Each block does ``camas = make_camas(tmp)`` and then calls
``camas("--list")``, ``camas("hello")`` etc — equivalent to typing
those at the shell from ``tmp``.

	>>> import os, subprocess, sys, tempfile
	>>> from pathlib import Path
	>>> from textwrap import dedent
	>>> def make_camas(tmp):
	...     def camas(*args):
	...         return subprocess.run(
	...             [sys.executable, "-m", "camas", *args],
	...             cwd=tmp, capture_output=True, text=True,
	...             encoding="utf-8", errors="replace",
	...             env={**os.environ, "NO_COLOR": "1"},
	...         )
	...     return camas

A single leaf — the smallest viable ``tasks.py``. Passing
``Summary(SummaryOptions(show_passing=True))`` prints every passing
task's stdout so the leaf's output is verifiable from the run record;
``camas --help`` lists the task by name:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from camas import Task
	...         hello = Task(("python", "-c", "print('hello world')"))
	...     '''))
	...     # Equivalent shell: $ camas --effects='(Summary(...))' hello
	...     run_ = camas("--effects=(Summary(SummaryOptions(show_passing=True)),)", "hello")
	...     helped = camas("--help")
	>>> run_.returncode, helped.returncode
	(0, 0)
	>>> "PASSED" in run_.stdout and "hello world" in run_.stdout
	True
	>>> "hello" in helped.stdout
	True

A composed tree mixes ``Sequential`` (run in order, short-circuit on
failure) and ``Parallel`` (run concurrently). Bare strings inside the
constructors coerce to anonymous Tasks. ``cwd`` and ``env`` are scoped
per-leaf. ``--dry-run`` prints the tree without running, ``--tree``
prints every defined task, and a trailing ``-- ARG`` forwards args to
the dispatched task's command:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     (Path(tmp) / "subdir").mkdir()
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from pathlib import Path
	...         from camas import Parallel, Sequential, Task
	...
	...         lint = Task(("python", "-c", "print('lint ok')"))
	...         test = Task(("python", "-c", "print('test ok')"), cwd=Path("subdir"))
	...         say = Task(
	...             ("python", "-c", "import os; print('VAR=' + os.environ['VAR'])"),
	...             env={"VAR": "scoped"},
	...         )
	...         echo = Task(("python", "-c", "import sys; print(*sys.argv[1:])"))
	...
	...         # A bare string inside Parallel/Sequential coerces to an
	...         # anonymous Task; its cmd is shlex-split, so prefer simple
	...         # forms (no embedded quotes). Use the tuple form for
	...         # complex commands to skip shlex entirely.
	...         ci = Sequential(
	...             lint,
	...             Parallel(test, "python --version"),
	...             say,
	...             name="ci",
	...         )
	...     '''))
	...     dry = camas("--dry-run", "ci")
	...     tree = camas("--tree")
	...     run_ = camas("--effects=(Summary(SummaryOptions(show_passing=True)),)", "ci")
	...     fwd = camas("echo", "--", "one", "two", "three")
	>>> [r.returncode for r in (dry, tree, run_, fwd)]
	[0, 0, 0, 0]
	>>> "lint" in dry.stdout and "python --version" in dry.stdout
	True
	>>> "ci" in tree.stdout and "echo" in tree.stdout
	True
	>>> all(s in run_.stdout for s in ("lint ok", "test ok", "Python", "VAR=scoped"))
	True
	>>> "one two three" in fwd.stdout
	True

``Sequential`` short-circuits on failure: when a leaf returns non-zero,
later leaves in that ``Sequential`` are reported ``Skipped`` rather
than executed, and the camas process exits non-zero:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from camas import Sequential, Task
	...         pipeline = Sequential(
	...             Task(("python", "-c", "raise SystemExit(2)"), name="boom"),
	...             Task(("python", "-c", "print('never reached')"), name="follow_up"),
	...             name="pipeline",
	...         )
	...     '''))
	...     r = camas("--effects=(Summary(),)", "pipeline")
	>>> r.returncode != 0
	True
	>>> "FAIL" in r.stdout and "SKIP" in r.stdout
	True
	>>> "never reached" not in r.stdout
	True

A one-axis matrix expanding over four names, plus a CLI override
pinning the axis to a single value, plus per-task ``--help`` that
documents the matrix override flag:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from camas import Parallel, Task
	...         greet = Parallel(
	...             Task(("python", "-c", "print('hello, {NAME}')")),
	...             matrix={"NAME": ("Jane", "Ada", "Wendy", "Alyssa")},
	...             name="greet",
	...         )
	...     '''))
	...     full = camas("--effects=(Summary(),)", "greet")
	...     pinned = camas("--effects=(Summary(),)", "greet", "--NAME=Ada")
	...     axes = camas("greet", "--help")
	>>> full.returncode, pinned.returncode, axes.returncode
	(0, 0, 0)
	>>> all(n in full.stdout for n in ("Jane", "Ada", "Wendy", "Alyssa"))
	True
	>>> "[NAME=Ada]" in pinned.stdout
	True
	>>> "--NAME" in axes.stdout
	True

A two-axis matrix with substitution across the cross-product, then the
same task with both axes overridden from the CLI to multi-value lists
that introduce Clara and Dolores; per-task ``--help`` documents both
axes:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from camas import Parallel, Task
	...         meet = Parallel(
	...             Task(("python", "-c", "print('hi {NAME1}, meet {NAME2}')")),
	...             matrix={
	...                 "NAME1": ("Jane", "Ada"),
	...                 "NAME2": ("Wendy", "Alyssa"),
	...             },
	...             name="meet",
	...         )
	...     '''))
	...     baseline = camas("--effects=(Summary(SummaryOptions(show_passing=True)),)", "meet")
	...     overridden = camas(
	...         "--effects=(Summary(SummaryOptions(show_passing=True)),)",
	...         "meet", "--NAME1=Clara,Dolores", "--NAME2=Jane,Ada",
	...     )
	...     axes = camas("meet", "--help")
	>>> [r.returncode for r in (baseline, overridden, axes)]
	[0, 0, 0]
	>>> all(s in baseline.stdout for s in (
	...     "hi Jane, meet Wendy",
	...     "hi Jane, meet Alyssa",
	...     "hi Ada, meet Wendy",
	...     "hi Ada, meet Alyssa",
	... ))
	True
	>>> all(s in overridden.stdout for s in (
	...     "hi Clara, meet Jane",
	...     "hi Clara, meet Ada",
	...     "hi Dolores, meet Jane",
	...     "hi Dolores, meet Ada",
	... ))
	True
	>>> "Wendy" not in overridden.stdout and "Alyssa" not in overridden.stdout
	True
	>>> "--NAME1" in axes.stdout and "--NAME2" in axes.stdout
	True

A custom ``Effect`` defined inline in ``tasks.py`` is discovered
automatically — ``camas --help`` lists it under *Available Effects*
alongside the built-ins (an LLM agent reading ``--help`` immediately
sees the plugin), ``--effects`` (no value) prints the effects-only
listing, ``<task> --help`` documents matrix override flags, and
``--effects=(Loud(),)`` invokes the user effect for real. ``--effects``
takes a *tuple* of any length, so multiple effects can be muxed in one
run — ``--effects=(Summary(), Loud())`` produces both the post-run
report and the streaming per-task lines:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from collections.abc import Sequence
	...         from camas.v0.effect import Effect
	...         from camas.v0.leaf_state import LeafState
	...         from camas.v0.task import Parallel, Task, TaskNode
	...         from camas.v0.task_event import OutputEvent, TaskEvent
	...
	...         class Loud(Effect[None]):
	...             async def setup(self, task: TaskNode) -> None: ...
	...             async def on_event(
	...                 self, event: TaskEvent, states: Sequence[LeafState], ctx: None,
	...             ) -> None:
	...                 if isinstance(event, OutputEvent):
	...                     print(f"!! {event.task.name}: {event.line.decode().strip()}")
	...             async def teardown(self, ctxs: tuple[None, ...]) -> None: ...
	...
	...         build = Parallel(
	...             Task(("python", "-c", "print('built {STAGE}')")),
	...             matrix={"STAGE": ("compile", "link")},
	...             name="build",
	...         )
	...     '''))
	...     top_help = camas("--help")
	...     listed = camas("--list")
	...     effects = camas("--effects")
	...     axes = camas("build", "--help")
	...     loud = camas("--effects=(Loud(),)", "build")
	...     muxed = camas("--effects=(Summary(), Loud())", "build")
	>>> [r.returncode for r in (top_help, listed, effects, axes, loud, muxed)]
	[0, 0, 0, 0, 0, 0]
	>>> "Available Effects" in top_help.stdout and "Loud" in top_help.stdout
	True
	>>> "build" in listed.stdout
	True
	>>> "Loud" in effects.stdout
	True
	>>> "--STAGE" in axes.stdout
	True
	>>> "built compile" in loud.stdout and "built link" in loud.stdout
	True
	>>> "PASS" in muxed.stdout and "!! " in muxed.stdout
	True

More patterns — canonical, full project layouts (these are not in your
local source tree; follow the GitHub links):

https://github.com/JPHutchins/camas/tree/main/examples

And advanced granular lib usage:

https://github.com/JPHutchins/camas/blob/main/tests

Effect protocol: subclass ``Effect`` (see ``examples/effect-plugin/``). Per-task
help: ``camas <task> --help``.
"""

import typing

from .v0 import Effect as Effect
from .v0 import Parallel as Parallel
from .v0 import Sequential as Sequential
from .v0 import Task as Task

if typing.TYPE_CHECKING:
	from .main.dispatch import run_cli as run_cli


def __getattr__(name: str) -> object:
	"""Lazily expose ``run_cli`` — the standalone ``run_cli(globals())`` entry point
	— without importing the engine at package load, so ``from camas import Task``
	stays light and the type layer keeps not depending on ``camas.main``.

	Raises:
		AttributeError: for any name other than ``run_cli``.
	"""
	if name == "run_cli":
		from .main.dispatch import run_cli

		return run_cli
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
