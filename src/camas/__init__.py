# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins
r"""camas — parallel and sequential task-tree runner.

The README documents camas, and the MCP tool `camas_docs` serves it. These
examples drive the CLI as a subprocess, with `python -c` leaves only because
every CI environment has Python.

`make_camas(tmp)` runs the CLI from a scratch directory:

	>>> import os, subprocess, sys, tempfile
	>>> from pathlib import Path
	>>> from textwrap import dedent
	>>> def make_camas(tmp):
	...     def camas(*args):
	...         return subprocess.run(
	...             [sys.executable, "-m", "camas", *args],
	...             cwd=tmp, capture_output=True, text=True,
	...             encoding="utf-8", errors="replace",
	...             env={**os.environ, "NO_COLOR": "1", "CLAUDECODE": "", "CAMAS_AGENT": ""},
	...         )
	...     return camas

One leaf, run and listed by `--help`:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from camas import Task
	...         hello = Task(("python", "-c", "print('hello world')"))
	...     '''))
	...     # Equivalent shell: $ camas --effects='(Summary(...))' hello
	...     run_ = camas("--effects=(Summary(show_passing=True),)", "hello")
	...     helped = camas("--help")
	>>> run_.returncode, helped.returncode
	(0, 0)
	>>> "PASSED" in run_.stdout and "hello world" in run_.stdout
	True
	>>> "hello" in helped.stdout
	True

A composed tree: `--dry-run`, `--tree`, a run, and `--` argument forwarding:

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
	...         ci = Sequential(
	...             lint,
	...             Parallel(test, "python --version"),
	...             say,
	...             name="ci",
	...         )
	...     '''))
	...     dry = camas("--dry-run", "ci")
	...     tree = camas("--tree")
	...     run_ = camas("--effects=(Summary(show_passing=True),)", "ci")
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

`Sequential` stops at the first failure and skips the rest:

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

`lint | test` is a `Parallel`, `fmt + check` a `Sequential`, and `check - test`
drops a child:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from camas import Task
	...         fmt = Task(("python", "-c", "print('formatted')"), mutates=True)
	...         lint = Task(("python", "-c", "print('linted')"))
	...         test = Task(("python", "-c", "print('tested')"))
	...         check = lint | test
	...         ci = fmt + check
	...         quick = check - test
	...     '''))
	...     tree = camas("--dry-run", "ci")
	...     quick_ = camas("--dry-run", "quick")
	...     run_ = camas("--effects=(Summary(show_passing=True),)", "ci")
	>>> [r.returncode for r in (tree, quick_, run_)]
	[0, 0, 0]
	>>> tree.stdout.index("fmt:") < tree.stdout.index("lint:") < tree.stdout.index("test:")
	True
	>>> "lint:" in quick_.stdout and "test:" not in quick_.stdout
	True
	>>> all(s in run_.stdout for s in ("formatted", "linted", "tested"))
	True

A `Pipe` feeds each stage's stdout to the next stage's stdin, and `>` pipes one
`Task` into another:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from camas import Pipe, Task
	...         upper = ("python", "-c", "import sys; print(sys.stdin.read().upper(), end='')")
	...         shout = Pipe(Task(("python", "-c", "print('hello pipe')")), Task(upper), name="shout")
	...         whisper = Task(("python", "-c", "print('quiet')")) > Task(upper)
	...     '''))
	...     piped = camas("--effects=(Summary(show_passing=True),)", "shout")
	...     chained = camas("--effects=(Summary(show_passing=True),)", "whisper")
	>>> piped.returncode, chained.returncode
	(0, 0)
	>>> "HELLO PIPE" in piped.stdout and "QUIET" in chained.stdout
	True

A matrix, an axis pinned from the CLI, and the task's `--help`:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from camas import Parallel, Task
	...         greet = Parallel(
	...             Task(("python", "-c", "print('hello, {NAME}')")),
	...             matrix={"NAME": ("Grace", "Ada", "Wendy", "Alyssa")},
	...             name="greet",
	...         )
	...     '''))
	...     full = camas("--effects=(Summary(),)", "greet")
	...     pinned = camas("--effects=(Summary(),)", "greet", "--NAME=Ada")
	...     axes = camas("greet", "--help")
	>>> full.returncode, pinned.returncode, axes.returncode
	(0, 0, 0)
	>>> all(n in full.stdout for n in ("Grace", "Ada", "Wendy", "Alyssa"))
	True
	>>> "[NAME=Ada]" in pinned.stdout
	True
	>>> "--NAME" in axes.stdout
	True

Two axes, overridden from the CLI with comma-separated values:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from camas import Parallel, Task
	...         meet = Parallel(
	...             Task(("python", "-c", "print('hi {NAME1}, meet {NAME2}')")),
	...             matrix={
	...                 "NAME1": ("Grace", "Ada"),
	...                 "NAME2": ("Wendy", "Alyssa"),
	...             },
	...             name="meet",
	...         )
	...     '''))
	...     baseline = camas("--effects=(Summary(show_passing=True),)", "meet")
	...     overridden = camas(
	...         "--effects=(Summary(show_passing=True),)",
	...         "meet", "--NAME1=Clara,Dolores", "--NAME2=Grace,Ada",
	...     )
	...     axes = camas("meet", "--help")
	>>> [r.returncode for r in (baseline, overridden, axes)]
	[0, 0, 0]
	>>> all(s in baseline.stdout for s in (
	...     "hi Grace, meet Wendy",
	...     "hi Grace, meet Alyssa",
	...     "hi Ada, meet Wendy",
	...     "hi Ada, meet Alyssa",
	... ))
	True
	>>> all(s in overridden.stdout for s in (
	...     "hi Clara, meet Grace",
	...     "hi Clara, meet Ada",
	...     "hi Dolores, meet Grace",
	...     "hi Dolores, meet Ada",
	... ))
	True
	>>> "Wendy" not in overridden.stdout and "Alyssa" not in overridden.stdout
	True
	>>> "--NAME1" in axes.stdout and "--NAME2" in axes.stdout
	True

Coupled `variants=`, a pin that filters them, and `--github-matrix`:

	>>> with tempfile.TemporaryDirectory() as tmp:
	...     camas = make_camas(tmp)
	...     _ = (Path(tmp) / "tasks.py").write_text(dedent('''\
	...         from camas import Parallel, Task
	...         port = Parallel(
	...             Task(("python", "-c", "print('build {TARGET} with {TOOL}')")),
	...             variants=(
	...                 {"TARGET": "wasm", "TOOL": "emcc"},
	...                 {"TARGET": "native", "TOOL": "clang"},
	...             ),
	...             name="port",
	...         )
	...     '''))
	...     full = camas("--effects=(Summary(show_passing=True),)", "port")
	...     pinned = camas("--effects=(Summary(show_passing=True),)", "port", "--TARGET=wasm")
	...     emitted = camas("port", "--github-matrix")
	>>> [r.returncode for r in (full, pinned, emitted)]
	[0, 0, 0]
	>>> all(s in full.stdout for s in ("build wasm with emcc", "build native with clang"))
	True
	>>> "clang" not in pinned.stdout
	True
	>>> emitted.stdout.strip()
	'{"include":[{"TARGET":"wasm","TOOL":"emcc"},{"TARGET":"native","TOOL":"clang"}]}'

An inline `Effect`: listed by `--help`, `--list`, and `--effects`, then run alone and beside `Summary`:

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
"""

import typing

from .v0 import GIT_PORCELAIN as GIT_PORCELAIN
from .v0 import Agent as Agent
from .v0 import AgentFormat as AgentFormat
from .v0 import Claude as Claude
from .v0 import Clean as Clean
from .v0 import Config as Config
from .v0 import Effect as Effect
from .v0 import Parallel as Parallel
from .v0 import Pipe as Pipe
from .v0 import Project as Project
from .v0 import Sequential as Sequential
from .v0 import Task as Task
from .v0 import by_glob as by_glob
from .v0 import by_suffix as by_suffix

if typing.TYPE_CHECKING:
	from .main.dispatch import run_cli as run_cli


def __getattr__(name: str) -> typing.Any:
	if name == "run_cli":
		from .main.dispatch import run_cli

		return run_cli
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
