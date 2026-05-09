"""Demonstrates a typed custom Effect discoverable from tasks.py scope."""

from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent
from typing import Final

from camas import Parallel, Sequential, Task
from camas.core.effect import Effect
from camas.core.leaf_state import LeafState
from camas.core.task import TaskNode
from camas.core.task_event import OutputEvent, StartedEvent, TaskEvent


class Tail(Effect[None]):
	"""Stream each line of task stdout as ``name: line`` pairs as it arrives —
	a less-sophisticated Termtree suited to CI logs where you want interleaved
	per-task output without ANSI redraws."""

	async def setup(self, task: TaskNode) -> None: ...

	async def on_event(
		self,
		event: TaskEvent,
		states: Sequence[LeafState],
		ctx: None,
	) -> None:
		match event:
			case OutputEvent(task=task, line=line):
				label: Final = task.name or (
					task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)
				)
				print(f"{label}: {line.decode('utf-8', errors='replace').rstrip()}")
			case _:
				pass

	async def teardown(self, ctxs: tuple[None, ...]) -> None: ...


def _safe(name: str) -> str:
	"""Reduce a task name to a filesystem-safe slug for log file names."""
	return "_".join(
		seg for seg in (
			"".join(c if c.isalnum() or c in "-_=." else " " for c in name).split()
		)
	)


class FileLog(Effect[Path | None]):
	"""Per-leaf log files under ``./logs/`` — one file per task, and one
	per matrix binding for matrix tasks (since matrix expansion produces a
	distinct ``task.name`` per binding). The per-leaf ctx tracks the file
	path; ``setup`` creates the directory once, ``StartedEvent`` derives
	the path, ``OutputEvent`` appends each line to it."""

	async def setup(self, task: TaskNode) -> Path | None:
		Path("logs").mkdir(exist_ok=True)
		return None

	async def on_event(
		self,
		event: TaskEvent,
		states: Sequence[LeafState],
		ctx: Path | None,
	) -> Path | None:
		match event:
			case StartedEvent(task=task):
				file_path: Final = Path("logs") / f"{_safe(task.name or 'anon')}.log"
				file_path.write_bytes(b"")
				return file_path
			case OutputEvent(line=line) if ctx is not None:
				with ctx.open("ab") as f:
					f.write(line)
				return ctx
			case _:
				return ctx

	async def teardown(self, ctxs: tuple[Path | None, ...]) -> None: ...


def _ticker(label: str, n: int, interval: float = 0.05) -> Task:
	script: Final = dedent(f"""\
		import time
		for i in range({n}):
			print(f"{label} {{i}}", flush=True)
			time.sleep({interval})
	""")
	return Task(("python", "-c", script), name=label)


fast = _ticker("fast", 2)
slow = _ticker("slow", 3)
done = Task(("python", "-c", "print('done')"), name="done")

check = Sequential(Parallel(fast, slow), done)

build = Parallel(
	Task(("python", "-c", "print('built {STAGE}')")),
	matrix={"STAGE": ("compile", "link")},
	name="build",
)
