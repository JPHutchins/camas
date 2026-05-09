"""Demonstrates a typed custom Effect discoverable from tasks.py scope."""

from collections.abc import Sequence
from textwrap import dedent
from typing import Final

from typing_extensions import override

from camas import Parallel, Sequential, Task
from camas.core.effect import Effect
from camas.core.leaf_state import LeafState
from camas.core.task import TaskNode
from camas.core.task_event import OutputEvent, TaskEvent


class Tail(Effect[None]):
	"""Stream each line of task stdout as ``name: line`` pairs as it arrives —
	a less-sophisticated Termtree suited to CI logs where you want interleaved
	per-task output without ANSI redraws."""

	@override
	async def setup(self, task: TaskNode) -> None: ...

	@override
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

	@override
	async def teardown(self, ctxs: tuple[None, ...]) -> None: ...


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
