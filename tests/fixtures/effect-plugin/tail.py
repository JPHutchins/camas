"""A custom Effect defined in a sibling module — imported by ``tasks.py``
to demonstrate that user effects don't have to live in the task file."""

from collections.abc import Sequence
from typing import Final

from camas.v0.effect import Effect
from camas.v0.leaf_state import LeafState
from camas.v0.task import TaskNode
from camas.v0.task_event import OutputEvent, TaskEvent


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
