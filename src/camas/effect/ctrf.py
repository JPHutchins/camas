# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

from ..core.traversal import flatten_leaves
from ..v0.leaf_state import Waiting

if TYPE_CHECKING:
	from collections.abc import Sequence
	from types import ModuleType

	from ..v0.leaf_state import LeafState
	from ..v0.task import TaskNode
	from ..v0.task_event import TaskEvent


def require_model() -> ModuleType:
	try:
		from . import _ctrf_model
	except ImportError as e:
		raise RuntimeError("Ctrf: requires feature camas[ctrf]") from e
	return _ctrf_model


def now_ms() -> int:
	return int(datetime.now(timezone.utc).timestamp() * 1000)


def emit_report(path: str | None, data: bytes) -> None:
	payload = data + b"\n"
	if path is None:
		sys.stdout.buffer.write(payload)
		sys.stdout.flush()
	else:
		Path(path).write_bytes(payload)


@dataclass
class ReportState:
	states: Sequence[LeafState]


class ReportContext(NamedTuple):
	start_ms: int
	state: ReportState


class Ctrf:
	"""A CTRF JSON test report."""

	def __init__(self, path: str | None = None, tail_bytes: int = 8192) -> None:
		self._path: Final = path
		self._tail_bytes: Final = tail_bytes
		self._model: ModuleType | None = None

	async def setup(self, task: TaskNode) -> ReportContext:
		self._model = require_model()
		return ReportContext(
			start_ms=now_ms(),
			state=ReportState(tuple(Waiting(info.task) for info in flatten_leaves(task))),
		)

	async def on_event(
		self, event: TaskEvent, states: Sequence[LeafState], ctx: ReportContext
	) -> ReportContext:
		ctx.state.states = states
		return ctx

	async def teardown(self, ctxs: tuple[ReportContext, ...]) -> None:
		model = self._model
		if model is None:  # pragma: no cover
			return
		ctx = ctxs[0]
		data: bytes = model.encode_run(
			ctx.state.states, ctx.start_ms, now_ms(), self._tail_bytes, version("camas")
		)
		await asyncio.to_thread(emit_report, self._path, data)
