# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

import pytest

from camas import Parallel, Task
from camas.effect.ctrf import Ctrf
from camas.v0.completion import Finished, Skipped, Stopped
from camas.v0.leaf_state import LeafState, Waiting
from camas.v0.task_event import CompletedEvent, StartedEvent, TaskEvent

if TYPE_CHECKING:
	from pathlib import Path

	from camas.v0.effect import Effect

pytest.importorskip("msgspec")

TS = datetime(2026, 5, 21, 14, 30, 0)

T = TypeVar("T")


def make_task(name: str, cmd: Any = ("python", "-c", "pass"), **kwargs: Any) -> Task:
	return Task(cmd, name=name, **kwargs)


async def drive(effect: Effect[T], task: Task | Parallel, events: list[TaskEvent]) -> None:
	"""Feed an effect through a full setup/on_event*/teardown lifecycle."""
	from camas.core.leaf_state import next_state
	from camas.core.traversal import flatten_leaves

	leaves = flatten_leaves(task)
	states: list[LeafState] = [Waiting(info.task) for info in leaves]
	initial = await effect.setup(task)
	ctxs: list[T] = [initial for _ in leaves]
	try:
		for event in events:
			states[event.leaf_index] = next_state(states[event.leaf_index], event)
			ctxs[event.leaf_index] = await effect.on_event(event, states, ctxs[event.leaf_index])
	finally:
		await effect.teardown(tuple(ctxs) or (initial,))


def test_ctrf_report_shape_and_schema_ref(capsys: pytest.CaptureFixture[str]) -> None:
	lint = make_task("lint", cmd="ruff check .", paths=".")
	boom = make_task("boom")
	task = Parallel(lint, boom)
	events: list[TaskEvent] = [
		StartedEvent(lint, 0, TS),
		StartedEvent(boom, 1, TS),
		CompletedEvent(lint, 0, Finished(0, 0.1, (b"clean\n",)), TS),
		CompletedEvent(boom, 1, Finished(1, 0.2, (b"error details\n",)), TS),
	]
	asyncio.run(drive(Ctrf(), task, events))
	report = json.loads(capsys.readouterr().out)

	assert report["reportFormat"] == "CTRF"
	assert report["specVersion"] == "1.0.0"
	assert report["generatedBy"].startswith("camas ")
	assert report["$schema"] == "https://ctrf.io/schema/ctrf.schema.json"

	results = report["results"]
	assert results["tool"]["name"] == "camas"
	assert results["tool"]["version"]
	summary = results["summary"]
	assert summary["tests"] == 2
	assert summary["passed"] == 1
	assert summary["failed"] == 1
	assert isinstance(summary["start"], int)
	assert isinstance(summary["stop"], int)

	by_name = {t["name"]: t for t in results["tests"]}
	assert by_name["lint"]["status"] == "passed"
	assert by_name["lint"]["duration"] == 100
	assert by_name["lint"]["rawStatus"] == "0"
	assert by_name["lint"]["stdout"] == ["clean"]
	assert by_name["lint"]["extra"] == {
		"command": "ruff check .",
		"exitCode": 0,
		"mutates": False,
		"paths": ".",
	}
	assert by_name["boom"]["status"] == "failed"
	assert by_name["boom"]["rawStatus"] == "1"
	assert by_name["boom"]["stdout"] == ["error details"]
	assert "paths" not in by_name["boom"]["extra"]


def test_ctrf_writes_to_file(tmp_path: Path) -> None:
	target = tmp_path / "ctrf-report.json"
	task = make_task("solo")
	events: list[TaskEvent] = [
		StartedEvent(task, 0, TS),
		CompletedEvent(task, 0, Finished(0, 0.05, (b"done\n",)), TS),
	]
	asyncio.run(drive(Ctrf(path=str(target)), task, events))
	report = json.loads(target.read_text(encoding="utf-8"))
	assert report["reportFormat"] == "CTRF"
	assert report["results"]["summary"]["passed"] == 1


def test_ctrf_tail_bytes_zero_omits_stdout(capsys: pytest.CaptureFixture[str]) -> None:
	task = make_task("noisy")
	events: list[TaskEvent] = [
		StartedEvent(task, 0, TS),
		CompletedEvent(task, 0, Finished(0, 0.1, (b"lots of output\n",)), TS),
	]
	asyncio.run(drive(Ctrf(tail_bytes=0), task, events))
	report = json.loads(capsys.readouterr().out)
	assert "stdout" not in report["results"]["tests"][0]


def test_ctrf_skipped_and_stopped_statuses(capsys: pytest.CaptureFixture[str]) -> None:
	a = make_task("fail")
	b = make_task("skip")
	c = make_task("stop")
	task = Parallel(a, b, c)
	events: list[TaskEvent] = [
		StartedEvent(a, 0, TS),
		CompletedEvent(a, 0, Finished(2, 0.1, (b"nope\n",)), TS),
		CompletedEvent(b, 1, Skipped(2), TS),
		CompletedEvent(c, 2, Stopped(130, 0.3, (b"partial\n",)), TS),
	]
	asyncio.run(drive(Ctrf(), task, events))
	report = json.loads(capsys.readouterr().out)
	summary = report["results"]["summary"]
	assert (summary["failed"], summary["skipped"], summary["other"]) == (1, 1, 1)
	by_name = {t["name"]: t for t in report["results"]["tests"]}
	assert by_name["skip"]["status"] == "skipped"
	assert by_name["skip"]["duration"] == 0
	assert "stdout" not in by_name["skip"]
	assert by_name["stop"]["status"] == "other"
	assert by_name["stop"]["stdout"] == ["partial"]


def test_ctrf_pending_leaf_never_completed(capsys: pytest.CaptureFixture[str]) -> None:
	done = make_task("done")
	never = make_task("never")
	task = Parallel(done, never)
	events: list[TaskEvent] = [
		StartedEvent(done, 0, TS),
		CompletedEvent(done, 0, Finished(0, 0.1, ()), TS),
	]
	asyncio.run(drive(Ctrf(), task, events))
	report = json.loads(capsys.readouterr().out)
	assert report["results"]["summary"]["pending"] == 1
	by_name = {t["name"]: t for t in report["results"]["tests"]}
	assert by_name["never"]["status"] == "pending"
	assert by_name["never"]["duration"] == 0


def test_ctrf_zero_leaf_run_emits_empty_report(capsys: pytest.CaptureFixture[str]) -> None:
	asyncio.run(drive(Ctrf(), Parallel(), []))
	report = json.loads(capsys.readouterr().out)
	assert report["$schema"] == "https://ctrf.io/schema/ctrf.schema.json"
	assert report["reportFormat"] == "CTRF"
	assert report["results"]["summary"]["tests"] == 0
	assert report["results"]["tests"] == []


def test_ctrf_requires_msgspec_extra(monkeypatch: pytest.MonkeyPatch) -> None:
	import camas.effect

	monkeypatch.delitem(sys.modules, "camas.effect._ctrf_model", raising=False)
	monkeypatch.delattr(camas.effect, "_ctrf_model", raising=False)
	monkeypatch.setitem(sys.modules, "msgspec", None)
	with pytest.raises(RuntimeError, match=r"camas\[ctrf\]"):
		asyncio.run(Ctrf().setup(make_task("x")))
