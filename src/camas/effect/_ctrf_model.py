# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The CTRF report model (msgspec Structs) and the camas-leaf → CTRF-test mapping.

Imported lazily by :mod:`camas.effect.ctrf` so the effect stays discoverable
without the ``camas[ctrf]`` extra installed.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final, Literal

import msgspec

from ..core.render import strip_ansi
from ..core.task import task_label
from ..v0.completion import Finished, Skipped, Stopped
from ..v0.leaf_state import Completed, Interrupting, Running, Waiting

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Sequence

	from ..v0.completion import Completion
	from ..v0.leaf_state import LeafState
	from ..v0.task import Task

SPEC_VERSION: Final = "1.0.0"
SCHEMA_URL: Final = "https://ctrf.io/schema/ctrf.schema.json"

CtrfStatus = Literal["passed", "failed", "skipped", "pending", "other"]


class Tool(msgspec.Struct, omit_defaults=True):
	"""The CTRF ``tool`` that produced the report."""

	name: str
	version: str | None = None


class Summary(msgspec.Struct):
	"""The CTRF run ``summary``."""

	tests: int
	passed: int
	failed: int
	skipped: int
	pending: int
	other: int
	start: int
	stop: int


class LeafExtra(msgspec.Struct, omit_defaults=True, rename="camel"):
	"""camas-specific fields carried on a test's ``extra``."""

	command: str
	mutates: bool
	exit_code: int | None = None
	paths: str | None = None


class Test(msgspec.Struct, omit_defaults=True, rename="camel"):
	"""A CTRF test — one camas leaf."""

	name: str
	status: CtrfStatus
	duration: int
	raw_status: str | None = None
	stdout: list[str] | None = None
	extra: LeafExtra | None = None


class ReportExtra(msgspec.Struct, rename={"schema_url": "$schema"}):
	"""The top-level ``extra``, carrying the schema reference."""

	schema_url: str


class Results(msgspec.Struct):
	"""The CTRF ``results`` object."""

	tool: Tool
	summary: Summary
	tests: list[Test]


class Report(msgspec.Struct, omit_defaults=True, rename="camel"):
	"""The top-level CTRF report document."""

	report_format: Literal["CTRF"]
	spec_version: str
	results: Results
	generated_by: str | None = None
	extra: ReportExtra | None = None


def cmd_str(task: Task) -> str:
	"""``task.cmd`` as one string, joining the tuple form with spaces.

	>>> from camas import Task
	>>> cmd_str(Task("ruff check ."))
	'ruff check .'
	>>> cmd_str(Task(("python", "-c", "pass")))
	'python -c pass'
	"""
	return task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)


def status_of(completion: Completion) -> CtrfStatus:
	"""Map a camas Completion to a CTRF status.

	>>> from camas.v0.completion import Finished, Skipped, Stopped
	>>> status_of(Finished(0, 0.1, ())), status_of(Finished(1, 0.1, ()))
	('passed', 'failed')
	>>> status_of(Skipped(2)), status_of(Stopped(130, 0.1, ()))
	('skipped', 'other')
	"""
	match completion:
		case Finished(returncode=rc):
			return "passed" if rc == 0 else "failed"
		case Skipped():
			return "skipped"
		case Stopped():
			return "other"
		case _:
			assert_never(completion)


def duration_ms(completion: Completion) -> int:
	"""A completion's elapsed time in whole milliseconds; 0 for a skip.

	>>> from camas.v0.completion import Finished, Skipped, Stopped
	>>> duration_ms(Finished(0, 1.5, ())), duration_ms(Stopped(130, 0.5, ())), duration_ms(Skipped(1))
	(1500, 500, 0)
	"""
	match completion:
		case Finished(elapsed=e) | Stopped(elapsed=e):
			return round(e * 1000)
		case Skipped():
			return 0
		case _:
			assert_never(completion)


def output_lines(completion: Completion, tail: int) -> list[str]:
	r"""The last ``tail`` bytes of a completion's output as ANSI-stripped lines.

	>>> from camas.v0.completion import Finished, Skipped, Stopped
	>>> output_lines(Finished(0, 0.1, (b"a\n", b"b\n")), 8192)
	['a', 'b']
	>>> output_lines(Stopped(130, 0.1, (b"x\n",)), 8192)
	['x']
	>>> output_lines(Finished(0, 0.1, (b"abcdef",)), 3)
	['def']
	>>> output_lines(Finished(0, 0.1, ()), 8192)
	[]
	>>> output_lines(Finished(0, 0.1, (b"hi\n",)), 0)
	[]
	>>> output_lines(Skipped(1), 8192)
	[]
	"""
	match completion:
		case Finished(output=o) | Stopped(output=o):
			buf = b"".join(o)
		case Skipped():
			return []
		case _:
			assert_never(completion)
	if tail <= 0 or not buf:
		return []
	clipped = buf[-tail:] if len(buf) > tail else buf
	return strip_ansi(clipped.decode("utf-8", errors="replace")).splitlines()


def status_of_state(state: LeafState) -> CtrfStatus:
	"""The CTRF status for a leaf's final state."""
	match state:
		case Completed(completion=c):
			return status_of(c)
		case Waiting() | Running() | Interrupting():
			return "pending"
		case _:
			assert_never(state)


def test_of(state: LeafState, tail: int) -> Test:
	"""A CTRF test for one leaf's final state."""
	match state:
		case Completed(task=task, completion=c):
			return Test(
				name=task_label(task),
				status=status_of(c),
				duration=duration_ms(c),
				raw_status=str(c.returncode),
				stdout=output_lines(c, tail) or None,
				extra=LeafExtra(
					command=cmd_str(task),
					mutates=task.mutates,
					exit_code=c.returncode,
					paths=None if task.paths is None else str(task.paths),
				),
			)
		case Waiting(task=task) | Running(task=task) | Interrupting(task=task):
			return Test(
				name=task_label(task),
				status="pending",
				duration=0,
				extra=LeafExtra(command=cmd_str(task), mutates=task.mutates),
			)
		case _:
			assert_never(state)


def summarize(states: Sequence[LeafState], start_ms: int, stop_ms: int) -> Summary:
	"""The CTRF ``summary`` for the run."""
	statuses = [status_of_state(s) for s in states]
	return Summary(
		tests=len(statuses),
		passed=statuses.count("passed"),
		failed=statuses.count("failed"),
		skipped=statuses.count("skipped"),
		pending=statuses.count("pending"),
		other=statuses.count("other"),
		start=start_ms,
		stop=stop_ms,
	)


def build(
	states: Sequence[LeafState], start_ms: int, stop_ms: int, tail: int, camas_version: str
) -> Report:
	"""The CTRF report for a run."""
	return Report(
		report_format="CTRF",
		spec_version=SPEC_VERSION,
		generated_by=f"camas {camas_version}",
		extra=ReportExtra(schema_url=SCHEMA_URL),
		results=Results(
			tool=Tool(name="camas", version=camas_version),
			summary=summarize(states, start_ms, stop_ms),
			tests=[test_of(s, tail) for s in states],
		),
	)


def encode_run(
	states: Sequence[LeafState], start_ms: int, stop_ms: int, tail: int, camas_version: str
) -> bytes:
	"""Build and encode the CTRF report as indented JSON bytes."""
	return msgspec.json.format(
		msgspec.json.encode(build(states, start_ms, stop_ms, tail, camas_version)), indent=2
	)
