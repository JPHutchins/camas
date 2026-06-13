# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Effect: streams plain-text per-task status lines as tasks start, stop, and
emit output. Counterpart to :class:`camas.effect.termtree.Termtree` (live,
cursor-redrawing) and :class:`camas.effect.summary.Summary` (one post-run
report) — suited for CI logs and any context where cursor control either
doesn't render or shouldn't.

See the :class:`Status` constructor for the mode and template arguments, and
the doctests on :func:`block_for` / :func:`fmt_started` / :func:`fmt_completed`
/ :func:`fmt_output` for the exact behavior of each.
"""

from __future__ import annotations

import secrets
import sys
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.color import CAMAS_VIOLET, GREEN, GREY, RED, RESET
from ..core.render import strip_ansi
from ..core.task import task_label
from ..v0.completion import Completion, Finished, Skipped
from ..v0.task_event import CompletedEvent, OutputEvent, StartedEvent, TaskEvent

if TYPE_CHECKING:
	from collections.abc import Sequence
	from datetime import datetime

	from ..v0.leaf_state import LeafState
	from ..v0.task import Task, TaskNode

OutputMode: TypeAlias = Literal["quiet", "all", "errors", "stream", "github"]


STARTED_FMT: Final = (
	f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} "
	f"{CAMAS_VIOLET}▶ [{{name}}] started{RESET}"
)
FINISHED_FMT: Final = (
	f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} "
	f"{GREEN}✓ [{{name}}] success{RESET} ({{elapsed:.3f}}s)"
)
FAILED_FMT: Final = (
	f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} "
	f"{RED}✗ [{{name}}] error{RESET} exit={{rc}} ({{elapsed:.3f}}s)"
)
SKIPPED_FMT: Final = (
	f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}]{RESET} "
	f"{GREY}⏭ [{{name}}] skipped{RESET} (prior rc={{rc}})"
)
OUTPUT_FMT: Final = (
	f"{GREY}[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}] · [{{name}}]{RESET} {{line}}"
)


class Idle(NamedTuple):
	"""Per-leaf ctx: ``StartedEvent`` has not fired yet.

	>>> Idle()
	Idle()
	"""


class Active(NamedTuple):
	r"""Per-leaf ctx: leaf is running.

	>>> Active(b"hello\n").output
	b'hello\n'
	"""

	output: bytes


class Done(NamedTuple):
	"""Per-leaf ctx: ``CompletedEvent`` processed; nothing remains to emit.

	>>> Done()
	Done()
	"""


LeafCtx: TypeAlias = Idle | Active | Done


def cmd_str(task: Task) -> str:
	"""Return ``task.cmd`` as a single string (joining tuple form with spaces)."""
	return task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)


def fmt_started(started_fmt: str, task: Task, ts: datetime) -> str | None:
	r"""Render the started-line, or ``None`` when ``started_fmt`` is empty.

	>>> from datetime import datetime
	>>> from camas import Task
	>>> t0 = datetime(2026, 5, 21, 14, 30, 0, 123000)
	>>> fmt_started(STARTED_FMT, Task("echo hi", name="greet"), t0)
	'\x1b[90m[2026-05-21 14:30:00.123]\x1b[0m \x1b[38;5;135m▶ [greet] started\x1b[0m'
	>>> fmt_started("", Task("echo hi"), t0) is None
	True
	>>> fmt_started("{cmd} @ {timestamp:%H:%M:%S}.{ms:03d}", Task(("python", "-c", "pass")), t0)
	'python -c pass @ 14:30:00.123'
	"""
	if not started_fmt:
		return None
	return started_fmt.format(
		name=task_label(task), cmd=cmd_str(task), timestamp=ts, ms=ts.microsecond // 1000
	)


def fmt_completed(
	finished_fmt: str,
	failed_fmt: str,
	skipped_fmt: str,
	task: Task,
	c: Completion,
	ts: datetime,
) -> str | None:
	r"""Render the completion line, or ``None`` when the matching template is empty.

	>>> from datetime import datetime
	>>> from camas import Task
	>>> t0 = datetime(2026, 5, 21, 14, 30, 0, 123000)
	>>> fmt_completed(FINISHED_FMT, FAILED_FMT, SKIPPED_FMT, Task("a", name="lint"), Finished(0, 1.5, ()), t0)
	'\x1b[90m[2026-05-21 14:30:00.123]\x1b[0m \x1b[32m✓ [lint] success\x1b[0m (1.500s)'
	>>> fmt_completed(FINISHED_FMT, FAILED_FMT, SKIPPED_FMT, Task("a", name="lint"), Finished(2, 0.5, ()), t0)
	'\x1b[90m[2026-05-21 14:30:00.123]\x1b[0m \x1b[31m✗ [lint] error\x1b[0m exit=2 (0.500s)'
	>>> fmt_completed(FINISHED_FMT, FAILED_FMT, SKIPPED_FMT, Task("a", name="follow"), Skipped(2), t0)
	'\x1b[90m[2026-05-21 14:30:00.123]\x1b[0m \x1b[90m⏭ [follow] skipped\x1b[0m (prior rc=2)'
	>>> fmt_completed("", FAILED_FMT, SKIPPED_FMT, Task("a"), Finished(0, 1.0, ()), t0) is None
	True
	"""
	name = task_label(task)
	cmd = cmd_str(task)
	ms = ts.microsecond // 1000
	match c:
		case Finished(returncode=rc, elapsed=e):
			tmpl = finished_fmt if rc == 0 else failed_fmt
			return (
				tmpl.format(name=name, cmd=cmd, rc=rc, elapsed=e, timestamp=ts, ms=ms)
				if tmpl
				else None
			)
		case Skipped(returncode=rc):
			return (
				skipped_fmt.format(name=name, cmd=cmd, rc=rc, timestamp=ts, ms=ms)
				if skipped_fmt
				else None
			)
		case _:
			assert_never(c)


def fmt_output(output_fmt: str, task: Task, line: bytes, ts: datetime) -> str | None:
	r"""Render a stream-mode output line (ANSI-stripped, trailing newline removed).

	>>> from datetime import datetime
	>>> from camas import Task
	>>> t0 = datetime(2026, 5, 21, 14, 30, 0, 123000)
	>>> fmt_output(OUTPUT_FMT, Task("a", name="lint"), b"hello\n", t0)
	'\x1b[90m[2026-05-21 14:30:00.123] · [lint]\x1b[0m hello'
	>>> fmt_output(OUTPUT_FMT, Task("a", name="lint"), b"\x1b[31mred\x1b[0m\n", t0)
	'\x1b[90m[2026-05-21 14:30:00.123] · [lint]\x1b[0m red'
	>>> fmt_output("", Task("a"), b"x\n", t0) is None
	True
	"""
	if not output_fmt:
		return None
	return output_fmt.format(
		name=task_label(task),
		cmd=cmd_str(task),
		timestamp=ts,
		ms=ts.microsecond // 1000,
		line=strip_ansi(line.decode("utf-8", errors="replace")).rstrip("\n"),
	)


def _github_safe_title(label: str) -> str:
	"""Strip CRs/newlines so a ``::group::`` title stays on one line."""
	return label.replace("\r", " ").replace("\n", " ")


def _github_stop_token(body: str) -> str:
	r"""Pick a stop-commands token that does not appear as ``::{token}::`` in ``body``.

	Raises:
		RuntimeError: if 32 random tokens all collide (practically unreachable).

	>>> tok = _github_stop_token("plain body\n")
	>>> len(tok) >= 8 and tok.isalnum()
	True
	"""
	for _ in range(32):
		token = secrets.token_hex(8)
		if f"::{token}::" not in body:  # pragma: no branch
			return token
	raise RuntimeError("could not pick a stop-commands token")  # pragma: no cover


def _format_github_group(title: str, body: str, token: str) -> str:
	"""Wrap ``body`` in a GitHub ``::group::`` with ``::stop-commands::`` neutralization.

	The stop-commands bracket disables workflow-command parsing for the body, so
	output lines like ``::endgroup::`` are emitted verbatim, not interpreted.
	"""
	return f"::group::{title}\n::stop-commands::{token}\n{body}::{token}::\n::endgroup::\n"


def block_for(mode: OutputMode, task: Task, c: Completion, output: bytes) -> str:
	r"""Return the completion-block text for ``mode`` (empty when nothing to dump).

	Output bytes pass through verbatim (ANSI preserved). For ``github`` mode the
	body is wrapped in ``::stop-commands::`` so task output starting with ``::``
	cannot prematurely close the group or inject workflow commands.

	>>> from camas import Task
	>>> block_for("all", Task("a", name="x"), Finished(0, 1.0, ()), b"ok\n")
	'ok\n'
	>>> block_for("all", Task("a"), Finished(0, 1.0, ()), b"\x1b[32mgreen\x1b[0m\n")
	'\x1b[32mgreen\x1b[0m\n'
	>>> block_for("errors", Task("a", name="x"), Finished(0, 1.0, ()), b"ok\n")
	''
	>>> block_for("errors", Task("a", name="x"), Finished(2, 1.0, ()), b"boom\n")
	'boom\n'
	>>> out = block_for("github", Task("a", name="x"), Finished(0, 1.0, ()), b"ok\n")
	>>> out.startswith("::group::x\n::stop-commands::") and out.endswith("::\n::endgroup::\n")
	True
	>>> "ok\n" in out
	True
	>>> block_for("quiet", Task("a"), Finished(0, 1.0, ()), b"ok\n")
	''
	>>> block_for("stream", Task("a"), Finished(0, 1.0, ()), b"ok\n")
	''
	>>> block_for("all", Task("a"), Finished(0, 1.0, ()), b"")
	''
	>>> block_for("github", Task("a", name="x"), Skipped(1), b"")
	''
	"""
	if not output:
		return ""
	body = output.decode("utf-8", errors="replace")
	if not body.endswith("\n"):
		body += "\n"
	failed = isinstance(c, Finished) and c.returncode != 0
	match mode:
		case "all":
			return body
		case "errors":
			return body if failed else ""
		case "github":
			return _format_github_group(
				_github_safe_title(task_label(task)), body, _github_stop_token(body)
			)
		case "quiet" | "stream":
			return ""
		case _:
			assert_never(mode)


def emit(text: str) -> None:
	"""Write ``text`` to ``stdout.buffer`` + flush. Caller controls newlines."""
	if not text:
		return
	sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
	sys.stdout.flush()


def emit_line(text: str | None) -> None:
	r"""Write ``text`` + ``\n`` when non-empty; no-op for ``None``/empty string."""
	if not text:
		return
	emit(text + "\n")


def next_ctx_on_output(mode: OutputMode, ctx: Active, line: bytes) -> Active:
	"""Next Active ctx for an OutputEvent — block modes accumulate, others don't.

	>>> next_ctx_on_output("all", Active(b"abc"), b"def").output
	b'abcdef'
	>>> next_ctx_on_output("stream", Active(b""), b"x").output
	b''
	"""
	match mode:
		case "all" | "errors" | "github":
			return Active(ctx.output + line)
		case "quiet" | "stream":
			return ctx
		case _:
			assert_never(mode)


class Status:
	"""Line-oriented status Effect; behavior is selected by the ``output_mode`` argument.

	See the module docstring for how it relates to ``Termtree`` and ``Summary``.
	"""

	def __init__(
		self,
		output_mode: OutputMode = "errors",
		started_fmt: str = STARTED_FMT,
		finished_fmt: str = FINISHED_FMT,
		failed_fmt: str = FAILED_FMT,
		skipped_fmt: str = SKIPPED_FMT,
		output_fmt: str = OUTPUT_FMT,
	) -> None:
		self._output_mode: Final = output_mode
		self._started_fmt: Final = started_fmt
		self._finished_fmt: Final = finished_fmt
		self._failed_fmt: Final = failed_fmt
		self._skipped_fmt: Final = skipped_fmt
		self._output_fmt: Final = output_fmt

	async def setup(self, task: TaskNode) -> LeafCtx:
		return Idle()

	async def on_event(
		self, event: TaskEvent, states: Sequence[LeafState], ctx: LeafCtx
	) -> LeafCtx:
		match event, ctx:
			case StartedEvent(task=t, timestamp=ts), Idle():
				emit_line(fmt_started(self._started_fmt, t, ts))
				return Active(b"")
			case OutputEvent(task=t, line=line, timestamp=ts), Active() as active:
				if self._output_mode == "stream":
					emit_line(fmt_output(self._output_fmt, t, line, ts))
				return next_ctx_on_output(self._output_mode, active, line)
			case CompletedEvent(task=t, completion=c, timestamp=ts), Active(output=buf):
				emit_line(
					fmt_completed(self._finished_fmt, self._failed_fmt, self._skipped_fmt, t, c, ts)
				)
				emit(block_for(self._output_mode, t, c, buf))
				return Done()
			case CompletedEvent(task=t, completion=c, timestamp=ts), Idle():
				emit_line(
					fmt_completed(self._finished_fmt, self._failed_fmt, self._skipped_fmt, t, c, ts)
				)
				return Done()
			case _:
				return ctx

	async def teardown(self, ctxs: tuple[LeafCtx, ...]) -> None:
		return None
