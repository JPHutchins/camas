# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Effect: streams plain-text per-task status lines as tasks start, stop, and
emit output. Counterpart to :class:`camas.effect.termtree.Termtree` (live,
cursor-redrawing) and :class:`camas.effect.summary.Summary` (one post-run
report) — suited for CI logs and any context where cursor control either
doesn't render or shouldn't.

See :class:`StatusOptions` for the mode and template fields, and the doctests
on :func:`block_for` / :func:`fmt_started` / :func:`fmt_completed` /
:func:`fmt_output` for the exact behavior of each.
"""

from __future__ import annotations

import secrets
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Final, Literal, NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.completion import Completion, Finished, Skipped
from ..core.leaf_state import LeafState
from ..core.render import GREEN, GREY, RED, RESET, VIOLET, strip_ansi
from ..core.task import Task, TaskNode, task_label
from ..core.task_event import (
	CompletedEvent,
	OutputEvent,
	StartedEvent,
	TaskEvent,
)

OutputMode: TypeAlias = Literal["quiet", "all", "errors", "stream", "github"]


class StatusOptions(NamedTuple):
	"""Configuration for the Status Effect.

	>>> StatusOptions().output_mode
	'errors'
	>>> StatusOptions(output_mode="github").output_mode
	'github'
	>>> StatusOptions(started_fmt="").started_fmt
	''
	"""

	output_mode: OutputMode = "errors"
	started_fmt: str = (
		f"[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}] {VIOLET}▶ [{{name}}] started{RESET}"
	)
	finished_fmt: str = (
		f"[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}] "
		f"{GREEN}✓ [{{name}}] success{RESET} ({{elapsed:.3f}}s)"
	)
	failed_fmt: str = (
		f"[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}] "
		f"{RED}✗ [{{name}}] error{RESET} exit={{rc}} ({{elapsed:.3f}}s)"
	)
	skipped_fmt: str = (
		f"[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}] "
		f"{GREY}⏭ [{{name}}] skipped{RESET} (prior rc={{rc}})"
	)
	output_fmt: str = (
		f"[{{timestamp:%Y-%m-%d %H:%M:%S}}.{{ms:03d}}] {GREY}·{RESET} [{{name}}] {{line}}"
	)


class Idle(NamedTuple):
	"""Per-leaf ctx: ``StartedEvent`` has not fired yet.

	>>> Idle()
	Idle()
	"""


class Active(NamedTuple):
	"""Per-leaf ctx: leaf is running.

	>>> Active(b"hello\\n").output
	b'hello\\n'
	"""

	output: bytes


class Done(NamedTuple):
	"""Per-leaf ctx: ``CompletedEvent`` processed; nothing remains to emit.

	>>> Done()
	Done()
	"""


LeafCtx: TypeAlias = Idle | Active | Done


def cmd_str(task: Task) -> str:
	"""Return ``task.cmd`` as a single string (joins tuple form with spaces).

	>>> cmd_str(Task("echo hi"))
	'echo hi'
	>>> cmd_str(Task(("python", "-c", "pass")))
	'python -c pass'
	"""
	return task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)


def fmt_started(opts: StatusOptions, task: Task, ts: datetime) -> str | None:
	"""Render the started-line, or ``None`` when ``started_fmt`` is empty.

	>>> t0 = datetime(2026, 5, 21, 14, 30, 0, 123000)
	>>> fmt_started(StatusOptions(), Task("echo hi", name="greet"), t0)
	'[2026-05-21 14:30:00.123] \\x1b[95m▶ [greet] started\\x1b[0m'
	>>> fmt_started(StatusOptions(started_fmt=""), Task("echo hi"), t0) is None
	True
	>>> fmt_started(
	...     StatusOptions(started_fmt="{cmd} @ {timestamp:%H:%M:%S}.{ms:03d}"),
	...     Task(("python", "-c", "pass")), t0,
	... )
	'python -c pass @ 14:30:00.123'
	"""
	if not opts.started_fmt:
		return None
	return opts.started_fmt.format(
		name=task_label(task), cmd=cmd_str(task), timestamp=ts, ms=ts.microsecond // 1000
	)


def fmt_completed(opts: StatusOptions, task: Task, c: Completion, ts: datetime) -> str | None:
	"""Render the completion line, or ``None`` when the matching template is empty.

	>>> t0 = datetime(2026, 5, 21, 14, 30, 0, 123000)
	>>> fmt_completed(StatusOptions(), Task("a", name="lint"), Finished(0, 1.5, ()), t0)
	'[2026-05-21 14:30:00.123] \\x1b[32m✓ [lint] success\\x1b[0m (1.500s)'
	>>> fmt_completed(StatusOptions(), Task("a", name="lint"), Finished(2, 0.5, ()), t0)
	'[2026-05-21 14:30:00.123] \\x1b[31m✗ [lint] error\\x1b[0m exit=2 (0.500s)'
	>>> fmt_completed(StatusOptions(), Task("a", name="follow"), Skipped(2), t0)
	'[2026-05-21 14:30:00.123] \\x1b[90m⏭ [follow] skipped\\x1b[0m (prior rc=2)'
	>>> fmt_completed(
	...     StatusOptions(finished_fmt=""), Task("a"), Finished(0, 1.0, ()), t0,
	... ) is None
	True
	>>> fmt_completed(
	...     StatusOptions(failed_fmt=""), Task("a"), Finished(2, 1.0, ()), t0,
	... ) is None
	True
	"""
	name = task_label(task)
	cmd = cmd_str(task)
	ms = ts.microsecond // 1000
	match c:
		case Finished(returncode=rc, elapsed=e):
			tmpl = opts.finished_fmt if rc == 0 else opts.failed_fmt
			return (
				tmpl.format(name=name, cmd=cmd, rc=rc, elapsed=e, timestamp=ts, ms=ms)
				if tmpl
				else None
			)
		case Skipped(returncode=rc):
			tmpl = opts.skipped_fmt
			return tmpl.format(name=name, cmd=cmd, rc=rc, timestamp=ts, ms=ms) if tmpl else None
		case _:
			assert_never(c)


def fmt_output(opts: StatusOptions, task: Task, line: bytes, ts: datetime) -> str | None:
	"""Render a stream-mode output line (ANSI-stripped, trailing newline removed).

	>>> t0 = datetime(2026, 5, 21, 14, 30, 0, 123000)
	>>> fmt_output(StatusOptions(), Task("a", name="lint"), b"hello\\n", t0)
	'[2026-05-21 14:30:00.123] \\x1b[90m·\\x1b[0m [lint] hello'
	>>> fmt_output(StatusOptions(), Task("a", name="lint"), b"\\x1b[31mred\\x1b[0m\\n", t0)
	'[2026-05-21 14:30:00.123] \\x1b[90m·\\x1b[0m [lint] red'
	>>> fmt_output(StatusOptions(output_fmt=""), Task("a"), b"x\\n", t0) is None
	True
	"""
	if not opts.output_fmt:
		return None
	return opts.output_fmt.format(
		name=task_label(task),
		cmd=cmd_str(task),
		timestamp=ts,
		ms=ts.microsecond // 1000,
		line=strip_ansi(line.decode("utf-8", errors="replace")).rstrip("\n"),
	)


def _github_safe_title(label: str) -> str:
	"""Strip newlines/CRs so a ``::group::`` title stays on one line.

	>>> _github_safe_title("normal")
	'normal'
	>>> _github_safe_title("a\\nb\\rc")
	'a b c'
	"""
	return label.replace("\r", " ").replace("\n", " ")


def _github_stop_token(body: str) -> str:
	"""Pick a stop-commands token that does not appear as ``::{token}::`` in ``body``.

	>>> tok = _github_stop_token("plain body\\n")
	>>> len(tok) >= 8 and tok.isalnum()
	True
	"""
	for _ in range(32):
		token = secrets.token_hex(8)
		if f"::{token}::" not in body:
			return token
	raise RuntimeError("could not pick a stop-commands token")  # pragma: no cover


def _format_github_group(title: str, body: str, token: str) -> str:
	"""Wrap ``body`` in a GitHub ``::group::`` with ``::stop-commands::`` neutralization.

	The stop-commands bracket disables workflow-command parsing for the body
	so output lines like ``::endgroup::`` or ``::add-mask::`` are emitted
	verbatim instead of being interpreted by Actions.

	>>> _format_github_group("x", "ok\\n", "T")
	'::group::x\\n::stop-commands::T\\nok\\n::T::\\n::endgroup::\\n'
	>>> _format_github_group("x", "::endgroup::\\n", "T")
	'::group::x\\n::stop-commands::T\\n::endgroup::\\n::T::\\n::endgroup::\\n'
	"""
	return f"::group::{title}\n::stop-commands::{token}\n{body}::{token}::\n::endgroup::\n"


def block_for(mode: OutputMode, task: Task, c: Completion, output: bytes) -> str:
	"""Return the completion-block text for ``mode`` (empty when nothing to dump).

	Output bytes are passed through verbatim — ANSI escapes are preserved so
	downstream viewers (terminals, Actions logs) render the original colors.
	For ``github`` mode the body is wrapped in ``::stop-commands::`` so task
	output that happens to start with ``::`` (e.g. another tool's
	``::endgroup::`` or ``::warning::``) cannot prematurely close the group
	or inject workflow commands.

	>>> block_for("all", Task("a", name="x"), Finished(0, 1.0, ()), b"ok\\n")
	'ok\\n'
	>>> block_for("all", Task("a"), Finished(0, 1.0, ()), b"\\x1b[32mgreen\\x1b[0m\\n")
	'\\x1b[32mgreen\\x1b[0m\\n'
	>>> block_for("errors", Task("a", name="x"), Finished(0, 1.0, ()), b"ok\\n")
	''
	>>> block_for("errors", Task("a", name="x"), Finished(2, 1.0, ()), b"boom\\n")
	'boom\\n'
	>>> out = block_for("github", Task("a", name="x"), Finished(0, 1.0, ()), b"ok\\n")
	>>> out.startswith("::group::x\\n::stop-commands::") and out.endswith("::\\n::endgroup::\\n")
	True
	>>> "ok\\n" in out
	True
	>>> block_for("quiet", Task("a"), Finished(0, 1.0, ()), b"ok\\n")
	''
	>>> block_for("stream", Task("a"), Finished(0, 1.0, ()), b"ok\\n")
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
	"""Write ``text`` + ``\\n`` when non-empty; no-op for ``None``/empty string."""
	if not text:
		return
	emit(text + "\n")


def next_ctx_on_output(mode: OutputMode, ctx: Active, line: bytes) -> Active:
	"""Next Active ctx for an OutputEvent — block modes accumulate, others don't.

	>>> next_ctx_on_output("all", Active(b"abc"), b"def").output
	b'abcdef'
	>>> next_ctx_on_output("stream", Active(b""), b"x").output
	b''
	>>> next_ctx_on_output("quiet", Active(b""), b"x").output
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
	def __init__(self, options: StatusOptions = StatusOptions()) -> None:
		self.options: Final = options

	async def setup(self, task: TaskNode) -> LeafCtx:
		return Idle()

	async def on_event(
		self, event: TaskEvent, states: Sequence[LeafState], ctx: LeafCtx
	) -> LeafCtx:
		opts = self.options
		match event, ctx:
			case StartedEvent(task=t, timestamp=ts), Idle():
				emit_line(fmt_started(opts, t, ts))
				return Active(b"")
			case OutputEvent(task=t, line=line, timestamp=ts), Active() as active:
				if opts.output_mode == "stream":
					emit_line(fmt_output(opts, t, line, ts))
				return next_ctx_on_output(opts.output_mode, active, line)
			case CompletedEvent(task=t, completion=c, timestamp=ts), Active(output=buf):
				emit_line(fmt_completed(opts, t, c, ts))
				emit(block_for(opts.output_mode, t, c, buf))
				return Done()
			case CompletedEvent(task=t, completion=c, timestamp=ts), Idle():
				emit_line(fmt_completed(opts, t, c, ts))
				return Done()
			case _:
				return ctx

	async def teardown(self, ctxs: tuple[LeafCtx, ...]) -> None:
		return None
