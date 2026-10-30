# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from mcp import types
from pydantic import ValidationError

from ..core.attached import scoped_body
from ..core.execution import run
from ..core.task import did_you_mean
from ..main.argv import apply_passthrough
from ..main.state import LoadErr, LoadOk
from ..v0.task import Pipe, Task
from . import wire
from .result import to_run_response

if TYPE_CHECKING:
	from collections.abc import Generator, Mapping

	from ..v0.command import Command
	from ..v0.task import TaskNode
	from .serve import Session


COMMAND_ANNOTATIONS: Final = types.ToolAnnotations(
	readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True
)

COMMAND_DESCRIPTION: Final = (
	"Run one of THIS project's Commands — side-effectful actions such as flashing firmware, "
	"deploying, or publishing — and return camas_run's per-leaf report. camas_list names them "
	"under commands. A Command must terminate; input=... answers its prompts."
)


def command_refusal(name: str) -> str:
	"""
	>>> command_refusal("flash")
	"'flash' is a Command: run it with camas_command"
	"""
	return f"{name!r} is a Command: run it with camas_command"


def resolve_command(commands: Mapping[str, Command], req: wire.CommandRequest) -> TaskNode:
	"""
	>>> from camas.v0.command import Command
	>>> resolve_command({"run": Command("cargo run")}, wire.CommandRequest(name="run", args=["-q"]))
	Task(cmd='cargo run -q', name=None, env={}, cwd=None)

	Raises:
		ValueError: on an unknown name, or ``input`` for a ``Sequential`` body.
	"""
	if req.name not in commands:
		known: Final = ", ".join(sorted(commands)) or "none"
		raise ValueError(
			f"no command named {req.name!r}{did_you_mean(req.name, commands)} (known: {known})"
		)
	command: Final = commands[req.name]
	if req.input is not None and not isinstance(command.body, (Task, Pipe)):
		raise ValueError("input applies to a single-leaf or Pipe command")
	body: Final = scoped_body(command)
	return apply_passthrough(body, tuple(req.args)) if req.args else body


@contextmanager
def stdin_of(text: str | None) -> Generator[int | None]:
	if text is None:
		yield None
		return
	with tempfile.TemporaryFile() as file:
		file.write(text.encode())
		file.seek(0)
		yield file.fileno()


async def command_call(session: Session, arguments: dict[str, Any]) -> types.CallToolResult:
	from .serve import error_result, leaf_color_of, load_error_hint

	match session.project:
		case LoadErr(source=source, exception=exception):
			return error_result(load_error_hint(session, source, exception))
		case LoadOk(commands=commands, config=config):
			try:
				req = wire.CommandRequest.model_validate(arguments)
				node = resolve_command(commands, req)
			except ValidationError as e:
				return error_result(f"invalid camas_command arguments:\n{e}")
			except ValueError as e:
				return error_result(str(e))
			return await run_command(session, req, node, leaf_color=leaf_color_of(config))
		case _:
			assert_never(session.project)


async def run_command(
	session: Session, req: wire.CommandRequest, node: TaskNode, *, leaf_color: bool
) -> types.CallToolResult:
	from .serve import (
		attach_logs,
		base_for,
		create_run_log_dir,
		failing_log_links,
		run_text,
		success,
		with_warning,
		write_logs,
	)

	with stdin_of(req.input) as stdin:
		result: Final = await run(
			node, interactive=False, base=base_for(session), leaf_color=leaf_color, stdin=stdin
		)
	logs: Final = write_logs(
		create_run_log_dir(session.camas_dir, req.name, session.reserve_run()), result
	)
	resp: Final = attach_logs(to_run_response(node, result, verbosity=req.verbosity), logs)
	return success(
		with_warning(session, run_text(req.name, resp, logs, tool="camas_command")),
		resp,
		session.compat,
		links=failing_log_links(resp, logs),
	)
