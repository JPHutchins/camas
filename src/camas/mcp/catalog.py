# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Build the MCP ``ListResponse`` catalog from a project's discovered tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.matrix import matrix_axes
from ..main.expression import to_expression
from . import wire

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.config import Config
	from ..v0.task import TaskNode


def to_list_response(tasks: Mapping[str, TaskNode], config: Config | None) -> wire.ListResponse:
	"""Assemble the ``camas_list`` catalog — one ``TaskInfo`` per task, sorted by
	name, with the default and CI-default names taken from ``config``.
	"""
	default = _task_name(config.default_task) if config is not None else None
	github_default = _task_name(config.github_task) if config is not None else None
	return wire.ListResponse(
		tasks=[
			_task_info(
				name,
				node,
				is_default=name == default,
				is_github_default=name == github_default,
			)
			for name, node in sorted(tasks.items())
		],
		default=default,
		github_default=github_default,
	)


def _task_name(node: TaskNode | None) -> str | None:
	"""The task's discovered name, or ``None`` when the field is unset."""
	return node.name if node is not None else None


def _task_info(
	name: str, node: TaskNode, *, is_default: bool, is_github_default: bool
) -> wire.TaskInfo:
	"""One ``TaskInfo``: help, a fully-typed command expression, and matrix axes as lists."""
	return wire.TaskInfo(
		name=name,
		help=node.help,
		command_preview=to_expression(node),
		matrix_axes={axis: list(values) for axis, values in matrix_axes(node).items()},
		is_default=is_default,
		is_github_default=is_github_default,
	)
