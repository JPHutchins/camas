# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The SA-delegation gate: scope the check node to the changed paths, run the checks, and
classify the residual ``green`` vs ``needs_reasoning``. The gate
never mutates — the deterministic fixers run separately on ``PostToolBatch`` (``camas mcp fix``).
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypeAlias

from ..v0.task import Group, Task
from .budget import plan_under
from .execution import run
from .matrix import expand_matrix
from .scope import scope_to_changed

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import TaskNode
	from .budget import BudgetPlan
	from .completion import RunResult
	from .timings import TaskLabel, TaskTiming


ResidualClass: TypeAlias = Literal["green", "needs_reasoning"]
Decision: TypeAlias = Literal["continue", "block"]

REPORT_TOKEN: Final = "{report}"
"""The literal in :attr:`camas.v0.task.AgentFormat.args` that switches a leaf to path mode."""

REPORT_DIR_PREFIX: Final = "camas-report-"
"""Prefix on this machine's path-mode report directories, one per gate run — swept by
:func:`prune_stale_report_dirs` once older than its max age.
"""

STALE_TEMP_MAX_AGE_S: Final = 3600.0
"""Age past which a prior run's leftovers in the system temp dir are swept — shared by
:func:`prune_stale_report_dirs` and the MCP nudge-marker sweep so both age out together."""


class GateOutcome(NamedTuple):
	"""A gate run's verdict and the check run that produced it."""

	residual_class: ResidualClass
	node: TaskNode | None
	"""The check node that ran and its result, paired — both set when a run happened (whether
	``green`` or ``needs_reasoning``), both ``None`` when nothing ran. Diagnostics derive from
	them only on ``needs_reasoning``.
	"""
	result: RunResult | None
	budget: BudgetPlan | None
	report_paths: tuple[Path | None, ...] = ()
	"""Each leaf's path-mode report file, DFS order aligned with ``node``'s leaves — set for a
	leaf whose ``agent_format.args`` used :data:`REPORT_TOKEN`, ``None`` for every other leaf.
	"""


def decision_of(residual_class: ResidualClass) -> Decision:
	"""The routing for a residual class: a surviving residual blocks, else continue.

	>>> decision_of("green"), decision_of("needs_reasoning")
	('continue', 'block')
	"""
	match residual_class:
		case "green":
			return "continue"
		case "needs_reasoning":
			return "block"
		case _:
			assert_never(residual_class)


class FormattedNode(NamedTuple):
	"""The result of :func:`with_agent_format`: the command-rewritten node, and each leaf's
	path-mode report file (``None`` for a leaf that isn't in path mode), DFS order aligned with
	the node's leaves.
	"""

	node: TaskNode
	report_paths: tuple[Path | None, ...]


def _allocate_report_path(report_dir: Path) -> Path:
	"""A fresh, empty file under ``report_dir`` for one path-mode leaf to write its report to."""
	fd, raw_path = tempfile.mkstemp(dir=report_dir, suffix=".report")
	os.close(fd)
	return Path(raw_path)


def _append_args(
	cmd: str | tuple[str, ...], args: str, report_path: Path | None
) -> str | tuple[str, ...]:
	"""Append ``args`` to ``cmd`` with :data:`REPORT_TOKEN` substituted — shell-quoted into a
	string command, per-token after splitting the template into a tuple command — so the
	substituted path never re-parses through POSIX ``shlex`` (a Windows path would lose its
	backslashes), mirroring ``{paths}`` injection (:func:`camas.core.scope._inject`).
	"""
	match cmd:
		case str():
			substituted = (
				args
				if report_path is None
				else args.replace(REPORT_TOKEN, shlex.quote(str(report_path)))
			)
			return f"{cmd} {substituted}"
		case tuple():
			tokens = shlex.split(args)
			return (
				*cmd,
				*(
					tokens
					if report_path is None
					else (tok.replace(REPORT_TOKEN, str(report_path)) for tok in tokens)
				),
			)
		case _:
			assert_never(cmd)


def with_agent_format(node: TaskNode, report_dir: Path) -> FormattedNode:
	"""Append each leaf's ``agent_format.args`` to its command — the agent-only structured-output
	variant the gate runs; a human run never applies this, so ``cmd`` is otherwise untouched.

	A leaf whose ``args`` contains :data:`REPORT_TOKEN` is in path mode: the token is replaced
	with a fresh file allocated under ``report_dir``, and that leaf's slot in ``report_paths``
	carries the allocated path — the gate reads it for the payload once the leaf has run.

	>>> from camas.v0.task import AgentFormat
	>>> import tempfile
	>>> tmp = Path(tempfile.mkdtemp())
	>>> with_agent_format(Task("ruff check .", agent_format=AgentFormat("--output-format sarif", "sarif")), tmp).node.cmd
	'ruff check . --output-format sarif'
	>>> with_agent_format(Task("ruff check ."), tmp).node.cmd
	'ruff check .'
	>>> fmt = with_agent_format(Task("pytest", agent_format=AgentFormat("--junitxml {report}", "junit")), tmp)
	>>> "{report}" in fmt.node.cmd
	False
	>>> fmt.report_paths[0].parent == tmp
	True
	"""
	match node:
		case Task():
			if node.agent_format is None:
				return FormattedNode(node, (None,))
			args = node.agent_format.args
			report_path = _allocate_report_path(report_dir) if REPORT_TOKEN in args else None
			return FormattedNode(
				dataclasses.replace(node, cmd=_append_args(node.cmd, args, report_path)),
				(report_path,),
			)
		case Group() as group:
			formatted = tuple(with_agent_format(c, report_dir) for c in group.tasks)
			return FormattedNode(
				type(group)(
					*(f.node for f in formatted),
					name=group.name,
					matrix=group.matrix,
					env=group.env,
					cwd=group.cwd,
					help=group.help,
					paths=group.paths,
					when=group.when,
				),
				tuple(p for f in formatted for p in f.report_paths),
			)
		case _:
			assert_never(node)


def uses_path_mode(node: TaskNode) -> bool:
	"""Whether any leaf in ``node`` puts :data:`REPORT_TOKEN` in its ``agent_format.args`` —
	gates the report-directory allocation in :func:`run_gate` so a tree with no path-mode leaf
	never touches disk for one.

	>>> from camas.v0.task import AgentFormat
	>>> uses_path_mode(Task("pytest", agent_format=AgentFormat("--junitxml {report}", "junit")))
	True
	>>> uses_path_mode(Task("ruff check .", agent_format=AgentFormat("--out sarif", "sarif")))
	False
	"""
	match node:
		case Task():
			return node.agent_format is not None and REPORT_TOKEN in node.agent_format.args
		case Group() as group:
			return any(uses_path_mode(c) for c in group.tasks)
		case _:
			assert_never(node)


def _rmtree_if_stale(path: Path, cutoff: float) -> None:
	"""Remove ``path`` when it's a directory older than ``cutoff``; best-effort, silent on
	races with a concurrent gate run or the OS reclaiming the temp dir first.
	"""
	try:
		if path.is_dir() and path.stat().st_mtime < cutoff:
			shutil.rmtree(path, ignore_errors=True)
	except OSError:  # pragma: no cover
		pass


def prune_stale_report_dirs(max_age_s: float = STALE_TEMP_MAX_AGE_S) -> None:
	"""Best-effort sweep of this machine's path-mode report directories from prior gate runs
	older than ``max_age_s`` — bounds their growth in the system temp dir without disturbing the
	current run's, which an agent may still need to inspect after an over-limit payload.
	"""
	base = Path(tempfile.gettempdir())
	cutoff = time.time() - max_age_s
	for stale in base.glob(f"{REPORT_DIR_PREFIX}*"):
		_rmtree_if_stale(stale, cutoff)


async def run_gate(
	node: TaskNode,
	changed: tuple[str, ...],
	*,
	under: float | None = None,
	jobs: int | None = None,
	base: Path | None = None,
	timings: Mapping[TaskLabel, TaskTiming] | None = None,
) -> GateOutcome:
	"""Run the check ``node`` over the ``changed`` paths and classify the residual.

	The check node is expanded, time-boxed (``under``), scoped to ``changed``, and run; the gate
	never mutates. Untimed leaves are run (and thereby measured); only leaves measured to exceed
	``under`` are skipped. ``green`` means the checks passed — or the change touched nothing the
	checks cover, or every leaf was measured too slow for ``under``; ``needs_reasoning`` means a
	check still fails. Budgeting precedes scoping so each leaf's estimate reuses its unscoped
	record (a scoped run is no slower than the whole).

	A path-mode leaf's report file is allocated under a fresh, machine-temp ``camas-report-*``
	directory for this call — left on disk (not cleaned up here) so an over-limit payload's
	pointer stays valid after the call returns; :func:`prune_stale_report_dirs` bounds their
	accumulation.
	"""
	expanded = expand_matrix(node)
	plan = plan_under(expanded, under, timings or {}) if under is not None else None
	budgeted = plan.node if plan is not None else expanded
	if budgeted is None:
		return GateOutcome("green", None, None, plan)
	scoped = scope_to_changed(budgeted, changed)
	if scoped is None:
		return GateOutcome("green", None, None, plan)
	if uses_path_mode(scoped):
		prune_stale_report_dirs()
		report_dir = Path(tempfile.mkdtemp(prefix=REPORT_DIR_PREFIX))
	else:
		report_dir = Path(tempfile.gettempdir())
	formatted = with_agent_format(scoped, report_dir)
	checks = await run(formatted.node, jobs=jobs, base=base, interactive=False)
	residual: ResidualClass = "needs_reasoning" if checks.returncode != 0 else "green"
	return GateOutcome(residual, formatted.node, checks, plan, formatted.report_paths)
