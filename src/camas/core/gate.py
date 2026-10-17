# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

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

from ..v0.task import Group, Pipe, Task, rebuilt
from .budget import BudgetRun, NothingToRun, plan_under, resolve_budget
from .execution import run
from .hook_event import STALE_TEMP_MAX_AGE_S
from .matrix import expand_matrix
from .scope import coverage_message
from .timings import observed, scope_of

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import TaskNode
	from .budget import BudgetSummary
	from .completion import RunResult
	from .timings import CacheKey, Observed, TaskTiming


ResidualClass: TypeAlias = Literal["green", "needs_reasoning"]
Decision: TypeAlias = Literal["continue", "block"]

REPORT_TOKEN: Final = "{report}"

REPORT_DIR_PREFIX: Final = "camas-report-"


class GateOutcome(NamedTuple):
	residual_class: ResidualClass
	node: TaskNode | None
	result: RunResult | None
	budget: BudgetSummary | None
	nothing_ran: str | None = None
	report_paths: tuple[Path | None, ...] = ()


def decision_of(residual_class: ResidualClass) -> Decision:
	"""
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
	node: TaskNode
	report_paths: tuple[Path | None, ...]


def _allocate_report_path(report_dir: Path) -> Path:
	fd, raw_path = tempfile.mkstemp(dir=report_dir, suffix=".report")
	os.close(fd)
	return Path(raw_path)


def _append_args(
	cmd: str | tuple[str, ...], args: str, report_path: Path | None
) -> str | tuple[str, ...]:
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
	"""
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
			if node.agent_format is None or not node.agent_format.args:
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
				rebuilt(group, *(f.node for f in formatted)),
				tuple(p for f in formatted for p in f.report_paths),
			)
		case _:
			assert_never(node)


def strip_agent_only_pipes(node: TaskNode) -> TaskNode:
	"""
	>>> strip_agent_only_pipes(Pipe("cargo clippy", "clippy-sarif", agent_only=True)).tasks[0].cmd
	'cargo clippy'
	>>> strip_agent_only_pipes(Pipe("a", "b")).tasks[0].cmd
	'a'
	"""
	if not _has_agent_only(node):
		return node
	match node:
		case Pipe(tasks=stages, agent_only=True):
			return rebuilt(node, stages[0], agent_only=False)
		case Group() as group:
			children = tuple(strip_agent_only_pipes(c) for c in group.tasks)
			if all(new is old for new, old in zip(children, group.tasks, strict=True)):
				return node  # pragma: no cover
			return rebuilt(group, *children)
		case Task():  # pragma: no cover
			return node
		case _:
			assert_never(node)


def _has_agent_only(node: TaskNode) -> bool:
	match node:
		case Pipe(agent_only=True):
			return True
		case Group() as group:
			return any(_has_agent_only(c) for c in group.tasks)
		case _:
			return False


def uses_path_mode(node: TaskNode) -> bool:
	"""
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
	try:
		if path.is_dir() and path.stat().st_mtime < cutoff:
			shutil.rmtree(path, ignore_errors=True)
	except OSError:  # pragma: no cover
		pass


def prune_stale_report_dirs(max_age_s: float = STALE_TEMP_MAX_AGE_S) -> None:
	base = Path(tempfile.gettempdir())
	cutoff = time.time() - max_age_s
	for stale in base.glob(f"{REPORT_DIR_PREFIX}*"):
		_rmtree_if_stale(stale, cutoff)


async def settle(
	fix: TaskNode | None,
	changed: tuple[str, ...],
	*,
	camas_dir: Path | None,
	jobs: int | None = None,
	base: Path | None = None,
	leaf_color: bool = True,
) -> RunResult | None:
	if fix is None:
		return None
	keying: Final = observed(camas_dir, expand_matrix(fix), changed)
	if keying.node is None:
		return None
	result: Final = await run(
		keying.node,
		jobs=jobs,
		base=base,
		interactive=False,
		leaf_color=leaf_color,
		identities=keying.identities,
	)
	keying.record(result)
	return result


async def run_gate(
	node: TaskNode,
	changed: tuple[str, ...],
	*,
	under: float | None = None,
	jobs: int | None = None,
	base: Path | None = None,
	timings: Mapping[CacheKey, TaskTiming] | None = None,
	leaf_color: bool = True,
) -> GateOutcome:
	expanded = expand_matrix(node)
	final_keying: Observed
	summary: BudgetSummary | None
	if under is None:
		final_keying, summary = observed(None, expanded, changed), None
		if final_keying.node is None:
			return GateOutcome("green", None, None, None, coverage_message(changed))
	else:
		outcome: Final = resolve_budget(
			plan_under(expanded, under, timings or {}, scope_of(changed)), None, changed
		)
		match outcome:
			case NothingToRun(summary=census, cause=cause):
				return GateOutcome("green", None, None, census, cause)
			case BudgetRun(keying=keying, summary=census):
				final_keying, summary = keying, census
			case _:
				assert_never(outcome)
	scoped = final_keying.node
	assert scoped is not None
	if uses_path_mode(scoped):
		prune_stale_report_dirs()
		report_dir = Path(tempfile.mkdtemp(prefix=REPORT_DIR_PREFIX))
	else:
		report_dir = Path(tempfile.gettempdir())
	formatted = with_agent_format(scoped, report_dir)
	checks = await run(
		formatted.node,
		jobs=jobs,
		base=base,
		interactive=False,
		leaf_color=leaf_color,
		identities=final_keying.identities,
	)
	residual: ResidualClass = "needs_reasoning" if checks.returncode != 0 else "green"
	return GateOutcome(
		residual,
		formatted.node,
		checks,
		summary,
		report_paths=formatted.report_paths,
	)
