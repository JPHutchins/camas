# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Effect: submits per-leaf check-run results to the GitHub Checks API.

Requires the 'github_checks' optional dependency::

	camas[github_checks]

Defaults read GitHub Actions env vars (``GITHUB_TOKEN``, ``GITHUB_REPOSITORY``,
``GITHUB_SHA``) so the typical CI invocation is just::

	--effects='(GitHubChecks(),)'

In ``pull_request`` events ``GITHUB_SHA`` points at the **merge commit**, not
the PR's head — checks attached to it don't render in the PR Checks panel.
Pass the head SHA explicitly::

	GitHubChecks(GitHubChecksOptions(sha="${{ github.event.pull_request.head.sha || github.sha }}"))

HTTP is fired-and-forgotten in ``on_event`` (``asyncio.create_task``) and
awaited only in ``teardown``, so the run isn't slowed by network latency.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from types import ModuleType
from typing import TYPE_CHECKING, Final, NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from builtins import BaseExceptionGroup
	from typing import assert_never
else:  # pragma: no cover
	from exceptiongroup import BaseExceptionGroup
	from typing_extensions import assert_never

if TYPE_CHECKING:
	import httpx

from ..core.completion import Completion, Finished, Skipped
from ..core.leaf_state import LeafState
from ..core.render import strip_ansi
from ..core.task import Task, TaskNode, task_label
from ..core.task_event import (
	CompletedEvent,
	OutputEvent,
	StartedEvent,
	TaskEvent,
)

GH_API_VERSION: Final = "2022-11-28"
GH_API_HOST: Final = "api.github.com"
OUTPUT_TEXT_LIMIT: Final = 65_000
NAME_LIMIT: Final = 100


class GitHubChecksOptions(NamedTuple):
	"""Configuration for the GitHubChecks Effect.

	>>> GitHubChecksOptions()
	GitHubChecksOptions(token=None, repository=None, sha=None, name_prefix='', tail_bytes=8192, fail_on_api_error=False)
	>>> GitHubChecksOptions(name_prefix="ubuntu/").name_prefix
	'ubuntu/'
	"""

	token: str | None = None
	"""Auth token. None → reads ``GITHUB_TOKEN`` env var at setup time."""
	repository: str | None = None
	"""``owner/repo`` slug. None → reads ``GITHUB_REPOSITORY``."""
	sha: str | None = None
	"""Commit SHA. None → reads ``GITHUB_SHA`` (see module docstring caveat)."""
	name_prefix: str = ""
	"""Prepended to each check-run name (use for matrix-cell discrimination)."""
	tail_bytes: int = 8192
	"""Max bytes of stdout/stderr included in the check-run output text."""
	fail_on_api_error: bool = False
	"""If True, HTTP failures surface in teardown as a BaseExceptionGroup."""


class ResolvedConfig(NamedTuple):
	"""Post-env-fallback config: every required field is a non-empty string.

	>>> ResolvedConfig("t", "o", "r", "s", "", 8192, False).owner
	'o'
	"""

	token: str
	owner: str
	repo: str
	sha: str
	name_prefix: str
	tail_bytes: int
	fail_on_api_error: bool


class PatchPayload(NamedTuple):
	"""Data captured at CompletedEvent for the background PATCH call.

	>>> PatchPayload("success", ("ok", "`cmd`", ""), "2026-01-01T00:00:00Z").conclusion
	'success'
	"""

	conclusion: str
	output: tuple[str, str, str]
	completed_at: str


class Pending(NamedTuple):
	"""No POST task spawned yet — StartedEvent not processed for this leaf.

	>>> Pending()
	Pending()
	"""


class Active(NamedTuple):
	"""POST task pending in background; accumulating output tail."""

	post_task: asyncio.Task[int]
	tail: bytes


class Closed(NamedTuple):
	"""All HTTP for this leaf is scheduled (awaited in teardown).

	``task`` is ``None`` when the leaf was Skipped before any Started fired."""

	task: asyncio.Task[None] | None


LeafCtx: TypeAlias = Pending | Active | Closed


class EffectState(NamedTuple):
	"""Effect-instance state populated by setup(); shared across leaves."""

	http: httpx.AsyncClient
	cfg: ResolvedConfig
	errors: tuple[type[BaseException], ...]


def require_httpx() -> ModuleType:
	"""Import httpx lazily; raise with the install hint when the extra is missing."""
	try:
		import httpx
	except ImportError as e:
		raise RuntimeError("GitHubChecks: requires feature camas[github_checks]") from e
	return httpx


def resolve_config(opts: GitHubChecksOptions) -> ResolvedConfig:
	"""Resolve options + env vars into a fully-specified ResolvedConfig.

	>>> resolve_config(GitHubChecksOptions(token="t", repository="o/r", sha="s")).repo
	'r'
	"""
	token: Final = opts.token or os.environ.get("GITHUB_TOKEN") or ""
	repository: Final = opts.repository or os.environ.get("GITHUB_REPOSITORY") or ""
	sha: Final = opts.sha or os.environ.get("GITHUB_SHA") or ""
	missing: Final = tuple(
		name
		for name, value in (
			("GITHUB_TOKEN (or options.token)", token),
			("GITHUB_REPOSITORY (or options.repository)", repository),
			("GITHUB_SHA (or options.sha)", sha),
		)
		if not value
	)
	if missing:
		raise RuntimeError(f"GitHubChecks: missing required configuration: {', '.join(missing)}")
	parts: Final = repository.split("/")
	if len(parts) != 2 or not parts[0] or not parts[1]:
		raise RuntimeError(f"GitHubChecks: repository must be 'owner/repo', got {repository!r}")
	owner, repo = parts
	return ResolvedConfig(
		token=token,
		owner=owner,
		repo=repo,
		sha=sha,
		name_prefix=opts.name_prefix,
		tail_bytes=opts.tail_bytes,
		fail_on_api_error=opts.fail_on_api_error,
	)


def auth_headers(cfg: ResolvedConfig) -> dict[str, str]:
	"""GitHub API auth + version headers shared by every request."""
	return {
		"Authorization": f"Bearer {cfg.token}",
		"Accept": "application/vnd.github+json",
		"X-GitHub-Api-Version": GH_API_VERSION,
		"Content-Type": "application/json",
		"User-Agent": "camas-github-checks",
	}


def build_name(prefix: str, task: Task) -> str:
	"""Compose check-run name; truncate to ``NAME_LIMIT`` with right-side ellipsis.

	>>> build_name("", Task("ruff check .", name="lint"))
	'lint'
	>>> build_name("ubuntu/", Task("ruff check .", name="lint"))
	'ubuntu/lint'
	>>> build_name("", Task("ruff check ."))
	'ruff check .'
	>>> len(build_name("x" * 200, Task("y", name="z")))
	100
	"""
	raw: Final = f"{prefix}{task_label(task)}"
	return raw if len(raw) <= NAME_LIMIT else raw[: NAME_LIMIT - 1] + "…"


def conclusion_for(completion: Completion) -> str:
	"""Map a camas Completion to a GitHub check-run conclusion string.

	>>> conclusion_for(Finished(0, 0.1, ()))
	'success'
	>>> conclusion_for(Finished(1, 0.1, ()))
	'failure'
	>>> conclusion_for(Skipped(0))
	'skipped'
	"""
	match completion:
		case Finished(returncode=rc):
			return "success" if rc == 0 else "failure"
		case Skipped():
			return "skipped"
		case _:
			assert_never(completion)


def tail_bytes(buf: bytes, limit: int) -> bytes:
	"""Return the last ``limit`` bytes of ``buf``; empty for non-positive ``limit``.

	>>> tail_bytes(b"abcdef", 3)
	b'def'
	>>> tail_bytes(b"abc", 10)
	b'abc'
	>>> tail_bytes(b"", 5)
	b''
	>>> tail_bytes(b"abc", 0)
	b''
	>>> tail_bytes(b"abc", -1)
	b''
	"""
	if limit <= 0:
		return b""
	return buf[-limit:] if len(buf) > limit else buf


_FENCE_OVERHEAD: Final = len("```\n\n```")


def render_body(task: Task, tail: bytes, completion: Completion | None) -> tuple[str, str, str]:
	"""Return ``(title, summary, text)`` for the check-run output object.

	ANSI escape sequences are stripped from ``tail`` — GitHub renders the
	output text as plain text in a fenced code block, so escapes would
	otherwise appear as literal ``\\x1b[...m`` noise. ``text`` is clamped to
	``OUTPUT_TEXT_LIMIT`` characters so an over-eager ``tail_bytes`` setting
	can't exceed GitHub's 65535-char ``output.text`` cap."""
	cmd_str = task.cmd if isinstance(task.cmd, str) else " ".join(task.cmd)
	decoded = strip_ansi(tail.decode("utf-8", errors="replace")) if tail else ""
	if len(decoded) > OUTPUT_TEXT_LIMIT - _FENCE_OVERHEAD:
		decoded = decoded[: OUTPUT_TEXT_LIMIT - _FENCE_OVERHEAD]
	text = f"```\n{decoded}\n```" if decoded else ""
	match completion:
		case None:
			return (f"🏃 Running: {task_label(task)}", f"`{cmd_str}`", "")
		case Finished(returncode=rc, elapsed=elapsed):
			status = "✅ Passed" if rc == 0 else f"❌ Failed (exit {rc})"
			return (f"{status} in {elapsed:.2f}s", f"`{cmd_str}`", text)
		case Skipped(returncode=rc):
			return ("⏭️ Skipped", f"Skipped because an earlier task exited {rc}.", "")
		case _:
			assert_never(completion)


def now_iso() -> str:
	"""Current UTC time as ISO-8601 (the format GitHub's Checks API expects)."""
	return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_http_error(msg: str, exc: BaseException) -> None:
	"""Single stderr-print path so tests can capture it."""
	print(f"{msg}: {exc}", file=sys.stderr)


async def post_check_run(
	http: httpx.AsyncClient,
	cfg: ResolvedConfig,
	name: str,
	external_id: str,
	started_at: str,
) -> int:
	"""POST /repos/{owner}/{repo}/check-runs — returns the new check_run_id.

	``external_id`` is a stable identifier per leaf (derived from name_prefix +
	task_label) — lets integrators correlate check_runs back to camas leaves
	via the API even when names collide."""
	response = await http.post(
		f"https://{GH_API_HOST}/repos/{cfg.owner}/{cfg.repo}/check-runs",
		json={
			"name": name,
			"head_sha": cfg.sha,
			"external_id": external_id,
			"status": "in_progress",
			"started_at": started_at,
		},
	)
	response.raise_for_status()
	return int(response.json()["id"])


async def patch_check_run(
	http: httpx.AsyncClient,
	cfg: ResolvedConfig,
	check_id: int,
	payload: PatchPayload,
) -> None:
	"""PATCH /repos/{owner}/{repo}/check-runs/{id}."""
	title, summary, text = payload.output
	response = await http.patch(
		f"https://{GH_API_HOST}/repos/{cfg.owner}/{cfg.repo}/check-runs/{check_id}",
		json={
			"status": "completed",
			"conclusion": payload.conclusion,
			"completed_at": payload.completed_at,
			"output": {"title": title, "summary": summary, "text": text},
		},
	)
	response.raise_for_status()


async def post_then_patch(
	http: httpx.AsyncClient,
	cfg: ResolvedConfig,
	post_task: asyncio.Task[int],
	payload: PatchPayload,
	errors: tuple[type[BaseException], ...],
) -> None:
	"""Await POST, then send PATCH. Logs errors; re-raises iff fail_on_api_error."""
	try:
		check_id = await post_task
	except errors as e:
		log_http_error("github_checks: POST failed", e)
		if cfg.fail_on_api_error:
			raise
		return
	try:
		await patch_check_run(http, cfg, check_id, payload)
	except errors as e:
		log_http_error(f"github_checks: PATCH check_id={check_id} failed", e)
		if cfg.fail_on_api_error:
			raise


async def post_then_cancel(
	http: httpx.AsyncClient,
	cfg: ResolvedConfig,
	post_task: asyncio.Task[int],
	errors: tuple[type[BaseException], ...],
) -> None:
	"""Teardown path for Active leaves whose event stream was truncated:
	wait for POST, then PATCH ``conclusion="cancelled"`` so the check
	doesn't linger ``in_progress`` in the GH UI."""
	await post_then_patch(
		http,
		cfg,
		post_task,
		PatchPayload(
			conclusion="cancelled",
			output=("Cancelled", "Run was aborted before this task completed.", ""),
			completed_at=now_iso(),
		),
		errors,
	)


def started_to_active(state: EffectState, task: Task) -> Active:
	"""Pure: build the next ctx for a StartedEvent on a Pending leaf.

	Creates the POST task; the returned Active holds the task handle, so the
	ctx itself is what keeps the task referenced (the runtime stores ctxs in
	ctx_grid for the duration of the run)."""
	name = build_name(state.cfg.name_prefix, task)
	return Active(
		post_task=asyncio.create_task(
			post_check_run(state.http, state.cfg, name, external_id_for(state.cfg, task), now_iso())
		),
		tail=b"",
	)


def external_id_for(cfg: ResolvedConfig, task: Task) -> str:
	"""Stable per-leaf identifier — same leaf on the same prefix yields the same id.

	>>> external_id_for(
	...     ResolvedConfig("t", "o", "r", "s", "ubuntu/", 8192, False),
	...     Task("ruff check .", name="lint"),
	... )
	'camas:ubuntu/lint'
	"""
	return f"camas:{cfg.name_prefix}{task_label(task)}"


def completed_to_closed(
	state: EffectState, task: Task, completion: Completion, prior: Active
) -> Closed:
	"""Pure: build the next ctx for a CompletedEvent on an Active leaf.

	Spawns the wrap task that awaits the prior POST and PATCHes the result.
	The wrap task is stored in the returned Closed."""
	return Closed(
		task=asyncio.create_task(
			post_then_patch(
				state.http,
				state.cfg,
				prior.post_task,
				PatchPayload(
					conclusion=conclusion_for(completion),
					output=render_body(task, prior.tail, completion),
					completed_at=now_iso(),
				),
				state.errors,
			)
		),
	)


def pipelines_from_ctxs(
	state: EffectState, ctxs: tuple[LeafCtx, ...]
) -> tuple[asyncio.Task[None], ...]:
	"""Pure: unfold every leaf ctx into its terminal pipeline task.

	- ``Closed(task=t)`` → ``t`` (already a pipeline)
	- ``Active(post_task=p, ...)`` → newly-spawned cancel pipeline awaiting ``p``
	- ``Pending()`` / ``Closed(task=None)`` → contributes nothing

	The cancel pipelines for Active leaves are created here, not held on the
	effect; their refs live in the returned tuple until ``teardown`` awaits
	them."""
	return tuple(extracted for ctx in ctxs if (extracted := pipeline_of(state, ctx)) is not None)


def pipeline_of(state: EffectState, ctx: LeafCtx) -> asyncio.Task[None] | None:
	match ctx:
		case Closed(task=task):
			return task
		case Active(post_task=post_task):
			return asyncio.create_task(
				post_then_cancel(state.http, state.cfg, post_task, state.errors)
			)
		case Pending():
			return None
		case _:
			assert_never(ctx)


class GitHubChecks:
	"""Submits per-leaf check-run results to the GitHub Checks API.

	See module docstring for env-var defaults, the PR head-SHA caveat, and the
	required ``checks: write`` workflow permission.
	"""

	def __init__(self, options: GitHubChecksOptions | None = None) -> None:
		self.options: Final = options if options is not None else GitHubChecksOptions()
		self.state: EffectState | None = None

	async def setup(self, task: TaskNode) -> LeafCtx:
		httpx_mod: Final = require_httpx()  # zuban: ignore[misc] # zuban defies PEP591
		cfg: Final = resolve_config(self.options)  # zuban: ignore[misc] # zuban defies PEP591
		self.state = EffectState(
			http=httpx_mod.AsyncClient(timeout=5.0, headers=auth_headers(cfg)),
			cfg=cfg,
			errors=(httpx_mod.HTTPError, OSError),
		)
		return Pending()

	async def on_event(
		self,
		event: TaskEvent,
		states: Sequence[LeafState],
		ctx: LeafCtx,
	) -> LeafCtx:
		if self.state is None:  # pragma: no cover
			return ctx
		state: Final = self.state  # zuban: ignore[misc] # zuban defies PEP591
		match event, ctx:
			case StartedEvent(task=task), Pending():
				return started_to_active(state, task)
			case OutputEvent(line=line), Active(post_task=post_task, tail=tail):
				return Active(
					post_task=post_task,
					tail=tail_bytes(tail + line, state.cfg.tail_bytes),
				)
			case CompletedEvent(task=task, completion=completion), Active() as prior:
				return completed_to_closed(state, task, completion, prior)
			case CompletedEvent(), Pending():
				return Closed(task=None)
			case _:
				return ctx

	async def teardown(self, ctxs: tuple[LeafCtx, ...]) -> None:
		if self.state is None:  # pragma: no cover
			return
		state: Final = self.state  # zuban: ignore[misc] # zuban defies PEP591
		pipelines: Final = (  # zuban: ignore[misc] # zuban defies PEP591
			pipelines_from_ctxs(state, ctxs)
		)
		errors: Final = tuple(  # zuban: ignore[misc] # zuban defies PEP591
			r
			for r in await asyncio.gather(*pipelines, return_exceptions=True)
			if isinstance(r, BaseException)
		)
		# HTTP/OS errors were already logged by post_then_patch; any other exception
		# type is a bug (KeyError, AttributeError, etc.) — log so it isn't silent.
		for err in errors:
			if not isinstance(err, state.errors):
				log_http_error("github_checks: unexpected error in pipeline", err)
		try:
			if state.cfg.fail_on_api_error and errors:
				raise BaseExceptionGroup("github_checks teardown failed", errors)
		finally:
			await state.http.aclose()
