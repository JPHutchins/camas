# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The ``GitHubChecks`` effect: one GitHub check run per leaf.

It needs the ``camas[github_checks]`` extra and the workflow permission ``checks: write``.
``token``, ``repository``, and ``sha`` default to ``GITHUB_TOKEN``, ``GITHUB_REPOSITORY``, and
``GITHUB_SHA``, so in CI it is just ``--effects='(GitHubChecks(),)'``. On ``pull_request``
events ``GITHUB_SHA`` is the merge commit, whose checks the PR Checks panel doesn't show; pass
the head commit::

    GitHubChecks(sha="${{ github.event.pull_request.head.sha || github.sha }}")
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, NamedTuple, TypeAlias

if sys.version_info >= (3, 11):
	from builtins import BaseExceptionGroup
	from typing import assert_never
else:  # pragma: no cover
	from exceptiongroup import BaseExceptionGroup
	from typing_extensions import assert_never

if TYPE_CHECKING:
	from collections.abc import Sequence
	from types import ModuleType

	import httpx

	from ..v0.leaf_state import LeafState
	from ..v0.task import Task, TaskNode

from ..core.render import strip_ansi
from ..core.task import task_label
from ..v0.completion import Completion, Errored, Finished, Skipped, Stopped
from ..v0.task_event import CompletedEvent, OutputEvent, StartedEvent, TaskEvent

GH_API_VERSION: Final = "2022-11-28"
GH_API_HOST: Final = "api.github.com"
OUTPUT_TEXT_LIMIT: Final = 65_000
NAME_LIMIT: Final = 100


class ResolvedConfig(NamedTuple):
	"""
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
	"""
	>>> PatchPayload("success", ("ok", "`cmd`", ""), "2026-01-01T00:00:00Z").conclusion
	'success'
	"""

	conclusion: str
	output: tuple[str, str, str]
	completed_at: str


class Pending(NamedTuple):
	"""
	>>> Pending()
	Pending()
	"""


class Active(NamedTuple):
	post_task: asyncio.Task[int]
	tail: bytes


class Closed(NamedTuple):
	task: asyncio.Task[None] | None


LeafCtx: TypeAlias = Pending | Active | Closed


class EffectState(NamedTuple):
	http: httpx.AsyncClient
	cfg: ResolvedConfig
	errors: tuple[type[BaseException], ...]


def require_httpx() -> ModuleType:
	try:
		import httpx
	except ImportError as e:
		raise RuntimeError("GitHubChecks: requires feature camas[github_checks]") from e
	return httpx


def resolve_config(
	token: str | None,
	repository: str | None,
	sha: str | None,
	name_prefix: str,
	tail_bytes: int,
	fail_on_api_error: bool,
) -> ResolvedConfig:
	"""
	Raises:
		RuntimeError: when token, repository, or sha is unset, or repository
			is not ``owner/repo``.

	>>> resolve_config("t", "o/r", "s", "", 8192, False).repo
	'r'
	"""
	resolved_token: Final = token or os.environ.get("GITHUB_TOKEN") or ""
	resolved_repository: Final = repository or os.environ.get("GITHUB_REPOSITORY") or ""
	resolved_sha: Final = sha or os.environ.get("GITHUB_SHA") or ""
	missing: Final = tuple(
		name
		for name, value in (
			("GITHUB_TOKEN (or the token argument)", resolved_token),
			("GITHUB_REPOSITORY (or the repository argument)", resolved_repository),
			("GITHUB_SHA (or the sha argument)", resolved_sha),
		)
		if not value
	)
	if missing:
		raise RuntimeError(f"GitHubChecks: missing required configuration: {', '.join(missing)}")
	parts: Final = resolved_repository.split("/")
	if len(parts) != 2 or not parts[0] or not parts[1]:
		raise RuntimeError(
			f"GitHubChecks: repository must be 'owner/repo', got {resolved_repository!r}"
		)
	owner, repo = parts
	return ResolvedConfig(
		token=resolved_token,
		owner=owner,
		repo=repo,
		sha=resolved_sha,
		name_prefix=name_prefix,
		tail_bytes=tail_bytes,
		fail_on_api_error=fail_on_api_error,
	)


def auth_headers(cfg: ResolvedConfig) -> dict[str, str]:
	return {
		"Authorization": f"Bearer {cfg.token}",
		"Accept": "application/vnd.github+json",
		"X-GitHub-Api-Version": GH_API_VERSION,
		"Content-Type": "application/json",
		"User-Agent": "camas-github-checks",
	}


def build_name(prefix: str, task: Task) -> str:
	"""
	>>> from camas import Task
	>>> build_name("ubuntu/", Task("ruff check .", name="lint"))
	'ubuntu/lint'
	>>> len(build_name("x" * 200, Task("y", name="z")))
	100
	"""
	raw: Final = f"{prefix}{task_label(task)}"
	return raw if len(raw) <= NAME_LIMIT else raw[: NAME_LIMIT - 1] + "…"


def conclusion_for(completion: Completion) -> str:
	"""
	>>> conclusion_for(Finished(0, 0.1, ()))
	'success'
	>>> conclusion_for(Finished(1, 0.1, ()))
	'failure'
	>>> conclusion_for(Skipped(0))
	'skipped'
	>>> conclusion_for(Stopped(130, 0.1, ()))
	'cancelled'
	>>> conclusion_for(Errored(127, "no such file or directory: x"))
	'failure'
	"""
	match completion:
		case Finished(returncode=rc):
			return "success" if rc == 0 else "failure"
		case Skipped():
			return "skipped"
		case Stopped():
			return "cancelled"
		case Errored():
			return "failure"
		case _:
			assert_never(completion)


def tail_bytes(buf: bytes, limit: int) -> bytes:
	"""
	>>> tail_bytes(b"abcdef", 3)
	b'def'
	>>> tail_bytes(b"abc", 10)
	b'abc'
	>>> tail_bytes(b"abc", 0)
	b''
	"""
	if limit <= 0:
		return b""
	return buf[-limit:] if len(buf) > limit else buf


_FENCE_OVERHEAD: Final = len("```\n\n```")


def render_body(task: Task, tail: bytes, completion: Completion | None) -> tuple[str, str, str]:
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
		case Stopped(returncode=rc, elapsed=elapsed):
			return (f"⏹️ Stopped (exit {rc}) in {elapsed:.2f}s", f"`{cmd_str}`", text)
		case Errored(message=message):
			return ("💥 Errored", message, "")
		case _:
			assert_never(completion)


def now_iso() -> str:
	return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_http_error(msg: str, exc: BaseException) -> None:
	print(f"{msg}: {exc}", file=sys.stderr)


async def post_check_run(
	http: httpx.AsyncClient,
	cfg: ResolvedConfig,
	name: str,
	external_id: str,
	started_at: str,
) -> int:
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
	name = build_name(state.cfg.name_prefix, task)
	return Active(
		post_task=asyncio.create_task(
			post_check_run(state.http, state.cfg, name, external_id_for(state.cfg, task), now_iso())
		),
		tail=b"",
	)


def external_id_for(cfg: ResolvedConfig, task: Task) -> str:
	return f"camas:{cfg.name_prefix}{task_label(task)}"


def completed_to_closed(
	state: EffectState, task: Task, completion: Completion, prior: Active
) -> Closed:
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
	"""Per-leaf check runs on the GitHub Checks API."""

	def __init__(
		self,
		token: str | None = None,
		repository: str | None = None,
		sha: str | None = None,
		name_prefix: str = "",
		tail_bytes: int = 8192,
		fail_on_api_error: bool = False,
	) -> None:
		self._token: Final = token
		self._repository: Final = repository
		self._sha: Final = sha
		self._name_prefix: Final = name_prefix
		self._tail_bytes: Final = tail_bytes
		self._fail_on_api_error: Final = fail_on_api_error
		self.state: EffectState | None = None

	async def setup(self, task: TaskNode) -> LeafCtx:
		httpx_mod: Final = require_httpx()  # zuban: ignore[misc] # zuban defies PEP591
		cfg: Final = resolve_config(  # zuban: ignore[misc] # zuban defies PEP591
			self._token,
			self._repository,
			self._sha,
			self._name_prefix,
			self._tail_bytes,
			self._fail_on_api_error,
		)
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
		for err in errors:
			if not isinstance(err, state.errors):
				log_http_error("github_checks: unexpected error in pipeline", err)
		try:
			if state.cfg.fail_on_api_error and errors:
				raise BaseExceptionGroup("github_checks teardown failed", errors)
		finally:
			await state.http.aclose()
