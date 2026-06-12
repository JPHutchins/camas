# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import asyncio
import json
import sys
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
	from builtins import BaseExceptionGroup
else:  # pragma: no cover
	from exceptiongroup import BaseExceptionGroup

from datetime import datetime

import httpx
import pytest

from camas import Parallel, Task
from camas.core.execution import run
from camas.effect.github_checks import (
	Active,
	Closed,
	GitHubChecks,
	GitHubChecksOptions,
	PatchPayload,
	Pending,
	ResolvedConfig,
	auth_headers,
	conclusion_for,
	patch_check_run,
	post_check_run,
	post_then_cancel,
	post_then_patch,
	render_body,
	resolve_config,
)
from camas.v0.completion import Finished, Skipped
from camas.v0.task_event import CompletedEvent, OutputEvent, StartedEvent

if TYPE_CHECKING:
	from collections.abc import Callable

TS = datetime(2026, 5, 21, 14, 30, 0)


def clear_gh_env(monkeypatch: pytest.MonkeyPatch) -> None:
	for k in ("GITHUB_TOKEN", "GITHUB_REPOSITORY", "GITHUB_SHA"):
		monkeypatch.delenv(k, raising=False)


def set_gh_env(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("GITHUB_TOKEN", "t")
	monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
	monkeypatch.setenv("GITHUB_SHA", "sha")


CFG = ResolvedConfig(
	token="t",
	owner="o",
	repo="r",
	sha="sha",
	name_prefix="",
	tail_bytes=64,
	fail_on_api_error=False,
)

ERRORS: tuple[type[BaseException], ...] = (httpx.HTTPError, OSError)


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
	return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def wrap_value(value: int) -> int:
	return value


async def wrap_raise(exc: BaseException) -> int:
	raise exc


async def wrap_raise_none(exc: BaseException) -> None:
	raise exc


def test_options_defaults() -> None:
	o = GitHubChecksOptions()
	assert (o.token, o.repository, o.sha) == (None, None, None)
	assert (o.name_prefix, o.tail_bytes, o.fail_on_api_error) == ("", 8192, False)


def test_resolve_config_reads_env_when_options_none(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("GITHUB_TOKEN", "env-token")
	monkeypatch.setenv("GITHUB_REPOSITORY", "env-owner/env-repo")
	monkeypatch.setenv("GITHUB_SHA", "env-sha")
	cfg = resolve_config(GitHubChecksOptions())
	assert cfg == ResolvedConfig("env-token", "env-owner", "env-repo", "env-sha", "", 8192, False)


def test_resolve_config_explicit_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("GITHUB_TOKEN", "env-token")
	monkeypatch.setenv("GITHUB_REPOSITORY", "env-o/env-r")
	monkeypatch.setenv("GITHUB_SHA", "env-sha")
	cfg = resolve_config(
		GitHubChecksOptions(
			token="explicit", repository="o/r", sha="s", name_prefix="p/", tail_bytes=10
		)
	)
	assert cfg.token == "explicit"
	assert (cfg.owner, cfg.repo) == ("o", "r")
	assert cfg.sha == "s"
	assert cfg.name_prefix == "p/"
	assert cfg.tail_bytes == 10


def test_resolve_config_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
	clear_gh_env(monkeypatch)
	with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
		resolve_config(GitHubChecksOptions(repository="o/r", sha="s"))


def test_resolve_config_missing_repository_raises(monkeypatch: pytest.MonkeyPatch) -> None:
	clear_gh_env(monkeypatch)
	with pytest.raises(RuntimeError, match="GITHUB_REPOSITORY"):
		resolve_config(GitHubChecksOptions(token="t", sha="s"))


def test_resolve_config_missing_sha_raises(monkeypatch: pytest.MonkeyPatch) -> None:
	clear_gh_env(monkeypatch)
	with pytest.raises(RuntimeError, match="GITHUB_SHA"):
		resolve_config(GitHubChecksOptions(token="t", repository="o/r"))


@pytest.mark.parametrize("repo", ["no-slash", "o/", "/r", "o/r/extra", "/", ""])
def test_resolve_config_malformed_repository_raises(
	monkeypatch: pytest.MonkeyPatch, repo: str
) -> None:
	clear_gh_env(monkeypatch)
	with pytest.raises(RuntimeError):
		resolve_config(GitHubChecksOptions(token="t", repository=repo, sha="s"))


def test_auth_headers_format() -> None:
	h = auth_headers(CFG)
	assert h["Authorization"] == "Bearer t"
	assert h["X-GitHub-Api-Version"] == "2022-11-28"
	assert h["Accept"] == "application/vnd.github+json"
	assert h["User-Agent"] == "camas-github-checks"


def test_conclusion_for_finished_negative_rc_is_failure() -> None:
	assert conclusion_for(Finished(-1, 0.0, ())) == "failure"


def test_render_body_pass_minimal() -> None:
	title, summary, text = render_body(Task("cmd", name="t"), b"", Finished(0, 1.23, ()))
	assert title == "✅ Passed in 1.23s"
	assert summary == "`cmd`"
	assert text == ""


def test_render_body_pass_includes_tail_when_present() -> None:
	title, _, text = render_body(Task("cmd", name="t"), b"all clean\n", Finished(0, 0.5, ()))
	assert title == "✅ Passed in 0.50s"
	assert "all clean" in text
	assert text.startswith("```\n")


def test_render_body_fail_with_invalid_utf8() -> None:
	title, summary, text = render_body(
		Task(("python", "-c", "x"), name="t"),
		b"\xff\xfe oops",
		Finished(2, 0.5, ()),
	)
	assert title.startswith("❌")
	assert "exit 2" in title
	assert "0.50s" in title
	assert summary == "`python -c x`"
	assert "�" in text
	assert text.startswith("```\n")


def test_render_body_fail_empty_output_no_fence() -> None:
	_, _, text = render_body(Task("cmd"), b"", Finished(1, 0.1, ()))
	assert text == ""


def test_render_body_skipped() -> None:
	title, summary, text = render_body(Task("cmd"), b"", Skipped(3))
	assert title == "⏭️ Skipped"
	assert "exited 3" in summary
	assert text == ""


def test_render_body_in_progress() -> None:
	title, summary, text = render_body(Task("cmd", name="t"), b"", None)
	assert title == "🏃 Running: t"
	assert summary == "`cmd`"
	assert text == ""


def test_render_body_clamps_text_to_output_limit() -> None:
	from camas.effect.github_checks import OUTPUT_TEXT_LIMIT

	huge = b"x" * (OUTPUT_TEXT_LIMIT * 2)
	_, _, text = render_body(Task("cmd"), huge, Finished(1, 0.1, ()))
	assert len(text) <= OUTPUT_TEXT_LIMIT
	assert text.startswith("```\n")
	assert text.endswith("\n```")


async def test_teardown_logs_unexpected_non_http_exception(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	set_gh_env(monkeypatch)
	effect = GitHubChecks()
	await effect.setup(Task("x"))
	bad = asyncio.create_task(wrap_raise_none(KeyError("missing-key")))
	await effect.teardown((Closed(task=bad),))
	assert "unexpected error in pipeline" in capsys.readouterr().err


async def test_teardown_skips_double_logging_of_http_errors(
	monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
	"""HTTP errors are logged at the source in post_then_patch; teardown shouldn't re-log."""
	set_gh_env(monkeypatch)
	effect = GitHubChecks()
	await effect.setup(Task("x"))
	http_err = asyncio.create_task(wrap_raise_none(httpx.ConnectError("boom")))
	await effect.teardown((Closed(task=http_err),))
	assert "unexpected error in pipeline" not in capsys.readouterr().err


async def test_post_check_run_posts_payload_and_returns_id() -> None:
	captured: list[httpx.Request] = []

	def handler(req: httpx.Request) -> httpx.Response:
		captured.append(req)
		return httpx.Response(201, json={"id": 12345})

	async with mock_client(handler) as http:
		assert (
			await post_check_run(http, CFG, "the-name", "camas:the-name", "2026-01-01T00:00:00Z")
			== 12345
		)

	req = captured[0]
	assert req.method == "POST"
	assert req.url.path == "/repos/o/r/check-runs"
	body = json.loads(req.content)
	assert body["name"] == "the-name"
	assert body["head_sha"] == "sha"
	assert body["external_id"] == "camas:the-name"
	assert body["status"] == "in_progress"


async def test_post_check_run_raises_on_4xx() -> None:
	def handler(req: httpx.Request) -> httpx.Response:
		return httpx.Response(404, json={"message": "not found"})

	async with mock_client(handler) as http:
		with pytest.raises(httpx.HTTPStatusError):
			await post_check_run(http, CFG, "n", "camas:n", "now")


async def test_patch_check_run_sends_completion_payload() -> None:
	captured: list[httpx.Request] = []

	def handler(req: httpx.Request) -> httpx.Response:
		captured.append(req)
		return httpx.Response(200, json={})

	async with mock_client(handler) as http:
		await patch_check_run(
			http,
			CFG,
			42,
			PatchPayload("success", ("title", "sum", "txt"), "2026-01-01T00:00:00Z"),
		)

	req = captured[0]
	assert req.method == "PATCH"
	assert req.url.path == "/repos/o/r/check-runs/42"
	body = json.loads(req.content)
	assert body["status"] == "completed"
	assert body["conclusion"] == "success"
	assert body["output"] == {"title": "title", "summary": "sum", "text": "txt"}


async def test_post_then_patch_happy_path() -> None:
	patches: list[httpx.Request] = []

	def handler(req: httpx.Request) -> httpx.Response:
		patches.append(req)
		return httpx.Response(200, json={})

	async with mock_client(handler) as http:
		post_task = asyncio.create_task(wrap_value(99))
		await post_then_patch(
			http, CFG, post_task, PatchPayload("success", ("t", "s", ""), "now"), ERRORS
		)

	assert patches[0].url.path == "/repos/o/r/check-runs/99"


async def test_post_then_patch_post_failure_logs(capsys: pytest.CaptureFixture[str]) -> None:
	def handler(req: httpx.Request) -> httpx.Response:  # pragma: no cover
		raise AssertionError("PATCH should not be reached when POST failed")

	async with mock_client(handler) as http:
		post_task = asyncio.create_task(wrap_raise(httpx.ConnectError("nope")))
		await post_then_patch(
			http, CFG, post_task, PatchPayload("success", ("t", "s", ""), "now"), ERRORS
		)

	assert "POST failed" in capsys.readouterr().err


async def test_post_then_patch_post_failure_reraises_when_fail_on_error() -> None:
	cfg_fail = CFG._replace(fail_on_api_error=True)

	def handler(req: httpx.Request) -> httpx.Response:  # pragma: no cover
		raise AssertionError("PATCH not expected")

	async with mock_client(handler) as http:
		post_task = asyncio.create_task(wrap_raise(httpx.ConnectError("x")))
		with pytest.raises(httpx.ConnectError):
			await post_then_patch(
				http, cfg_fail, post_task, PatchPayload("success", ("t", "s", ""), "now"), ERRORS
			)


async def test_post_then_patch_patch_failure_logs(capsys: pytest.CaptureFixture[str]) -> None:
	def handler(req: httpx.Request) -> httpx.Response:
		return httpx.Response(500, json={"message": "boom"})

	async with mock_client(handler) as http:
		post_task = asyncio.create_task(wrap_value(7))
		await post_then_patch(
			http, CFG, post_task, PatchPayload("failure", ("t", "s", ""), "now"), ERRORS
		)

	assert "PATCH check_id=7 failed" in capsys.readouterr().err


async def test_post_then_patch_patch_failure_reraises_when_fail_on_error() -> None:
	cfg_fail = CFG._replace(fail_on_api_error=True)

	def handler(req: httpx.Request) -> httpx.Response:
		return httpx.Response(500, json={})

	async with mock_client(handler) as http:
		post_task = asyncio.create_task(wrap_value(7))
		with pytest.raises(httpx.HTTPStatusError):
			await post_then_patch(
				http, cfg_fail, post_task, PatchPayload("failure", ("t", "s", ""), "now"), ERRORS
			)


async def test_post_then_cancel_sends_cancelled_patch() -> None:
	patches: list[dict[str, Any]] = []

	def handler(req: httpx.Request) -> httpx.Response:
		patches.append(json.loads(req.content))
		return httpx.Response(200, json={})

	async with mock_client(handler) as http:
		post_task = asyncio.create_task(wrap_value(55))
		await post_then_cancel(http, CFG, post_task, ERRORS)

	assert patches[0]["conclusion"] == "cancelled"


async def test_setup_returns_pending_with_state(monkeypatch: pytest.MonkeyPatch) -> None:
	set_gh_env(monkeypatch)
	effect = GitHubChecks()
	ctx = await effect.setup(Task("x"))
	assert isinstance(ctx, Pending)
	assert effect.state is not None
	await effect.state.http.aclose()


async def test_setup_propagates_missing_httpx_as_setup_error(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	set_gh_env(monkeypatch)
	monkeypatch.setitem(sys.modules, "httpx", None)
	with pytest.raises(BaseExceptionGroup) as ei:
		await run(Task(("python", "-c", "pass")), effects=[GitHubChecks()])
	(inner,) = ei.value.exceptions
	assert isinstance(inner, RuntimeError)
	assert "github_checks" in str(inner)


async def test_setup_propagates_missing_config_as_setup_error(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	clear_gh_env(monkeypatch)
	with pytest.raises(BaseExceptionGroup) as ei:
		await run(Task(("python", "-c", "pass")), effects=[GitHubChecks()])
	(inner,) = ei.value.exceptions
	assert isinstance(inner, RuntimeError)
	assert "GITHUB_TOKEN" in str(inner)


async def test_on_event_started_spawns_post_task(monkeypatch: pytest.MonkeyPatch) -> None:
	set_gh_env(monkeypatch)
	posted: list[tuple[str, str]] = []

	async def fake_post(http: Any, cfg: Any, name: str, external_id: str, started_at: str) -> int:
		posted.append((name, external_id))
		return 42

	monkeypatch.setattr("camas.effect.github_checks.post_check_run", fake_post)

	effect = GitHubChecks(GitHubChecksOptions(name_prefix="pfx/"))
	ctx = await effect.setup(Task("x"))
	t = Task("cmd", name="lint")
	ctx = await effect.on_event(StartedEvent(t, 0, TS), [], ctx)
	assert isinstance(ctx, Active)
	assert ctx.tail == b""
	await ctx.post_task
	await effect.teardown((Closed(task=None),))
	assert posted == [("pfx/lint", "camas:pfx/lint")]


async def test_on_event_output_accumulates_and_trims_tail(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	set_gh_env(monkeypatch)

	async def fake_post(*args: Any, **kwargs: Any) -> int:
		return 1

	monkeypatch.setattr("camas.effect.github_checks.post_check_run", fake_post)

	effect = GitHubChecks(GitHubChecksOptions(tail_bytes=5))
	ctx = await effect.setup(Task("x"))
	t = Task("cmd", name="t")
	ctx = await effect.on_event(StartedEvent(t, 0, TS), [], ctx)
	ctx = await effect.on_event(OutputEvent(t, 0, b"abc", TS), [], ctx)
	ctx = await effect.on_event(OutputEvent(t, 0, b"defgh", TS), [], ctx)
	assert isinstance(ctx, Active)
	assert ctx.tail == b"defgh"
	await effect.teardown((Closed(task=None),))


async def test_on_event_completed_finished_spawns_patch_success(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	set_gh_env(monkeypatch)
	patches: list[PatchPayload] = []

	async def fake_post(*args: Any, **kwargs: Any) -> int:
		return 7

	async def fake_then_patch(
		http: Any, cfg: Any, post_task: Any, payload: PatchPayload, errors: Any
	) -> None:
		await post_task
		patches.append(payload)

	monkeypatch.setattr("camas.effect.github_checks.post_check_run", fake_post)
	monkeypatch.setattr("camas.effect.github_checks.post_then_patch", fake_then_patch)

	effect = GitHubChecks()
	ctx = await effect.setup(Task("x"))
	t = Task("cmd", name="t")
	ctx = await effect.on_event(StartedEvent(t, 0, TS), [], ctx)
	ctx = await effect.on_event(CompletedEvent(t, 0, Finished(0, 1.0, ()), TS), [], ctx)
	assert isinstance(ctx, Closed)
	assert ctx.task is not None
	await effect.teardown((ctx,))
	assert len(patches) == 1
	assert patches[0].conclusion == "success"


async def test_on_event_completed_skipped_on_pending_returns_closed_none(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	set_gh_env(monkeypatch)
	effect = GitHubChecks()
	ctx = await effect.setup(Task("x"))
	t = Task("cmd", name="t")
	ctx = await effect.on_event(CompletedEvent(t, 0, Skipped(1), TS), [], ctx)
	assert isinstance(ctx, Closed)
	assert ctx.task is None
	await effect.teardown((ctx,))


async def test_teardown_no_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
	"""All-Pending ctxs: gather receives no pipelines, no errors."""
	set_gh_env(monkeypatch)
	effect = GitHubChecks()
	await effect.setup(Task("x"))
	await effect.teardown((Pending(), Pending()))


async def test_teardown_active_leaf_sends_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
	set_gh_env(monkeypatch)
	cancels: list[str] = []

	async def fake_then_cancel(http: Any, cfg: Any, post_task: Any, errors: Any) -> None:
		cancels.append("cancelled")
		await post_task

	monkeypatch.setattr("camas.effect.github_checks.post_then_cancel", fake_then_cancel)

	effect = GitHubChecks()
	await effect.setup(Task("x"))
	post_task = asyncio.create_task(wrap_value(1))
	await effect.teardown((Active(post_task=post_task, tail=b""),))
	assert cancels == ["cancelled"]


async def test_teardown_fail_on_api_error_raises_exceptiongroup(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	set_gh_env(monkeypatch)
	effect = GitHubChecks(GitHubChecksOptions(fail_on_api_error=True))
	await effect.setup(Task("x"))
	bad = asyncio.create_task(wrap_raise_none(RuntimeError("boom")))
	with pytest.raises(BaseExceptionGroup) as ei:
		await effect.teardown((Closed(task=bad),))
	(inner,) = ei.value.exceptions
	assert isinstance(inner, RuntimeError)
	assert "boom" in str(inner)


async def test_teardown_logs_errors_without_raising_by_default(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	set_gh_env(monkeypatch)
	effect = GitHubChecks()
	await effect.setup(Task("x"))
	bad = asyncio.create_task(wrap_raise_none(RuntimeError("ignored")))
	await effect.teardown((Closed(task=bad),))


async def test_end_to_end_creates_post_and_patch_per_leaf(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	set_gh_env(monkeypatch)

	calls: list[tuple[str, str]] = []
	next_id = [100]

	def handler(req: httpx.Request) -> httpx.Response:
		calls.append((req.method, req.url.path))
		if req.method == "POST":
			next_id[0] += 1
			return httpx.Response(201, json={"id": next_id[0]})
		return httpx.Response(200, json={})

	real_client = httpx.AsyncClient

	def fake_async_client(**kwargs: Any) -> httpx.AsyncClient:
		return real_client(transport=httpx.MockTransport(handler))

	monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)

	a = Task(("python", "-c", "pass"), name="alpha")
	b = Task(("python", "-c", "pass"), name="beta")
	result = await run(Parallel(a, b), effects=[GitHubChecks()])
	assert result.returncode == 0
	methods = [m for m, _ in calls]
	assert methods.count("POST") == 2
	assert methods.count("PATCH") == 2
