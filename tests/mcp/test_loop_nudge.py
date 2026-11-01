# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp import types

from camas import Task
from camas.main.state import LoadOk
from camas.mcp import serve
from camas.mcp.serve import Compat, Session

if TYPE_CHECKING:
	from pathlib import Path

FORMATLESS_TRUNCATED_FAILURE = Task(
	("python", "-c", "[print(i) for i in range(60)]; raise SystemExit(1)"), name="bad"
)


def _session(base: Path) -> Session:
	state = LoadOk(
		tasks={"bad": FORMATLESS_TRUNCATED_FAILURE},
		source=base / "tasks.py",
		scope_effects={},
		config=None,
	)
	return Session(state, base, Compat(emit_structured=False))


def _text(result: types.CallToolResult) -> str:
	return "\n".join(block.text for block in result.content if isinstance(block, types.TextContent))


def test_run_loop_nudge_is_silent_without_a_formatless_failure() -> None:
	assert serve.run_loop_nudge(any_failing_without_agent_format=False) == ""


async def test_camas_run_points_a_formatless_failure_at_the_gate(tmp_path: Path) -> None:
	text = _text(await serve.run_call(_session(tmp_path), {"task": "bad"}))
	assert "so camas_gate reports its output as unstructured raw text" in text
	assert "Tip: output was truncated" not in text


async def test_camas_run_under_a_budget_points_a_formatless_failure_at_the_gate(
	tmp_path: Path,
) -> None:
	text = _text(await serve.run_call(_session(tmp_path), {"task": "bad", "under": 60.0}))
	assert "so camas_gate reports its output as unstructured raw text" in text
	assert "Tip: output was truncated" not in text
