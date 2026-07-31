# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Guards that the tiered camas-fixer agent and gate skill templates shipped in the wheel are
well-formed: each agent gates via the MCP tool (not bare CLI), carries the correct frontmatter,
budgets enough turns to finish the workflow it mandates, and the skill documents the escalation
ladder and the Stop-hook nudge."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TEMPLATES = _REPO / "src" / "camas" / "main"

_AGENT_TEMPLATES = (
	("claude_agent_lint_haiku.md", "camas-lint-fixer-haiku", "haiku"),
	("claude_agent_lint_sonnet.md", "camas-lint-fixer-sonnet", "sonnet"),
	("claude_agent_test_fixer.md", "camas-test-fixer", "sonnet"),
)


_MANDATED_TURNS = (
	("claude_agent_lint_haiku.md", 5),
	("claude_agent_lint_sonnet.md", 5),
	("claude_agent_test_fixer.md", 7),
)
"""``(template, the longest chain of turns its own steps mandate)``: the conditional
``camas_gate``, the ``Read`` that Edit's contract forces before an edit, the edit, ``camas_fix``,
and the final report — plus, for the test tier, its closing re-gate and a second read, since its
step 2 diagnoses across the failing test *and* the source, so the file it must edit is then not the
file it read. Each step depends on the previous one's result, so each costs an assistant turn.

``maxTurns`` caps those turns and stops the agent before its final message: a budget *below* this
count hands the delegating agent no report at all, and one exactly equal to it completes with no
turn to spare for a second file or an edit retried on a non-unique match."""


def _max_turns(filename: str) -> int:
	declared = [
		line.removeprefix("maxTurns:")
		for line in (_TEMPLATES / filename).read_text().splitlines()
		if line.startswith("maxTurns:")
	]
	assert len(declared) == 1, f"{filename} declares {len(declared)} maxTurns lines, want exactly 1"
	return int(declared[0])


@pytest.mark.parametrize(("filename", "mandated"), _MANDATED_TURNS)
def test_agent_template_budgets_more_turns_than_its_own_steps_mandate(
	filename: str, mandated: int
) -> None:
	assert _max_turns(filename) > mandated


def test_every_shipped_agent_template_has_a_mandated_turn_count() -> None:
	assert sorted(f for f, _ in _MANDATED_TURNS) == sorted(f for f, _, _ in _AGENT_TEMPLATES)


def test_templates_mandating_the_same_chain_budget_the_same_turns() -> None:
	"""The asymmetry #275 reported, stated as the invariant rather than as a pair of tiers: two
	templates whose steps mandate the same chain have the same workflow to finish, so a difference
	in budget is a difference in nothing. Derived from ``_MANDATED_TURNS``, so it covers whatever
	tiers exist rather than the two that existed when it was written.
	"""
	budgets = {
		mandated: sorted({_max_turns(f) for f, m in _MANDATED_TURNS if m == mandated})
		for _, mandated in _MANDATED_TURNS
	}
	assert all(len(seen) == 1 for seen in budgets.values()), budgets


@pytest.mark.parametrize(("filename", "name", "model"), _AGENT_TEMPLATES)
def test_agent_template_gates_via_mcp_tool_not_bare_cli(
	filename: str, name: str, model: str
) -> None:
	agent = (_TEMPLATES / filename).read_text()
	assert "camas_gate" in agent
	assert "camas mcp gate" not in agent


@pytest.mark.parametrize(("filename", "name", "model"), _AGENT_TEMPLATES)
def test_agent_template_has_correct_frontmatter(filename: str, name: str, model: str) -> None:
	agent = (_TEMPLATES / filename).read_text()
	assert f"name: {name}" in agent
	assert f"model: {model}" in agent
	assert "tools: Read, Edit, mcp__camas__camas_gate, mcp__camas__camas_fix" in agent


def test_skill_template_has_correct_frontmatter_name() -> None:
	skill = (_TEMPLATES / "claude_gate_skill.md").read_text()
	assert "name: gate" in skill


def test_skill_template_documents_the_escalation_ladder_and_stop_nudge() -> None:
	skill = (_TEMPLATES / "claude_gate_skill.md").read_text()
	assert "camas-lint-fixer-haiku" in skill
	assert "camas-lint-fixer-sonnet" in skill
	assert "camas-test-fixer" in skill
	assert "Stop" in skill
