# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Guards that the tiered camas-fixer agent and gate skill templates shipped in the wheel are
well-formed: each agent gates via the MCP tool (not bare CLI), carries the correct frontmatter,
budgets enough turns to finish the workflow it mandates, and the skill documents the escalation
ladder and the Stop-hook nudge."""

from __future__ import annotations

from itertools import takewhile
from pathlib import Path

import pytest

from camas.mcp.scaffold import AGENT_TEMPLATES

_REPO = Path(__file__).resolve().parents[2]
_TEMPLATES = _REPO / "src" / "camas" / "main"

_GUARDED = (
	("claude_agent_lint_haiku.md", "camas-lint-fixer-haiku", "haiku", 5),
	("claude_agent_lint_sonnet.md", "camas-lint-fixer-sonnet", "sonnet", 5),
	("claude_agent_test_fixer.md", "camas-test-fixer", "sonnet", 7),
)
"""One row per shipped template, cross-checked against ``AGENT_TEMPLATES`` — the list the wheel
actually writes — so a tier added there cannot land unguarded here."""

_AGENT_TEMPLATES = tuple((source, name, model) for source, name, model, _ in _GUARDED)

_MANDATED_TURNS = tuple((source, mandated) for source, _, _, mandated in _GUARDED)
"""``(template, the longest chain of turns its own steps mandate)``: the conditional
``camas_gate``, the ``Read`` that Edit's contract forces before an edit, the edit, ``camas_fix``,
and the final report — plus, for the test tier, its closing re-gate and a second read, since its
step 2 diagnoses across the failing test *and* the source, so the file it must edit is then not the
file it read. Each step depends on the previous one's result, so each costs an assistant turn.

``maxTurns`` caps those turns and stops the agent before its final message: a budget *below* this
count hands the delegating agent no report at all, and one exactly equal to it completes with no
turn to spare for a second file or an edit retried on a non-unique match."""

_SPARE_ROUND = 2
"""The read and the edit one more file in the scope costs, or a retry of an edit whose
``old_string`` was not unique. A budget has to clear the mandated chain by this much rather than
merely exceed it: one spare turn buys neither, since each needs a read before an edit."""


def _frontmatter(text: str) -> list[str]:
	"""A template's frontmatter lines, so a field is read as a field rather than matched against the
	body prose the agent is meant to follow. Cut by lines rather than by a literal ``\\n---\\n``,
	which a CRLF checkout — the repo has no ``.gitattributes`` pinning the line ending — would not
	contain, silently widening the search back to the whole file.
	"""
	return list(takewhile(lambda line: line != "---", text.splitlines()[1:]))


def _max_turns(filename: str) -> int:
	declared = [
		line.removeprefix("maxTurns:")
		for line in _frontmatter((_TEMPLATES / filename).read_text())
		if line.startswith("maxTurns:")
	]
	assert len(declared) == 1, f"{filename} declares {len(declared)} maxTurns lines, want exactly 1"
	return int(declared[0])


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_frontmatter_cuts_at_the_closing_delimiter_whatever_the_line_ending(newline: str) -> None:
	text = newline.join(("---", "name: x", "maxTurns: 7", "---", "", "maxTurns: not a field"))
	assert _frontmatter(text) == ["name: x", "maxTurns: 7"]


@pytest.mark.parametrize(("filename", "mandated"), _MANDATED_TURNS)
def test_agent_template_budgets_its_mandated_chain_plus_a_spare_round(
	filename: str, mandated: int
) -> None:
	assert _max_turns(filename) >= mandated + _SPARE_ROUND


def test_the_guards_cover_every_agent_template_the_wheel_ships() -> None:
	assert sorted(source for source, *_ in _GUARDED) == sorted(
		source for source, _ in AGENT_TEMPLATES
	)


def test_templates_mandating_the_same_chain_budget_the_same_turns() -> None:
	"""The asymmetry #275 reported, stated as the invariant rather than as a pair of tiers: two
	templates whose steps mandate the same chain have the same workflow to finish, so a difference
	in budget is a difference in nothing. Derived from ``_MANDATED_TURNS``, so it covers whatever
	tiers exist rather than the two that existed when it was written.
	"""
	budgets = {
		mandated: sorted({_max_turns(f) for f, m in _MANDATED_TURNS if m == mandated})
		for mandated in {m for _, m in _MANDATED_TURNS}
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
