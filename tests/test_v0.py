# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Pin the ``camas.v0`` contract: the exact export list and re-export identity.

``camas.v0`` is the public surface, semver-zero loose until 1.0 — it may
change, but only on purpose. These tests make that intent mechanical: an
accidental addition, removal, or re-binding fails here and forces a
deliberate edit of ``V0_CONTRACT``.
"""

from __future__ import annotations

from types import ModuleType
from typing import Final

import camas
import camas.v0
from camas.v0 import completion, effect, leaf_state, task, task_event

V0_CONTRACT: Final = frozenset(
	{
		"Completed",
		"CompletedEvent",
		"Completion",
		"Effect",
		"Finished",
		"Group",
		"LeafState",
		"OutputEvent",
		"Parallel",
		"Running",
		"Sequential",
		"Skipped",
		"StartedEvent",
		"Task",
		"TaskEvent",
		"TaskNode",
		"Waiting",
	}
)


def test_v0_exports_exactly_the_contract() -> None:
	assert {
		name
		for name, val in vars(camas.v0).items()
		if not name.startswith("_") and not isinstance(val, ModuleType)
	} == V0_CONTRACT


def test_v0_reexports_the_defining_modules() -> None:
	defined = {
		**vars(completion),
		**vars(effect),
		**vars(leaf_state),
		**vars(task),
		**vars(task_event),
	}
	for name in sorted(V0_CONTRACT):
		assert vars(camas.v0)[name] is defined[name], name


def test_contract_types_are_defined_in_v0() -> None:
	for name in sorted(V0_CONTRACT):
		obj = vars(camas.v0)[name]
		if isinstance(obj, type):
			assert obj.__module__.startswith("camas.v0."), f"{name}: {obj.__module__}"


def test_top_level_definers_are_v0_objects() -> None:
	assert camas.Task is camas.v0.Task
	assert camas.Sequential is camas.v0.Sequential
	assert camas.Parallel is camas.v0.Parallel
	assert camas.Effect is camas.v0.Effect
