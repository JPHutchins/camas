# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Pin the ``camas.v0`` contract: the export list, re-export identity, and the
1:1 relationship with the top-level ``camas`` namespace.

``camas.v0`` is the public surface, semver-zero loose until 1.0 — it may
change, but only on purpose. These tests make that intent mechanical: an
accidental addition, removal, or re-binding fails here and forces a
deliberate edit of ``V0_CONTRACT``. The top-level ``camas`` namespace is
the unversioned alias for the latest generation and must expose the
identical set, bound to the identical objects.
"""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType
from typing import Final

import camas
import camas.v0
from camas.v0 import completion, effect, leaf_state, task, task_event


def public_names(module: object) -> set[str]:
	"""Exported names of a namespace module: public, non-submodule bindings."""
	return {
		name
		for name, val in vars(module).items()
		if not name.startswith("_") and not isinstance(val, ModuleType)
	}


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
	assert public_names(camas.v0) == V0_CONTRACT


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


def test_importing_v0_does_not_load_the_engine() -> None:
	"""``camas.core`` / ``camas.main`` / ``camas.effect`` consume the v0 types,
	never the reverse, so importing ``camas.v0`` must not drag the engine in.
	Pins the one-directional layering the module docstring and README assert.
	"""
	probe = (
		"import sys, camas.v0\n"
		"engine = sorted(\n"
		"    m for m in sys.modules\n"
		"    if m.split('.')[:2] in (['camas', 'core'], ['camas', 'main'], ['camas', 'effect'])\n"
		")\n"
		"print(','.join(engine))\n"
		"raise SystemExit(1 if engine else 0)\n"
	)
	result = subprocess.run(
		[sys.executable, "-c", probe],
		capture_output=True,
		text=True,
		encoding="utf-8",
		check=False,
	)
	assert result.returncode == 0, f"importing camas.v0 loaded the engine: {result.stdout}"


def test_top_level_namespace_is_one_to_one_with_v0() -> None:
	assert public_names(camas) == public_names(camas.v0)
	for name in sorted(V0_CONTRACT):
		assert vars(camas)[name] is vars(camas.v0)[name], name
