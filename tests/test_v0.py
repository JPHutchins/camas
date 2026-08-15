# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import inspect
import subprocess
import sys
from types import ModuleType
from typing import Final

import pytest

import camas
import camas.v0
from camas.v0.completion import Completion, Errored, Finished, Skipped
from camas.v0.config import Agent, Claude, Config
from camas.v0.effect import Effect
from camas.v0.leaf_state import Completed, LeafState, Running, Waiting
from camas.v0.task import GROUP_FIELDS, Group, Parallel, Sequential, Task, TaskNode, rebuilt
from camas.v0.task_event import CompletedEvent, OutputEvent, StartedEvent, TaskEvent

HEADLINE: Final = frozenset(
	{
		"Agent",
		"AgentFormat",
		"Claude",
		"Config",
		"Effect",
		"Parallel",
		"Project",
		"Sequential",
		"Task",
		"by_glob",
		"by_suffix",
	}
)
"""The unversioned definers re-exported by both ``camas`` and ``camas.v0``."""

PUBLIC_API: Final = (
	Completion,
	Errored,
	Finished,
	Skipped,
	Agent,
	Claude,
	Config,
	Effect,
	Completed,
	LeafState,
	Running,
	Waiting,
	Group,
	Parallel,
	Sequential,
	Task,
	TaskNode,
	CompletedEvent,
	OutputEvent,
	StartedEvent,
	TaskEvent,
)
"""Every public type, imported above from its canonical ``camas.v0`` submodule
— this module failing to import is the signal a public type moved or vanished."""


def public_names(module: object) -> set[str]:
	"""Exported names of a namespace package: public, non-submodule bindings."""
	return {
		name
		for name, val in vars(module).items()
		if not name.startswith("_") and not isinstance(val, ModuleType)
	}


def test_top_level_and_v0_expose_the_same_headline() -> None:
	assert public_names(camas) == HEADLINE
	assert public_names(camas.v0) == HEADLINE


def test_headline_names_are_the_same_objects() -> None:
	for name in sorted(HEADLINE):
		assert vars(camas)[name] is vars(camas.v0)[name], name


def test_public_types_are_defined_in_the_version_package() -> None:
	for obj in PUBLIC_API:
		if isinstance(obj, type):
			assert obj.__module__.startswith("camas.v0."), obj


def test_group_fields_track_every_group_constructor_kwarg() -> None:
	"""Drift guard for :func:`camas.v0.task.rebuilt`: a new Group kwarg — on Group itself or on
	either subclass — fails here until ``GROUP_FIELDS`` names it, so every rebuild site carries
	it by construction instead of by hand (the silently-dropped-field bug class #270 kills)."""
	for cls in (Group, Sequential, Parallel):
		params = tuple(inspect.signature(cls).parameters.values())
		keywords = tuple(p.name for p in params if p.kind is p.KEYWORD_ONLY)
		assert keywords == GROUP_FIELDS, cls
		positional = tuple((p.name, p.kind) for p in params if p.kind is not p.KEYWORD_ONLY)
		assert positional == (("tasks", inspect.Parameter.VAR_POSITIONAL),), cls


def test_rebuilt_rejects_unknown_fields() -> None:
	"""A misspelled override fails loudly, as the spelled-out constructors did before."""
	with pytest.raises(TypeError, match="nam"):
		rebuilt(Sequential("a"), Task("b"), nam="x")


def test_run_cli_lazy_export_is_engine_function() -> None:
	"""``run_cli`` is reachable from both namespaces and is the engine's function —
	a lazy ``__getattr__`` export, so it stays out of ``vars()`` and off the
	eager-import path that :func:`test_importing_v0_does_not_load_the_engine` pins."""
	from camas.main.dispatch import run_cli as canonical

	assert camas.run_cli is canonical
	assert camas.v0.run_cli is canonical
	assert "run_cli" not in public_names(camas)


@pytest.mark.parametrize("module", [camas, camas.v0])
def test_unknown_attribute_raises(module: object) -> None:
	missing = "does_not_exist"
	with pytest.raises(AttributeError):
		getattr(module, missing)


def test_importing_v0_does_not_load_the_engine() -> None:
	"""The version namespace defines the public types; the engine consumes them,
	never the reverse — so importing ``camas.v0`` must not pull ``camas.core`` /
	``camas.main`` / ``camas.effect`` in. Pins the one-directional layering.
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
