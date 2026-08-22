# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import inspect
import subprocess
import sys
from types import MappingProxyType, ModuleType
from typing import Final, cast

import pytest
from typing_extensions import assert_type

import camas
import camas.v0
from camas.v0.completion import Completion, Errored, Finished, Skipped
from camas.v0.config import Agent, Claude, Config
from camas.v0.effect import Effect
from camas.v0.leaf_state import Completed, LeafState, Running, Waiting
from camas.v0.task import (
	GROUP_FIELDS,
	Group,
	Parallel,
	Project,
	Sequential,
	Task,
	TaskNode,
	rebuilt,
)
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


def test_a_fresh_plain_left_operand_adopts_the_right_fields() -> None:
	"""The tie-break's default table, pinned through public behavior: a freshly constructed
	plain group counts as fieldless, so the right operand's fields adopt — a new Group field
	whose stored default is not ``None`` (or, for ``env``, not empty) makes the fresh left
	fieldful and breaks this assert."""
	assert Parallel("a") | Parallel("b", name="n") == Parallel("a", "b", name="n")
	assert Sequential("a") + Sequential("b", name="n") == Sequential("a", "b", name="n")
	assert Parallel("a", env=cast("dict[str, str]", MappingProxyType({}))) | Parallel(
		"b", name="n"
	) == Parallel("a", "b", name="n")


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


def test_or_appends_to_a_parallel() -> None:
	assert Parallel("format", "lint") | "integration" == Parallel("format", "lint", "integration")


def test_operators_assert_their_declared_types() -> None:
	"""#298's acceptance contract: each operator's static return type — ``assert_type`` is a
	runtime no-op enforced by every checker in the CI battery, so this test fails at analysis
	time if an operator's declared type drifts."""
	check = Parallel("format", "lint", "types", "tests")
	assert_type(check | "integration", Parallel)
	assert_type(check + "integration", Sequential)
	assert_type(Task("format") | Task("lint"), Parallel)
	assert_type(Task("build") + Task("test"), Sequential)
	assert_type(Sequential("build") | "lint", Parallel)
	assert_type(Project("libs") | "lint", Parallel)
	assert_type(Project("libs") + "lint", Sequential)


def test_or_builds_a_parallel_from_two_leaves() -> None:
	assert Task("a") | Task("b") == Parallel("a", "b")


def test_or_flattens_a_right_parallel() -> None:
	assert Parallel("a") | Parallel("b", "c") == Parallel("a", "b", "c")


def test_or_nests_a_sequential_as_one_child() -> None:
	seq = Sequential("build", "test")
	assert seq | "lint" == Parallel(seq, Task("lint"))


def test_add_appends_to_a_sequential() -> None:
	assert Sequential("build") + "test" == Sequential("build", "test")


def test_add_flattens_a_right_sequential() -> None:
	assert Sequential("a") + Sequential("b", "c") == Sequential("a", "b", "c")


def test_add_coerces_a_parallel_to_a_sequential() -> None:
	"""``Parallel(...) + integration`` runs the whole group first, then the new node —
	the coercion #298 asks for."""
	check = Parallel("format", "lint")
	assert check + "integration" == Sequential(check, Task("integration"))


def test_composition_is_associative() -> None:
	a, b, c = Task("a"), Task("b"), Task("c")
	assert (a | b) | c == a | (b | c)
	assert (a + b) + c == a + (b + c)
	left_named = Parallel("a", name="n")
	assert (left_named | Parallel("b")) | Parallel("c") == left_named | (
		Parallel("b") | Parallel("c")
	)
	assert (a | b) | Parallel("c", name="n") == a | (b | Parallel("c", name="n"))
	assert (a + b) + Sequential("c", name="n") == a + (b + Sequential("c", name="n"))
	assert (a | b) | Parallel("c", matrix={}) == a | (b | Parallel("c", matrix={}))
	assert (a + b) + Sequential("c", matrix={}) == a + (b + Sequential("c", matrix={}))

	class Named(Parallel):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	class Staged(Sequential):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	assert (a | b) | Named("c") == a | (b | Named("c"))
	assert (a + b) + Staged("c") == a + (b + Staged("c"))


def test_operators_leave_their_operands_unchanged() -> None:
	check = Parallel("format")
	_ = check | "lint"
	_ = check + "integration"
	assert check == Parallel("format")


def test_parallel_operand_carries_its_fields_and_type() -> None:
	class Named(Parallel):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	check = Named("format", name="check", paths=".")
	ci = check | "lint"
	assert type(ci) is Named
	assert ci == Named("format", "lint", name="check", paths=".")


def test_left_parallel_wins_when_both_sides_are_parallels() -> None:
	"""Both sides flatten; the fields of the right-side Parallel have no home, so the
	left's carry — the documented tie-break."""
	assert Parallel("a", name="left") | Parallel("b", name="right") == Parallel(
		"a", "b", name="left"
	)


def test_right_parallel_carries_when_the_left_is_a_leaf() -> None:
	assert Task("a") | Parallel("b", name="check") == Parallel("a", "b", name="check")


def test_sequential_operand_carries_its_fields_and_type() -> None:
	class Staged(Sequential):  # pyrefly: ignore[bad-class-definition]
		__slots__ = ()

	staged = Staged("b", name="stage")
	assert Parallel("a") + staged == Staged(Parallel("a"), "b", name="stage")


def test_composition_rejects_non_nodes_loudly() -> None:
	with pytest.raises(TypeError, match="None"):
		_ = Task("a") | cast("TaskNode", None)


def test_composition_reguards_a_group_operands_children() -> None:
	"""A group built through the lenient constructor can hold a non-node child; composition
	re-checks its children instead of carrying the broken tree along."""
	with pytest.raises(TypeError, match="None"):
		_ = Parallel("a") | Parallel(cast("TaskNode", None))


def test_or_composes_a_project_reference() -> None:
	"""A :class:`Project` reference is a node for composition — the loader resolves it."""
	assert Task("a") | Project("libs") == Parallel(Task("a"), Project("libs"))


def test_project_references_compose_on_the_left() -> None:
	assert Project("libs") | Task("lint") == Parallel(Project("libs"), Task("lint"))
	assert Project("libs") + "lint" == Sequential(Project("libs"), "lint")
