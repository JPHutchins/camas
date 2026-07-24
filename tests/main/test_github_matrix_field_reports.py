# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Regression tests keyed to the alpha-migration findings the #258 umbrella grouped.

Each test encodes the shape a real migration actually hit — the tree from the issue, not a
minimal stand-in — so the loop on that finding is provably closed and stays closed. The generic
behaviour of each mechanism lives in ``test_github_matrix.py`` and ``tests/core/test_matrix.py``;
this module is the field report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from camas import Config, Parallel, Sequential, Task
from camas.main.dispatch import dispatch
from camas.main.github_matrix import declared_cells, to_matrix_object
from camas.main.state import LoadOk

if TYPE_CHECKING:
	from collections.abc import Mapping

	from camas.v0.effect import Effect
	from camas.v0.task import TaskNode


def _state(tasks: Mapping[str, TaskNode], config: Config | None = None) -> LoadOk:
	effects: dict[str, type[Effect[Any]]] = {}
	return LoadOk(tasks=dict(tasks), source=None, scope_effects=effects, config=config)


def _crate(name: str, *args: str) -> Sequential:
	"""The CosmWasm migration's per-crate helper, verbatim from #239."""
	return Sequential(*(Task(f"cargo {a} -p {name}") for a in args), name=name)


def test_237_mixed_parallel_is_rejected_not_partially_emitted() -> None:
	"""The reported tree: a matrixed leaf beside a plain one. Emitting the matrixed axis alone
	drops the plain leaf or runs it redundantly in every cell, so it must be rejected.
	"""
	mixed = Parallel(
		Parallel(Task("echo {X}"), matrix={"X": ("a", "b")}, name="matrixed"),
		Task("echo plain", name="plain"),
		name="mixed",
	)
	with pytest.raises(ValueError, match=r"does not cover every leaf \(plain\)"):
		to_matrix_object(mixed)


def test_237_mixed_parallel_exits_non_zero(capsys: pytest.CaptureFixture[str]) -> None:
	"""#237's headline: it exited 0. The CLI must exit 2 with the reason on stderr."""
	mixed = Parallel(
		Parallel(Task("echo {X}"), matrix={"X": ("a", "b")}, name="matrixed"),
		Task("echo plain", name="plain"),
	)
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"mixed": mixed}), ["mixed", "--github-matrix"])
	assert "does not cover every leaf" in capsys.readouterr().err


def test_255_empty_axis_lists_as_required(capsys: pytest.CaptureFixture[str]) -> None:
	"""``matrix={"version": ()}`` used to raise IndexError in ``format_axis``; it now renders as
	a required input.
	"""
	release = Sequential(Task("release {version}"), matrix={"version": ()})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"release": release}), ["--list"])
	assert "version=required" in capsys.readouterr().out


def test_255_empty_axis_run_errors_instead_of_doing_nothing(
	capsys: pytest.CaptureFixture[str],
) -> None:
	release = Sequential(Task("release {version}"), matrix={"version": ()})
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"release": release}), ["release"])
	assert "required but unset" in capsys.readouterr().err


def test_255_empty_axis_emission_errors() -> None:
	with pytest.raises(ValueError, match=r"'version' has no values"):
		to_matrix_object(Sequential(Task("release {version}"), matrix={"version": ()}))


def test_266_user_env_key_named_like_an_axis_adds_no_phantom_cell() -> None:
	"""The emitted cells come from the declarations, so a hand-set ``env={"PY": ...}`` — which
	matrix expansion writes the same way — cannot contribute a value camas never runs.
	"""
	task = Sequential(
		Parallel(Task("test"), Task("lint", env={"PY": "hand-set"})),
		matrix={"PY": ("3.13",)},
	)
	assert declared_cells(task).cells == ({"PY": "3.13"},)
	assert to_matrix_object(task) == {"PY": ["3.13"]}


def test_266_user_env_key_does_not_pass_the_coverage_guard() -> None:
	"""The same inference backed #237's guard: a plain leaf that merely *sets* an axis-named env
	var used to look covered. It is uncovered, and rejected.
	"""
	task = Parallel(
		Parallel(Task("test {PY}"), matrix={"PY": ("3.13",)}),
		Task("lint", name="lint", env={"PY": "hand-set"}),
	)
	with pytest.raises(ValueError, match=r"does not cover every leaf \(lint\)"):
		to_matrix_object(task)


def _openmeter() -> tuple[dict[str, TaskNode], TaskNode]:
	"""OpenMeter's shape: one binding per job, composed into a matrix-less top-level Parallel."""
	jobs = {
		name: Task(f"make {name}")
		for name in ("build", "check", "migrations", "generators", "prek", "test")
	}
	ci = Parallel(*jobs.values())
	return {**jobs, "ci": ci}, ci


def test_234_matrixless_parallel_emits_one_job_per_child() -> None:
	tasks, ci = _openmeter()
	assert to_matrix_object(ci, tasks) == {
		"task": ["build", "check", "migrations", "generators", "prek", "test"]
	}


def test_234_emitted_names_all_dispatch() -> None:
	"""The emitted values are what ``camas <task>`` runs in each job, so each must resolve."""
	tasks, ci = _openmeter()
	emitted = to_matrix_object(ci, tasks)["task"]
	assert all(name in tasks for name in emitted)


def test_234_cli_emits_the_job_list(capsys: pytest.CaptureFixture[str]) -> None:
	tasks, _ = _openmeter()
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state(tasks), ["ci", "--github-matrix"])
	assert '"task":["build","check","migrations","generators","prek","test"]' in (
		capsys.readouterr().out
	)


def test_234_anonymous_child_is_rejected() -> None:
	"""An inline child has no name for a job to dispatch — say so at emit time, not in CI."""
	build = Task("make build")
	with pytest.raises(ValueError, match="not reachable as `camas <name>`"):
		to_matrix_object(Parallel(build, Task("make lint")), {"build": build})


def test_234_sequential_is_rejected() -> None:
	"""Exploding a Sequential into matrix jobs would drop its ordering."""
	build, test = Task("make build"), Task("make test")
	with pytest.raises(ValueError, match="run in order"):
		to_matrix_object(Sequential(build, test), {"build": build, "test": test})


def test_234_parent_state_a_job_would_not_inherit_is_rejected() -> None:
	"""``env=`` on the Parallel is baked into its leaves locally but absent from a per-child job,
	so the fan-out would not reproduce the run — reject rather than emit a lie.
	"""
	build, test = Task("make build"), Task("make test")
	ci = Parallel(build, test, env={"CI": "1"})
	with pytest.raises(ValueError, match="would not inherit"):
		to_matrix_object(ci, {"build": build, "test": test, "ci": ci})


RTIC_VARIANTS = (
	{"backend": "thumbv7", "platform": "lm3s6965", "target": "thumbv7m-none-eabi"},
	{
		"backend": "riscv32-imc-clint",
		"platform": "hifive1",
		"target": "riscv32imc-unknown-none-elf",
	},
	{"backend": "riscv-esp32-c3", "platform": "esp32-c3", "target": "riscv32imc-unknown-none-elf"},
)


def _rtic_qemu() -> Parallel:
	return Parallel(
		Task("cargo run --target {target} -b {backend} --example {platform}"),
		variants=RTIC_VARIANTS,
		name="qemu",
	)


def test_236_coupled_variants_run_only_the_declared_cells() -> None:
	"""Three coupled triples, not backends × platforms × targets: a riscv backend never pairs
	with the lm3s6965 platform.
	"""
	from camas.core.matrix import expand_matrix
	from camas.core.traversal import flatten_leaves

	leaves = flatten_leaves(expand_matrix(_rtic_qemu()))
	assert len(leaves) == len(RTIC_VARIANTS)
	commands = {info.task.cmd for info in leaves}
	assert "cargo run --target thumbv7m-none-eabi -b thumbv7 --example lm3s6965" in commands
	assert not any("riscv" in str(c) and "lm3s6965" in str(c) for c in commands)


def test_236_coupled_variants_emit_as_include_entries() -> None:
	"""The fan-out #17 could not emit at all: it now emits, cell for cell."""
	assert to_matrix_object(_rtic_qemu()) == {"include": [dict(v) for v in RTIC_VARIANTS]}


def test_236_variants_cross_a_profile_axis_without_decoupling() -> None:
	"""RTIC's real matrix is the coupled triples × profiles: 3 × 2 cells, not 3 × 2 × 3."""
	task = Parallel(
		Task("cargo build --target {target} -b {backend} --{PROFILE}"),
		variants=RTIC_VARIANTS,
		matrix={"PROFILE": ("debug", "release")},
	)
	emitted = to_matrix_object(task)
	assert len(emitted["include"]) == len(RTIC_VARIANTS) * 2
	assert {"PROFILE": "debug", **RTIC_VARIANTS[0]} in emitted["include"]
	assert not any(
		cell["backend"] == "thumbv7" and cell["platform"] != "lm3s6965"
		for cell in emitted["include"]
	)


def test_236_pinning_a_coupled_key_selects_its_whole_bundle(
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""A coupled axis can't be *replaced* from the CLI — pinning one key filters to the bundles
	that bind it, so the platform and target travel with the backend.
	"""
	with pytest.raises(SystemExit, match="0"):
		dispatch(
			_state({"qemu": _rtic_qemu()}), ["qemu", "--backend", "riscv-esp32-c3", "--dry-run"]
		)
	out = capsys.readouterr().out
	assert "esp32-c3" in out
	assert "lm3s6965" not in out
	assert "hifive1" not in out


def test_236_pinning_an_undeclared_pair_errors(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="2"):
		dispatch(
			_state({"qemu": _rtic_qemu()}),
			["qemu", "--platform", "lm3s6965", "--backend", "riscv-esp32-c3"],
		)
	assert "no variant matches" in capsys.readouterr().err


def test_239_per_crate_jobs_emit_dispatchable_binding_names() -> None:
	"""No hand-built ``tuple(n.name for n in CRATES if n.name is not None)``: the axis is the
	crates themselves, so nothing needs narrowing from ``str | None``.
	"""
	cosmwasm_core = _crate("cosmwasm-core", "check", "test")
	cosmwasm_crypto = _crate("cosmwasm-crypto", "check", "test")
	tasks = {"cosmwasm_core": cosmwasm_core, "cosmwasm_crypto": cosmwasm_crypto}
	ci = Parallel(cosmwasm_core, cosmwasm_crypto)
	assert to_matrix_object(ci, {**tasks, "ci": ci}) == {
		"task": ["cosmwasm_core", "cosmwasm_crypto"]
	}


def test_239_emits_the_binding_not_the_display_name() -> None:
	"""The reported 404: ``camas`` dispatches on the binding, but the old emitter emitted
	``name=``. A short binding with a long ``name=`` now emits the binding — the value that
	resolves — so CI cannot 404 on a name that only exists as a label.
	"""
	core = _crate("cosmwasm-core", "check")
	ci = Parallel(core)
	emitted = to_matrix_object(ci, {"core": core, "ci": ci})
	assert emitted == {"task": ["core"]}
	assert core.name == "cosmwasm-core"


def test_239_every_emitted_value_resolves_through_dispatch(
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""Round-trip the guarantee: dispatch each emitted value as CI would, and it runs."""
	core = _crate("cosmwasm-core", "check")
	crypto = _crate("cosmwasm-crypto", "check")
	tasks: dict[str, TaskNode] = {"core": core, "crypto": crypto}
	ci = Parallel(core, crypto)
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({**tasks, "ci": ci}), ["ci", "--github-matrix"])
	emitted = capsys.readouterr().out
	for name in tasks:
		with pytest.raises(SystemExit, match="0"):
			dispatch(_state({**tasks, "ci": ci}), [name, "--dry-run"])
		assert f'"{name}"' in emitted


def test_239_crate_jobs_cross_an_os_axis_in_yaml() -> None:
	"""11 crates × 3 OS stays expressible: the ``tasks`` shape is object-of-arrays, so the OS
	axis composes YAML-side (``task: ${{ fromJSON(...).task }}``) with no camas-side change.
	"""
	crates = tuple(_crate(f"crate-{i}", "check") for i in range(11))
	tasks: dict[str, TaskNode] = {f"crate_{i}": c for i, c in enumerate(crates)}
	assert to_matrix_object(Parallel(*crates), tasks) == {"task": list(tasks)}


def test_239_check_warns_when_a_fanned_out_name_does_not_dispatch(
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""#239's residual guard, for anyone keeping the recursive-dispatch pattern: a matrix whose
	values are dispatched as `camas {crate}` is checked at author time, not at CI runtime.
	"""
	from camas.main.check import unresolved_dispatch_warnings

	core = _crate("cosmwasm-core", "check")
	fan = Parallel(Task("camas {crate}"), matrix={"crate": ("core", "cosmwasm-core")})
	warnings = unresolved_dispatch_warnings({"core": core, "fan": fan})
	assert len(warnings) == 1
	assert "camas cosmwasm-core" in warnings[0]
	assert "no task named 'cosmwasm-core'" in warnings[0]
