# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from camas import Parallel, Sequential, Task
from camas.main.dispatch import dispatch
from camas.main.github_matrix import (
	Fanout,
	declared_cells,
	emit,
	format_matrix_json,
	is_cross_product,
	to_matrix_object,
)
from camas.main.parser import (
	RESERVED_DESTS,
	RESERVED_FLAGS,
	build_parser,
	normalize_github_matrix,
)
from camas.main.state import LoadOk

if TYPE_CHECKING:
	from collections.abc import Mapping
	from types import ModuleType

	from camas.v0.config import Config
	from camas.v0.effect import Effect
	from camas.v0.task import TaskNode


SCHEMA_PATH = Path(__file__).parent.parent / "fixtures" / "github-actions-matrix-schema.json"


@pytest.fixture(scope="module")
def jsonschema() -> ModuleType:
	"""Python 3.15 can't yet install ``jsonschema`` (its transitive ``rpds-py`` needs a PyO3
	release without 3.15 support), so importing it in a fixture skips only the tests that
	request it, not the whole module.
	"""
	return cast("ModuleType", pytest.importorskip("jsonschema"))


@pytest.fixture(scope="module")
def matrix_schema() -> Mapping[str, Any]:
	"""GHA matrix subschema vendored from SchemaStore github-workflow.json."""
	schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())
	assert isinstance(schema, dict), f"{SCHEMA_PATH}: schema fixture must be a JSON object"
	return schema


def _state(tasks: Mapping[str, TaskNode], config: Config | None = None) -> LoadOk:
	effects: dict[str, type[Effect[Any]]] = {}
	return LoadOk(tasks=dict(tasks), source=None, scope_effects=effects, config=config)


def test_declared_cells_single_axis() -> None:
	task = Parallel(Task("t"), matrix={"PY": ("3.12", "3.13")})
	assert declared_cells(task).cells == ({"PY": "3.12"}, {"PY": "3.13"})


def test_declared_cells_no_matrix_is_the_uncovered_leaf() -> None:
	assert declared_cells(Task("t")) == Fanout((), ("t",))


def test_declared_cells_full_two_axis_product() -> None:
	task = Parallel(Task("t"), matrix={"PY": ("3.12", "3.13"), "PROFILE": ("debug", "release")})
	assert declared_cells(task).cells == (
		{"PY": "3.12", "PROFILE": "debug"},
		{"PY": "3.12", "PROFILE": "release"},
		{"PY": "3.13", "PROFILE": "debug"},
		{"PY": "3.13", "PROFILE": "release"},
	)


def test_declared_cells_nested_distinct_axes_is_rectangular() -> None:
	task = Sequential(
		Parallel(Task("t"), matrix={"PROFILE": ("debug", "release")}),
		matrix={"PY": ("3.12", "3.13")},
	)
	assert declared_cells(task).cells == (
		{"PY": "3.12", "PROFILE": "debug"},
		{"PY": "3.12", "PROFILE": "release"},
		{"PY": "3.13", "PROFILE": "debug"},
		{"PY": "3.13", "PROFILE": "release"},
	)


def test_declared_cells_heterogeneous_keeps_only_declared_cells() -> None:
	task = Parallel(
		Parallel(Task("t"), matrix={"PROFILE": ("release",), "PY": ("3.13",)}),
		Parallel(Task("t"), matrix={"PROFILE": ("debug",), "PY": ("3.12", "3.13")}),
	)
	assert declared_cells(task).cells == (
		{"PROFILE": "release", "PY": "3.13"},
		{"PROFILE": "debug", "PY": "3.12"},
		{"PROFILE": "debug", "PY": "3.13"},
	)


def test_declared_cells_dedupes_identical_cells() -> None:
	task = Parallel(
		Parallel(Task("a"), matrix={"PY": ("3.13",)}),
		Parallel(Task("b"), matrix={"PY": ("3.13",)}),
	)
	assert declared_cells(task).cells == ({"PY": "3.13"},)


def test_declared_cells_reports_a_leaf_outside_any_matrix() -> None:
	task = Sequential(
		Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")}),
		Task("lint"),
	)
	assert declared_cells(task) == Fanout(({"PY": "3.12"}, {"PY": "3.13"}), ("lint",))


def test_declared_cells_ignores_a_user_env_key_named_like_an_axis() -> None:
	"""The #266 root fix: cells come from the declarations, never from a leaf's env, so a
	hand-set env={"PY": ...} cannot contribute a phantom axis value."""
	task = Parallel(
		Parallel(Task("test"), matrix={"PY": ("3.12",)}),
		Parallel(Task("lint"), env={"PY": "hand-set"}, matrix={"PY": ("3.12",)}),
	)
	assert declared_cells(task).cells == ({"PY": "3.12"},)


def test_is_cross_product_single_axis() -> None:
	assert is_cross_product(({"PY": "3.12"}, {"PY": "3.13"}), ("PY",)) is True


def test_is_cross_product_full_two_axis() -> None:
	combos = (
		{"PY": "3.12", "PROFILE": "debug"},
		{"PY": "3.12", "PROFILE": "release"},
		{"PY": "3.13", "PROFILE": "debug"},
		{"PY": "3.13", "PROFILE": "release"},
	)
	assert is_cross_product(combos, ("PY", "PROFILE")) is True


def test_is_cross_product_heterogeneous_is_false() -> None:
	combos = ({"PY": "3.12", "PROFILE": "debug"}, {"PY": "3.13", "PROFILE": "release"})
	assert is_cross_product(combos, ("PY", "PROFILE")) is False


def test_is_cross_product_independent_keys_is_false() -> None:
	assert is_cross_product(({"PY": "3.12"}, {"PROFILE": "debug"}), ("PY", "PROFILE")) is False


def test_is_cross_product_empty_is_false() -> None:
	assert is_cross_product((), ("PY",)) is False


def test_to_matrix_object_single_axis() -> None:
	task = Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")})
	assert to_matrix_object(task) == {"PY": ["3.12", "3.13"]}


def test_to_matrix_object_multi_axis_preserves_order() -> None:
	task = Parallel(Task("t"), matrix={"PY": ("3.13",), "PROFILE": ("debug", "release")})
	result = to_matrix_object(task)
	assert result == {"PY": ["3.13"], "PROFILE": ["debug", "release"]}
	assert list(result.keys()) == ["PY", "PROFILE"]


def test_to_matrix_object_nested_distinct_axes_are_rectangular() -> None:
	task = Sequential(
		Parallel(Task("t"), matrix={"PROFILE": ("debug", "release")}),
		matrix={"PY": ("3.12", "3.13")},
	)
	assert to_matrix_object(task) == {"PY": ["3.12", "3.13"], "PROFILE": ["debug", "release"]}


def test_to_matrix_object_no_matrix_errors() -> None:
	with pytest.raises(ValueError, match="no matrix axes"):
		to_matrix_object(Task("hi"))


def test_to_matrix_object_empty_axis_errors() -> None:
	with pytest.raises(ValueError, match=r"'PY' has no values"):
		to_matrix_object(Parallel(Task("t"), matrix={"PY": ()}))


def test_to_matrix_object_heterogeneous_errors() -> None:
	task = Parallel(
		Parallel(Task("t"), matrix={"PROFILE": ("release",), "PY": ("3.13",)}),
		Parallel(Task("t"), matrix={"PROFILE": ("debug",), "PY": ("3.12", "3.13")}),
	)
	with pytest.raises(ValueError, match="not a clean cross-product"):
		to_matrix_object(task)


def test_to_matrix_object_independent_fanouts_error() -> None:
	task = Parallel(
		Parallel(Task("test {PY}"), matrix={"PY": ("3.12", "3.13")}),
		Parallel(Task("lint {TOOLCHAIN}"), matrix={"TOOLCHAIN": ("stable", "nightly")}),
	)
	with pytest.raises(ValueError, match="not a clean cross-product"):
		to_matrix_object(task)


def test_to_matrix_object_plain_leaf_beside_matrix_errors() -> None:
	"""The #237 repro: a plain leaf beside a matrixed one in a Parallel is not covered by the
	emitted axes, so emitting would drop or duplicate it — reject instead of exiting 0.
	"""
	matrixed = Parallel(Task("echo {X}"), matrix={"X": ("a", "b")}, name="matrixed")
	mixed = Parallel(matrixed, Task("echo plain", name="plain"), name="mixed")
	with pytest.raises(ValueError, match=r"does not cover every leaf \(plain\)"):
		to_matrix_object(mixed)


def test_to_matrix_object_sequential_sibling_plain_leaf_errors() -> None:
	"""Same gap in a Sequential: a lint step beside a matrixed Parallel runs once, not per-axis,
	so the emitted PY axis cannot represent it.
	"""
	task = Sequential(Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")}), Task("lint"))
	with pytest.raises(ValueError, match="does not cover every leaf"):
		to_matrix_object(task)


def test_to_matrix_object_outer_matrix_covers_all_leaves() -> None:
	"""A matrix on the outer node bakes the axis into every leaf, including plain steps, so the
	whole run-set is a clean cross-product and emits — the shape the project's own CI emits from.
	"""
	task = Sequential(Task("uv sync"), Task("test"), matrix={"PY": ("3.12", "3.13")})
	assert to_matrix_object(task) == {"PY": ["3.12", "3.13"]}


def test_format_compact_has_no_spaces() -> None:
	assert format_matrix_json({"PY": ["3.12", "3.13"]}, pretty=False) == '{"PY":["3.12","3.13"]}'


def test_format_pretty_is_indented_and_multiline() -> None:
	out = format_matrix_json({"PY": ["3.12"]}, pretty=True)
	assert out == '{\n  "PY": [\n    "3.12"\n  ]\n}'


def test_format_compact_is_single_line() -> None:
	out = format_matrix_json({"PY": ["3.10", "3.11"], "PROFILE": ["debug"]}, pretty=False)
	assert "\n" not in out


def test_format_compact_round_trips_through_json() -> None:
	original = {"PY": ["3.10", "3.11"], "PROFILE": ["debug", "release"]}
	assert json.loads(format_matrix_json(original, pretty=False)) == original


def test_emit_smoke_compact() -> None:
	assert emit(Parallel(Task("t"), matrix={"PY": ("3.12",)}), pretty=False) == '{"PY":["3.12"]}'


def test_emit_smoke_pretty() -> None:
	out = emit(Parallel(Task("t"), matrix={"PY": ("3.12",)}), pretty=True)
	assert out == '{\n  "PY": [\n    "3.12"\n  ]\n}'


@pytest.mark.parametrize(
	"task",
	[
		Parallel(Task("t"), matrix={"PY": ("3.12",)}),
		Parallel(Task("t"), matrix={"PY": ("3.10", "3.11", "3.12", "3.13", "3.14", "3.15")}),
		Parallel(Task("t"), matrix={"PY": ("3.13",), "PROFILE": ("debug", "release")}),
		Sequential(
			Parallel(Task("t"), matrix={"PROFILE": ("debug", "release")}),
			matrix={"PY": ("3.12", "3.13")},
		),
		Parallel(Task("t"), matrix={"FLAG": ("-- --debug", "")}),
		Parallel(Task("t"), matrix={"VAL": ("with space", "1+2", "kebab-case")}),
	],
	ids=["single", "many", "two-axis", "nested", "shell-quoted", "weird-values"],
)
def test_emitted_json_validates_against_schema(
	task: TaskNode, matrix_schema: Mapping[str, Any], jsonschema: ModuleType
) -> None:
	parsed: dict[str, Any] = json.loads(emit(task, pretty=False))
	assert isinstance(parsed, dict)
	jsonschema.validate(parsed, matrix_schema)


def test_emitted_pretty_json_validates_against_schema(
	matrix_schema: Mapping[str, Any], jsonschema: ModuleType
) -> None:
	task = Parallel(Task("t"), matrix={"PY": ("3.12", "3.13"), "PROFILE": ("debug",)})
	parsed: dict[str, Any] = json.loads(emit(task, pretty=True))
	assert isinstance(parsed, dict)
	jsonschema.validate(parsed, matrix_schema)


def test_schema_fixture_rejects_empty_axis(
	matrix_schema: Mapping[str, Any], jsonschema: ModuleType
) -> None:
	"""Guards fixture drift: an empty axis array (minItems: 1 violation) must be rejected, since
	:func:`to_matrix_object` relies on the schema to reject what its own empty-axis guard blocks.
	"""
	with pytest.raises(jsonschema.ValidationError):
		jsonschema.validate({"PY": []}, matrix_schema)


def test_schema_fixture_rejects_empty_object(
	matrix_schema: Mapping[str, Any], jsonschema: ModuleType
) -> None:
	with pytest.raises(jsonschema.ValidationError):
		jsonschema.validate({}, matrix_schema)


def test_cli_github_matrix_emits_valid_json(
	capsys: pytest.CaptureFixture[str], matrix_schema: Mapping[str, Any], jsonschema: ModuleType
) -> None:
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["check", "--github-matrix"])
	matrix: dict[str, Any] = json.loads(capsys.readouterr().out)
	jsonschema.validate(matrix, matrix_schema)
	assert matrix == {"PY": ["3.12", "3.13"]}


def test_cli_github_matrix_compact_when_non_tty(capsys: pytest.CaptureFixture[str]) -> None:
	"""capsys captures a non-TTY stream, so the compact form is what comes out — the shape
	``$(camas ... --github-matrix) >> $GITHUB_OUTPUT`` needs.
	"""
	task = Parallel(Task("t"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["check", "--github-matrix"])
	out = capsys.readouterr().out.strip()
	assert out == '{"PY":["3.12","3.13"]}'
	assert "\n" not in out


def test_cli_github_matrix_pretty_when_tty(
	capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
	task = Parallel(Task("t"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["check", "--github-matrix"])
	assert "\n  " in capsys.readouterr().out


def test_cli_github_matrix_applies_override(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.10", "3.11", "3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["check", "--github-matrix", "--PY", "3.13"])
	assert json.loads(capsys.readouterr().out) == {"PY": ["3.13"]}


def test_cli_github_matrix_applies_override_multi_value(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.10", "3.11", "3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["check", "--github-matrix", "--PY", "3.13,3.14"])
	assert json.loads(capsys.readouterr().out) == {"PY": ["3.13", "3.14"]}


def test_cli_github_matrix_no_matrix_errors(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"plain": Task("echo hi")}), ["plain", "--github-matrix"])
	assert "no matrix axes" in capsys.readouterr().err


def test_cli_github_matrix_heterogeneous_errors(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(
		Parallel(Task("t"), matrix={"PROFILE": ("release",), "PY": ("3.13",)}),
		Parallel(Task("t"), matrix={"PROFILE": ("debug",), "PY": ("3.12", "3.13")}),
	)
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"check": task}), ["check", "--github-matrix"])
	assert "not a clean cross-product" in capsys.readouterr().err


def test_cli_github_matrix_mixed_leaf_errors(capsys: pytest.CaptureFixture[str]) -> None:
	"""The #237 repro end-to-end: `camas mixed --github-matrix` exits 2 with a naming error
	instead of emitting a partial axis and exiting 0.
	"""
	matrixed = Parallel(Task("echo {X}"), matrix={"X": ("a", "b")}, name="matrixed")
	mixed = Parallel(matrixed, Task("echo plain", name="plain"), name="mixed")
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"mixed": mixed}), ["mixed", "--github-matrix"])
	assert "does not cover every leaf" in capsys.readouterr().err


def test_cli_dry_run_and_github_matrix_mutually_exclusive(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"check": task}), ["--dry-run", "--github-matrix", "check"])
	assert "not allowed" in capsys.readouterr().err.lower()


def test_cli_github_matrix_ignores_run_only_flags(capsys: pytest.CaptureFixture[str]) -> None:
	"""``--paths``/``--under``/``--jobs`` and ``--`` passthrough modify a run; ``--github-matrix``
	never runs, so it emits the full matrix and ignores them rather than erroring.
	"""
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.12", "3.13")})
	argv = ["check", "--github-matrix", "--paths", "x.py", "--jobs", "2", "--", "extra"]
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), argv)
	assert json.loads(capsys.readouterr().out) == {"PY": ["3.12", "3.13"]}


def test_cli_github_matrix_emits_before_effects_resolution(
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""A bad ``--effects`` must not block matrix emission, which never uses effects."""
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["check", "--github-matrix", "--effects", "(Bogus())"])
	assert json.loads(capsys.readouterr().out) == {"PY": ["3.12"]}


def test_parser_github_matrix_flag() -> None:
	"""The bare flag before the task keeps the task: dispatch normalizes the optional value so
	argparse cannot swallow ``check`` as a shape (:func:`normalize_github_matrix`)."""
	args = build_parser().parse_args(normalize_github_matrix(["--github-matrix", "check"]))
	assert args.github_matrix == "auto"
	assert args.expression == "check"
	assert args.dry_run is False


def test_parser_github_matrix_pins_a_shape() -> None:
	for spelling in (["--github-matrix", "tasks"], ["--github-matrix=tasks"]):
		args = build_parser().parse_args(normalize_github_matrix(spelling))
		assert args.github_matrix == "tasks"
		assert args.expression is None


def test_parser_github_matrix_rejects_an_unknown_shape() -> None:
	with pytest.raises(SystemExit, match="2"):
		build_parser().parse_args(normalize_github_matrix(["--github-matrix=bogus"]))


def test_parser_github_matrix_in_reserved_flags() -> None:
	assert "github-matrix" in RESERVED_FLAGS


def test_reserved_dests_covers_flag_dests_and_positional() -> None:
	"""dispatch compares axis names against ``RESERVED_DESTS``, so every built-in flag's
	underscore dest plus the positional ``expression`` must be present.
	"""
	assert {"github_matrix", "dry_run", "expression"} <= RESERVED_DESTS


def test_parser_mutex_dry_run_and_github_matrix(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="2"):
		build_parser().parse_args(
			normalize_github_matrix(["--dry-run", "--github-matrix", "check"])
		)
	assert "not allowed" in capsys.readouterr().err.lower()


def test_dispatch_skips_axis_whose_dest_collides_with_builtin(
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""A matrix axis named ``github_matrix`` shares argparse's derived dest with the built-in
	flag; it must be filtered out of ``--AXIS`` registration so neither clobbers the other.
	"""
	task = Parallel(Task("echo {github_matrix}"), matrix={"github_matrix": ("a", "b")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["--dry-run", "check"])
	out = capsys.readouterr().out
	assert "[github_matrix=a]" in out
	assert "[github_matrix=b]" in out


def test_task_help_shows_mutex_when_axes_exist(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["check", "--help"])
	assert "[--dry-run | --github-matrix [SHAPE]]" in capsys.readouterr().out


def test_task_help_omits_github_matrix_when_no_axes(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"plain": Task("echo hi")}), ["plain", "--help"])
	out = capsys.readouterr().out
	assert "--github-matrix" not in out
	assert "[--dry-run]" in out


def test_task_help_filters_reserved_axis_from_flags_and_block(
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""A reserved-dest axis (``dry_run``) is dropped from both the usage line's ``--AXIS`` list
	and the "Matrix axes" override block — matching what dispatch registers — while a normal axis
	(``PY``) stays in both.
	"""
	task = Parallel(Task("echo {PY} {dry_run}"), matrix={"PY": ("3.12",), "dry_run": ("a", "b")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"check": task}), ["check", "--help"])
	out = capsys.readouterr().out
	assert "--dry_run" not in out
	assert "[--PY VAL[,VAL...]]" in out
	assert "Matrix axes" in out


def test_shape_variants_on_a_matrix_enumerates_cells_as_include() -> None:
	"""A pinned shape wins over the derived one: an axes-only task can still emit include cells."""
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.13", "3.14")})
	assert to_matrix_object(task, {}, "variants") == {"include": [{"PY": "3.13"}, {"PY": "3.14"}]}


def test_shape_axes_on_a_variants_task_points_at_the_variants_shape() -> None:
	task = Parallel(Task("t {b}"), variants=({"b": "x"},))
	with pytest.raises(ValueError, match="--github-matrix=variants"):
		to_matrix_object(task, {}, "axes")


def test_shape_variants_on_a_plain_task_errors() -> None:
	with pytest.raises(ValueError, match="no matrix axes or variants"):
		to_matrix_object(Task("hi"), {}, "variants")


def test_empty_variants_emission_errors() -> None:
	with pytest.raises(ValueError, match="declares no variant"):
		to_matrix_object(Parallel(Task("t"), variants=(), name="qemu"))


def test_sibling_axes_disagreeing_on_values_are_rejected() -> None:
	"""Both siblings declare PY, with different values: object-of-arrays looks clean (the axis
	merges outermost-first), but pinning PY=3.12 would drag the 3.13-only sibling along — the
	latent unfaithfulness the run-set check now catches.
	"""
	task = Parallel(
		Parallel(Task("test {PY}"), matrix={"PY": ("3.12", "3.13")}),
		Parallel(Task("lint {PY}"), matrix={"PY": ("3.13",)}),
	)
	with pytest.raises(ValueError, match=r"not a faithful fan-out.*never runs it"):
		to_matrix_object(task)


def test_duplicate_axis_value_is_rejected() -> None:
	"""A repeated axis value (a duplicated line in a .python-version) runs the leaf twice locally
	but dedupes to one job, so the fan-out is not faithful — named directly rather than left to
	the job-counting fallback.
	"""
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.13", "3.13")})
	with pytest.raises(ValueError, match=r"axis 'PY' repeats a value"):
		to_matrix_object(task)


def test_duplicate_variant_still_emits_because_the_override_filters() -> None:
	"""The asymmetry with a duplicated axis value: pinning a variants key *filters*, so the one
	emitted job still runs both copies — the run-set is preserved and the emission is faithful,
	where replacing a duplicated axis value would have dropped one.
	"""
	task = Parallel(Task("t {b}"), variants=({"b": "x"}, {"b": "x"}))
	assert to_matrix_object(task) == {"include": [{"b": "x"}]}


def test_axis_shadowed_by_an_outer_of_the_same_name_is_rejected() -> None:
	"""A nested axis whose values differ from the outer axis shadowing it: the merged axis emits
	one cell, but pinning it narrows the inner fan-out too, so a leaf would go unrun.
	"""
	task = Sequential(
		Parallel(Task("t {PY}"), matrix={"PY": ("3.13", "3.14")}),
		matrix={"PY": ("3.13",)},
	)
	with pytest.raises(ValueError, match=r"would run in 0 of the 1 emitted job\(s\)"):
		to_matrix_object(task)


def test_variants_beside_an_independent_axis_is_rejected() -> None:
	"""A variants node beside a matrixed sibling: each cell pins only its own node, so every cell
	would also run the other sibling's leaves.
	"""
	task = Parallel(
		Parallel(Task("build {b}"), variants=({"b": "x"},)),
		Parallel(Task("test {PY}"), matrix={"PY": ("3.13",)}),
	)
	with pytest.raises(ValueError, match="not a faithful fan-out"):
		to_matrix_object(task)


def test_cell_that_cannot_be_pinned_alone_is_rejected() -> None:
	"""Two nodes declaring the same coupled key with different bundles: pinning b=y filters the
	second node to nothing, so no single job can run that cell.
	"""
	task = Parallel(
		Parallel(Task("build {b}"), variants=({"b": "x"}, {"b": "y"})),
		Parallel(Task("test {b}"), variants=({"b": "x"},)),
	)
	with pytest.raises(ValueError, match="cannot be pinned on its own"):
		to_matrix_object(task)


def test_dispatch_name_falls_back_to_value_equality() -> None:
	"""A child rebuilt by composition (a Project rebase, an override pass) is value-equal to its
	binding rather than identical — it still dispatches, so it still emits.
	"""
	build = Task("make build")
	twin = Task("make build")
	assert to_matrix_object(Parallel(twin), {"build": build}) == {"task": ["build"]}


def test_cli_pins_a_shape(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("t {b}"), variants=({"b": "x"}, {"b": "y"}))
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"port": task}), ["port", "--github-matrix=variants"])
	assert capsys.readouterr().out.strip() == '{"include":[{"b":"x"},{"b":"y"}]}'


def test_cli_empty_variants_run_errors(capsys: pytest.CaptureFixture[str]) -> None:
	task = Parallel(Task("t"), variants=(), name="qemu")
	with pytest.raises(SystemExit, match="2"):
		dispatch(_state({"qemu": task}), ["qemu"])
	assert "declares no variant" in capsys.readouterr().err


def test_task_help_offers_github_matrix_for_a_matrixless_parallel(
	capsys: pytest.CaptureFixture[str],
) -> None:
	build = Task("make build")
	with pytest.raises(SystemExit, match="0"):
		dispatch(_state({"build": build, "ci": Parallel(build)}), ["ci", "--help"])
	assert "--github-matrix" in capsys.readouterr().out


def test_emitted_variants_json_validates_against_schema(
	jsonschema: ModuleType, matrix_schema: Mapping[str, Any]
) -> None:
	emitted = json.loads(emit(Parallel(Task("t {b}"), variants=({"b": "x"},)), pretty=False))
	jsonschema.validate(emitted, matrix_schema)


def test_emitted_jobs_json_validates_against_schema(
	jsonschema: ModuleType, matrix_schema: Mapping[str, Any]
) -> None:
	build = Task("make build")
	emitted = json.loads(emit(Parallel(build), {"build": build}, pretty=False))
	jsonschema.validate(emitted, matrix_schema)
