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
	distinct_combinations,
	emit,
	format_matrix_json,
	is_cross_product,
	matrix_combinations,
	to_matrix_object,
)
from camas.main.parser import RESERVED_DESTS, RESERVED_FLAGS, build_parser
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


def test_matrix_combinations_single_axis() -> None:
	task = Parallel(Task("t"), matrix={"PY": ("3.12", "3.13")})
	assert matrix_combinations(task) == ({"PY": "3.12"}, {"PY": "3.13"})


def test_matrix_combinations_no_matrix_is_empty() -> None:
	assert matrix_combinations(Task("t")) == ()


def test_matrix_combinations_full_two_axis_product() -> None:
	task = Parallel(Task("t"), matrix={"PY": ("3.12", "3.13"), "PROFILE": ("debug", "release")})
	assert matrix_combinations(task) == (
		{"PY": "3.12", "PROFILE": "debug"},
		{"PY": "3.12", "PROFILE": "release"},
		{"PY": "3.13", "PROFILE": "debug"},
		{"PY": "3.13", "PROFILE": "release"},
	)


def test_matrix_combinations_nested_distinct_axes_is_rectangular() -> None:
	task = Sequential(
		Parallel(Task("t"), matrix={"PROFILE": ("debug", "release")}),
		matrix={"PY": ("3.12", "3.13")},
	)
	assert matrix_combinations(task) == (
		{"PY": "3.12", "PROFILE": "debug"},
		{"PY": "3.12", "PROFILE": "release"},
		{"PY": "3.13", "PROFILE": "debug"},
		{"PY": "3.13", "PROFILE": "release"},
	)


def test_matrix_combinations_heterogeneous_keeps_only_real_runs() -> None:
	task = Parallel(
		Parallel(Task("t"), matrix={"PROFILE": ("release",), "PY": ("3.13",)}),
		Parallel(Task("t"), matrix={"PROFILE": ("debug",), "PY": ("3.12", "3.13")}),
	)
	assert matrix_combinations(task) == (
		{"PROFILE": "release", "PY": "3.13"},
		{"PROFILE": "debug", "PY": "3.12"},
		{"PROFILE": "debug", "PY": "3.13"},
	)


def test_matrix_combinations_dedupes_identical_runs() -> None:
	task = Parallel(
		Parallel(Task("a"), matrix={"PY": ("3.13",)}),
		Parallel(Task("b"), matrix={"PY": ("3.13",)}),
	)
	assert matrix_combinations(task) == ({"PY": "3.13"},)


def test_matrix_combinations_ignores_leaf_outside_any_matrix() -> None:
	task = Sequential(
		Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")}),
		Task("lint"),
	)
	assert matrix_combinations(task) == ({"PY": "3.12"}, {"PY": "3.13"})


def test_distinct_combinations_drops_leaf_under_no_axis() -> None:
	"""The extracted dedup core drops a leaf carrying none of the axes (an empty binding); the
	coverage of such a leaf is `to_matrix_object`'s concern, so this stays a pure binding dedup."""
	from camas.core.matrix import expand_matrix
	from camas.core.traversal import flatten_leaves

	tree = Parallel(Parallel(Task("t"), matrix={"PY": ("3.12",)}), Task("plain"))
	leaves = flatten_leaves(expand_matrix(tree))
	assert distinct_combinations(leaves, ("PY",)) == ({"PY": "3.12"},)


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
	args = build_parser().parse_args(["--github-matrix", "check"])
	assert args.github_matrix is True
	assert args.dry_run is False


def test_parser_github_matrix_in_reserved_flags() -> None:
	assert "github-matrix" in RESERVED_FLAGS


def test_reserved_dests_covers_flag_dests_and_positional() -> None:
	"""dispatch compares axis names against ``RESERVED_DESTS``, so every built-in flag's
	underscore dest plus the positional ``expression`` must be present.
	"""
	assert {"github_matrix", "dry_run", "expression"} <= RESERVED_DESTS


def test_parser_mutex_dry_run_and_github_matrix(capsys: pytest.CaptureFixture[str]) -> None:
	with pytest.raises(SystemExit, match="2"):
		build_parser().parse_args(["--dry-run", "--github-matrix", "check"])
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
	assert "[--dry-run | --github-matrix]" in capsys.readouterr().out


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
