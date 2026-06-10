# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from camas import Parallel, Sequential, Task
from camas.core.effect import Effect
from camas.core.task import TaskNode
from camas.main.dispatch import dispatch
from camas.main.github_matrix import emit, format_matrix_json, to_matrix_object
from camas.main.parser import RESERVED_DESTS, RESERVED_FLAGS, build_parser
from camas.main.state import LoadOk

jsonschema = pytest.importorskip("jsonschema")
"""Python 3.15 currently cannot install ``jsonschema`` because its transitive
``rpds-py`` dependency relies on a PyO3 release that doesn't yet support 3.15.
Schema-validation tests skip cleanly on those interpreters instead of erroring
at import."""

SCHEMA_PATH = Path(__file__).parent.parent / "fixtures" / "github-actions-matrix-schema.json"


@pytest.fixture(scope="module")
def matrix_schema() -> Mapping[str, Any]:
	"""GHA matrix subschema (vendored from SchemaStore github-workflow.json).

	``json.loads`` is inherently dynamic; the ``assert isinstance`` is the
	static-type guarantee that callers can rely on a dict-shaped payload.
	"""
	schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())
	assert isinstance(schema, dict), f"{SCHEMA_PATH}: schema fixture must be a JSON object"
	return schema


def make_state(tasks: Mapping[str, TaskNode]) -> LoadOk:
	loaded: dict[str, TaskNode] = dict(tasks)
	effects: dict[str, type[Effect[Any]]] = {}
	return LoadOk(tasks=loaded, source=None, scope_effects=effects)


def test_to_matrix_object_single_axis() -> None:
	task = Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")})
	assert to_matrix_object(task) == {"PY": ["3.12", "3.13"]}


def test_to_matrix_object_multi_axis_preserves_order() -> None:
	task = Parallel(Task("t"), matrix={"PY": ("3.13",), "OS": ("linux", "macos")})
	result = to_matrix_object(task)
	assert result == {"PY": ["3.13"], "OS": ["linux", "macos"]}
	assert list(result.keys()) == ["PY", "OS"]


def test_to_matrix_object_collects_nested_axes_outermost_wins() -> None:
	"""A Sequential matrix at the root and a Parallel matrix inside both
	contribute axes; on duplicate keys, the outermost (root) wins — the same
	behavior as ``matrix_axes`` for CLI overrides."""
	task = Sequential(
		Parallel(Task("t"), matrix={"PY": ("3.10",)}),
		matrix={"PY": ("3.12", "3.13"), "OS": ("linux",)},
	)
	assert to_matrix_object(task) == {"PY": ["3.12", "3.13"], "OS": ["linux"]}


def test_to_matrix_object_no_matrix_errors() -> None:
	with pytest.raises(ValueError, match="no matrix axes"):
		to_matrix_object(Task("hi"))


def test_to_matrix_object_empty_axis_errors() -> None:
	task = Parallel(Task("t"), matrix={"PY": ()})
	with pytest.raises(ValueError, match=r"'PY' has no values"):
		to_matrix_object(task)


def test_format_compact_has_no_spaces() -> None:
	assert format_matrix_json({"PY": ["3.12", "3.13"]}, pretty=False) == '{"PY":["3.12","3.13"]}'


def test_format_pretty_is_indented_and_multiline() -> None:
	out = format_matrix_json({"PY": ["3.12"]}, pretty=True)
	assert out == '{\n  "PY": [\n    "3.12"\n  ]\n}'


def test_format_compact_is_single_line() -> None:
	out = format_matrix_json({"PY": ["3.10", "3.11"], "OS": ["linux"]}, pretty=False)
	assert "\n" not in out


def test_format_compact_round_trips_through_json() -> None:
	original = {"PY": ["3.10", "3.11"], "OS": ["linux", "macos"]}
	assert json.loads(format_matrix_json(original, pretty=False)) == original


def test_format_pretty_round_trips_through_json() -> None:
	original = {"PY": ["3.10", "3.11"]}
	assert json.loads(format_matrix_json(original, pretty=True)) == original


def test_emit_smoke_compact() -> None:
	task = Parallel(Task("t"), matrix={"PY": ("3.12",)})
	assert emit(task, pretty=False) == '{"PY":["3.12"]}'


def test_emit_smoke_pretty() -> None:
	task = Parallel(Task("t"), matrix={"PY": ("3.12",)})
	assert emit(task, pretty=True) == '{\n  "PY": [\n    "3.12"\n  ]\n}'


@pytest.mark.parametrize(
	"task",
	[
		Parallel(Task("t"), matrix={"PY": ("3.12",)}),
		Parallel(Task("t"), matrix={"PY": ("3.10", "3.11", "3.12", "3.13", "3.14", "3.15")}),
		Parallel(Task("t"), matrix={"PY": ("3.13",), "OS": ("linux", "macos", "windows")}),
		Sequential(
			Parallel(Task("t"), matrix={"PY": ("3.10",)}),
			matrix={"PY": ("3.12", "3.13")},
		),
		Parallel(Task("t"), matrix={"FLAG": ("-- --debug", "")}),
		Parallel(Task("t"), matrix={"VAL": ("with space", "1+2", "kebab-case")}),
	],
	ids=["single", "many", "two-axis", "nested", "shell-quoted", "weird-values"],
)
def test_emitted_json_validates_against_schema(
	task: TaskNode, matrix_schema: Mapping[str, Any]
) -> None:
	out = emit(task, pretty=False)
	parsed: dict[str, Any] = json.loads(out)
	assert isinstance(parsed, dict)
	jsonschema.validate(parsed, matrix_schema)


def test_emitted_pretty_json_validates_against_schema(
	matrix_schema: Mapping[str, Any],
) -> None:
	task = Parallel(Task("t"), matrix={"PY": ("3.12", "3.13"), "OS": ("linux",)})
	out = emit(task, pretty=True)
	parsed: dict[str, Any] = json.loads(out)
	assert isinstance(parsed, dict)
	jsonschema.validate(parsed, matrix_schema)


def test_schema_fixture_rejects_empty_axis(matrix_schema: Mapping[str, Any]) -> None:
	"""Sanity check on the vendored fixture: an empty axis array (minItems: 1
	violation) must be rejected. If this test breaks, the fixture is stale or
	the upstream schema has loosened, and our ``to_matrix_object`` guard
	(which rejects empty axes before emission) may need revisiting."""
	with pytest.raises(jsonschema.ValidationError):
		jsonschema.validate({"PY": []}, matrix_schema)


def test_schema_fixture_rejects_empty_object(matrix_schema: Mapping[str, Any]) -> None:
	with pytest.raises(jsonschema.ValidationError):
		jsonschema.validate({}, matrix_schema)


def test_cli_github_matrix_emits_valid_json(
	capsys: pytest.CaptureFixture[str], matrix_schema: Mapping[str, Any]
) -> None:
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(make_state({"check": task}), ["check", "--github-matrix"])
	out = capsys.readouterr().out.strip()
	matrix: dict[str, Any] = json.loads(out)
	assert isinstance(matrix, dict)
	jsonschema.validate(matrix, matrix_schema)
	assert matrix == {"PY": ["3.12", "3.13"]}


def test_cli_github_matrix_compact_when_non_tty(
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""capsys captures to a non-TTY stream, so the compact form is what comes out —
	exactly the shape ``$(camas ... --github-matrix) >> $GITHUB_OUTPUT`` needs."""
	task = Parallel(Task("t"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(make_state({"check": task}), ["check", "--github-matrix"])
	out = capsys.readouterr().out.strip()
	assert out == '{"PY":["3.12","3.13"]}'
	assert "\n" not in out


def test_cli_github_matrix_pretty_when_tty(
	capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
	task = Parallel(Task("t"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="0"):
		dispatch(make_state({"check": task}), ["check", "--github-matrix"])
	out = capsys.readouterr().out
	assert "\n  " in out


def test_cli_github_matrix_applies_override(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.10", "3.11", "3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(make_state({"check": task}), ["check", "--github-matrix", "--PY", "3.13"])
	matrix: dict[str, Any] = json.loads(capsys.readouterr().out)
	assert isinstance(matrix, dict)
	assert matrix == {"PY": ["3.13"]}


def test_cli_github_matrix_applies_override_multi_value(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.10", "3.11", "3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(make_state({"check": task}), ["check", "--github-matrix", "--PY", "3.13,3.14"])
	matrix: dict[str, Any] = json.loads(capsys.readouterr().out)
	assert isinstance(matrix, dict)
	assert matrix == {"PY": ["3.13", "3.14"]}


def test_cli_github_matrix_no_matrix_errors(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Task("echo hi")
	with pytest.raises(SystemExit, match="2"):
		dispatch(make_state({"plain": task}), ["plain", "--github-matrix"])
	err = capsys.readouterr().err
	assert "no matrix axes" in err


def test_cli_dry_run_and_github_matrix_mutually_exclusive(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="2"):
		dispatch(make_state({"check": task}), ["--dry-run", "--github-matrix", "check"])
	assert "not allowed" in capsys.readouterr().err.lower()


def test_cli_github_matrix_rejects_passthrough(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(Task("t {PY}"), matrix={"PY": ("3.12",)})
	with pytest.raises(SystemExit, match="2"):
		dispatch(make_state({"check": task}), ["check", "--github-matrix", "--", "extra"])
	assert "passthrough" in capsys.readouterr().err


def test_parser_github_matrix_flag() -> None:
	args = build_parser().parse_args(["--github-matrix", "check"])
	assert args.github_matrix is True
	assert args.dry_run is False


def test_parser_github_matrix_in_reserved_flags() -> None:
	assert "github-matrix" in RESERVED_FLAGS


def test_reserved_dests_normalizes_hyphens_to_underscores() -> None:
	"""``RESERVED_DESTS`` is what dispatch compares axis names against, so the
	underscore form of every built-in flag must be present — an axis literally
	named ``github_matrix`` shares argparse's auto-derived dest with
	``--github-matrix`` and must be filtered out before registration."""
	assert "github_matrix" in RESERVED_DESTS
	assert "dry_run" in RESERVED_DESTS


def test_parser_mutex_dry_run_and_github_matrix(
	capsys: pytest.CaptureFixture[str],
) -> None:
	parser = build_parser()
	with pytest.raises(SystemExit, match="2"):
		parser.parse_args(["--dry-run", "--github-matrix", "check"])
	assert "not allowed" in capsys.readouterr().err.lower()


def test_dispatch_skips_axis_whose_dest_collides_with_builtin(
	capsys: pytest.CaptureFixture[str],
) -> None:
	"""A matrix axis named ``github_matrix`` normalizes to the same dest as the
	built-in ``--github-matrix`` flag; the axis must be filtered out of CLI
	registration so the two don't overwrite each other in ``args``."""
	task = Parallel(Task("echo {github_matrix}"), matrix={"github_matrix": ("a", "b")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(make_state({"check": task}), ["--dry-run", "check"])
	out = capsys.readouterr().out
	assert "[github_matrix=a]" in out
	assert "[github_matrix=b]" in out


def test_task_help_shows_mutex_when_axes_exist(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Parallel(Task("echo {PY}"), matrix={"PY": ("3.12", "3.13")})
	with pytest.raises(SystemExit, match="0"):
		dispatch(make_state({"check": task}), ["check", "--help"])
	out = capsys.readouterr().out
	assert "[--dry-run | --github-matrix]" in out


def test_task_help_omits_github_matrix_when_no_axes(
	capsys: pytest.CaptureFixture[str],
) -> None:
	task = Task("echo hi")
	with pytest.raises(SystemExit, match="0"):
		dispatch(make_state({"plain": task}), ["plain", "--help"])
	out = capsys.readouterr().out
	assert "--github-matrix" not in out
	assert "[--dry-run]" in out
