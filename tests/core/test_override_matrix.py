# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import pytest

from camas import Parallel, Sequential, Task
from camas.core.matrix import matrix_axes, override_matrix, unfilled_required_axes
from camas.main.argv import parse_matrix_kv


def test_matrix_axes_empty_for_task() -> None:
	assert matrix_axes(Task("hi")) == {}


def test_matrix_axes_single_axis() -> None:
	t = Parallel(Task("x"), matrix={"PY": ("3.12", "3.13")})
	assert matrix_axes(t) == {"PY": ("3.12", "3.13")}


def test_matrix_axes_collects_from_nested_tree() -> None:
	t = Sequential(
		Parallel(Task("a"), matrix={"OS": ("linux", "macos")}),
		Task("b"),
		matrix={"PY": ("3.12",)},
	)
	axes = matrix_axes(t)
	assert axes == {"PY": ("3.12",), "OS": ("linux", "macos")}


def test_matrix_axes_outermost_wins_on_duplicate_key() -> None:
	t = Sequential(
		Parallel(Task("a"), matrix={"PY": ("3.12",)}),
		matrix={"PY": ("3.13", "3.14")},
	)
	assert matrix_axes(t) == {"PY": ("3.13", "3.14")}


def test_unfilled_required_axes_none_for_task() -> None:
	assert unfilled_required_axes(Task("hi")) == ()


def test_unfilled_required_axes_none_when_filled() -> None:
	assert unfilled_required_axes(Parallel(Task("t"), matrix={"PY": ("3.13",)})) == ()


def test_unfilled_required_axes_reports_empty_axis() -> None:
	assert unfilled_required_axes(Parallel(Task("t"), matrix={"version": ()})) == ("version",)


def test_unfilled_required_axes_mixed_filled_and_empty() -> None:
	t = Sequential(Task("t"), matrix={"PY": ("3.13",), "version": ()})
	assert unfilled_required_axes(t) == ("version",)


def test_unfilled_required_axes_empty_after_override_fills_it() -> None:
	t = Parallel(Task("t"), matrix={"version": ()})
	assert unfilled_required_axes(override_matrix(t, {"version": ("1.2.3",)})) == ()


def test_unfilled_required_axes_finds_inner_axis_masked_by_outer() -> None:
	"""matrix_axes' outermost-wins merge would hide the inner empty PY behind the outer filled
	PY, but the inner node still expands to zero leaves — the whole-tree walk catches it."""
	t = Sequential(Parallel(Task("t"), matrix={"PY": ()}), matrix={"PY": ("3.13",)})
	assert unfilled_required_axes(t) == ("PY",)


def test_override_matrix_replaces_top_level() -> None:
	t = Parallel(Task("x"), matrix={"PY": ("3.12", "3.13", "3.14")})
	result = override_matrix(t, {"PY": ("3.13",)})
	assert isinstance(result, Parallel)
	assert result.matrix == {"PY": ("3.13",)}


def test_override_matrix_replaces_at_every_node() -> None:
	t = Sequential(
		Parallel(Task("a"), matrix={"PY": ("3.12", "3.13")}),
		matrix={"PY": ("3.12", "3.13", "3.14")},
	)
	result = override_matrix(t, {"PY": ("3.13",)})
	assert isinstance(result, Sequential)
	assert result.matrix == {"PY": ("3.13",)}
	inner = result.tasks[0]
	assert isinstance(inner, Parallel)
	assert inner.matrix == {"PY": ("3.13",)}


def test_override_matrix_leaves_unrelated_axes_alone() -> None:
	t = Parallel(Task("x"), matrix={"PY": ("3.12",), "DB": ("sqlite",)})
	result = override_matrix(t, {"PY": ("3.13",)})
	assert isinstance(result, Parallel)
	assert result.matrix == {"PY": ("3.13",), "DB": ("sqlite",)}


def test_override_matrix_unknown_axis_raises() -> None:
	t = Parallel(Task("x"), matrix={"PY": ("3.12",)})
	with pytest.raises(ValueError, match="unknown matrix axis 'XX'"):
		override_matrix(t, {"XX": ("1",)})


def test_override_matrix_no_overrides_returns_input() -> None:
	t = Parallel(Task("x"), matrix={"PY": ("3.12",)})
	assert override_matrix(t, {}) is t


def test_override_matrix_permissive_on_values() -> None:
	t = Parallel(Task("x"), matrix={"PY": ("3.12", "3.13")})
	result = override_matrix(t, {"PY": ("3.99",)})
	assert isinstance(result, Parallel)
	assert result.matrix == {"PY": ("3.99",)}


def test_override_matrix_walks_through_node_without_matrix() -> None:
	inner = Parallel(Task("x"), matrix={"PY": ("3.12", "3.13")})
	tree = Sequential(inner)
	result = override_matrix(tree, {"PY": ("3.99",)})
	assert isinstance(result, Sequential)
	assert result.matrix is None
	assert isinstance(result.tasks[0], Parallel)
	assert result.tasks[0].matrix == {"PY": ("3.99",)}


def test_parse_matrix_kv_missing_equals() -> None:
	with pytest.raises(ValueError, match="--matrix expects KEY=VAL"):
		parse_matrix_kv("PY")


def test_parse_matrix_kv_empty_key() -> None:
	with pytest.raises(ValueError, match="--matrix expects KEY=VAL"):
		parse_matrix_kv("=3.13")


def test_parse_matrix_kv_empty_values() -> None:
	with pytest.raises(ValueError, match="at least one value required"):
		parse_matrix_kv("PY=")
