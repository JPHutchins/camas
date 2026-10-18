# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The `fields <module>.<Class>` directive: a field table, then the field docs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable


def test_named_tuple_rows_follow_the_fields(generated: Callable[[str], tuple[str, ...]]) -> None:
	assert generated("fields demo.shapes.Options") == (
		"| Field | Type | Default |",
		"|---|---|---|",
		"| `path` | `str` | required |",
		"| `mode` | `int \\| None` | `None` |",
		'| `directory` | `str` | `".x"` |',
		"| `limit` | `int` | `8_000` |",
		"",
		"<details><summary><code>Options</code> fields</summary>",
		"",
		"- `path`: Where the output goes.",
		"",
		"  A second paragraph.",
		"",
		"</details>",
	)


def test_init_rows_follow_the_parameters_with_inherited_docs(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("fields demo.shapes.Child") == (
		"| Field | Type | Default |",
		"|---|---|---|",
		"| `*items` | `str` → `tuple[str, ...]` |  |",
		"| `extra` | `bool` | `False` |",
		"| `label` | `str \\| None` | `None` |",
		"",
		"<details><summary><code>Child</code> fields</summary>",
		"",
		"- `*items`: The `a | b` items.",
		"- `extra`: An extra.",
		"",
		"</details>",
	)


def test_a_class_without_init_or_fields_inherits_both(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	child = generated("fields demo.shapes.Child")
	assert generated("fields demo.shapes.Grandchild") == tuple(
		line.replace("<code>Child</code>", "<code>Grandchild</code>") for line in child
	)


def test_undocumented_fields_get_no_details(generated: Callable[[str], tuple[str, ...]]) -> None:
	assert generated("fields demo.shapes.Leaf") == (
		"| Field | Type | Default |",
		"|---|---|---|",
		"| `cwd` | `str \\| Path \\| None` → `Path \\| None` | `None` |",
		"| `env` | `Mapping[str, str]` | `MappingProxyType({})` |",
	)
