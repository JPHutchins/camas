# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The `fields <module>.<Type>` directive: the type's docstring around a table of its fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable


def test_a_named_tuple_renders_its_docstring_around_its_field_table(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("fields demo.shapes.Options") == (
		"Options.",
		"",
		"| `Options` | Type | Default | |",
		"|---|---|---|---|",
		"| `path` | `str` | required | Where the output goes. |",
		"| `mode` | `int \\| None` | `None` |  |",
		'| `directory` | `str` | `".x"` |  |',
		"| `limit` | `int` | `8_000` |  |",
		"",
		"<details><summary><code>Options</code> fields</summary>",
		"",
		"- `path`: Where the output goes.",
		"",
		"  A second paragraph.",
		"",
		"</details>",
		"",
		"<details><summary><code>Options</code> examples</summary>",
		"",
		"```pycon",
		'>>> Options("out").path',
		"'out'",
		"```",
		"",
		"</details>",
	)


def test_init_rows_follow_the_parameters_with_inherited_docs(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("fields demo.shapes.Child") == (
		"| `Child` | Type | Default | |",
		"|---|---|---|---|",
		"| `*items` | `str` → `tuple[str, ...]` |  | The `a \\| b` items. |",
		"| `extra` | `bool` | `False` | An extra. |",
		"| `label` | `str \\| None` | `None` |  |",
	)


def test_a_type_without_init_or_fields_inherits_both(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	child = generated("fields demo.shapes.Child")
	assert generated("fields demo.shapes.Grandchild") == tuple(
		line.replace("`Child`", "`Grandchild`") for line in child
	)


def test_undocumented_fields_get_an_empty_description(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("fields demo.shapes.Leaf") == (
		"| `Leaf` | Type | Default | |",
		"|---|---|---|---|",
		"| `cwd` | `str \\| Path \\| None` → `Path \\| None` | `None` |  |",
		"| `env` | `Mapping[str, str]` | `MappingProxyType({})` |  |",
	)
