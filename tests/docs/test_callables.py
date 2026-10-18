# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The `function`, `protocol`, and `alias` directives."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable


def test_a_function_renders_its_signature_prose_raises_and_examples(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("function demo.callables.documented") == (
		"```python",
		"def documented(path: str, *, strict: bool = False, fallback: str = make()) -> str",
		"```",
		"",
		"Resolve `path`.",
		"",
		"Between the examples.",
		"",
		"Raises:",
		"",
		"- `ValueError`: on an empty `path`.",
		"- a bare entry",
		"",
		"<details><summary><code>documented</code> examples</summary>",
		"",
		"```pycon",
		'>>> documented("a")',
		"'a'",
		">>> for x in ():",
		"...     pass",
		"```",
		"",
		"```pycon",
		'>>> documented("b", strict=True)',
		"'b'",
		"```",
		"",
		"</details>",
	)


def test_every_parameter_kind_renders_in_the_signature(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("function demo.callables.bare") == (
		"```python",
		"async def bare(a, /, b=1, *args: int, c: int, d: int = 2, **kwargs: str)",
		"```",
	)


def test_a_function_without_examples_gets_no_details(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("function demo.callables.keyword_only") == (
		"```python",
		"def keyword_only(*, flag: bool) -> None",
		"```",
		"",
		"A function without examples.",
	)


def test_a_protocol_renders_its_public_methods_and_their_docs(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("protocol demo.callables.Plugin") == (
		"```python",
		"class Plugin(Protocol):",
		"    def start(self) -> int: ...",
		"    async def stop(self, code: int) -> None: ...",
		"```",
		"",
		"<details><summary><code>Plugin</code> methods</summary>",
		"",
		"- `start`: Begin.",
		"",
		"</details>",
	)


def test_a_class_without_bases_renders_bare(generated: Callable[[str], tuple[str, ...]]) -> None:
	assert generated("protocol demo.callables.Plain") == (
		"```python",
		"class Plain:",
		"    def run(self) -> None: ...",
		"```",
	)


def test_an_alias_renders_its_value_unquoted_and_its_doc(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("alias demo.callables.Scope") == (
		"```python",
		"Scope: TypeAlias = Callable[[str], str]",
		"```",
		"",
		"Maps a path to",
		"another path.",
	)


def test_an_undocumented_alias_renders_its_expression(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("alias demo.callables.Kind") == (
		"```python",
		'Kind: TypeAlias = Literal["a", "b"]',
		"```",
	)
