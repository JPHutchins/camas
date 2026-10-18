# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""`python -m docs.readme`: render README.md from docs/README.md.in; `--check` it."""

from __future__ import annotations

import runpy
import sys
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from docs.readme import BANNER, main

if TYPE_CHECKING:
	from pathlib import Path

SHAPES = dedent(
	'''\
	from pathlib import Path
	from types import MappingProxyType
	from typing import Final, Mapping, NamedTuple

	DEFAULT_DIR: Final = ".x"
	_EMPTY: Mapping[str, str] = MappingProxyType({})
	LIMIT = 8_000


	class Options(NamedTuple):
		"""Options."""

		path: str
		"""Where the output goes.

		A second paragraph."""
		mode: int | None = None
		directory: str = DEFAULT_DIR
		limit: int = LIMIT
		_hidden: int = 0


	class Base:
		items: tuple[str, ...]
		"""The `a | b` items."""
		label: str | None

		def __init__(self, *items: str, label: str | None = None) -> None: ...


	class Child(Base):
		extra: bool
		"""An extra."""

		def __init__(self, *items: str, extra: bool = False, label: str | None = None) -> None: ...


	class Grandchild(Child):
		pass


	class Leaf:
		cwd: Path | None
		env: Mapping[str, str]

		def __init__(self, cwd: str | Path | None = None, env: Mapping[str, str] = _EMPTY) -> None: ...
	'''
)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "docs").mkdir()
	(tmp_path / "src" / "demo").mkdir(parents=True)
	(tmp_path / "src" / "demo" / "__init__.py").write_text(
		'class Thing:\n\tsize: int\n\t"""How big."""\n', encoding="utf-8"
	)
	(tmp_path / "src" / "demo" / "shapes.py").write_text(SHAPES, encoding="utf-8")
	return tmp_path


def _template(project: Path, text: str) -> None:
	(project / "docs" / "README.md.in").write_text(text, encoding="utf-8", newline="\n")


def _readme(project: Path) -> str:
	return (project / "README.md").read_text(encoding="utf-8")


def _generated(project: Path, directive: str) -> tuple[str, ...]:
	_template(project, f"<!-- GENERATED: {directive} -->")
	assert main([]) == 0
	return tuple(_readme(project).splitlines()[1:])


def test_write_renders_the_template_under_a_banner(
	project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	_template(
		project, "# Title\n\nProse.\n\n<!-- GENERATED: fields demo.Thing -->\n\nMore prose.\n"
	)
	assert main([]) == 0
	assert capsys.readouterr().out == "README.md: regenerated from docs/README.md.in\n"
	assert _readme(project) == dedent(
		f"""\
		{BANNER}
		# Title

		Prose.

		| Field | Type | Default |
		|---|---|---|
		| `size` | `int` | required |

		<details><summary><code>Thing</code> fields</summary>

		- `size`: How big.

		</details>

		More prose.
		"""
	)


def test_a_second_write_changes_nothing(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
	_template(project, "<!-- GENERATED: fields demo.Thing -->\n")
	main([])
	written = _readme(project)
	capsys.readouterr()
	assert main([]) == 0
	assert (capsys.readouterr().out, _readme(project)) == ("", written)


def test_check_passes_on_a_current_readme(
	project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	_template(project, "prose\n")
	main([])
	capsys.readouterr()
	assert main(["--check"]) == 0
	assert tuple(capsys.readouterr()) == ("", "")


def test_check_passes_on_a_crlf_checkout(project: Path) -> None:
	_template(project, "prose\n<!-- GENERATED: fields demo.Thing -->\n")
	main([])
	readme = project / "README.md"
	readme.write_bytes(readme.read_bytes().replace(b"\n", b"\r\n"))
	assert main(["--check"]) == 0


def test_check_prints_a_diff_and_writes_nothing(
	project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	_template(project, "prose\n")
	main([])
	_template(project, "new prose\n")
	capsys.readouterr()
	assert main(["--check"]) == 1
	out, err = capsys.readouterr()
	assert out.splitlines() == [
		"--- README.md (committed)",
		"+++ README.md (rendered from docs/README.md.in)",
		"@@ -1,2 +1,2 @@",
		f" {BANNER}",
		"-prose",
		"+new prose",
	]
	assert err == "README.md: stale; run `python -m docs.readme`\n"
	assert _readme(project) == f"{BANNER}\nprose\n"


def test_check_fails_without_a_readme_and_write_creates_it(project: Path) -> None:
	_template(project, "prose\n")
	assert main(["--check"]) == 1
	assert not (project / "README.md").exists()
	assert main([]) == 0
	assert _readme(project) == f"{BANNER}\nprose\n"


@pytest.mark.parametrize(
	("template", "error"),
	[
		("x\n<!-- generated: fields demo.Thing -->\n", "docs/README.md.in:2: malformed directive"),
		("<!-- GENERATED: -->\n", "docs/README.md.in:1: malformed directive"),
		(
			"Text <!-- GENERATED: fields demo.Thing -->\n",
			"docs/README.md.in:1: malformed directive",
		),
		(
			"<!-- GENERATED: feilds demo.Thing -->\n",
			"docs/README.md.in:1: unknown directive 'feilds demo.Thing'",
		),
		("<!-- GENERATED: fields demo.Thing extra -->\n", "docs/README.md.in:1: unknown directive"),
		(
			"<!-- GENERATED: fields demo.Nope -->\n",
			"docs/README.md.in:1: no class 'Nope' in src/demo/__init__.py",
		),
		(
			"<!-- GENERATED: fields demo.nope.X -->\n",
			"docs/README.md.in:1: no module 'demo.nope' under src/",
		),
		("<!-- GENERATED: fields Thing -->\n", "docs/README.md.in:1: no module '' under src/"),
	],
)
def test_a_bad_directive_exits_2_naming_its_line(
	project: Path, capsys: pytest.CaptureFixture[str], template: str, error: str
) -> None:
	_template(project, template)
	for argv in ([], ["--check"]):
		assert main(argv) == 2
		assert capsys.readouterr().err.startswith(error)
	assert not (project / "README.md").exists()


def test_every_bad_directive_is_reported(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
	_template(
		project,
		"<!-- GENERATED: fields demo.Nope -->\nok\n<!-- GENERATED: fields demo.nope.X -->\n",
	)
	assert main(["--check"]) == 2
	assert [line.split(":")[1] for line in capsys.readouterr().err.splitlines()] == ["1", "3"]


def test_an_unknown_argument_is_a_usage_error(
	project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
	assert main(["--bogus"]) == 2
	assert capsys.readouterr().err == "usage: python -m docs.readme [--check]\n"


def test_the_module_runs_as_a_script(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_template(project, "prose\n")
	monkeypatch.setattr(sys, "argv", ["readme", "--check"])
	with pytest.raises(SystemExit) as exited:
		runpy.run_module("docs.readme.__main__", run_name="__main__")
	assert exited.value.code == 1


def test_named_tuple_rows_follow_the_fields(project: Path) -> None:
	assert _generated(project, "fields demo.shapes.Options") == (
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


def test_init_rows_follow_the_parameters_with_inherited_docs(project: Path) -> None:
	assert _generated(project, "fields demo.shapes.Child") == (
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


def test_a_class_without_init_or_fields_inherits_both(project: Path) -> None:
	child = _generated(project, "fields demo.shapes.Child")
	assert _generated(project, "fields demo.shapes.Grandchild") == tuple(
		line.replace("<code>Child</code>", "<code>Grandchild</code>") for line in child
	)


def test_undocumented_fields_get_no_details(project: Path) -> None:
	assert _generated(project, "fields demo.shapes.Leaf") == (
		"| Field | Type | Default |",
		"|---|---|---|",
		"| `cwd` | `str \\| Path \\| None` → `Path \\| None` | `None` |",
		"| `env` | `Mapping[str, str]` | `MappingProxyType({})` |",
	)
