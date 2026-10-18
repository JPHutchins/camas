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
	from collections.abc import Callable
	from pathlib import Path


def test_write_renders_the_template_under_a_banner(
	project: Path, template: Callable[[str], None], capsys: pytest.CaptureFixture[str]
) -> None:
	template("# Title\n\nProse.\n\n<!-- GENERATED: fields demo.Thing -->\n\nMore prose.\n")
	assert main([]) == 0
	assert capsys.readouterr().out == "README.md: regenerated from docs/README.md.in\n"
	assert (project / "README.md").read_text(encoding="utf-8") == dedent(
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


def test_a_second_write_changes_nothing(
	project: Path, template: Callable[[str], None], capsys: pytest.CaptureFixture[str]
) -> None:
	template("<!-- GENERATED: fields demo.Thing -->\n")
	main([])
	written = (project / "README.md").read_text(encoding="utf-8")
	capsys.readouterr()
	assert main([]) == 0
	assert (capsys.readouterr().out, (project / "README.md").read_text(encoding="utf-8")) == (
		"",
		written,
	)


def test_check_passes_on_a_current_readme(
	template: Callable[[str], None], capsys: pytest.CaptureFixture[str]
) -> None:
	template("prose\n")
	main([])
	capsys.readouterr()
	assert main(["--check"]) == 0
	assert tuple(capsys.readouterr()) == ("", "")


def test_check_passes_on_a_crlf_checkout(project: Path, template: Callable[[str], None]) -> None:
	template("prose\n<!-- GENERATED: fields demo.Thing -->\n")
	main([])
	readme = project / "README.md"
	readme.write_bytes(readme.read_bytes().replace(b"\n", b"\r\n"))
	assert main(["--check"]) == 0


def test_check_prints_a_diff_and_writes_nothing(
	project: Path, template: Callable[[str], None], capsys: pytest.CaptureFixture[str]
) -> None:
	template("prose\n")
	main([])
	template("new prose\n")
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
	assert (project / "README.md").read_text(encoding="utf-8") == f"{BANNER}\nprose\n"


def test_check_fails_without_a_readme_and_write_creates_it(
	project: Path, template: Callable[[str], None]
) -> None:
	template("prose\n")
	assert main(["--check"]) == 1
	assert not (project / "README.md").exists()
	assert main([]) == 0
	assert (project / "README.md").read_text(encoding="utf-8") == f"{BANNER}\nprose\n"


@pytest.mark.parametrize(
	("text", "error"),
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
		(
			"<!-- GENERATED: function demo.nope.f -->\n",
			"docs/README.md.in:1: no module 'demo.nope'",
		),
		(
			"<!-- GENERATED: function demo.callables.Plain -->\n",
			"docs/README.md.in:1: no function 'Plain' in",
		),
		(
			"<!-- GENERATED: protocol demo.nope.P -->\n",
			"docs/README.md.in:1: no module 'demo.nope'",
		),
		(
			"<!-- GENERATED: protocol demo.callables.bare -->\n",
			"docs/README.md.in:1: no class 'bare' in",
		),
		("<!-- GENERATED: alias demo.nope.A -->\n", "docs/README.md.in:1: no module 'demo.nope'"),
		(
			"<!-- GENERATED: alias demo.callables._DEFAULT -->\n",
			"docs/README.md.in:1: no alias '_DEFAULT' in",
		),
		(
			"<!-- GENERATED: alias demo.callables.Missing -->\n",
			"docs/README.md.in:1: no alias 'Missing' in",
		),
	],
)
def test_a_bad_directive_exits_2_naming_its_line(
	project: Path,
	template: Callable[[str], None],
	capsys: pytest.CaptureFixture[str],
	text: str,
	error: str,
) -> None:
	template(text)
	for argv in ([], ["--check"]):
		assert main(argv) == 2
		assert capsys.readouterr().err.startswith(error)
	assert not (project / "README.md").exists()


def test_every_bad_directive_is_reported(
	template: Callable[[str], None], capsys: pytest.CaptureFixture[str]
) -> None:
	template("<!-- GENERATED: fields demo.Nope -->\nok\n<!-- GENERATED: fields demo.nope.X -->\n")
	assert main(["--check"]) == 2
	assert [line.split(":")[1] for line in capsys.readouterr().err.splitlines()] == ["1", "3"]


@pytest.mark.usefixtures("project")
def test_an_unknown_argument_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
	assert main(["--bogus"]) == 2
	assert capsys.readouterr().err == "usage: python -m docs.readme [--check]\n"


def test_the_module_runs_as_a_script(
	template: Callable[[str], None], monkeypatch: pytest.MonkeyPatch
) -> None:
	template("prose\n")
	monkeypatch.setattr(sys, "argv", ["readme", "--check"])
	with pytest.raises(SystemExit) as exited:
		runpy.run_module("docs.readme.__main__", run_name="__main__")
	assert exited.value.code == 1
