# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""A scratch project for `python -m docs.readme`: a `src/demo` package and a template writer."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from docs.readme import main

if TYPE_CHECKING:
	from collections.abc import Callable
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
		"""Options.

		>>> Options("out").path
		'out'
		"""

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

CALLABLES = dedent(
	'''\
	from typing import Protocol, TypeAlias

	_DEFAULT: Final = make()

	Scope: TypeAlias = "Callable[[str], str]"
	"""Maps a path to
	another path."""

	Kind: TypeAlias = Literal["a", "b"]


	def documented(path: str, *, strict: bool = False, fallback: str = _DEFAULT) -> str:
		"""Resolve `path`.

		>>> documented("a")
		'a'
		>>> for x in ():
		...     pass

		Between the examples.

		>>> documented("b", strict=True)
		'b'

		Raises:
			ValueError: on an empty
				`path`.
			a bare entry
		"""


	async def bare(a, /, b=1, *args: int, c: int, d: int = 2, **kwargs: str):
		pass


	def keyword_only(*, flag: bool) -> None:
		"""A function without examples."""


	class Plugin(Protocol):
		"""Not rendered."""

		def start(self) -> int:
			"""Begin."""
			...

		async def stop(self, code: int) -> None: ...

		def _private(self) -> None: ...


	class Plain:
		def run(self) -> None: ...
	'''
)

EXAMPLES = (
	'r"""Examples, parsed and never run.\n'
	"\n"
	"\t>>> def make_camas(tmp): ...\n"
	"\n"
	"One:\n"
	"\n"
	"\t>>> with tempfile.TemporaryDirectory() as tmp:\n"
	"\t...     camas = make_camas(tmp)\n"
	"\t...     _ = (Path(tmp) / 'tasks.py').write_text(dedent('''\\\n"
	"\t...         from camas import Task\n"
	'\t...         hello = Task("echo hi")\n'
	"\t...     '''))\n"
	'\t...     ran = camas("--effects=(Summary(),)", "hello")\n'
	'\t...     listed = camas("--list")\n'
	"\n"
	"Two, with no camas calls:\n"
	"\n"
	"\t>>> _ = dedent('''\\\n"
	"\t...     from camas import Task\n"
	"\t...     class Loud: ...\n"
	'\t...     again = Task("echo again")\n'
	"\t... ''')\n"
	"\n"
	"Three, binding `again` too:\n"
	"\n"
	"\t>>> _ = dedent('''\\\n"
	"\t...     again = 1\n"
	"\t... ''')\n"
	'"""\n'
)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
	monkeypatch.chdir(tmp_path)
	(tmp_path / "docs").mkdir()
	demo = tmp_path / "src" / "demo"
	demo.mkdir(parents=True)
	(demo / "__init__.py").write_text(
		'class Thing:\n\tsize: int\n\t"""How big."""\n', encoding="utf-8"
	)
	(demo / "shapes.py").write_text(SHAPES, encoding="utf-8")
	(demo / "callables.py").write_text(CALLABLES, encoding="utf-8")
	(demo / "examples.py").write_text(EXAMPLES, encoding="utf-8")
	(demo / "api.py").write_text(
		"import typing\nfrom .shapes import Options as Options, Leaf\nfrom . import Thing as Thing\n"
		"if typing.TYPE_CHECKING:\n\tfrom .callables import bare as bare\n",
		encoding="utf-8",
	)
	return tmp_path


@pytest.fixture
def template(project: Path) -> Callable[[str], None]:
	def write(text: str) -> None:
		(project / "docs" / "README.in.md").write_text(text, encoding="utf-8", newline="\n")

	return write


@pytest.fixture
def generated(project: Path, template: Callable[[str], None]) -> Callable[[str], tuple[str, ...]]:
	def generate(directive: str) -> tuple[str, ...]:
		template(f"<!-- GENERATED: {directive} -->")
		assert main([]) == 0
		return tuple((project / "README.md").read_text(encoding="utf-8").splitlines()[1:])

	return generate
