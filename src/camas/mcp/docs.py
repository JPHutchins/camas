# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import itertools
import re
from importlib.metadata import distributions
from pathlib import Path
from typing import Final

from ..paths import camas_package_dir
from . import wire

FENCE: Final = re.compile(r"\s*(```|~~~)")
HEADING: Final = re.compile(r"#{1,6}\s")


def to_docs_response() -> wire.DocsResponse:
	source = camas_package_dir()
	return wire.DocsResponse(source=str(source), tutorial=outline(readme_path(source)))


def readme_path(source: Path) -> Path:
	checkout = source.parent.parent / "README.md"
	if source.parent.name == "src" and checkout.is_file():
		return checkout
	return next(
		Path(file.locate())
		for found in distributions(name="camas")
		for file in found.files or ()
		if file.name == "METADATA" and file.parent.name.endswith(".dist-info")
	)


def outline(readme: Path) -> str:
	r"""
	>>> import tempfile
	>>> with tempfile.TemporaryDirectory() as tmp:
	...     path = Path(tmp) / "README.md"
	...     _ = path.write_text("# Camas\n\n```python\n# a comment\n```\n## Install\n", encoding="utf-8")
	...     print(outline(path).replace(str(path), "README.md"))
	camas's documentation is README.md; read the sections you need from it.
	Each heading, after its line number in that file:
	1: # Camas
	6: ## Install
	"""
	lines = tuple(readme.read_text(encoding="utf-8").split("\n"))
	fences = tuple(FENCE.match(line) is not None for line in lines)
	fenced = itertools.accumulate(fences, lambda inside, fence: inside != fence)
	return "\n".join(
		(
			f"camas's documentation is {readme}; read the sections you need from it.",
			"Each heading, after its line number in that file:",
			*(
				f"{number}: {line}"
				for number, (line, fence, inside) in enumerate(
					zip(lines, fences, fenced, strict=True), 1
				)
				if not fence and not inside and HEADING.match(line)
			),
		)
	)
