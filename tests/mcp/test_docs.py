# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path

from camas.mcp.docs import outline, readme_path, to_docs_response
from camas.paths import camas_package_dir


def _headings(text: str) -> list[tuple[int, str]]:
	return [
		(int(number), heading)
		for number, _, heading in (line.partition(": ") for line in text.splitlines()[2:])
	]


def test_camas_package_dir_points_at_installed_package() -> None:
	source = camas_package_dir()
	assert source.name == "camas"
	assert (source / "__init__.py").is_file()


def test_docs_outline_the_readme_by_line_number() -> None:
	resp = to_docs_response()
	readme = readme_path(Path(resp.source))
	assert str(readme) in resp.tutorial.splitlines()[0]
	lines = readme.read_text(encoding="utf-8").split("\n")
	headings = _headings(resp.tutorial)
	assert "## Getting started" in dict(headings).values()
	assert all(lines[number - 1] == heading for number, heading in headings)


def test_a_checkout_is_a_src_layout_with_a_readme(tmp_path: Path) -> None:
	(tmp_path / "src" / "camas").mkdir(parents=True)
	(tmp_path / "README.md").write_text("# from the checkout\n", encoding="utf-8")
	assert readme_path(tmp_path / "src" / "camas") == tmp_path / "README.md"


def test_an_installed_package_points_at_its_metadata(tmp_path: Path) -> None:
	(tmp_path / "site-packages" / "camas").mkdir(parents=True)
	metadata = readme_path(tmp_path / "site-packages" / "camas")
	assert metadata.name == "METADATA"
	assert metadata.parent.name.endswith(".dist-info")
	lines = metadata.read_text(encoding="utf-8").split("\n")
	assert all(lines[number - 1] == heading for number, heading in _headings(outline(metadata)))
