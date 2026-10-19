# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from importlib.metadata import metadata
from pathlib import Path

from camas.mcp.docs import readme, to_docs_response
from camas.paths import camas_package_dir


def test_camas_package_dir_points_at_installed_package() -> None:
	source = camas_package_dir()
	assert source.name == "camas"
	assert (source / "__init__.py").is_file()


def test_to_docs_response_serves_the_checkouts_readme_in_development() -> None:
	resp = to_docs_response()
	assert resp.source == str(camas_package_dir())
	assert resp.tutorial == (Path(resp.source).parent.parent / "README.md").read_text(
		encoding="utf-8"
	)


def test_a_checkout_is_a_src_layout_with_a_readme(tmp_path: Path) -> None:
	(tmp_path / "src" / "camas").mkdir(parents=True)
	(tmp_path / "README.md").write_text("# from the checkout\n", encoding="utf-8")
	assert readme(tmp_path / "src" / "camas") == "# from the checkout\n"


def test_an_installed_package_serves_the_readme_from_its_metadata(tmp_path: Path) -> None:
	(tmp_path / "site-packages" / "camas").mkdir(parents=True)
	assert readme(tmp_path / "site-packages" / "camas") == metadata("camas")["Description"]
