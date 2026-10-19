# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from importlib.metadata import metadata
from typing import TYPE_CHECKING

from ..paths import camas_package_dir
from . import wire

if TYPE_CHECKING:
	from pathlib import Path


def to_docs_response() -> wire.DocsResponse:
	source = camas_package_dir()
	return wire.DocsResponse(source=str(source), tutorial=readme(source))


def readme(source: Path) -> str:
	checkout = source.parent.parent / "README.md"
	if source.parent.name == "src" and checkout.is_file():
		return checkout.read_text(encoding="utf-8")
	return metadata("camas")["Description"]
