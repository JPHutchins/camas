# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path
from typing import Final

from .source import Unresolved

LANGUAGES: Final = {
	".py": "python",
	".toml": "toml",
	".yaml": "yaml",
	".yml": "yaml",
	".json": "json",
	".sh": "sh",
}


def file_section(name: str) -> tuple[str, ...] | Unresolved:
	path = Path(name)
	if not path.is_file():
		return Unresolved(f"no file {path.as_posix()!r}")
	return (
		f"```{LANGUAGES.get(path.suffix, '')}",
		*path.read_text(encoding="utf-8").expandtabs(4).rstrip("\n").split("\n"),
		"```",
	)
