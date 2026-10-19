# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""`python -m docs.readme --watch`: re-render README.md whenever its inputs change."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from docs.readme import BANNER, main

if TYPE_CHECKING:
	from collections.abc import Callable
	from pathlib import Path

	import pytest


def test_watch_rerenders_on_each_change_and_stops_on_ctrl_c(
	project: Path,
	template: Callable[[str], None],
	monkeypatch: pytest.MonkeyPatch,
	capsys: pytest.CaptureFixture[str],
) -> None:
	template("first\n")
	naps = iter(range(3))

	def nap(_: float) -> None:
		match next(naps):
			case 0:
				template("second\n")
				later = (project / "docs" / "README.in.md").stat().st_mtime_ns + 10**9
				os.utime(project / "docs" / "README.in.md", ns=(later, later))
			case 1:
				pass
			case _:
				raise KeyboardInterrupt

	monkeypatch.setattr("docs.readme.time.sleep", nap)
	assert main(["--watch"]) == 0
	out, err = capsys.readouterr()
	assert (project / "README.md").read_text(encoding="utf-8") == f"{BANNER}\nsecond\n"
	assert out.count("regenerated") == 2
	assert err == "watching docs/README.in.md and src/; Ctrl-C stops\n"
