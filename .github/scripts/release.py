#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Cut a release: assert a clean, synced main; write VERSION; commit; tag.

Pushing (main + the tag) is left to the operator — the tag push publishes.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


def git(*args: str) -> str:
	return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def die(message: str) -> NoReturn:
	print(f"release: {message}", file=sys.stderr)
	raise SystemExit(1)


def main(argv: list[str]) -> None:
	if len(argv) != 1:
		die("usage: release.py <version>  (e.g. release.py 0.1.22)")
	version = argv[0]
	if re.fullmatch(r"\d+(\.\d+)*", version) is None:
		die(f"{version!r} is not a release version (digits and dots)")
	if (branch := git("rev-parse", "--abbrev-ref", "HEAD")) != "main":
		die(f"on {branch!r}, not main")
	if git("status", "--porcelain"):
		die("working tree is not clean")
	git("fetch", "--quiet", "origin", "main")
	if git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
		die("main is not in sync with origin/main (pull or push first)")
	version_file = Path(git("rev-parse", "--show-toplevel")) / "VERSION"
	current = version_file.read_text(encoding="utf-8").strip()
	if version == current:
		die(f"VERSION is already {current}")
	tag_probe = subprocess.run(
		["git", "rev-parse", "--quiet", "--verify", f"refs/tags/{version}"],
		capture_output=True,
		check=False,
	)
	if tag_probe.returncode == 0:
		die(f"tag {version} already exists")
	version_file.write_text(f"{version}\n", encoding="utf-8", newline="\n")
	git("commit", "--quiet", "-m", f"release: {version}", "--", str(version_file))
	git("tag", version)
	print(f"release: {current} -> {version} committed and tagged")
	print(f"publish with: git push origin main {version}")


if __name__ == "__main__":
	main(sys.argv[1:])
