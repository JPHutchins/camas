# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Hook probe for the Claude Code contract test: append a JSON record of exactly what a hook
command received — its ``argv`` (post shell-expansion, so ``${file_path}`` interpolation is
visible) and the stdin payload — to ``$CAMAS_PROBE_LOG``. Run as a subprocess by the hooks under
test, never imported, so it is outside the coverage run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
	stdin_raw = "" if sys.stdin.isatty() else sys.stdin.read()
	try:
		stdin_json = json.loads(stdin_raw) if stdin_raw.strip() else None
	except json.JSONDecodeError:
		stdin_json = None
	record = {"argv": sys.argv, "stdin_json": stdin_json}
	with Path(os.environ["CAMAS_PROBE_LOG"]).open("a", encoding="utf-8") as fh:
		fh.write(json.dumps(record) + "\n")
	return 0


if __name__ == "__main__":
	sys.exit(main())
