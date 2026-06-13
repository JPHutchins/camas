# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``--init``: scaffold a commented starter ``tasks.py``."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
	from pathlib import Path

STARTER: Final = '''\
# /// script
# requires-python = ">=3.10"
# dependencies = ["camas"]
# ///
"""Project tasks — run with ``camas``, or standalone via ``uv run tasks.py``.

Every task below is a cross-platform placeholder (``python -c ...``);
replace the placeholders with your real commands:

	lint = Task("ruff check .")
	test = Task("cargo test", cwd=Path("rust"))
	build = Task("npm run build")
"""

from camas import Config, Parallel, Sequential, Task, run_cli

# A leaf is any shell command. Tuple form runs as-is; string form is
# shlex-split. A bare string inside Sequential/Parallel coerces to an
# anonymous Task.
hello = Task(("python", "-c", "print('hello from camas')"))

# matrix= clones the subtree per axis value, interpolating {NAME} into
# cmd/env/cwd; env= is scoped to the subtree. Pin an axis from the CLI:
# camas greet --NAME Grace
greet = Parallel(
	Task(("python", "-c", "import os; print(os.environ['GREETING'] + ', {NAME}!')")),
	matrix={"NAME": ("Ada", "Grace")},
	env={"GREETING": "hello"},
	help="say hello to everyone at once",
)

# Sequential runs in order, short-circuiting on the first failure;
# Parallel runs concurrently. They nest freely.
ci = Sequential(
	hello,
	Parallel(greet, "python --version"),
	name="ci",
)

# Discovered by type (the binding name never matters): bare `camas` runs
# default_task — or github_task under GitHub Actions, falling back to
# default_task when unset.
_ = Config(default_task=ci)

# Optional: with the PEP 723 header above, any PEP 723-aware tool runs
# this file directly (`uv run tasks.py --list`). camas auto-discovery
# ignores this block.
if __name__ == "__main__":
	run_cli(globals())
'''


def write_starter_tasks_py(directory: Path) -> int:
	"""Write :data:`STARTER` to ``directory/tasks.py`` — the exit code for
	``camas --init``. Refuses to overwrite an existing file.
	"""
	target: Final = directory / "tasks.py"
	if target.exists():
		print(f"error: {target} already exists", file=sys.stderr)
		return 2
	target.write_text(STARTER, encoding="utf-8")
	print(f"Wrote {target}\n\nTry:\n  camas --list\n  camas greet --help\n  camas")
	return 0
