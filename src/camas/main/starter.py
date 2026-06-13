# /// script
# requires-python = ">=3.10"
# dependencies = ["camas"]
# ///
"""Project tasks — run with ``camas``.

Every task below is a cross-platform placeholder (``python -c ...``);
replace the placeholders with your real commands:

	lint = Task("ruff check .")
	test = Task("cargo test", cwd=Path("rust"))
	build = Task("npm run build")

The ``# /// script`` block above is optional PEP 723 inline metadata. With
it (plus the ``__main__`` block at the bottom), PEP 723-aware tools can run
this file standalone — ``uv run tasks.py``, ``pipx run tasks.py`` — building
a throwaway env with camas installed; handy for non-Python projects. It is
also where camas gets version-pinned (``dependencies = ["camas>=X.Y"]``).
Both are inert under plain ``camas``; delete them if you don't want the
standalone path.
"""

from camas import Config, Parallel, Sequential, Task, run_cli

# A leaf is any shell command, shlex-split (so quote the -c payload). A bare
# string inside Sequential/Parallel coerces to an anonymous Task; tuple form
# Task(("python", "-c", "...")) skips shlex when quoting gets hairy.
hello = Task("python -c \"print('hello from camas')\"")

# matrix= clones the subtree per axis value, interpolating {NAME} into
# cmd/env/cwd; env= is scoped to the subtree. Pin an axis from the CLI:
# camas greet --NAME Grace
greet = Parallel(
	Task("python -c \"import os; print(os.environ['GREETING'] + ', {NAME}!')\""),
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

# The PEP 723 standalone flow (see the docstring): running this file directly
# (`uv run tasks.py <task>`) dispatches through camas. Inert when the `camas`
# command discovers this file itself.
if __name__ == "__main__":
	run_cli(globals())
