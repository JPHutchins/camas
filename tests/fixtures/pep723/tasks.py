# /// script
# requires-python = ">=3.11"
# dependencies = ["camas"]
# ///
"""Standalone PEP 723 tasks.py fixture.

The inline-metadata header sits above the imports and the ``__main__`` block wraps
the bindings; camas's loader must read this module identically to a header-less
``tasks.py``, and ``run_cli(globals())`` must dispatch when run as a script.
"""

from camas import Task, run_cli

hello = Task(("python", "-c", "print('hello from pep723')"))

if __name__ == "__main__":
	run_cli(globals())
