"""camas task definitions for the history library."""

from camas import Task

lint = Task(
	("python", "-c", "import sys; print('history-lint', *sys.argv[1:])", "{paths}"),
	paths=".",
)
