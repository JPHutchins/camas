"""camas task definitions in a hyphenated directory."""

from camas import Task

check = Task(("python", "-c", "import os; print('my-lib-check', os.getcwd())"))
