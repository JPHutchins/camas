"""camas task definitions discovered through the foreign legacy/tasks.py above."""

from camas import Task

work = Task(("python", "-c", "import os; print('queue-work', os.getcwd())"))
