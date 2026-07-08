"""camas task definitions for the monorepo fixture root."""

from camas import Config, Task

hello = Task(("python", "-c", "import os; print('hello', os.getcwd())"))

_ = Config(default_task=hello)
