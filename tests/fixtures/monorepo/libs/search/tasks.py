"""camas task definitions for the search library."""

from camas import Config, Task

lint = Task(("python", "-c", "import os; print('search-lint', os.getcwd())"))
test = Task(("python", "-c", "import os; print('search-test', os.getcwd())"), cwd="src")

_ = Config(default_task=lint)
