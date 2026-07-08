"""camas task definitions for the libs tree."""

from camas import Config, Task

build = Task(("python", "-c", "import os; print('libs-build', os.getcwd())"))

_ = Config(default_task=build)
