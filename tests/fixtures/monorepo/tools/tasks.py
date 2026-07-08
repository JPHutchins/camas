"""camas task definitions that opt out of composing their own descendants."""

from camas import Config, Task

info = Task(("python", "-c", "import os; print('tools-info', os.getcwd())"))

_ = Config(discover=False)
