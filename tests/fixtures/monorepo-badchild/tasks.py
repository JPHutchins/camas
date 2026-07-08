"""A healthy root whose composed load is poisoned by a broken child."""

from camas import Config, Task

ok = Task(("python", "-c", "print('ok')"))

_ = Config(default_task=ok)
