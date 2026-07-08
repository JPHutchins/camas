"""The colliding child: its default binds the bare ``search`` key the root already owns."""

from camas import Config, Task

build = Task(("python", "-c", "print('child build')"))

_ = Config(default_task=build)
