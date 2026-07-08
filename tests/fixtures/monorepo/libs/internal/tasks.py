"""A private tasks file: runnable here, never composed into an ancestor."""

from camas import Config, Task

secret = Task(("python", "-c", "import os; print('internal-secret', os.getcwd())"))

_ = Config(default_task=secret, discoverable=False)
