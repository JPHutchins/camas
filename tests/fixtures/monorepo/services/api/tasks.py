"""camas task definitions for the api service (services/ has no tasks.py: the gap collapses)."""

from camas import Config, Task

deploy = Task(("python", "-c", "import os; print('api-deploy', os.getcwd())"))

_ = Config(default_task=deploy)
