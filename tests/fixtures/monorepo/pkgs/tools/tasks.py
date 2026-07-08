"""camas task definitions whose natural namespace collides with the root-level tools/."""

from camas import Task

check = Task(("python", "-c", "import os; print('pkgs-tools-check', os.getcwd())"))
