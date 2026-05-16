from pathlib import Path

from camas import Parallel, Sequential, Task

lint = Task("ruff check .")
format_fix = Task("ruff format .")
typecheck = Task("mypy .")
test = Task("pytest", cwd=Path("."))

fix = Sequential(format_fix, Task("ruff check --fix ."))
check = Parallel(lint, typecheck, test)
