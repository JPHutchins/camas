from camas import Claude, Config, Task

build = Task(
	("python", "-c", "import sys; print('web-build', *sys.argv[1:])", "{paths}"),
	name="build",
	paths=".",
)
ship = Task(("python", "-c", "import os; print('web-ship', os.getcwd())"), name="ship")
fix = Task(("python", "-c", "import os; print('web-fix', os.getcwd())"), name="fix")
check = Task(("python", "-c", "import os; print('web-check', os.getcwd())"), name="check")

_ = Config(default_task=build, github_task=ship, agent=Claude(fix=fix, check=check))
