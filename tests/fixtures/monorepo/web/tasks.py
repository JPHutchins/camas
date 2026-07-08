from camas import Config, Task

build = Task(
	("python", "-c", "import sys; print('web-build', *sys.argv[1:])", "{paths}"),
	name="build",
	paths=".",
)
ship = Task(("python", "-c", "import os; print('web-ship', os.getcwd())"), name="ship")

_ = Config(default_task=build, github_task=ship)
