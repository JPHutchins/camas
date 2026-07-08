from camas import Config, Task

lint = Task(
	("python", "-c", "import os; print('search-lint', os.getcwd())"), name="lint", cwd="src"
)
test = Task(("python", "-c", "import os; print('search-test', os.getcwd())"), name="test")

_ = Config(default_task=lint)
