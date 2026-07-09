from camas import Claude, Config, Parallel, Project, Task

hello = Task(("python", "-c", "import os; print('hello', os.getcwd())"), name="hello")
libs = Project("libs")
api = Project("services/api")
web = Project("web")

_ = Config(
	default_task=Parallel(libs, api, web, name="all"),
	github_task=Parallel(libs, api, web, name="ci"),
	agent=Claude(
		fix=Parallel(libs, api, web, name="fix-all"),
		check=Parallel(libs, api, web, name="check-all"),
	),
)
