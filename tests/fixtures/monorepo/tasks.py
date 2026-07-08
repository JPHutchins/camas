from camas import Config, Parallel, Project, Task

hello = Task(("python", "-c", "import os; print('hello', os.getcwd())"), name="hello")
libs = Project("libs")
api = Project("services/api")
web = Project("web")

_ = Config(default_task=hello, github_task=Parallel(libs, api, web))
