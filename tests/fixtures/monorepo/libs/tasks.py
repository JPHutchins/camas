from camas import Config, Project, Task

build = Task(("python", "-c", "import os; print('libs-build', os.getcwd())"), name="build")
search = Project("search")

_ = Config(default_task=build)
