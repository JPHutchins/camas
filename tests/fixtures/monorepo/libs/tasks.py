from camas import Claude, Config, Project, Task

build = Task(("python", "-c", "import os; print('libs-build', os.getcwd())"), name="build")
fix = Task(("python", "-c", "import os; print('libs-fix', os.getcwd())"), name="fix")
check = Task(("python", "-c", "import os; print('libs-check', os.getcwd())"), name="check")
search = Project("search")

_ = Config(default_task=build, agent=Claude(fix=fix, check=check))
