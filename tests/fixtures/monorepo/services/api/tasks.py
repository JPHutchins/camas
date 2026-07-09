from camas import Claude, Config, Task

deploy = Task(("python", "-c", "import os; print('api-deploy', os.getcwd())"), name="deploy")
fix = Task(("python", "-c", "import os; print('api-fix', os.getcwd())"), name="fix")
check = Task(("python", "-c", "import os; print('api-check', os.getcwd())"), name="check")

_ = Config(default_task=deploy, agent=Claude(fix=fix, check=check))
