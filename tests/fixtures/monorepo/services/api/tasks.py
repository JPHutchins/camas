from camas import Config, Task

deploy = Task(("python", "-c", "import os; print('api-deploy', os.getcwd())"), name="deploy")

_ = Config(default_task=deploy)
