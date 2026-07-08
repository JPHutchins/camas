from camas import Config, Project, Task

ok = Task(("python", "-c", "print('ok')"), name="ok")
broken = Project("broken")

_ = Config(default_task=ok)
