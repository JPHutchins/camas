from pathlib import Path

from camas import Parallel, Task


def test_task_accepts_path():
	task = Task("echo {paths}", paths=Path("src"))
	assert task.paths == "src"


def test_parallel_accepts_path():
	group = Parallel(Task("echo {paths}"), paths=Path("src"))
	assert group.paths == "src"
