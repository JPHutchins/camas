"""Below a Config(discover=False) parent: never composed into any ancestor."""

from camas import Task

hidden = Task(("python", "-c", "print('tools-child-hidden')"))
