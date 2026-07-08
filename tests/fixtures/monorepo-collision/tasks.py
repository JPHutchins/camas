"""A root binding that collides with a discovered namespace of the same name."""

from camas import Task

search = Task(("python", "-c", "print('root search')"))
