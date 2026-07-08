"""A foreign task module (Celery-style): no camas import, so discovery never executes it."""

from celery import shared_task


@shared_task
def add(x, y):
	return x + y
