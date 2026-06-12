# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from camas import Config, Parallel, Sequential, Task

lint = Task(("python", "-c", "print('lint ran')"), name="lint")
test = Task(("python", "-c", "print('test ran')"), name="test")
ci = Sequential(lint, test, name="ci")
ci_full = Parallel(
	lint,
	test,
	Task(("python", "-c", "print('cov ran')"), name="cov"),
	name="ci_full",
)

_ = Config(default_task=ci, github_task=ci_full)
