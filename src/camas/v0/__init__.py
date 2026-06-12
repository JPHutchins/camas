# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins
"""The v0 generation of the public API.

This package is the entirety of the public surface for the v0
generation — everything needed to write a ``tasks.py`` or an effect.
The four headline definers (:class:`~camas.v0.task.Task`,
:class:`~camas.v0.task.Sequential`, :class:`~camas.v0.task.Parallel`,
:class:`~camas.v0.effect.Effect`) are re-exported here and aliased,
unversioned, as ``camas``; the rest of the surface lives in this
package's submodules (:mod:`~camas.v0.task`, :mod:`~camas.v0.task_event`,
:mod:`~camas.v0.leaf_state`, :mod:`~camas.v0.completion`).

Import from ``camas.v0`` to pin this generation — right for an effect
plugin you distribute or a ``tasks.py`` you won't revisit. ``v0`` is
semver-zero loose while camas is 0.x: the surface prefers to grow,
breaking changes stay possible until 1.0 (deliberate, release-noted),
and at 1.0 it freezes — a breaking change then forces ``camas.v1``,
while ``v0`` keeps shipping.

The headline definers:

	>>> from camas.v0 import Parallel, Sequential, Task
	>>> from camas.v0.task import Group
	>>> ci = Sequential(Task("ruff check ."), Parallel("mypy .", "pytest"), name="ci")
	>>> ci.name
	'ci'
	>>> isinstance(ci, Group)
	True

The plugin contract — what an :class:`~camas.v0.effect.Effect` observes:
the :data:`~camas.v0.task.TaskNode` tree it was set up with, the
:data:`~camas.v0.task_event.TaskEvent` stream, every leaf's
:data:`~camas.v0.leaf_state.LeafState`, and each completed leaf's
:data:`~camas.v0.completion.Completion`:

	>>> from camas.v0.completion import Completion, Finished, Skipped
	>>> def verdict(completion: Completion) -> str:
	...     match completion:
	...         case Finished(returncode=rc):
	...             return f"exited {rc}"
	...         case Skipped(returncode=rc):
	...             return f"skipped (prior rc={rc})"
	>>> verdict(Finished(0, 0.5, (b"ok",)))
	'exited 0'
	>>> verdict(Skipped(1))
	'skipped (prior rc=1)'

See ``examples/effect-plugin/`` for a complete out-of-tree effect.
"""

from .effect import Effect as Effect
from .task import Parallel as Parallel
from .task import Sequential as Sequential
from .task import Task as Task
