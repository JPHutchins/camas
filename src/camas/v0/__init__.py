# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins
"""The v0 public API — ``from camas.v0 import ...`` is the stability contract.

This module is the public API, versioned the way semver versions the
package: ``v0`` pairs with camas 0.x and is exactly as loose as semver
says 0.x is. The surface prefers to grow — new names, fields appended
with defaults — but breaking changes remain possible until 1.0, made
deliberately and noted in releases, never by accident
(``tests/test_v0.py`` pins the surface). At 1.0 the contract hardens:
a stable-era namespace never removes or changes an exported name, a
breaking change forces the next ``camas.vN``, and old namespaces keep
shipping, so files written against one keep working across upgrades.

The types are *defined* in this package — ``camas.core`` and
``camas.main`` consume them, never the reverse — so the diff history of
``src/camas/v0/`` is the complete history of the public API.

The top-level ``camas`` namespace re-exports the task definers from the
latest version namespace — best effort across major generations, fine for
a ``tasks.py`` that lives next to its dev environment. Import from
``camas.v0`` when a file should outlive that: an effect plugin you
distribute, a standalone task file, a ``tasks.py`` nobody revisits.

The task definers:

	>>> from camas.v0 import Group, Parallel, Sequential, Task
	>>> ci = Sequential(Task("ruff check ."), Parallel("mypy .", "pytest"), name="ci")
	>>> ci.name
	'ci'
	>>> isinstance(ci, Group)
	True

The plugin contract — what an :class:`Effect` observes: the
:data:`TaskNode` tree it was set up with, the :data:`TaskEvent` stream
(``StartedEvent | OutputEvent | CompletedEvent``), every leaf's
:data:`LeafState` (``Waiting | Running | Completed``), and each completed
leaf's :data:`Completion` (``Finished | Skipped``):

	>>> from camas.v0 import Completion, Finished, Skipped
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

See :class:`camas.v0.effect.Effect` for the protocol an effect
implements, and ``examples/effect-plugin/`` for a complete out-of-tree
example.
"""

from .completion import Completion as Completion
from .completion import Finished as Finished
from .completion import Skipped as Skipped
from .effect import Effect as Effect
from .leaf_state import Completed as Completed
from .leaf_state import LeafState as LeafState
from .leaf_state import Running as Running
from .leaf_state import Waiting as Waiting
from .task import Group as Group
from .task import Parallel as Parallel
from .task import Sequential as Sequential
from .task import Task as Task
from .task import TaskNode as TaskNode
from .task_event import CompletedEvent as CompletedEvent
from .task_event import OutputEvent as OutputEvent
from .task_event import StartedEvent as StartedEvent
from .task_event import TaskEvent as TaskEvent
