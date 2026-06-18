# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Completion outcomes: a finished task ran to exit; a skipped task never ran."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple, TypeAlias

if TYPE_CHECKING:
	from collections.abc import Sequence


INTERRUPT_RC: Final = 130
"""Exit code for a signal-interrupted run (128 + SIGINT)."""


class Finished(NamedTuple):
	"""Completion outcome: task ran to exit with a returncode.

	>>> Finished(0, 1.234, (b"all clean",))
	Finished(returncode=0, elapsed=1.234, output=(b'all clean',))
	"""

	returncode: int
	elapsed: float
	output: Sequence[bytes]


class Skipped(NamedTuple):
	"""Completion outcome: task was skipped due to a prior Sequential failure.

	Carries the returncode of the task that caused the skip — an Either-like
	propagation so callers that need an rc (e.g. the overall run's exit code)
	can read it uniformly across completion variants — and ``blocked_by``, the
	label of that task, so consumers can point at the real failure, not the skip.

	>>> Skipped(1)
	Skipped(returncode=1, blocked_by=None)
	>>> Skipped(1, "lint").blocked_by
	'lint'
	"""

	returncode: int
	blocked_by: str | None = None
	"""Label of the leaf whose non-zero exit short-circuited this one; ``None`` when not derivable."""


class Stopped(NamedTuple):
	"""Completion outcome: the runner forwarded the task a signal while it ran.

	>>> Stopped(130, 0.5, (b"caught SIGINT",))
	Stopped(returncode=130, elapsed=0.5, output=(b'caught SIGINT',))
	"""

	returncode: int
	elapsed: float
	output: Sequence[bytes]


Completion: TypeAlias = Finished | Skipped | Stopped
