# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""How a leaf's run ended."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple, TypeAlias

if TYPE_CHECKING:
	from collections.abc import Sequence


INTERRUPT_RC: Final = 130

NOT_FOUND_RC: Final = 127


class Finished(NamedTuple):
	"""A leaf that ran to exit.

	>>> Finished(0, 1.234, (b"all clean",))
	Finished(returncode=0, elapsed=1.234, output=(b'all clean',))
	"""

	returncode: int
	elapsed: float
	output: Sequence[bytes]


class Skipped(NamedTuple):
	"""A leaf skipped after an earlier failure.

	>>> Skipped(1)
	Skipped(returncode=1, blocked_by=None)
	>>> Skipped(1, "lint").blocked_by
	'lint'
	"""

	returncode: int
	"""The failed leaf's return code."""
	blocked_by: str | None = None
	"""The failed leaf's label, when known."""


class Stopped(NamedTuple):
	"""A leaf the runner signalled while it ran.

	>>> Stopped(130, 0.5, (b"caught SIGINT",))
	Stopped(returncode=130, elapsed=0.5, output=(b'caught SIGINT',))
	"""

	returncode: int
	elapsed: float
	output: Sequence[bytes]


class Errored(NamedTuple):
	"""A leaf whose command could not start.

	>>> Errored(127, "no such file or directory: does-not-exist")
	Errored(returncode=127, message='no such file or directory: does-not-exist')
	"""

	returncode: int
	message: str


Completion: TypeAlias = Finished | Skipped | Stopped | Errored
