# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins
"""The v0 generation of the public API."""

import typing

from .command import Command as Command
from .config import Agent as Agent
from .config import Claude as Claude
from .config import Config as Config
from .effect import Effect as Effect
from .task import GIT_PORCELAIN as GIT_PORCELAIN
from .task import AgentFormat as AgentFormat
from .task import Clean as Clean
from .task import Parallel as Parallel
from .task import Pipe as Pipe
from .task import Project as Project
from .task import Sequential as Sequential
from .task import Task as Task
from .task import by_glob as by_glob
from .task import by_suffix as by_suffix

if typing.TYPE_CHECKING:
	from ..main.dispatch import run_cli as run_cli


def __getattr__(name: str) -> typing.Any:
	if name == "run_cli":
		from ..main.dispatch import run_cli

		return run_cli
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
