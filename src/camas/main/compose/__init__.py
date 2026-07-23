# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Load a ``tasks.py`` scope, resolving its ``Project`` references into a composed task tree."""

from __future__ import annotations

from .errors import ProjectLoadError
from .scope import (
	load_dhall_tasks_state,
	load_py_tasks_state,
	load_scope,
	state_from_scope,
)

__all__ = [
	"ProjectLoadError",
	"load_dhall_tasks_state",
	"load_py_tasks_state",
	"load_scope",
	"state_from_scope",
]
