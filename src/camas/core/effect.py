# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Engine-side effect plumbing: the per-leaf event dispatcher type."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeAlias

from ..v0 import TaskEvent

EventSink: TypeAlias = Callable[[int, TaskEvent], Awaitable[None]]
"""Per-leaf event dispatcher: await sink(leaf_idx, event)."""
