# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeAlias

from ..v0.task_event import TaskEvent

EventSink: TypeAlias = Callable[[int, TaskEvent], Awaitable[None]]
