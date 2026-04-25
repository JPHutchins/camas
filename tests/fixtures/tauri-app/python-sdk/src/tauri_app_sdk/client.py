"""Minimal client for invoking tauri-app commands from Python."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Client:
    base_url: str

    def invoke(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Stub: would normally POST to the tauri-app IPC bridge."""
        return {"command": command, "args": args or {}}
