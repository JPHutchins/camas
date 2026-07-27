# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Fixtures shared across the suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def unforced_color(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Clear the color environment, so a test reads camas's own decision rather than whatever the
	developer's shell or the CI runner exported into it.
	"""
	for name in ("NO_COLOR", "FORCE_COLOR", "CLICOLOR_FORCE"):
		monkeypatch.delenv(name, raising=False)
