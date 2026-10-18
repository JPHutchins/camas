# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from typing import NamedTuple


class Ref(NamedTuple):
	"""
	>>> Ref("lint")
	Ref(name='lint')
	"""

	name: str
