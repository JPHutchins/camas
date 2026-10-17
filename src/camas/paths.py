# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path


def camas_package_dir() -> Path:
	"""
	>>> (camas_package_dir() / "__init__.py").is_file()
	True
	"""
	import camas

	return Path(camas.__file__).parent
