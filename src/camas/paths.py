# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Where the running camas lives."""

from __future__ import annotations

from pathlib import Path


def camas_package_dir() -> Path:
	"""The installed ``camas`` package directory.

	Resolved at call time, not import time: mypyc-compiled modules may not define ``__file__``
	while the module body executes (nixpkgs' mypyc doesn't), which is the install both callers —
	``camas_docs`` and ``--check``'s search path — exist to serve.

	>>> (camas_package_dir() / "__init__.py").is_file()
	True
	"""
	import camas

	return Path(camas.__file__).parent
