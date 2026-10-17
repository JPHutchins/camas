# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import shutil
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Literal

LocalEnvironment = Literal["venv", "nix"]


def in_nix_store(executable: str) -> bool:
	r"""
	>>> in_nix_store("/nix/store/9k1zdwqp-camas-0.1.27/bin/camas")
	True
	>>> in_nix_store("/usr/bin/camas"), in_nix_store(r"C:\Program Files\camas.exe")
	(False, False)
	"""
	return PurePosixPath(executable).is_relative_to("/nix/store")


def in_virtualenv(executable: str) -> bool:
	with suppress(OSError):
		return (Path(executable).parent.parent / "pyvenv.cfg").is_file()
	return False


def local_environment(executable: str) -> LocalEnvironment | None:
	if in_nix_store(executable):
		return "nix"
	if in_virtualenv(executable):
		return "venv"
	return None


def local_camas() -> LocalEnvironment | None:
	found = shutil.which("camas")
	return None if found is None else local_environment(found)
