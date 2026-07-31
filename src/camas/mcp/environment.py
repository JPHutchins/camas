# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Whether the ``camas`` on PATH belongs to the ecosystem environment holding the project's tools —
a virtual environment or a Nix devShell — which is what decides whether a launcher written into
``.mcp.json`` can run them at all.
"""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from typing import Literal

LocalEnvironment = Literal["venv", "nix"]
"""An ecosystem environment a ``camas`` executable can belong to, and whose tools it therefore
shares: a Python virtual environment, or the Nix store a flake devShell builds from."""


def in_nix_store(executable: str) -> bool:
	r"""Whether ``executable`` came out of the Nix store, and so from a devShell or profile that
	provides the project's toolchain alongside it.

	>>> in_nix_store("/nix/store/9k1zdwqp-camas-0.1.27/bin/camas")
	True
	>>> in_nix_store("/usr/bin/camas"), in_nix_store(r"C:\Program Files\camas.exe")
	(False, False)
	"""
	return PurePosixPath(executable).is_relative_to("/nix/store")


def in_virtualenv(executable: str) -> bool:
	"""Whether ``executable`` sits in a virtual environment's script directory — ``bin``, or
	``Scripts`` on Windows — and so shares an interpreter with every tool installed into it. Decided
	by the ``pyvenv.cfg`` beside that directory rather than by a ``.venv`` name, so ``venv/``,
	``env/`` and ``~/.virtualenvs/x`` count too and a plain directory someone named ``.venv`` does
	not.
	"""
	return (Path(executable).parent.parent / "pyvenv.cfg").is_file()


def local_environment(executable: str) -> LocalEnvironment | None:
	"""Which ecosystem environment ``executable`` belongs to, or ``None`` for a global or
	tool-isolated install — ``uv tool install``, pipx, a system package — that shares no tools with
	the project.

	Only where the executable lives is consulted, never ``VIRTUAL_ENV`` or ``IN_NIX_SHELL``: those
	report that a shell was entered, not that this camas came from it. Measured on nix 2.32.1,
	``nix develop`` sets ``IN_NIX_SHELL=impure`` in a shell where ``camas`` still resolved to a
	global ``~/.local/bin/camas`` — as isolated from the devShell's tools as ``uvx`` is, and without
	its pin.

	Neither test resolves symlinks, in both directions deliberately. A devShell puts literal store
	paths on PATH, so an unresolved store path is what distinguishes it from a ``nix profile``
	install behind ``~/.nix-profile/bin``; and resolving would follow a ``uv tool install`` shim in
	``~/.local/bin`` into a tool environment that carries its own ``pyvenv.cfg``, classifying an
	isolated install as a project venv.
	"""
	if in_nix_store(executable):
		return "nix"
	if in_virtualenv(executable):
		return "venv"
	return None


def local_camas() -> LocalEnvironment | None:
	"""Which environment the PATH ``camas`` belongs to — the one a bare ``camas`` command written
	into ``.mcp.json`` or a hook will resolve to — or ``None`` when there is none on PATH, or the one
	there belongs to no environment.
	"""
	found = shutil.which("camas")
	return None if found is None else local_environment(found)
