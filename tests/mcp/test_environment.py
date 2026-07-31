# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from camas.mcp.environment import in_virtualenv, local_camas, local_environment

if TYPE_CHECKING:
	from collections.abc import Callable

	import pytest


def _outside_any_environment(tmp_path: Path) -> Path:
	"""A ``camas`` in a plain directory — a global or tool-isolated install. Built under ``tmp_path``
	rather than named ``/usr/bin/camas`` so the verdict rests on a directory the test owns: the venv
	test stats the ``pyvenv.cfg`` beside it, and a real system path could in principle have one.
	"""
	(tmp_path / "global").mkdir()
	return tmp_path / "global" / "camas"


def test_in_virtualenv_reads_the_pyvenv_cfg(
	tmp_path: Path, venv_camas: Callable[[Path], Path]
) -> None:
	assert in_virtualenv(str(venv_camas(tmp_path / ".venv")))


def test_in_virtualenv_rejects_a_directory_merely_named_venv(tmp_path: Path) -> None:
	(tmp_path / ".venv" / "bin").mkdir(parents=True)
	assert not in_virtualenv(str(tmp_path / ".venv" / "bin" / "camas"))


def test_local_environment_nix_store() -> None:
	assert local_environment("/nix/store/9k1zdwqp-camas-0.1.27/bin/camas") == "nix"


def test_local_environment_virtualenv(tmp_path: Path, venv_camas: Callable[[Path], Path]) -> None:
	assert local_environment(str(venv_camas(tmp_path / ".venv"))) == "venv"


def test_local_environment_global_install_belongs_to_none(tmp_path: Path) -> None:
	assert local_environment(str(_outside_any_environment(tmp_path))) is None


def _camas_resolving_to(path: Path | None) -> Callable[[str], str | None]:
	"""A ``shutil.which`` that finds ``camas`` at ``path``, or finds nothing — the only question
	:func:`local_camas` asks of PATH.
	"""
	return lambda _name: None if path is None else str(path)


def test_local_camas_none_when_off_path(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr("shutil.which", _camas_resolving_to(None))
	assert local_camas() is None


def test_local_camas_none_for_a_global_install(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	monkeypatch.setattr("shutil.which", _camas_resolving_to(_outside_any_environment(tmp_path)))
	assert local_camas() is None


def test_local_camas_finds_the_venv(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch, venv_camas: Callable[[Path], Path]
) -> None:
	monkeypatch.setattr("shutil.which", _camas_resolving_to(venv_camas(tmp_path / ".venv")))
	assert local_camas() == "venv"


def test_in_virtualenv_answers_rather_than_raising_on_an_unreadable_parent(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Whatever the probe raises, classifying a candidate launcher answers — ``camas mcp init`` has
	to fall through to the next one, not die on a directory it merely wanted to look at.

	The raise is injected into ``Path.is_file`` rather than into ``os.stat`` so this holds on every
	supported interpreter: measured, an unreadable parent makes ``Path.is_file`` raise
	``PermissionError`` on 3.10 through 3.13 but return ``False`` on 3.14, so patching underneath it
	would leave the guard unexercised on the newer ones.
	"""

	def denied(_self: Path, **_kwargs: object) -> bool:
		raise PermissionError(13, "Permission denied")

	monkeypatch.setattr(Path, "is_file", denied)
	assert in_virtualenv("/locked/bin/camas") is False
