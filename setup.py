"""Optionally compile the hot modules with mypyc (CAMAS_USE_MYPYC=1)."""

import os
import sys
from pathlib import Path

from setuptools import setup

is_editable = any("editable" in arg or arg == "develop" for arg in sys.argv)
use_mypyc = os.environ.get("CAMAS_USE_MYPYC") == "1" and not is_editable

if use_mypyc:
	from mypyc.build import mypycify

	# core/ stays interpreted: Effect's @runtime_checkable Protocol and Group's
	# @dataclass-generated __eq__ don't survive compilation.
	# github_checks.py stays interpreted: its optional httpx dep isn't available in
	# the isolated build env, so mypy/mypyc compilation can't resolve it. ctrf.py
	# stays interpreted for the same reason via the msgspec-backed _ctrf_model it
	# lazy-imports; _ctrf_model itself is already skipped by the [!_] glob below.
	# check.py and state.py stay interpreted: mypyc's NamedTuple codegen rejects
	# the built-in ``Exception`` as a field type (KeyError: 'Exception' at import).
	# starter.py stays interpreted: it is the --init template, shipped as plain
	# source and read back as text by init.py.
	# __init__.py stays interpreted: a compiled package __init__ runs its body in
	# create_module, before the import machinery sets __path__, so its eager
	# ``from .entrypoint import …`` chain reaches the interpreted ``.check``
	# sibling while camas.main is not yet a package (mypy >= 1.20).
	_main_excluded = {"__init__.py", "check.py", "state.py", "starter.py"}
	_effect_excluded = {"github_checks.py", "ctrf.py"}
	ext_modules = mypycify(
		[
			*(str(p) for p in Path("src/camas/main").glob("*.py") if p.name not in _main_excluded),
			*(
				str(p)
				for p in Path("src/camas/effect").glob("[!_]*.py")
				if p.name not in _effect_excluded
			),
		],
		opt_level="3",
	)
else:
	ext_modules = []

setup(ext_modules=ext_modules)
