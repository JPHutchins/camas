"""Optionally compile the hot modules with mypyc (CAMAS_USE_MYPYC=1)."""

import os
import sys
from pathlib import Path

from setuptools import setup

is_editable = any("editable" in arg or arg == "develop" for arg in sys.argv)
use_mypyc = os.environ.get("CAMAS_USE_MYPYC") == "1" and not is_editable

if use_mypyc:
	from mypyc.build import mypycify

	_init_templates_read_as_text = {"starter.py", "starter_verbose.py"}
	_named_tuples_with_exception_fields = {"check.py", "state.py"}
	_package_init_with_eager_imports = {"__init__.py"}
	_importers_of_optional_dependencies = {"github_checks.py", "ctrf.py"}
	_main_excluded = (
		_init_templates_read_as_text
		| _named_tuples_with_exception_fields
		| _package_init_with_eager_imports
	)
	ext_modules = mypycify(
		[
			*(str(p) for p in Path("src/camas/main").glob("*.py") if p.name not in _main_excluded),
			*(
				str(p)
				for p in Path("src/camas/effect").glob("[!_]*.py")
				if p.name not in _importers_of_optional_dependencies
			),
		],
		opt_level="3",
	)
else:
	ext_modules = []

setup(ext_modules=ext_modules)
