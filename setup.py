import glob
import os
import sys

from setuptools import setup

is_editable = any("editable" in arg or arg == "develop" for arg in sys.argv)
use_mypyc = os.environ.get("CAMAS_USE_MYPYC") == "1" and not is_editable

if use_mypyc:
	from mypyc.build import mypycify

	# core/ stays interpreted: Effect's @runtime_checkable Protocol and Group's
	# @dataclass-generated __eq__ don't survive compilation.
	ext_modules = mypycify(
		[
			*glob.glob("src/camas/main/*.py"),
			*glob.glob("src/camas/effect/[!_]*.py"),
		],
		opt_level="3",
	)
else:
	ext_modules = []

setup(ext_modules=ext_modules)
