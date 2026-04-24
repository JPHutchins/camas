import os
import sys

from setuptools import setup

is_editable = any("editable" in arg or arg == "develop" for arg in sys.argv)
use_mypyc = os.environ.get("CAMAS_USE_MYPYC") == "1" and not is_editable

if use_mypyc:
	from mypyc.build import mypycify

	ext_modules = mypycify(
		[
			"src/camas/main.py",
			"src/camas/effect/summary.py",
			"src/camas/effect/termtree.py",
		],
		opt_level="3",
	)
else:
	ext_modules = []

setup(ext_modules=ext_modules)
