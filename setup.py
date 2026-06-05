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
	# github_checks.py stays interpreted: its optional httpx dep isn't available
	# in the isolated build env, so mypy/mypyc compilation can't resolve it.
	# check.py and state.py stay interpreted: mypyc's NamedTuple codegen rejects
	# the built-in ``Exception`` as a field type (KeyError: 'Exception' at import).
	# parser.py stays interpreted: its ``Cli`` argtree spec is parsed at runtime
	# via ``typing.get_type_hints``, which mypyc's annotation erasure defeats.
	_main_excluded = {"check.py", "state.py", "parser.py"}
	ext_modules = mypycify(
		[
			*(
				p
				for p in glob.glob("src/camas/main/*.py")
				if os.path.basename(p) not in _main_excluded
			),
			*(
				p
				for p in glob.glob("src/camas/effect/[!_]*.py")
				if os.path.basename(p) != "github_checks.py"
			),
		],
		opt_level="3",
	)
else:
	ext_modules = []

setup(ext_modules=ext_modules)
