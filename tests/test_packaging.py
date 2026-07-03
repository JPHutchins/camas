# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Guard: setup.py's mypyc effect-exclusion set stays derivable from the imports.

The compiled (``CAMAS_USE_MYPYC=1``) build keeps every effect module with an
optional dependency interpreted, because those deps are absent from the isolated
build env. ``setup.py`` hardcodes that set; this test derives it from the actual
effect-module imports and the pyproject extras and asserts they agree, so drift
fails the fast ``check`` job instead of the slow wheels build.
"""

from __future__ import annotations

import ast
import itertools
import string
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 11):
	import tomllib
else:  # pragma: no cover
	import tomli as tomllib

if TYPE_CHECKING:
	from collections.abc import Mapping

ROOT: Final = Path(__file__).resolve().parent.parent
EFFECT_DIR: Final = ROOT / "src" / "camas" / "effect"
NAME_CHARS: Final = frozenset(string.ascii_letters + string.digits + "._-")


def dist_name(spec: str) -> str:
	"""The leading PEP 508 distribution name of a requirement spec."""
	return "".join(itertools.takewhile(lambda char: char in NAME_CHARS, spec.strip()))


def optional_dist_names(pyproject: Mapping[str, Any]) -> frozenset[str]:
	"""Distribution names reachable only through a ``[project.optional-dependencies]`` extra."""
	project = pyproject["project"]
	core = frozenset(dist_name(spec) for spec in project["dependencies"])
	optional = frozenset(
		dist_name(spec)
		for specs in project["optional-dependencies"].values()
		for spec in specs
		if not spec.startswith("camas[")
	)
	return optional - core


def import_roots(node: ast.AST) -> tuple[str, ...]:
	"""Absolute (level-0) import roots contributed by a single AST node."""
	match node:
		case ast.Import(names=names):
			return tuple(alias.name.split(".", 1)[0] for alias in names)
		case ast.ImportFrom(module=str(module), level=0):
			return (module.split(".", 1)[0],)
		case _:
			return ()


def sibling_imports(node: ast.AST) -> tuple[str, ...]:
	"""Sibling names a single AST node pulls in via ``from . import X``."""
	match node:
		case ast.ImportFrom(module=None, level=1, names=names):
			return tuple(alias.name for alias in names)
		case _:
			return ()


def module_roots(tree: ast.AST) -> frozenset[str]:
	return frozenset(root for node in ast.walk(tree) for root in import_roots(node))


def module_siblings(tree: ast.AST) -> frozenset[str]:
	return frozenset(name for node in ast.walk(tree) for name in sibling_imports(node))


def parse_module(path: Path) -> ast.Module:
	return ast.parse(path.read_text(encoding="utf-8"))


def combined_roots(path: Path) -> frozenset[str]:
	"""Import roots of an effect module plus those of its ``from . import`` siblings."""
	tree = parse_module(path)
	return module_roots(tree) | frozenset(
		root
		for sibling in module_siblings(tree)
		for root in module_roots(parse_module(EFFECT_DIR / f"{sibling}.py"))
	)


def public_effect_files() -> tuple[Path, ...]:
	return tuple(path for path in sorted(EFFECT_DIR.glob("*.py")) if not path.name.startswith("_"))


def derived_exclusion(optional: frozenset[str]) -> frozenset[str]:
	"""Public effect modules whose own or sibling imports reach an optional dependency."""
	return frozenset(path.name for path in public_effect_files() if combined_roots(path) & optional)


def effect_excluded_members(node: ast.AST) -> tuple[str, ...]:
	"""The members of the ``_effect_excluded`` set literal at a single AST node."""
	match node:
		case ast.Assign(targets=[ast.Name(id="_effect_excluded")], value=value):
			return tuple(ast.literal_eval(value))
		case _:
			return ()


def literal_exclusion(source: str) -> frozenset[str]:
	"""The ``_effect_excluded`` set literal declared in setup.py source."""
	return frozenset(
		member for node in ast.walk(ast.parse(source)) for member in effect_excluded_members(node)
	)


def test_effect_exclusion_matches_derived_set() -> None:
	pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
	optional = optional_dist_names(pyproject)
	derived = derived_exclusion(optional)
	literal = literal_exclusion((ROOT / "setup.py").read_text(encoding="utf-8"))
	assert derived == literal, (
		"setup.py _effect_excluded has drifted from the imports; "
		f"symmetric difference = {sorted(derived ^ literal)} "
		f"(derived = {sorted(derived)}, literal = {sorted(literal)})"
	)
