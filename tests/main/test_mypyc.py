# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
import sys
import types
from typing import TYPE_CHECKING, NamedTuple

from camas.main.effects import signature_fields
from camas.main.mypyc import MISSING, resolve_in_namespace, signature_fields_from_source

if TYPE_CHECKING:
	from pathlib import Path

	import pytest


def test_signature_fields_from_source_namedtuple() -> None:
	from camas.effect.summary import SummaryOptions

	out = signature_fields_from_source(SummaryOptions)
	assert out is not None
	names = [n for n, _, _, _ in out]
	assert names == ["term_width", "show_passing"]


def test_signature_fields_from_source_regular_class() -> None:
	from camas.effect.summary import Summary, SummaryOptions

	out = signature_fields_from_source(Summary)
	assert out is not None
	assert len(out) == 1
	name, _kind, annot, default = out[0]
	assert name == "options"
	assert "SummaryOptions" in str(annot)
	assert SummaryOptions.__name__ in str(annot)
	assert default == SummaryOptions()


def test_signature_fields_from_source_module_not_loaded() -> None:
	class Stranded: ...

	Stranded.__module__ = "no_such_module_xyz"
	assert signature_fields_from_source(Stranded) is None


def test_signature_fields_from_source_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
	mod = types.ModuleType("fake_no_file")
	monkeypatch.setitem(sys.modules, "fake_no_file", mod)

	class C: ...

	C.__module__ = "fake_no_file"
	assert signature_fields_from_source(C) is None


def test_signature_fields_from_source_missing_py(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	fake_so = tmp_path / "ghost.cpython-314-x86_64-linux-gnu.so"
	fake_so.write_bytes(b"")
	mod = types.ModuleType("ghost")
	mod.__file__ = str(fake_so)
	monkeypatch.setitem(sys.modules, "ghost", mod)

	class C: ...

	C.__module__ = "ghost"
	assert signature_fields_from_source(C) is None


def test_signature_fields_from_source_class_not_in_file(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	source = tmp_path / "tinymod.py"
	source.write_text("class Other: pass\n")
	mod = types.ModuleType("tinymod")
	mod.__file__ = str(source)
	monkeypatch.setitem(sys.modules, "tinymod", mod)

	class Missing: ...

	Missing.__module__ = "tinymod"
	assert signature_fields_from_source(Missing) is None


def test_signature_fields_from_source_no_init(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	source = tmp_path / "noinit.py"
	source.write_text("class C:\n    pass\n")
	mod = types.ModuleType("noinit")
	mod.__file__ = str(source)
	monkeypatch.setitem(sys.modules, "noinit", mod)

	class C: ...

	C.__module__ = "noinit"
	assert signature_fields_from_source(C) is None


def test_signature_fields_from_source_parse_error(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	source = tmp_path / "broken.py"
	source.write_text("def x(:\n")
	mod = types.ModuleType("broken")
	mod.__file__ = str(source)
	monkeypatch.setitem(sys.modules, "broken", mod)

	class C: ...

	C.__module__ = "broken"
	assert signature_fields_from_source(C) is None


def test_signature_fields_from_source_with_required_arg(
	tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
	"""Required (non-defaulted) ``__init__`` params return ``MISSING``."""
	source = tmp_path / "required.py"
	source.write_text(
		"class C:\n"
		"    def __init__(self, must_have: int, optional: str = 'x') -> None:\n"
		"        ...\n"
	)
	mod = types.ModuleType("required")
	mod.__file__ = str(source)
	monkeypatch.setitem(sys.modules, "required", mod)

	class C: ...

	C.__module__ = "required"
	out = signature_fields_from_source(C)
	assert out is not None
	(must_have, optional) = out
	assert must_have[0] == "must_have"
	assert must_have[3] is MISSING
	assert optional[0] == "optional"
	assert optional[3] == "x"


def test_signature_fields_falls_back_for_compiled_namedtuple(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""When ``get_type_hints`` returns an ``Any``/``type`` annotation (the mypyc
	fingerprint for NamedTuples), ``signature_fields`` falls through to the
	source-parse path."""
	import typing as _typing

	from camas.effect.summary import SummaryOptions

	def fake_hints(_obj: object) -> dict[str, object]:
		return {"term_width": type, "show_passing": bool}

	monkeypatch.setattr(_typing, "get_type_hints", fake_hints)
	out = signature_fields(SummaryOptions)
	annot_term = next(annot for n, _, annot, _ in out if n == "term_width")
	assert "Auto" in str(annot_term) or "Fixed" in str(annot_term)


def test_signature_fields_falls_back_for_compiled_class(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""When ``inspect.signature`` returns all-``Any`` annotations (the mypyc
	fingerprint for regular classes), ``signature_fields`` falls through to
	source-parse."""
	import inspect as _inspect

	from camas.effect.summary import Summary

	def fake_signature(_obj: object) -> _inspect.Signature:
		return _inspect.Signature(
			parameters=[
				_inspect.Parameter(
					"options", _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
				)
			]
		)

	monkeypatch.setattr(_inspect, "signature", fake_signature)
	out = signature_fields(Summary)
	annot = next(annot for n, _, annot, _ in out if n == "options")
	assert "SummaryOptions" in str(annot)


def test_signature_fields_namedtuple_fallback_returns_none(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""When the NamedTuple inspect path returns ``Any``/``type`` and the
	source-parse fallback also fails (class not loaded as module), keep the
	original inspected fields rather than crashing."""
	import typing as _typing

	class Stranded(NamedTuple):
		x: int = 1

	def fake_hints(_obj: object) -> dict[str, object]:
		return {"x": type}

	monkeypatch.setattr(_typing, "get_type_hints", fake_hints)
	Stranded.__module__ = "no_such_module_xyz"
	out = signature_fields(Stranded)
	assert [n for n, _, _, _ in out] == ["x"]


def test_signature_fields_class_fallback_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Same defensive path for ordinary classes."""
	import inspect as _inspect

	class Stranded:
		def __init__(self, x: int = 1) -> None: ...

	def fake_signature(_o: object) -> _inspect.Signature:
		return _inspect.Signature(
			parameters=[
				_inspect.Parameter("x", _inspect.Parameter.POSITIONAL_OR_KEYWORD, default=1)
			]
		)

	monkeypatch.setattr(_inspect, "signature", fake_signature)
	Stranded.__module__ = "no_such_module_xyz"
	out = signature_fields(Stranded)
	assert [n for n, _, _, _ in out] == ["x"]


def test_resolve_in_namespace_evaluates() -> None:
	node = ast.parse("1 + 2", mode="eval").body
	assert resolve_in_namespace(node, {}) == 3


def test_resolve_in_namespace_falls_back_to_string() -> None:
	node = ast.parse("undefined_name_xyz", mode="eval").body
	assert resolve_in_namespace(node, {}) == "undefined_name_xyz"


def test_signature_fields_falls_back_when_inspect_signature_raises(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""When ``inspect.signature`` raises (the mypyc fingerprint for compiled
	classes whose ``__init__`` is a C function with no inspectable signature),
	``signature_fields`` falls through to the source-parse path."""
	import inspect as _inspect

	from camas.effect.summary import Summary

	def boom(*_a: object, **_kw: object) -> _inspect.Signature:
		raise ValueError("no signature")

	monkeypatch.setattr(_inspect, "signature", boom)
	out = signature_fields(Summary)
	annot = next(annot for n, _, annot, _ in out if n == "options")
	assert "SummaryOptions" in str(annot)


def test_signature_fields_namedtuple_unresolvable_hints(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	import typing as _typing

	from camas.effect.summary import Auto

	def boom(_obj: object) -> dict[str, object]:
		raise NameError("forward ref")

	monkeypatch.setattr(_typing, "get_type_hints", boom)
	fields = signature_fields(Auto)
	assert fields == []
