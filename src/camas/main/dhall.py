# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Load task definitions from a ``tasks.dhall`` scope (requires the ``camas[dhall]`` extra).

Dhall has no recursive types, so a group names its children (``refs``) instead of nesting
them — the same by-name model as ``[tool.camas.tasks]``. :func:`build_scope` turns the marshalled
record into the very binding scope a ``tasks.py`` would produce (names → ``Task``/``Sequential``/
``Parallel``/``ProjectRef``, plus a ``Config``), which :func:`camas.main.compose.scope._compose_scope`
then composes exactly as it does a Python scope — matrix expansion and monorepo ``Project`` mounting
included. Marshalling returns ``Any``; every value is validated back to a concrete type here so no
``Any`` escapes, mirroring the ``tomllib`` boundary in :mod:`camas.main.tasks`.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypeGuard, cast

from ..v0.config import Claude, Config
from ..v0.task import AgentFormat, Parallel, Project, Sequential, Task

if TYPE_CHECKING:
	from ..v0.task import OutputKind, TaskNode


_OUTPUT_KINDS: Final[frozenset[str]] = frozenset({"sarif", "rdjson", "lsp", "junit", "tap", "raw"})


def _is_str_dict(value: Any) -> TypeGuard[dict[str, Any]]:
	return isinstance(value, dict) and all(isinstance(k, str) for k in value)  # pyright: ignore[reportUnknownVariableType]


def _rec(value: Any, ctx: str) -> dict[str, Any]:
	"""``value`` as a string-keyed record.

	Raises:
		ValueError: when ``value`` is not a record.

	>>> _rec({"cmd": "x"}, "task")
	{'cmd': 'x'}
	>>> _rec("nope", "task")
	Traceback (most recent call last):
	ValueError: task: expected a record, got str
	"""
	if _is_str_dict(value):
		return value
	raise ValueError(f"{ctx}: expected a record, got {type(value).__name__}")


def _str(rec: dict[str, Any], key: str, ctx: str) -> str:
	"""A text field, defaulting to ``""`` when absent (the prelude's unset sentinel).

	Raises:
		ValueError: when the field is present but not text.
	"""
	value = rec.get(key, "")
	if isinstance(value, str):
		return value
	raise ValueError(f"{ctx}.{key}: expected text, got {type(value).__name__}")


def _opt_str(rec: dict[str, Any], key: str, ctx: str) -> str | None:
	"""A text field, with the ``""`` sentinel collapsed to ``None``."""
	return _str(rec, key, ctx) or None


def _bool(rec: dict[str, Any], key: str, ctx: str) -> bool:
	"""A boolean field, defaulting to ``False``.

	Raises:
		ValueError: when the field is present but not a boolean.
	"""
	value = rec.get(key, False)
	if isinstance(value, bool):
		return value
	raise ValueError(f"{ctx}.{key}: expected a boolean, got {type(value).__name__}")


def _str_map(rec: dict[str, Any], key: str, ctx: str) -> dict[str, str]:
	"""A ``Map Text`` field (marshalled to a dict), defaulting to ``{}``.

	Raises:
		ValueError: when the field is not a text→text mapping.
	"""
	value = rec.get(key, {})
	if _is_str_dict(value) and all(isinstance(v, str) for v in value.values()):
		return {k: v for k, v in value.items() if isinstance(v, str)}
	raise ValueError(f"{ctx}.{key}: expected a Map Text, got {value!r}")


def _str_list(value: Any, ctx: str) -> tuple[str, ...]:
	"""``value`` as a tuple of text.

	Raises:
		ValueError: when ``value`` is not a list of text.
	"""
	if isinstance(value, list) and all(isinstance(v, str) for v in cast("list[object]", value)):
		return tuple(cast("list[str]", value))
	raise ValueError(f"{ctx}: expected a list of text, got {value!r}")


def _refs(rec: dict[str, Any], ctx: str) -> tuple[str, ...]:
	return _str_list(rec.get("refs", []), f"{ctx}.refs")


def _when(rec: dict[str, Any], ctx: str) -> tuple[str, ...] | None:
	"""The ``when`` predicate: an empty list is the unset sentinel (``None``)."""
	return _str_list(rec.get("when", []), f"{ctx}.when") or None


def _matrix(rec: dict[str, Any], ctx: str) -> dict[str, tuple[str, ...]] | None:
	"""A ``Map (List Text)`` axis mapping; an empty map is the unset sentinel (``None``).

	Raises:
		ValueError: when the field is not an axis→values mapping.
	"""
	value = rec.get("matrix", {})
	if not _is_str_dict(value):
		raise ValueError(f"{ctx}.matrix: expected a Map (List Text), got {value!r}")
	if not value:
		return None
	return {axis: _str_list(value[axis], f"{ctx}.matrix.{axis}") for axis in value}


def _agent_format(rec: dict[str, Any], ctx: str) -> AgentFormat | None:
	"""An optional ``AgentFormat`` record (marshalled to a dict or ``None``).

	Raises:
		ValueError: when ``kind`` is not one of the supported output kinds.
	"""
	value = rec.get("agent_format")
	if value is None:
		return None
	af = _rec(value, f"{ctx}.agent_format")
	kind = _str(af, "kind", f"{ctx}.agent_format")
	if kind not in _OUTPUT_KINDS:
		raise ValueError(
			f"{ctx}.agent_format.kind: expected one of {', '.join(sorted(_OUTPUT_KINDS))}, got {kind!r}"
		)
	return AgentFormat(
		_str(af, "args", f"{ctx}.agent_format"), cast("OutputKind", kind), _limit(af, ctx)
	)


def _limit(af: dict[str, Any], ctx: str) -> int:
	"""The ``AgentFormat`` character limit, defaulting to the model's own default.

	Raises:
		ValueError: when the limit is present but not a positive integer.
	"""
	value = af.get("limit", AgentFormat._field_defaults["limit"])
	if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
		raise ValueError(f"{ctx}.agent_format.limit: expected a positive integer, got {value!r}")
	return value


def _leaf(rec: dict[str, Any], ctx: str) -> Task:
	return Task(
		cmd=_str(rec, "cmd", ctx),
		name=_opt_str(rec, "name", ctx),
		env=_str_map(rec, "env", ctx),
		cwd=_opt_str(rec, "cwd", ctx),
		help=_opt_str(rec, "help", ctx),
		mutates=_bool(rec, "mutates", ctx),
		paths=_opt_str(rec, "paths", ctx),
		when=_when(rec, ctx),
		agent_format=_agent_format(rec, ctx),
	)


def _group(rec: dict[str, Any], kind: str, children: tuple[TaskNode, ...], ctx: str) -> TaskNode:
	ctor = Sequential if kind == "sequential" else Parallel
	return ctor(
		*children,
		name=_opt_str(rec, "name", ctx),
		matrix=_matrix(rec, ctx),
		env=_str_map(rec, "env", ctx),
		cwd=_opt_str(rec, "cwd", ctx),
		help=_opt_str(rec, "help", ctx),
		paths=_opt_str(rec, "paths", ctx),
		when=_when(rec, ctx),
	)


def build_scope(data: object, path: Path) -> dict[str, object]:
	"""The marshalled ``tasks.dhall`` record as a camas binding scope.

	Each ``tasks`` entry becomes a ``Task``/``Sequential``/``Parallel``/``ProjectRef`` bound under
	its key; group ``refs`` are resolved by name to the *same* interned instances (so downstream
	name propagation and identity checks behave as they do for a ``tasks.py``). The ``config``
	record, when present, is bound under ``_``. A malformed record, an unknown ref, or a ref cycle
	raises ``ValueError`` (from the field validators and ref resolution).
	"""
	root = _rec(data, str(path))
	specs = _rec(root.get("tasks", {}), f"{path}: tasks")
	memo: dict[str, TaskNode] = {}

	def resolve(name: str, visiting: frozenset[str]) -> TaskNode:
		if name in memo:
			return memo[name]
		if name in visiting:
			raise ValueError(f"cycle in task refs: {' -> '.join([*sorted(visiting), name])}")
		if name not in specs:
			known = ", ".join(sorted(specs)) or "none"
			raise ValueError(f"unknown task ref {name!r} (known: {known})")
		rec = _rec(specs[name], f"{path}: task {name!r}")
		kind = _str(rec, "kind", name)
		match kind:
			case "task":
				node: TaskNode = _leaf(rec, name)
			case "project":
				node = Project(_str(rec, "path", name))
			case "sequential" | "parallel":
				children = tuple(resolve(ref, visiting | {name}) for ref in _refs(rec, name))
				node = _group(rec, kind, children, name)
			case _:
				raise ValueError(f"task {name!r}: unknown kind {kind!r}")
		memo[name] = node
		return node

	scope: dict[str, object] = {name: resolve(name, frozenset()) for name in specs}
	config_raw = root.get("config")
	if config_raw is not None:
		scope["_"] = _build_config(config_raw, memo, str(path))
	return scope


def _config_ref(memo: dict[str, TaskNode], name: str, ctx: str) -> TaskNode | None:
	"""A ``Config`` field's task reference resolved to its binding; ``""`` means unset.

	Raises:
		ValueError: when the referenced task does not exist.
	"""
	if not name:
		return None
	if name not in memo:
		known = ", ".join(sorted(memo)) or "none"
		raise ValueError(f"{ctx} references unknown task {name!r} (known: {known})")
	return memo[name]


def _build_config(raw: object, memo: dict[str, TaskNode], path: str) -> Config:
	"""A ``Config`` (and optional ``Claude`` agent) from the marshalled ``config`` record.

	Raises:
		ValueError: when the record shape is wrong or ``agent.fix`` is missing.
	"""
	rec = _rec(raw, f"{path}: config")
	agent_raw = rec.get("agent")
	agent: Claude | None = None
	if agent_raw is not None:
		arec = _rec(agent_raw, f"{path}: config.agent")
		fix = _config_ref(memo, _str(arec, "fix", "config.agent"), "config.agent.fix")
		if fix is None:
			raise ValueError(f"{path}: config.agent.fix is required")
		agent = Claude(
			fix=fix,
			check=_config_ref(memo, _str(arec, "check", "config.agent"), "config.agent.check"),
			default=_config_ref(
				memo, _str(arec, "default", "config.agent"), "config.agent.default"
			),
		)
	return Config(
		default_task=_config_ref(memo, _str(rec, "default_task", "config"), "config.default_task"),
		github_task=_config_ref(memo, _str(rec, "github_task", "config"), "config.github_task"),
		camas_dir=_str(rec, "camas_dir", "config") or ".camas",
		agent=agent,
	)


def evaluate_dhall(path: Path) -> object:
	"""``path`` evaluated to native Python data by the ``dhall`` binding.

	Raises:
		RuntimeError: when the ``camas[dhall]`` extra is not installed.
	"""
	try:
		module: Any = importlib.import_module("dhall")
	except ImportError as e:
		raise RuntimeError(
			"tasks.dhall requires the 'camas[dhall]' extra (pip install 'camas[dhall]')"
		) from e
	with path.open() as handle:
		return cast("object", module.load(handle))


def prelude_path() -> Path:
	"""Filesystem path to the packaged Dhall prelude — the ``camas`` schema a ``tasks.dhall``
	imports (``let camas = <this file>``).
	"""
	return Path(__file__).resolve().parent.parent / "data" / "prelude.dhall"
