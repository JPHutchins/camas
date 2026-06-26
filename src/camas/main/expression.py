# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""AST-evaluate the CLI task-expression mini-language into a task tree."""

from __future__ import annotations

import ast
import json
import re
import sys
from typing import TYPE_CHECKING, Final, NamedTuple, cast

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..v0.task import AgentFormat, OutputKind, Parallel, Sequential, Task, TaskNode

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import PathScope


class Ref(NamedTuple):
	"""Parser-only sentinel for a task referenced by name inside a config expression.

	>>> Ref("lint")
	Ref(name='lint')
	"""

	name: str


CONSTRUCTORS: Final = {
	Task.__name__: Task,
	Sequential.__name__: Sequential,
	Parallel.__name__: Parallel,
}

CONFIG_CONSTRUCTORS: Final = CONSTRUCTORS | {Ref.__name__: Ref}

EXPRESSION_PATTERN: Final = re.compile(r"^\s*(?:(?:Task|Sequential|Parallel|Ref)\s*\(|[(\{])")


def format_syntax_error(source: str, err: SyntaxError) -> str:
	"""Format a SyntaxError against ``source`` with a caret pointing at the offending column.

	>>> import ast
	>>> try: ast.parse("Task(", mode="eval")
	... except SyntaxError as e: print(format_syntax_error("Task(", e))
	invalid syntax (line 1, column 5): '(' was never closed
	    Task(
	        ^
	"""
	lineno = err.lineno or 0
	offset = err.offset or 0
	msg = err.msg or "invalid syntax"
	lines = source.splitlines()
	if not (1 <= lineno <= len(lines)) or offset < 1:
		return f"invalid syntax: {msg}"
	line = lines[lineno - 1]
	expanded = line.expandtabs(4)
	caret_col = len(line[: offset - 1].expandtabs(4))
	return (
		f"invalid syntax (line {lineno}, column {offset}): {msg}\n"
		f"    {expanded}\n"
		f"    {' ' * caret_col}^"
	)


def eval_str_lit(node: ast.expr) -> str:
	match node:
		case ast.Constant(value=str() as s):
			return s
		case _:
			raise ValueError(f"expected string literal, got {ast.dump(node)}")


def eval_cmd(node: ast.expr) -> str | tuple[str, ...]:
	match node:
		case ast.Constant(value=str() as s):
			return s
		case ast.Tuple(elts=elts):
			return tuple(eval_str_lit(e) for e in elts)
		case _:
			raise ValueError(f"cmd must be str or tuple of str, got {ast.dump(node)}")


def eval_opt_str(node: ast.expr | None) -> str | None:
	match node:
		case None:
			return None
		case ast.Constant(value=str() as s):
			return s
		case ast.Constant(value=None):
			return None
		case _:
			raise ValueError(f"expected str or None, got {ast.dump(node)}")


def eval_opt_bool(node: ast.expr | None) -> bool:
	match node:
		case None:
			return False
		case ast.Constant(value=bool() as b):
			return b
		case _:
			raise ValueError(f"expected bool literal, got {ast.dump(node)}")


def _output_kind(node: ast.expr) -> OutputKind:
	match node:
		case ast.Constant(value="sarif" | "rdjson" | "lsp" | "junit" | "tap" | "raw" as kind):
			return kind
		case _:
			raise ValueError(
				f"AgentFormat kind must be sarif/rdjson/lsp/junit/tap/raw, got {ast.dump(node)}"
			)


def eval_agent_format(node: ast.expr | None) -> AgentFormat | None:
	match node:
		case None:
			return None
		case ast.Call(func=ast.Name(id="AgentFormat"), args=call_args, keywords=call_kw):
			kw = {k.arg: k.value for k in call_kw if k.arg is not None}
			args_node = call_args[0] if call_args else kw.get("args")
			kind_node = call_args[1] if len(call_args) > 1 else kw.get("kind")
			if args_node is None or kind_node is None:
				raise ValueError("AgentFormat requires args and kind")
			return AgentFormat(eval_str_lit(args_node), _output_kind(kind_node))
		case ast.Tuple(elts=[args_node, kind_node]):
			return AgentFormat(eval_str_lit(args_node), _output_kind(kind_node))
		case _:
			raise ValueError(f"agent_format must be AgentFormat(args, kind), got {ast.dump(node)}")


def eval_env(node: ast.expr | None) -> dict[str, str]:
	match node:
		case None:
			return {}
		case ast.Dict(keys=keys, values=values):
			if any(k is None for k in keys):
				raise ValueError("env: ** unpacking is not supported")
			return {
				eval_str_lit(k): eval_str_lit(v)
				for k, v in zip(keys, values, strict=True)
				if k is not None
			}
		case _:
			raise ValueError(f"env must be a dict of str→str, got {ast.dump(node)}")


def eval_str_tuple(node: ast.expr) -> tuple[str, ...]:
	match node:
		case ast.Tuple(elts=elts):
			return tuple(eval_str_lit(e) for e in elts)
		case _:
			raise ValueError(f"expected tuple of str, got {ast.dump(node)}")


def eval_matrix(node: ast.expr | None) -> dict[str, tuple[str, ...]] | None:
	match node:
		case None:
			return None
		case ast.Dict(keys=keys, values=values):
			if any(k is None for k in keys):
				raise ValueError("matrix: ** unpacking is not supported")
			return {
				eval_str_lit(k): eval_str_tuple(v)
				for k, v in zip(keys, values, strict=True)
				if k is not None
			}
		case _:
			raise ValueError(f"matrix must be a dict of str→tuple[str,...], got {ast.dump(node)}")


def children(elts: list[ast.expr], allow_refs: bool) -> tuple[TaskNode, ...]:
	"""Evaluate task-position children. ``Ref``s pass through here at the type level
	via ``cast`` — they're resolved later by ``resolve_refs``.
	"""
	return cast("tuple[TaskNode, ...]", tuple(eval_task_pos(e, allow_refs) for e in elts))


def eval_task_pos(node: ast.expr, allow_refs: bool) -> TaskNode | Ref:
	"""Evaluate an AST node at a *task position*, coercing literals to nodes; other
	forms delegate to ``eval_node``.

	>>> import ast
	>>> eval_task_pos(ast.parse('"echo hi"', mode="eval").body, allow_refs=False)
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> eval_task_pos(ast.parse('(Task("a"), Task("b"))', mode="eval").body, allow_refs=False).tasks  # type: ignore[union-attr]
	(Task(cmd='a', name=None, env={}, cwd=None), Task(cmd='b', name=None, env={}, cwd=None))
	>>> isinstance(eval_task_pos(ast.parse('{"a", "b"}', mode="eval").body, allow_refs=False), Parallel)
	True
	"""
	match node:
		case ast.Constant(value=str() as s):
			return Task(cmd=s)
		case ast.Tuple(elts=elts):
			return Sequential(*children(elts, allow_refs))
		case ast.Set(elts=elts):
			return Parallel(*children(elts, allow_refs))
		case _:
			return eval_node(node, allow_refs)


def eval_node(
	node: ast.expr,
	allow_refs: bool = False,
) -> TaskNode | Ref:
	"""Walk an AST node and construct the corresponding TaskNode or Ref safely (no eval).

	Raises:
		ValueError: on names or call shapes outside the expression language.

	>>> import ast
	>>> eval_node(ast.parse('Task("echo hi")', mode="eval").body)
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> eval_node(ast.parse('lint', mode="eval").body, allow_refs=True)
	Ref(name='lint')
	>>> eval_node(ast.parse('Ref("x")', mode="eval").body, allow_refs=True)
	Ref(name='x')
	>>> eval_node(ast.parse('Sequential(Task("a"), "b")', mode="eval").body).tasks  # type: ignore[union-attr]
	(Task(cmd='a', name=None, env={}, cwd=None), Task(cmd='b', name=None, env={}, cwd=None))
	"""
	match node:
		case ast.Call(func=ast.Name(id=name), args=args, keywords=keywords):
			constructors = CONFIG_CONSTRUCTORS if allow_refs else CONSTRUCTORS
			if name not in constructors:
				raise ValueError(f"unknown type: {name!r} (expected {', '.join(constructors)})")
			kw = {k.arg: k.value for k in keywords if k.arg is not None}
			match name:
				case "Task":
					cmd_node = args[0] if args else kw.get("cmd")
					if cmd_node is None:
						raise ValueError("Task requires 'cmd'")
					return Task(
						cmd=eval_cmd(cmd_node),
						name=eval_opt_str(kw.get("name")),
						env=eval_env(kw.get("env")),
						cwd=eval_opt_str(kw.get("cwd")),
						help=eval_opt_str(kw.get("help")),
						mutates=eval_opt_bool(kw.get("mutates")),
						paths=eval_opt_str(kw.get("paths")),
						agent_format=eval_agent_format(kw.get("agent_format")),
					)
				case "Sequential" | "Parallel":
					ctor = Sequential if name == "Sequential" else Parallel
					return ctor(
						*children(args, allow_refs),
						name=eval_opt_str(kw.get("name")),
						matrix=eval_matrix(kw.get("matrix")),
						env=eval_env(kw.get("env")),
						cwd=eval_opt_str(kw.get("cwd")),
						help=eval_opt_str(kw.get("help")),
					)
				case "Ref":
					ref_name_node = args[0] if args else kw.get("name")
					if ref_name_node is None:
						raise ValueError("Ref requires 'name'")
					return Ref(eval_str_lit(ref_name_node))
				case _:
					raise ValueError(f"unknown type: {name!r}")
		case ast.Name(id=name) if allow_refs:
			return Ref(name)
		case _:
			raise ValueError(f"unsupported syntax: {ast.dump(node)}")


def parse_expression(expr: str, tasks: Mapping[str, TaskNode] | None = None) -> TaskNode:
	"""Parse a typed Python expression string into a TaskNode tree using AST (no eval).

	When ``tasks`` is provided, bare identifiers in the expression become Refs that are
	resolved against ``tasks`` after parsing. Bare strings, tuple literals, and set
	literals are coerced into ``Task`` / ``Sequential`` / ``Parallel`` respectively.

	>>> parse_expression('Task("echo hi")')
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> parse_expression('Parallel(a)', tasks={"a": Task("x")})
	Parallel(tasks=(Task(cmd='x', name=None, env={}, cwd=None),), name=None, matrix=None, env={}, cwd=None)
	>>> parse_expression('"echo hi"')
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> parse_expression('("a", "b")').tasks  # type: ignore[union-attr]
	(Task(cmd='a', name=None, env={}, cwd=None), Task(cmd='b', name=None, env={}, cwd=None))
	"""
	try:
		tree = ast.parse(expr, mode="eval")
	except SyntaxError as e:
		print(f"error: {format_syntax_error(expr, e)}", file=sys.stderr)
		sys.exit(2)

	try:
		result = eval_task_pos(tree.body, allow_refs=tasks is not None)
	except (ValueError, TypeError) as e:
		print(f"error: {e}", file=sys.stderr)
		sys.exit(2)

	if tasks is None:
		match result:
			case Task() | Sequential() | Parallel():
				return result
			case _:
				print(
					f"error: expression must be {', '.join(CONSTRUCTORS)}, got {type(result).__name__}",
					file=sys.stderr,
				)
				sys.exit(2)
	try:
		return resolve_refs(result, tasks, frozenset())
	except ValueError as e:
		print(f"error: {e}", file=sys.stderr)
		sys.exit(2)


def to_expression(node: TaskNode) -> str:
	"""Render a task tree to a fully-typed camas expression — the verbose inverse of
	:func:`parse_expression`, in the same notation as ``tasks.py``.

	Every node is its constructor call with the command/children expanded and the
	``name`` shown, so an agent reading it sees the exact structure, the literal
	commands, and the handle of every task — and could paste it back into ``tasks.py``
	or a ``camas`` expression unchanged.

	>>> to_expression(Task("echo hi"))
	'Task("echo hi")'
	>>> to_expression(Task(("ruff", "check", ".")))
	'Task("ruff check .")'
	>>> to_expression(Task("ruff", name="lint"))
	'Task("ruff", name="lint")'
	>>> to_expression(Task("ruff format .", mutates=True))
	'Task("ruff format .", mutates=True)'
	>>> to_expression(Sequential(Task("a"), Task("b")))
	'Sequential(Task("a"), Task("b"))'
	>>> to_expression(Parallel(Task("a"), Task("b"), name="checks"))
	'Parallel(Task("a"), Task("b"), name="checks")'
	>>> to_expression(Sequential(Parallel(Task("a"), name="p"), Task("b"), name="s"))
	'Sequential(Parallel(Task("a"), name="p"), Task("b"), name="s")'
	>>> to_expression(Task("ruff check {paths}", name="lint", paths="src"))
	'Task("ruff check {paths}", name="lint", paths="src")'
	>>> from camas.v0.task import AgentFormat
	>>> to_expression(Task("ruff check .", agent_format=AgentFormat("--output-format sarif", "sarif")))
	'Task("ruff check .", agent_format=AgentFormat("--output-format sarif", "sarif"))'
	"""
	match node:
		case Task(cmd=cmd, name=name, mutates=mutates, paths=paths, agent_format=agent_format):
			return (
				f"Task({quote(join_command(cmd))}{name_kwarg(name)}{mutates_kwarg(mutates)}"
				f"{paths_kwarg(paths)}{agent_format_kwarg(agent_format)})"
			)
		case Sequential(tasks=tasks, name=name):
			return f"Sequential({render_members(tasks)}{name_kwarg(name)})"
		case Parallel(tasks=tasks, name=name):
			return f"Parallel({render_members(tasks)}{name_kwarg(name)})"
		case _:
			assert_never(node)


def render_members(tasks: tuple[TaskNode, ...]) -> str:
	return ", ".join(to_expression(child) for child in tasks)


def name_kwarg(name: str | None) -> str:
	return f", name={quote(name)}" if name is not None else ""


def mutates_kwarg(mutates: bool) -> str:
	return ", mutates=True" if mutates else ""


def paths_kwarg(paths: str | PathScope | None) -> str:
	"""A str scope round-trips; a callable scope has no source, so it renders as a
	non-parseable marker (this preview is informational, never re-evaluated by camas).
	"""
	if paths is None:
		return ""
	if isinstance(paths, str):
		return f", paths={quote(paths)}"
	return ", paths=<callable>"


def agent_format_kwarg(agent_format: AgentFormat | None) -> str:
	if agent_format is None:
		return ""
	return f", agent_format=AgentFormat({quote(agent_format.args)}, {quote(agent_format.kind)})"


def quote(value: str) -> str:
	return json.dumps(value, ensure_ascii=False)


def join_command(cmd: str | tuple[str, ...]) -> str:
	return cmd if isinstance(cmd, str) else " ".join(cmd)


def parse_task_value(raw: str) -> TaskNode | Ref:
	r"""Parse a single pyproject.toml task value. Bare strings become ``Task(cmd)``;
	a leading ``Task``/``Sequential``/``Parallel``/``Ref`` call, ``(``, or ``{`` triggers
	AST parsing (so ``(a, b)`` is a Sequential and ``{a, b}`` a Parallel).

	Raises:
		ValueError: when an expression-like value fails to parse.

	>>> parse_task_value("ruff check .")
	Task(cmd='ruff check .', name=None, env={}, cwd=None)
	>>> parse_task_value("(a, b)")
	Sequential(tasks=(Ref(name='a'), Ref(name='b')), name=None, matrix=None, env={}, cwd=None)
	>>> isinstance(parse_task_value("{a, b}"), Parallel)
	True
	"""
	if not EXPRESSION_PATTERN.match(raw):
		return Task(cmd=raw)
	try:
		tree = ast.parse(raw, mode="eval")
	except SyntaxError as e:
		raise ValueError(format_syntax_error(raw, e)) from e
	return eval_task_pos(tree.body, allow_refs=True)


def resolve_refs(
	node: TaskNode | Ref,
	defs: Mapping[str, TaskNode | Ref],
	visiting: frozenset[str],
) -> TaskNode:
	"""Recursively substitute Ref(name) with its defined TaskNode. Detects cycles.

	Raises:
		ValueError: on an unknown ref or a reference cycle.

	>>> resolve_refs(Task("x"), {}, frozenset())
	Task(cmd='x', name=None, env={}, cwd=None)
	>>> resolve_refs(Ref("a"), {"a": Task("hi")}, frozenset())
	Task(cmd='hi', name=None, env={}, cwd=None)
	"""
	match node:
		case Ref(name=name):
			if name in visiting:
				chain = " -> ".join([*sorted(visiting), name])
				raise ValueError(f"cycle in task refs: {chain}")
			if name not in defs:
				known = ", ".join(sorted(defs)) or "none"
				raise ValueError(f"unknown task ref {name!r} (known: {known})")
			return resolve_refs(defs[name], defs, visiting | {name})
		case Task():
			return node
		case Sequential(tasks=tasks, name=n, matrix=m, env=e, cwd=c, help=h):
			return Sequential(
				*(resolve_refs(t, defs, visiting) for t in tasks),
				name=n,
				matrix=m,
				env=e,
				cwd=c,
				help=h,
			)
		case Parallel(tasks=tasks, name=n, matrix=m, env=e, cwd=c, help=h):
			return Parallel(
				*(resolve_refs(t, defs, visiting) for t in tasks),
				name=n,
				matrix=m,
				env=e,
				cwd=c,
				help=h,
			)
		case _:
			assert_never(node)
