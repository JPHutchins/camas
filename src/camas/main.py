# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.metadata
import io
import re
import runpy
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple, TypeGuard, cast

if sys.version_info >= (3, 11):
	from typing import assert_never

	import tomllib
else:  # pragma: no cover
	import tomli as tomllib
	from typing_extensions import assert_never

from camas import Effect, Parallel, Sequential, Task, TaskNode, run
from camas.effect.summary import Auto, Fixed, Summary, SummaryOptions
from camas.effect.termtree import Termtree, TermtreeOptions, print_tree


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

EXAMPLES: Final = """\
examples:
    camas                            # list defined tasks
    camas <task>                     # run a defined task
    camas '<expression>'             # run an inline expression

    camas 'Parallel("ruff check .", "mypy .")'
    camas '("ruff format --check .", {"mypy .", "pyright ."}, "pytest")'
    camas 'Parallel("pytest --python {PY}", matrix={"PY": ("3.12", "3.13")})'

named tasks (tasks.py or [tool.camas.tasks] in pyproject.toml):
    lint = "ruff check ."
    test = "pytest"
    ci   = '(lint, test)'           # tuple → Sequential, set → Parallel
"""


def _format_syntax_error(source: str, err: SyntaxError) -> str:
	"""Format a SyntaxError against ``source`` with a caret pointing at the offending column.

	>>> import ast
	>>> try: ast.parse("Task(", mode="eval")
	... except SyntaxError as e: print(_format_syntax_error("Task(", e))
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


def _eval_str_lit(node: ast.expr) -> str:
	match node:
		case ast.Constant(value=str() as s):
			return s
		case _:
			raise ValueError(f"expected string literal, got {ast.dump(node)}")


def _eval_cmd(node: ast.expr) -> str | tuple[str, ...]:
	match node:
		case ast.Constant(value=str() as s):
			return s
		case ast.Tuple(elts=elts):
			return tuple(_eval_str_lit(e) for e in elts)
		case _:
			raise ValueError(f"cmd must be str or tuple of str, got {ast.dump(node)}")


def _eval_opt_str(node: ast.expr | None) -> str | None:
	match node:
		case None:
			return None
		case ast.Constant(value=str() as s):
			return s
		case ast.Constant(value=None):
			return None
		case _:
			raise ValueError(f"expected str or None, got {ast.dump(node)}")


def _eval_env(node: ast.expr | None) -> dict[str, str]:
	match node:
		case None:
			return {}
		case ast.Dict(keys=keys, values=values):
			return {
				_eval_str_lit(k): _eval_str_lit(v) for k, v in zip(keys, values) if k is not None
			}
		case _:
			raise ValueError(f"env must be a dict of str→str, got {ast.dump(node)}")


def _eval_str_tuple(node: ast.expr) -> tuple[str, ...]:
	match node:
		case ast.Tuple(elts=elts):
			return tuple(_eval_str_lit(e) for e in elts)
		case _:
			raise ValueError(f"expected tuple of str, got {ast.dump(node)}")


def _eval_matrix(node: ast.expr | None) -> dict[str, tuple[str, ...]] | None:
	match node:
		case None:
			return None
		case ast.Dict(keys=keys, values=values):
			return {
				_eval_str_lit(k): _eval_str_tuple(v) for k, v in zip(keys, values) if k is not None
			}
		case _:
			raise ValueError(f"matrix must be a dict of str→tuple[str,...], got {ast.dump(node)}")


def children(elts: list[ast.expr], allow_refs: bool) -> tuple[TaskNode, ...]:
	"""Evaluate task-position children. ``Ref``s pass through here at the type level
	via ``cast`` — they're resolved later by ``resolve_refs``."""
	return cast(tuple[TaskNode, ...], tuple(eval_task_pos(e, allow_refs) for e in elts))


def eval_task_pos(node: ast.expr, allow_refs: bool) -> TaskNode | Ref:
	"""Evaluate an AST node at a *task position* — inside ``Sequential``/``Parallel`` args
	or as a top-level expression — coercing literals to nodes.

	Coercion: ``str`` → ``Task(cmd=str)``; tuple literal → ``Sequential``;
	set literal → ``Parallel``. Other forms delegate to ``eval_node``.

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
						cmd=_eval_cmd(cmd_node),
						name=_eval_opt_str(kw.get("name")),
						env=_eval_env(kw.get("env")),
					)
				case "Sequential" | "Parallel":
					ctor = Sequential if name == "Sequential" else Parallel
					return ctor(
						*children(args, allow_refs),
						name=_eval_opt_str(kw.get("name")),
						matrix=_eval_matrix(kw.get("matrix")),
						env=_eval_env(kw.get("env")),
					)
				case "Ref":
					ref_name_node = args[0] if args else kw.get("name")
					if ref_name_node is None:
						raise ValueError("Ref requires 'name'")
					return Ref(_eval_str_lit(ref_name_node))
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
		print(f"error: {_format_syntax_error(expr, e)}", file=sys.stderr)
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


def parse_task_value(raw: str) -> TaskNode | Ref:
	"""Parse a single pyproject.toml task value. Bare strings become Task(cmd).

	A leading ``Task``/``Sequential``/``Parallel``/``Ref`` call, ``(``, or ``{`` triggers
	AST-based parsing — letting users use the fluent ``(a, b)`` (Sequential) and
	``{a, b}`` (Parallel) literals as well as explicit constructor calls.

	>>> parse_task_value("ruff check .")
	Task(cmd='ruff check .', name=None, env={}, cwd=None)
	>>> parse_task_value('Task("pytest")')
	Task(cmd='pytest', name=None, env={}, cwd=None)
	>>> parse_task_value("Ref(\\"lint\\")")
	Ref(name='lint')
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
		raise ValueError(_format_syntax_error(raw, e)) from e
	return eval_task_pos(tree.body, allow_refs=True)


def resolve_refs(
	node: TaskNode | Ref,
	defs: Mapping[str, TaskNode | Ref],
	visiting: frozenset[str],
) -> TaskNode:
	"""Recursively substitute Ref(name) with its defined TaskNode. Detects cycles.

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
		case Sequential(tasks=tasks, name=n, matrix=m, env=e, cwd=c):
			return Sequential(
				*(resolve_refs(t, defs, visiting) for t in tasks),
				name=n,
				matrix=m,
				env=e,
				cwd=c,
			)
		case Parallel(tasks=tasks, name=n, matrix=m, env=e, cwd=c):
			return Parallel(
				*(resolve_refs(t, defs, visiting) for t in tasks),
				name=n,
				matrix=m,
				env=e,
				cwd=c,
			)
		case _:
			assert_never(node)


def find_pyproject(start: Path) -> Path | None:
	"""Walk upward from ``start`` looking for a pyproject.toml."""
	for candidate in (start, *start.parents):
		if (pyproject := candidate / "pyproject.toml").is_file():
			return pyproject
	return None


def _is_str_dict(value: Any) -> TypeGuard[dict[str, Any]]:
	return isinstance(value, dict)


def _dig(data: Any, key: str) -> Any:
	return data[key] if _is_str_dict(data) and key in data else {}


def _assign_key_name(node: TaskNode | Ref, key: str) -> TaskNode | Ref:
	"""Set the TOML key as the node's name unless the expression already named it.

	>>> _assign_key_name(Task("x"), "foo")
	Task(cmd='x', name='foo', env={}, cwd=None)
	>>> _assign_key_name(Task("x", name="explicit"), "foo")
	Task(cmd='x', name='explicit', env={}, cwd=None)
	>>> _assign_key_name(Ref("bar"), "foo")
	Ref(name='bar')
	"""
	match node:
		case Task(cmd=cmd, name=None, env=env, cwd=cwd):
			return Task(cmd=cmd, name=key, env=env, cwd=cwd)
		case Sequential(tasks=tasks, name=None, matrix=matrix, env=env, cwd=cwd):
			return Sequential(*tasks, name=key, matrix=matrix, env=env, cwd=cwd)
		case Parallel(tasks=tasks, name=None, matrix=matrix, env=env, cwd=cwd):
			return Parallel(*tasks, name=key, matrix=matrix, env=env, cwd=cwd)
		case _:
			return node


def load_tasks(path: Path) -> dict[str, TaskNode]:
	"""Read [tool.camas.tasks] from a pyproject.toml and resolve all refs."""
	parsed: dict[str, Any] = tomllib.loads(path.read_text())
	raw: Any = _dig(_dig(_dig(parsed, "tool"), "camas"), "tasks")
	if not _is_str_dict(raw):
		raise ValueError(f"[tool.camas.tasks] must be a table, got {type(raw).__name__}")
	pre: dict[str, TaskNode | Ref] = {}
	for name, value in raw.items():
		if not isinstance(value, str):
			raise ValueError(f"task {name!r} must be a string, got {type(value).__name__}")
		try:
			pre[name] = _assign_key_name(parse_task_value(value), name)
		except ValueError as e:
			raise ValueError(f"task {name!r}: {e}") from e
	return {name: resolve_refs(tree, pre, frozenset({name})) for name, tree in pre.items()}


def find_tasks_py(start: Path) -> Path | None:
	"""Walk upward from ``start`` looking for a ``tasks.py``."""
	for candidate in (start, *start.parents):
		if (tasks_py := candidate / "tasks.py").is_file():
			return tasks_py
	return None


def _name_scope_bindings(scope: Mapping[str, object]) -> dict[str, TaskNode]:
	"""Collect public ``Task``/``Sequential``/``Parallel`` bindings from a module's
	globals and propagate each top-level binding's name (by id) into nested
	references — ``Parallel(mypy)`` where ``mypy`` is itself a top-level binding
	inherits ``mypy``'s name, matching ``[tool.camas.tasks]`` naming.
	"""
	bindings: Final = {
		name: val
		for name, val in scope.items()
		if not name.startswith("_") and isinstance(val, Task | Sequential | Parallel)
	}
	named_by_id: Final = {id(val): _assign_key_name(val, name) for name, val in bindings.items()}

	def promote(node: TaskNode) -> TaskNode:
		source = cast("TaskNode", named_by_id.get(id(node), node))
		match source:
			case Task():
				return source
			case Sequential(tasks=children, name=n, matrix=m, env=e, cwd=c):
				return Sequential(*(promote(ch) for ch in children), name=n, matrix=m, env=e, cwd=c)
			case Parallel(tasks=children, name=n, matrix=m, env=e, cwd=c):
				return Parallel(*(promote(ch) for ch in children), name=n, matrix=m, env=e, cwd=c)
			case _:
				assert_never(source)

	return {name: promote(val) for name, val in bindings.items()}


def load_tasks_from_py(path: Path) -> dict[str, TaskNode]:
	"""Execute a Python task-definition file and collect module-level TaskNode bindings."""
	return _name_scope_bindings(runpy.run_path(str(path)))


def dispatch_arg(arg: str, tasks: Mapping[str, TaskNode]) -> TaskNode:
	"""Interpret a CLI arg: task name (possibly hyphenated) ⇒ lookup, else inline expression."""
	if arg in tasks:
		return tasks[arg]
	try:
		parsed = ast.parse(arg, mode="eval")
	except SyntaxError:
		return parse_expression(arg, tasks=tasks)
	if isinstance(parsed.body, ast.Name):
		name = parsed.body.id
		known = ", ".join(sorted(tasks)) or "none"
		print(f"error: no task named {name!r} (known: {known})", file=sys.stderr)
		sys.exit(2)
	return parse_expression(arg, tasks=tasks)


def build_parser(tasks: Mapping[str, TaskNode] | None = None) -> argparse.ArgumentParser:
	"""Build the CLI argument parser.

	When ``tasks`` is provided, known task names appear in the positional
	metavar so the usage line reads like a list of subcommands.

	>>> parser = build_parser()
	>>> args = parser.parse_args(['Task("echo hi")'])
	>>> args.expression
	'Task("echo hi")'
	>>> parser.parse_args(["--list"]).list
	True
	>>> "task | expression" in build_parser({"all": Task("x")}).format_usage()
	True
	"""
	parser: Final = argparse.ArgumentParser(
		prog="camas",
		description="Generic parallel/sequential task runner with TUI output.",
		epilog=EXAMPLES,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument(
		"expression",
		nargs="?",
		metavar=_expression_metavar(tasks),
		help="name of a defined task, or a typed Python expression",
	)
	parser.add_argument(
		"--version",
		action="version",
		version=f"camas {importlib.metadata.version('camas')}",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="print the task tree without executing",
	)
	parser.add_argument(
		"--list",
		action="store_true",
		help="list all defined tasks and exit (also the default with no args)",
	)
	parser.add_argument(
		"--tree",
		action="store_true",
		help="print every defined task's expanded tree and exit",
	)
	parser.add_argument(
		"--effects",
		default="(Termtree(),)",
		help=(
			"tuple expression of Effect instances, e.g. '(Summary(),)' for CI."
			f" known: {', '.join(EFFECT_CONSTRUCTORS)}. default: '(Termtree(),)'"
		),
	)
	return parser


def _expression_metavar(tasks: Mapping[str, TaskNode] | None) -> str:
	"""Build the positional metavar.

	>>> _expression_metavar(None)
	'expression'
	>>> _expression_metavar({"all": Task("x")})
	'task | expression'
	"""
	return "task | expression" if tasks else "expression"


def _is_named_ref(node: TaskNode, names: frozenset[str]) -> bool:
	return node.name is not None and node.name in names


def _par_child_summary(node: TaskNode, names: frozenset[str]) -> str:
	"""Render a Parallel child, parenthesising an anonymous Sequential because
	``,`` binds looser than ``|``."""
	rendered = task_summary(node, names, is_root=False)
	if isinstance(node, Sequential) and not _is_named_ref(node, names) and len(node.tasks) > 1:
		return f"({rendered})"
	return rendered


def task_summary(node: TaskNode, names: frozenset[str], is_root: bool = True) -> str:
	"""One-line representation of a task tree using ``,`` for Sequential and
	``|`` for Parallel. Children whose name appears in ``names`` render as a
	bare reference. Precedence: ``|`` binds tighter than ``,``.

	>>> task_summary(Task("echo hi"), frozenset())
	'echo hi'
	>>> task_summary(Task(("python", "-c", "pass")), frozenset())
	'python -c pass'
	>>> task_summary(Sequential(Task("a"), Task("b")), frozenset())
	'a, b'
	>>> task_summary(Parallel(Task("a"), Task("b")), frozenset())
	'a | b'
	>>> task_summary(Sequential(Task("a"), Parallel(Task("b"), Task("c"))), frozenset())
	'a, b | c'
	>>> task_summary(Parallel(Sequential(Task("a"), Task("b")), Task("c")), frozenset())
	'(a, b) | c'
	>>> task_summary(Sequential(Task("a", name="lint"), Task("b")), frozenset({"lint"}))
	'lint, b'
	"""
	if not is_root and _is_named_ref(node, names):
		assert node.name is not None
		return node.name
	match node:
		case Task(cmd=cmd):
			return cmd if isinstance(cmd, str) else " ".join(cmd)
		case Sequential(tasks=tasks):
			return ", ".join(task_summary(t, names, is_root=False) for t in tasks)
		case Parallel(tasks=tasks):
			return " | ".join(_par_child_summary(t, names) for t in tasks)
		case _:
			assert_never(node)


def _summary_annotation(node: TaskNode) -> str:
	"""Annotate a top-level entry with its matrix axes — important context that
	doesn't appear in the body otherwise."""
	if isinstance(node, Task) or node.matrix is None:
		return ""
	axes = ", ".join(node.matrix)
	return f"  [matrix: {axes}]"


def print_task_summary_listing(tasks: Mapping[str, TaskNode], source: Path | None) -> None:
	"""Print the ``Tasks from <source>`` listing — one compact line per task."""
	if not tasks:
		if source is not None:
			print(f"No tasks defined in {source}.")
		else:
			print("No tasks file found in this directory or any parent.")
			print("Define tasks in tasks.py or [tool.camas.tasks] in pyproject.toml,")
			print('or pass an expression directly: camas \'Parallel("ruff check .", "mypy .")\'')
		return
	names = frozenset(tasks)
	items = sorted(tasks.items())
	width = max(len(n) for n, _ in items)
	header = f"Available tasks from {source}:" if source is not None else "Tasks:"
	print(header)
	for name, node in items:
		body = task_summary(node, names)
		print(f"  {(name).ljust(width + 1)} {body}{_summary_annotation(node)}")


def print_task_trees(tasks: Mapping[str, TaskNode], source: Path | None) -> None:
	"""Print every defined task's tree with commands expanded — verbose ``--tree`` output."""
	if not tasks:
		print_task_summary_listing(tasks, source)
		return
	header = f"Available tasks from {source}:" if source is not None else "Tasks:"
	print(header)
	print()
	for _, task in sorted(tasks.items()):
		print_tree(task, show_cmd=True)
		print()


def print_task_help(name: str, task: TaskNode) -> None:
	"""Print subcommand help for a single task: its expanded tree."""
	print(f"usage: camas {name} [-h] [--dry-run] [--effects EFFECTS]")
	print()
	print(f"runs the {name!r} task:")
	print_tree(task, show_cmd=True)


def run_cli(scope: Mapping[str, object]) -> None:
	"""Collect Task/Sequential/Parallel values from ``scope`` and dispatch CLI args.

	Names starting with ``_`` and non-task values are skipped. Public bindings
	are named by their binding identifier, and nested references inherit those
	names (consistent with ``[tool.camas.tasks]`` in pyproject.toml).
	"""
	dispatch(_name_scope_bindings(scope))


EFFECT_CONSTRUCTORS: Final[Mapping[str, Callable[..., Any]]] = {
	Auto.__name__: Auto,
	Fixed.__name__: Fixed,
	Summary.__name__: Summary,
	SummaryOptions.__name__: SummaryOptions,
	Termtree.__name__: Termtree,
	TermtreeOptions.__name__: TermtreeOptions,
}


def parse_effects(expr: str) -> tuple[Effect[Any], ...]:
	"""Parse a ``--effects`` expression into a tuple of Effect instances.

	>>> [type(e).__name__ for e in parse_effects("(Summary(),)")]
	['Summary']
	>>> [type(e).__name__ for e in parse_effects("(Termtree(), Summary())")]
	['Termtree', 'Summary']
	>>> parse_effects("(Summary(options=SummaryOptions(term_width=Fixed(80))),)")[0].options.term_width
	Fixed(columns=80)
	>>> parse_effects("()")
	()
	"""
	try:
		tree = ast.parse(expr, mode="eval")
	except SyntaxError as e:
		raise ValueError(_format_syntax_error(expr, e)) from e
	if not isinstance(tree.body, ast.Tuple):
		raise ValueError(f"--effects must be a tuple, got {type(tree.body).__name__}")
	return tuple(_expect_effect(_eval_value(elt)) for elt in tree.body.elts)


def _eval_value(node: ast.expr) -> Any:
	"""Evaluate a node in the --effects mini-language. Returns Any since the
	concrete type depends on the AST shape; ``_expect_effect`` validates at the
	tuple level."""
	match node:
		case ast.Constant(value=val):
			return val
		case ast.Call(func=ast.Name(id=name), args=args, keywords=keywords) if (
			name in EFFECT_CONSTRUCTORS
		):
			return EFFECT_CONSTRUCTORS[name](
				*(_eval_value(a) for a in args),
				**{kw.arg: _eval_value(kw.value) for kw in keywords if kw.arg is not None},
			)
		case _:
			raise ValueError(
				f"unsupported syntax (known: {', '.join(EFFECT_CONSTRUCTORS)}): {ast.dump(node)}"
			)


def _expect_effect(value: Any) -> Effect[Any]:
	if not isinstance(value, Summary | Termtree):
		raise ValueError(f"expected an Effect instance, got {type(value).__name__}")
	return value


def dispatch(
	tasks: Mapping[str, TaskNode],
	argv: list[str] | None = None,
	source: Path | None = None,
) -> None:
	"""Parse ``argv`` (defaulting to sys.argv) against ``tasks`` and run the dispatched task.

	When invoked with no expression and tasks are defined, prints a compact listing
	of available tasks and exits 0 — the ``camas`` (no-args) ergonomic default,
	mirroring ``just`` and ``task``.
	"""
	raw: Final = sys.argv[1:] if argv is None else argv

	if len(raw) >= 2 and raw[0] in tasks and any(a in ("-h", "--help") for a in raw[1:]):
		print_task_help(raw[0], tasks[raw[0]])
		sys.exit(0)

	parser: Final = build_parser(tasks)
	args: Final = parser.parse_args(raw)

	if args.list:
		print_task_summary_listing(tasks, source)
		sys.exit(0)

	if args.tree:
		print_task_trees(tasks, source)
		sys.exit(0)

	if args.expression is None:
		if tasks:
			print(parser.format_usage().rstrip())
			print()
			print_task_summary_listing(tasks, source)
			print()
			print("Try running a sequence of tasks using ( ):")
			print('  camas \'("echo Hello", "echo world!")\'')
			print("or parallel tasks using { }:")
			print('  camas \'{"echo Hello", "echo world!"}\'')
			print()
			sys.stdout.flush()
			print(f"{parser.prog}: error: task or expression is required", file=sys.stderr)
			sys.exit(2)
		parser.error("task or expression is required (no tasks defined)")

	try:
		effects: Final = parse_effects(args.effects)
	except ValueError as e:
		print(f"error: --effects: {e}", file=sys.stderr)
		sys.exit(2)

	task: Final = dispatch_arg(args.expression, tasks)
	if args.dry_run:
		print_tree(task, show_cmd=True)
		sys.exit(0)
	sys.exit(asyncio.run(run(task, effects=effects)).returncode)


def _looks_like_py_file(arg: str) -> bool:
	"""A CLI arg refers to a tasks file when it ends in ``.py`` — expressions always end in ``)``.

	>>> _looks_like_py_file("tasks.py")
	True
	>>> _looks_like_py_file("./sub/my_tasks.py")
	True
	>>> _looks_like_py_file('Task("pytest x.py")')
	False
	>>> _looks_like_py_file("lint")
	False
	"""
	return arg.endswith(".py")


def _resolve_tasks_source(
	argv: list[str],
) -> tuple[dict[str, TaskNode], list[str], Path | None]:
	"""Locate the tasks source and return (tasks, remaining_argv, source_path).

	If ``argv[0]`` ends in ``.py`` it is consumed as an explicit file path.
	Otherwise walks upward from cwd and returns tasks from the nearest directory
	that defines any; ``tasks.py`` wins over ``pyproject.toml`` at the same level.
	A ``pyproject.toml`` without ``[tool.camas.tasks]`` is not a match — the walk
	continues upward. ``source_path`` is the file the tasks were loaded from, or
	``None`` when no tasks file was found.
	"""
	if argv and _looks_like_py_file(argv[0]):
		path = Path(argv[0])
		if not path.is_file():
			print(f"error: {path}: no such file", file=sys.stderr)
			sys.exit(2)
		try:
			return load_tasks_from_py(path), argv[1:], path
		except Exception as e:
			print(f"error: {path}: {e}", file=sys.stderr)
			sys.exit(2)

	start: Final = Path.cwd()
	for candidate in (start, *start.parents):
		tasks_py = candidate / "tasks.py"
		if tasks_py.is_file():
			try:
				return load_tasks_from_py(tasks_py), argv, tasks_py
			except Exception as e:
				print(f"error: {tasks_py}: {e}", file=sys.stderr)
				sys.exit(2)
		pyproject = candidate / "pyproject.toml"
		if pyproject.is_file():
			try:
				tasks = load_tasks(pyproject)
			except ValueError as e:
				print(f"error: {pyproject}: {e}", file=sys.stderr)
				sys.exit(2)
			if tasks:
				return tasks, argv, pyproject

	return {}, argv, None


def main() -> None:
	"""Console script entry: resolves tasks source and dispatches.

	Reconfigures stdout/stderr to UTF-8 so Windows consoles (cp1252 by default) can
	render the box-drawing characters used in the tree output.
	"""
	for stream in (sys.stdout, sys.stderr):
		cast(io.TextIOWrapper, stream).reconfigure(encoding="utf-8", errors="replace")
	tasks, argv, source = _resolve_tasks_source(sys.argv[1:])
	dispatch(tasks, argv, source)


if __name__ == "__main__":
	main()
