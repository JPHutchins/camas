# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.metadata
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

EXPRESSION_PATTERN: Final = re.compile(r"^\s*(Task|Sequential|Parallel|Ref)\s*\(")

EXAMPLES: Final = """\
examples:
    camas 'Parallel(tasks=(Task("ruff check ."), Task("mypy .")))'

    camas 'Sequential(tasks=(
        Task("ruff format . --check"),
        Parallel(tasks=(Task("mypy ."), Task("pyright ."))),
        Task("pytest"),
    ))'

    camas 'Parallel(
        tasks=(Task("pytest --python {PY}"),),
        matrix={"PY": ("3.12", "3.13", "3.14")},
    )'

named tasks (pyproject.toml):
    [tool.camas.tasks]
    lint = "ruff check ."
    test = "pytest"
    ci   = 'Sequential(tasks=(Ref("lint"), Ref("test")))'

    camas ci            # run a named task
    camas --list        # show all defined tasks
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


def _eval_tasks(node: ast.expr, allow_refs: bool) -> tuple[TaskNode | Ref, ...]:
	match node:
		case ast.Tuple(elts=elts):
			return tuple(eval_node(e, allow_refs) for e in elts)
		case _:
			raise ValueError(f"tasks must be a tuple, got {ast.dump(node)}")


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
					tasks_node = kw.get("tasks")
					if tasks_node is None:
						raise ValueError(f"{name} requires 'tasks'")
					tasks = cast(tuple[TaskNode, ...], _eval_tasks(tasks_node, allow_refs))
					if name == "Sequential":
						return Sequential(
							tasks=tasks,
							name=_eval_opt_str(kw.get("name")),
							matrix=_eval_matrix(kw.get("matrix")),
							env=_eval_env(kw.get("env")),
						)
					return Parallel(
						tasks=tasks,
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
		case ast.Tuple():
			raise ValueError(
				"expected Task/Sequential/Parallel/Ref, got bare tuple"
				" — did you mean Parallel(tasks=(...)) or Sequential(tasks=(...))?",
			)
		case _:
			raise ValueError(f"unsupported syntax: {ast.dump(node)}")


def parse_expression(expr: str, tasks: Mapping[str, TaskNode] | None = None) -> TaskNode:
	"""Parse a typed Python expression string into a TaskNode tree using AST (no eval).

	When ``tasks`` is provided, bare identifiers in the expression become Refs that are
	resolved against ``tasks`` after parsing.

	>>> parse_expression('Task("echo hi")')
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> parse_expression('Parallel(tasks=(a,))', tasks={"a": Task("x")})
	Parallel(tasks=(Task(cmd='x', name=None, env={}, cwd=None),), name=None, matrix=None, env={}, cwd=None)
	"""
	try:
		tree = ast.parse(expr, mode="eval")
	except SyntaxError as e:
		print(f"error: {_format_syntax_error(expr, e)}", file=sys.stderr)
		sys.exit(2)

	try:
		result = eval_node(tree.body, allow_refs=tasks is not None)
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

	>>> parse_task_value("ruff check .")
	Task(cmd='ruff check .', name=None, env={}, cwd=None)
	>>> parse_task_value('Task("pytest")')
	Task(cmd='pytest', name=None, env={}, cwd=None)
	>>> parse_task_value("Ref(\\"lint\\")")
	Ref(name='lint')
	"""
	if not EXPRESSION_PATTERN.match(raw):
		return Task(cmd=raw)
	try:
		tree = ast.parse(raw, mode="eval")
	except SyntaxError as e:
		raise ValueError(_format_syntax_error(raw, e)) from e
	return eval_node(tree.body, allow_refs=True)


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
				tasks=tuple(resolve_refs(t, defs, visiting) for t in tasks),
				name=n,
				matrix=m,
				env=e,
				cwd=c,
			)
		case Parallel(tasks=tasks, name=n, matrix=m, env=e, cwd=c):
			return Parallel(
				tasks=tuple(resolve_refs(t, defs, visiting) for t in tasks),
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
			return Sequential(tasks=tasks, name=key, matrix=matrix, env=env, cwd=cwd)
		case Parallel(tasks=tasks, name=None, matrix=matrix, env=env, cwd=cwd):
			return Parallel(tasks=tasks, name=key, matrix=matrix, env=env, cwd=cwd)
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
	"""Collect public TaskNode bindings from a mapping of globals and propagate names.

	Each public (non-underscore) ``Task``/``Sequential``/``Parallel`` becomes a named
	task; nested references by identity (e.g. ``Parallel(tasks=(mypy,))`` where
	``mypy`` is itself a top-level binding) inherit the binding's name, matching the
	naming behavior of ``[tool.camas.tasks]`` in pyproject.toml.
	"""
	bindings: Final = {
		name: val
		for name, val in scope.items()
		if not name.startswith("_") and isinstance(val, Task | Sequential | Parallel)
	}
	named_by_id: Final = {id(val): _assign_key_name(val, name) for name, val in bindings.items()}

	def promote(node: TaskNode) -> TaskNode:
		source = cast(TaskNode, named_by_id.get(id(node), node))
		match source:
			case Task():
				return source
			case Sequential(tasks=children, name=n, matrix=m, env=e, cwd=c):
				return Sequential(
					tasks=tuple(promote(ch) for ch in children), name=n, matrix=m, env=e, cwd=c
				)
			case Parallel(tasks=children, name=n, matrix=m, env=e, cwd=c):
				return Parallel(
					tasks=tuple(promote(ch) for ch in children), name=n, matrix=m, env=e, cwd=c
				)
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
	>>> "all|check" in build_parser({"all": Task("x"), "check": Task("y")}).format_usage()
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
		help="list all defined tasks and exit",
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
	"""Build the positional metavar, summarising known task names when present.

	>>> _expression_metavar(None)
	'expression'
	>>> _expression_metavar({"all": Task("x"), "check": Task("y"), "lint": Task("z")})
	'{all|check|lint}'
	"""
	if not tasks:
		return "expression"
	return f"{{{'|'.join(sorted(tasks))}}}"


def print_tasks(tasks: Mapping[str, TaskNode]) -> None:
	"""Print each named task's tree with commands expanded."""
	if not tasks:
		print("(no tasks defined)")
		return
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


def dispatch(tasks: Mapping[str, TaskNode], argv: list[str] | None = None) -> None:
	"""Parse ``argv`` (defaulting to sys.argv) against ``tasks`` and run the dispatched task."""
	raw: Final = sys.argv[1:] if argv is None else argv

	if len(raw) >= 2 and raw[0] in tasks and any(a in ("-h", "--help") for a in raw[1:]):
		print_task_help(raw[0], tasks[raw[0]])
		sys.exit(0)

	parser: Final = build_parser(tasks)
	args: Final = parser.parse_args(raw)

	if args.list:
		print_tasks(tasks)
		sys.exit(0)

	if args.expression is None:
		parser.error("expression or --list is required")

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


def _resolve_tasks_source(argv: list[str]) -> tuple[dict[str, TaskNode], list[str]]:
	"""Locate the tasks source and return (tasks, remaining_argv).

	If ``argv[0]`` ends in ``.py`` it is consumed as an explicit file path.
	Otherwise auto-discovers ``tasks.py`` and/or ``pyproject.toml``; if both
	define tasks, ``tasks.py`` wins and a warning is printed.
	"""
	if argv and _looks_like_py_file(argv[0]):
		path = Path(argv[0])
		if not path.is_file():
			print(f"error: {path}: no such file", file=sys.stderr)
			sys.exit(2)
		try:
			return load_tasks_from_py(path), argv[1:]
		except Exception as e:
			print(f"error: {path}: {e}", file=sys.stderr)
			sys.exit(2)

	tasks_py: Final = find_tasks_py(Path.cwd())
	pyproject: Final = find_pyproject(Path.cwd())

	if tasks_py is not None:
		if pyproject is not None:
			try:
				pyproject_tasks = load_tasks(pyproject)
			except ValueError:
				pyproject_tasks = {}
			if pyproject_tasks:
				print(
					f"warning: {tasks_py} and [tool.camas.tasks] in {pyproject} both define tasks;"
					f" using {tasks_py.name}",
					file=sys.stderr,
				)
		try:
			return load_tasks_from_py(tasks_py), argv
		except Exception as e:
			print(f"error: {tasks_py}: {e}", file=sys.stderr)
			sys.exit(2)

	if pyproject is not None:
		try:
			return load_tasks(pyproject), argv
		except ValueError as e:
			print(f"error: {pyproject}: {e}", file=sys.stderr)
			sys.exit(2)

	return {}, argv


def main() -> None:
	"""Console script entry: resolves tasks source and dispatches."""
	tasks, argv = _resolve_tasks_source(sys.argv[1:])
	dispatch(tasks, argv)


if __name__ == "__main__":
	main()
