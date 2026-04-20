# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.metadata
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple, TypeGuard, assert_never, cast

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
	Task(cmd='echo hi', name=None, env={})
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
						)
					return Parallel(
						tasks=tasks,
						name=_eval_opt_str(kw.get("name")),
						matrix=_eval_matrix(kw.get("matrix")),
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
	Task(cmd='echo hi', name=None, env={})
	>>> parse_expression('Parallel(tasks=(a,))', tasks={"a": Task("x")})
	Parallel(tasks=(Task(cmd='x', name=None, env={}),), name=None, matrix=None)
	"""
	try:
		tree = ast.parse(expr, mode="eval")
	except SyntaxError as e:
		print(f"error: invalid syntax: {e}", file=sys.stderr)
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
	Task(cmd='ruff check .', name=None, env={})
	>>> parse_task_value('Task("pytest")')
	Task(cmd='pytest', name=None, env={})
	>>> parse_task_value("Ref(\\"lint\\")")
	Ref(name='lint')
	"""
	if not EXPRESSION_PATTERN.match(raw):
		return Task(cmd=raw)
	try:
		tree = ast.parse(raw, mode="eval")
	except SyntaxError as e:
		raise ValueError(f"invalid expression {raw!r}: {e}") from e
	return eval_node(tree.body, allow_refs=True)


def resolve_refs(
	node: TaskNode | Ref,
	defs: Mapping[str, TaskNode | Ref],
	visiting: frozenset[str],
) -> TaskNode:
	"""Recursively substitute Ref(name) with its defined TaskNode. Detects cycles.

	>>> resolve_refs(Task("x"), {}, frozenset())
	Task(cmd='x', name=None, env={})
	>>> resolve_refs(Ref("a"), {"a": Task("hi")}, frozenset())
	Task(cmd='hi', name=None, env={})
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
		case Sequential(tasks=tasks, name=n, matrix=m):
			return Sequential(
				tasks=tuple(resolve_refs(t, defs, visiting) for t in tasks),
				name=n,
				matrix=m,
			)
		case Parallel(tasks=tasks, name=n, matrix=m):
			return Parallel(
				tasks=tuple(resolve_refs(t, defs, visiting) for t in tasks),
				name=n,
				matrix=m,
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
		pre[name] = parse_task_value(value)
	return {name: resolve_refs(tree, pre, frozenset({name})) for name, tree in pre.items()}


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


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI argument parser.

	>>> parser = build_parser()
	>>> args = parser.parse_args(['Task("echo hi")'])
	>>> args.expression
	'Task("echo hi")'
	>>> parser.parse_args(["--list"]).list
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
		help="typed Python expression, or the name of a task defined in pyproject.toml",
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
		help="list tasks defined in pyproject.toml and exit",
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


def print_tasks(tasks: Mapping[str, TaskNode]) -> None:
	"""Print each named task and its tree."""
	if not tasks:
		print("(no tasks defined)")
		return
	for name, task in sorted(tasks.items()):
		print(f"{name}:")
		print_tree(task)
		print()


def run_cli(scope: Mapping[str, object]) -> None:
	"""Collect Task/Sequential/Parallel from ``scope`` and dispatch CLI args.

	Intended for user-owned ``check.py`` scripts::

	    from camas import Parallel, Task
	    from camas.main import run_cli

	    lint = Task("ruff check .")
	    ci = Parallel(tasks=(lint,))

	    if __name__ == "__main__":
	        run_cli(globals())

	Names starting with ``_`` and non-task values are skipped.
	"""
	dispatch(
		{
			name: val
			for name, val in scope.items()
			if not name.startswith("_") and isinstance(val, Task | Sequential | Parallel)
		}
	)


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
		raise ValueError(f"invalid syntax: {e}") from e
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


def dispatch(tasks: Mapping[str, TaskNode]) -> None:
	"""Parse sys.argv against ``tasks`` and run the dispatched task. Always exits."""
	# Tree-rendering uses UTF-8 box-drawing chars; Windows defaults stdout to cp1252
	# in non-TTY contexts (captured subprocesses, piped CI logs), so reconfigure.
	if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding.lower() != "utf-8":
		sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]  # pragma: no cover
	parser: Final = build_parser()
	args: Final = parser.parse_args()

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
		print_tree(task)
		sys.exit(0)
	sys.exit(asyncio.run(run(task, effects=effects)).returncode)


def main() -> None:
	"""Console script entry: loads pyproject.toml tasks and dispatches."""
	tasks: dict[str, TaskNode] = {}
	if (pyproject := find_pyproject(Path.cwd())) is not None:
		try:
			tasks = load_tasks(pyproject)
		except ValueError as e:
			print(f"error: {pyproject}: {e}", file=sys.stderr)
			sys.exit(2)
	dispatch(tasks)


if __name__ == "__main__":
	main()
