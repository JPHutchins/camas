from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.metadata
import sys
from typing import Final

from camas import Parallel, Sequential, Task, TaskNode, run
from camas.effect.termtree import Termtree, TermtreeOptions, print_tree

CONSTRUCTORS: Final = {
	Task.__name__: Task,
	Sequential.__name__: Sequential,
	Parallel.__name__: Parallel,
}

EXAMPLES: Final = """\
examples:
    python -m camas 'Parallel(tasks=(Task("ruff check ."), Task("mypy .")))'

    python -m camas 'Sequential(tasks=(
        Task("ruff format . --check"),
        Parallel(tasks=(Task("mypy ."), Task("pyright ."))),
        Task("pytest"),
    ))'

    python -m camas 'Parallel(
        tasks=(Task("pytest --python {PY}"),),
        matrix={"PY": ("3.12", "3.13", "3.14")},
    )'
"""


def eval_node(
	node: ast.expr,
) -> str | int | float | bool | None | tuple[object, ...] | dict[str, object] | TaskNode:
	"""Walk an AST node and construct the corresponding Python value safely (no eval).

	>>> import ast
	>>> eval_node(ast.parse('"hello"', mode="eval").body)
	'hello'
	>>> eval_node(ast.parse('(1, 2)', mode="eval").body)
	(1, 2)
	"""
	match node:
		case ast.Constant(value=val) if isinstance(val, str | int | float | bool) or val is None:
			return val
		case ast.Tuple(elts=elts):
			return tuple(eval_node(e) for e in elts)
		case ast.Dict(keys=keys, values=values):
			return {
				expect_str(eval_node(k)): eval_node(v)
				for k, v in zip(keys, values)
				if k is not None
			}
		case ast.Call(func=ast.Name(id=name), args=args, keywords=keywords):
			kwargs = {kw.arg: eval_node(kw.value) for kw in keywords if kw.arg is not None}
			if name not in CONSTRUCTORS:
				raise ValueError(f"unknown type: {name!r} (expected {', '.join(CONSTRUCTORS)})")
			constructor = CONSTRUCTORS[name]
			if args:
				kwargs["cmd"] = eval_node(args[0])
			return constructor(**kwargs)  # type: ignore[no-any-return]
		case _:
			raise ValueError(f"unsupported syntax: {ast.dump(node)}")


def expect_str(val: object) -> str:
	"""Validate that a value is a string, raising ValueError otherwise.

	>>> expect_str("hello")
	'hello'
	>>> expect_str(42)
	Traceback (most recent call last):
	    ...
	ValueError: expected str key, got int
	"""
	if not isinstance(val, str):
		raise ValueError(f"expected str key, got {type(val).__name__}")
	return val


def parse_expression(expr: str) -> TaskNode:
	"""Parse a typed Python expression string into a TaskNode tree using AST (no eval).

	>>> parse_expression('Task("echo hi")')
	Task(cmd='echo hi', name=None, env={})
	"""
	try:
		tree = ast.parse(expr, mode="eval")
	except SyntaxError as e:
		print(f"error: invalid syntax: {e}", file=sys.stderr)
		sys.exit(2)

	try:
		result = eval_node(tree.body)
	except (ValueError, TypeError) as e:
		print(f"error: {e}", file=sys.stderr)
		sys.exit(2)

	if not isinstance(result, Task | Sequential | Parallel):
		print(
			f"error: expression must be {', '.join(CONSTRUCTORS)}, got {type(result).__name__}",
			file=sys.stderr,
		)
		sys.exit(2)
	return result


def build_parser() -> argparse.ArgumentParser:
	"""Build the CLI argument parser.

	>>> parser = build_parser()
	>>> args = parser.parse_args(['Task("echo hi")'])
	>>> args.expression
	'Task("echo hi")'
	"""
	parser: Final = argparse.ArgumentParser(
		prog="camas",
		description="Generic parallel/sequential task runner with TUI output.",
		epilog=EXAMPLES,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument(
		"expression",
		help="typed Python expression constructing a TaskNode tree",
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
	return parser


def main() -> None:
	"""CLI entry point: parse args, build task tree, and execute or dry-run."""
	args: Final = build_parser().parse_args()
	task: Final = parse_expression(args.expression)
	if args.dry_run:
		print_tree(task)
		sys.exit(0)
	sys.exit(asyncio.run(run(task, effects=(Termtree(TermtreeOptions()),))).returncode)


if __name__ == "__main__":
	main()
