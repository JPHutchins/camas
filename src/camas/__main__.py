from __future__ import annotations

import ast
import sys
from typing import Final

from camas import Parallel, Sequential, Task, TaskNode, run

_USAGE: Final = """\
usage: python -m camas '<expression>'

  expression is a typed Python expression constructing a TaskNode tree, e.g.:

    Parallel(tasks=(Task("ruff check ."), Task("mypy .")))

    Sequential(tasks=(
        Task("ruff format . --check"),
        Parallel(tasks=(Task("mypy ."), Task("pyright ."))),
        Task("pytest"),
    ))

    Parallel(
        tasks=(Task("pytest --python {PY}"),),
        matrix={"PY": ("3.12", "3.13", "3.14")},
    )
"""


def _eval_node(
	node: ast.expr,
) -> str | int | float | bool | None | tuple[object, ...] | dict[str, object] | TaskNode:
	match node:
		case ast.Constant(value=val) if isinstance(val, str | int | float | bool) or val is None:
			return val
		case ast.Tuple(elts=elts):
			return tuple(_eval_node(e) for e in elts)
		case ast.Dict(keys=keys, values=values):
			return {
				_expect_str(_eval_node(k)): _eval_node(v)
				for k, v in zip(keys, values)
				if k is not None
			}
		case ast.Call(func=ast.Name(id=name), args=args, keywords=keywords):
			kwargs = {kw.arg: _eval_node(kw.value) for kw in keywords if kw.arg is not None}
			match name:
				case "Task":
					if args:
						kwargs["cmd"] = _eval_node(args[0])
					return Task(**kwargs)  # type: ignore[arg-type]
				case "Sequential":
					return Sequential(**kwargs)  # type: ignore[arg-type]
				case "Parallel":
					return Parallel(**kwargs)  # type: ignore[arg-type]
				case _:
					raise ValueError(
						f"unknown type: {name!r} (expected Task, Sequential, or Parallel)"
					)
		case _:
			raise ValueError(f"unsupported syntax: {ast.dump(node)}")


def _expect_str(val: object) -> str:
	if not isinstance(val, str):
		raise ValueError(f"expected str key, got {type(val).__name__}")
	return val


def parse_expression(expr: str) -> TaskNode:
	try:
		tree = ast.parse(expr, mode="eval")
	except SyntaxError as e:
		print(f"error: invalid syntax: {e}", file=sys.stderr)
		print(_USAGE, file=sys.stderr)
		sys.exit(2)

	try:
		result = _eval_node(tree.body)
	except (ValueError, TypeError) as e:
		print(f"error: {e}", file=sys.stderr)
		print(_USAGE, file=sys.stderr)
		sys.exit(2)

	if not isinstance(result, Task | Sequential | Parallel):
		print(
			f"error: expression must be Task, Sequential, or Parallel, got {type(result).__name__}",
			file=sys.stderr,
		)
		sys.exit(2)
	return result


def main() -> None:
	if len(sys.argv) != 2:
		print(_USAGE, file=sys.stderr)
		sys.exit(2)
	task = parse_expression(sys.argv[1])
	sys.exit(run(task))


if __name__ == "__main__":
	main()
