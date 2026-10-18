# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast

from .source import Module, constants, segment


def signature(function: ast.FunctionDef | ast.AsyncFunctionDef, module: Module) -> str:
	prefix = "async def" if isinstance(function, ast.AsyncFunctionDef) else "def"
	returns = (
		f" -> {segment(module.source, function.returns)}" if function.returns is not None else ""
	)
	return f"{prefix} {function.name}({', '.join(parameters(function.args, module))}){returns}"


def parameters(arguments: ast.arguments, module: Module) -> tuple[str, ...]:
	values = constants(module)
	positional = (*arguments.posonlyargs, *arguments.args)
	defaults = (*(None,) * (len(positional) - len(arguments.defaults)), *arguments.defaults)
	shown = tuple(
		parameter(a, d, module.source, values) for a, d in zip(positional, defaults, strict=True)
	)
	boundary = len(arguments.posonlyargs)
	return (
		*shown[:boundary],
		*(("/",) if boundary else ()),
		*shown[boundary:],
		*(
			(f"*{parameter(arguments.vararg, None, module.source, values)}",)
			if arguments.vararg is not None
			else ("*",)
			if arguments.kwonlyargs
			else ()
		),
		*(
			parameter(a, d, module.source, values)
			for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
		),
		*(
			(f"**{parameter(arguments.kwarg, None, module.source, values)}",)
			if arguments.kwarg is not None
			else ()
		),
	)


def parameter(
	argument: ast.arg, default: ast.expr | None, source: str, values: dict[str, str]
) -> str:
	annotation = "" if argument.annotation is None else f": {segment(source, argument.annotation)}"
	if default is None:
		return f"{argument.arg}{annotation}"
	shown = values.get(segment(source, default), segment(source, default))
	return f"{argument.arg}{annotation} = {shown}" if annotation else f"{argument.arg}={shown}"
