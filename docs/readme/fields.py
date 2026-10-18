# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

from __future__ import annotations

import ast
import inspect
from typing import NamedTuple

from .markdown import code
from .source import Module, Unresolved, constants, resolve, segment


class Attribute(NamedTuple):
	annotation: str
	value: str | None
	doc: str | None


class Parameter(NamedTuple):
	accepts: str
	default: str | None
	variadic: bool


class Field(NamedTuple):
	name: str
	accepts: str
	stored: str
	default: str | None
	doc: str | None
	variadic: bool


def fields_section(dotted: str) -> tuple[str, ...] | Unresolved:
	target = resolve(dotted)
	if isinstance(target, Unresolved):
		return target
	classes = {
		node.name: node for node in target.module.tree.body if isinstance(node, ast.ClassDef)
	}
	if target.name not in classes:
		return Unresolved(f"no class {target.name!r} in {target.module.path.as_posix()}")
	fields = class_fields(classes[target.name], classes, target.module)
	return (*table(fields), *details(target.name, fields))


def table(fields: tuple[Field, ...]) -> tuple[str, ...]:
	return ("| Field | Type | Default |", "|---|---|---|", *(row(field) for field in fields))


def row(field: Field) -> str:
	accepts = code(field.accepts)
	shown = accepts if field.accepts == field.stored else f"{accepts} → {code(field.stored)}"
	default = "" if field.variadic else "required" if field.default is None else code(field.default)
	return f"| `{label(field)}` | {shown} | {default} |"


def details(name: str, fields: tuple[Field, ...]) -> tuple[str, ...]:
	documented = tuple((field, field.doc) for field in fields if field.doc)
	if not documented:
		return ()
	return (
		"",
		f"<details><summary><code>{name}</code> fields</summary>",
		"",
		*(line for field, doc in documented for line in bullet(label(field), doc)),
		"",
		"</details>",
	)


def bullet(name: str, doc: str) -> tuple[str, ...]:
	first, *rest = inspect.cleandoc(doc).splitlines()
	return (f"- `{name}`: {first}", *(f"  {line}" if line else "" for line in rest))


def label(field: Field) -> str:
	return f"*{field.name}" if field.variadic else field.name


def class_fields(
	cls: ast.ClassDef, classes: dict[str, ast.ClassDef], module: Module
) -> tuple[Field, ...]:
	ancestry = lineage(cls, classes)
	attributes = {
		name: attribute
		for ancestor in reversed(ancestry)
		for name, attribute in attribute_docs(ancestor, module.source).items()
	}
	init = next(
		(method for ancestor in ancestry if (method := init_of(ancestor)) is not None), None
	)
	values = constants(module)
	shown = (
		parameters(init, module.source)
		if init is not None
		else {
			name: Parameter(a.annotation, a.value, variadic=False) for name, a in attributes.items()
		}
	)
	return tuple(
		Field(
			name,
			parameter.accepts,
			attributes[name].annotation if name in attributes else parameter.accepts,
			values.get(parameter.default, parameter.default)
			if parameter.default is not None
			else None,
			attributes[name].doc if name in attributes else None,
			parameter.variadic,
		)
		for name, parameter in shown.items()
	)


def lineage(cls: ast.ClassDef, classes: dict[str, ast.ClassDef]) -> tuple[ast.ClassDef, ...]:
	return (
		cls,
		*(
			ancestor
			for base in cls.bases
			if isinstance(base, ast.Name) and base.id in classes
			for ancestor in lineage(classes[base.id], classes)
		),
	)


def attribute_docs(cls: ast.ClassDef, source: str) -> dict[str, Attribute]:
	return {
		statement.target.id: Attribute(
			segment(source, statement.annotation),
			segment(source, statement.value) if statement.value is not None else None,
			following.value.value
			if isinstance(following, ast.Expr)
			and isinstance(following.value, ast.Constant)
			and isinstance(following.value.value, str)
			else None,
		)
		for statement, following in zip(cls.body, (*cls.body[1:], None), strict=True)
		if isinstance(statement, ast.AnnAssign)
		and isinstance(statement.target, ast.Name)
		and not statement.target.id.startswith("_")
	}


def init_of(cls: ast.ClassDef) -> ast.FunctionDef | None:
	return next(
		(
			statement
			for statement in cls.body
			if isinstance(statement, ast.FunctionDef) and statement.name == "__init__"
		),
		None,
	)


def parameters(init: ast.FunctionDef, source: str) -> dict[str, Parameter]:
	arguments = init.args
	positional = (*arguments.posonlyargs, *arguments.args)
	defaults = (*(None,) * (len(positional) - len(arguments.defaults)), *arguments.defaults)
	return {
		argument.arg: Parameter(
			segment(source, argument.annotation) if argument.annotation is not None else "",
			segment(source, default) if default is not None else None,
			variadic=argument is arguments.vararg,
		)
		for argument, default in (
			*zip(positional, defaults, strict=True),
			*(((arguments.vararg, None),) if arguments.vararg is not None else ()),
			*zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True),
		)
		if argument.arg != "self"
	}
