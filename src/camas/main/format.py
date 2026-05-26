# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Plain-text formatting for task listings, summaries, and load-error hints."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core import timings
from ..core.color import BOLD_CYAN, CAMAS_LIGHT_PINK, CAMAS_VIOLET, GREY, RESET
from ..core.matrix import matrix_axes
from ..core.render import color_on, print_tree
from ..v0.task import Parallel, Sequential, Task, TaskNode
from .color import maybe_color, wrap_ansi
from .effects import available_effects, flatten_annotation, signature_fields
from .mypyc import MISSING

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..core.timings import Estimate
	from ..v0.effect import Effect


def is_named_ref(node: TaskNode, names: frozenset[str]) -> bool:
	return node.name is not None and node.name in names


def par_child_summary(node: TaskNode, names: frozenset[str]) -> str:
	"""Render a Parallel child, parenthesising an anonymous Sequential because
	``,`` binds looser than ``|``.
	"""
	rendered = task_summary(node, names, is_root=False)
	if isinstance(node, Sequential) and not is_named_ref(node, names) and len(node.tasks) > 1:
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
	>>> task_summary(Parallel(Sequential(Task("a"), Task("b")), Task("c")), frozenset())
	'(a, b) | c'
	>>> task_summary(Sequential(Task("a", name="lint"), Task("b")), frozenset({"lint"}))
	'lint, b'
	"""
	if not is_root and is_named_ref(node, names):
		assert node.name is not None
		return node.name
	match node:
		case Task(cmd=cmd):
			return cmd if isinstance(cmd, str) else " ".join(cmd)
		case Sequential(tasks=tasks):
			return ", ".join(task_summary(t, names, is_root=False) for t in tasks)
		case Parallel(tasks=tasks):
			return " | ".join(par_child_summary(t, names) for t in tasks)
		case _:
			assert_never(node)


def format_axis(name: str, values: tuple[str, ...]) -> str:
	"""Render a matrix axis for the listing annotation (``PY=3.13`` or ``PY×6 (lo..hi)``).

	>>> format_axis("PY", ("3.10", "3.11", "3.12", "3.13", "3.14", "3.15"))
	'PY×6 (3.10..3.15)'
	"""
	if len(values) == 1:
		return f"{name}={values[0]}"
	return f"{name}×{len(values)} ({values[0]}..{values[-1]})"


def colorize_summary(body: str, color: bool) -> str:
	"""Grey out the structural ``,`` and ``|`` operators in a task summary."""
	if not color:
		return body
	return body.replace(", ", f"{GREY},{RESET} ").replace(" | ", f" {GREY}|{RESET} ")


def format_load_error_hint(source: Path, exception: Exception) -> str:
	"""One-line hint shown in place of the task listing when ``tasks.py`` fails
	to evaluate — points the user at ``camas --check`` for the full diagnostic.

	>>> "camas --check" in format_load_error_hint(Path("/x/tasks.py"), RuntimeError("boom"))
	True
	"""
	return (
		f"Tasks unavailable — {source} failed to evaluate "
		f"({type(exception).__name__}: {exception}). "
		"Run `camas --check` for the full diagnostic."
	)


def format_task_summary_listing(
	tasks: Mapping[str, TaskNode],
	source: Path | None,
	color: bool,
	default_task_name: str | None = None,
	camas_dir: Path | None = None,
) -> str:
	"""Build the ``Available tasks from <source>`` listing as a string.

	``default_task_name`` marks the task a bare ``camas`` runs with ``(default)``;
	``camas_dir`` is the project's resolved camas directory, read for timing estimates.
	"""
	if not tasks:
		if source is not None:
			return f"No tasks defined in {source}."
		return (
			"No tasks file found in this directory or any parent.\n"
			"Define tasks in tasks.py or [tool.camas.tasks] in pyproject.toml,\n"
			'or pass an expression directly: camas \'Parallel("ruff check .", "mypy .")\''
		)
	names = frozenset(tasks)
	items = sorted(tasks.items())
	observed = timings.load(camas_dir) if camas_dir is not None else {}
	marker = " (default)"
	width = max(len(n) + (len(marker) if n == default_task_name else 0) for n, _ in items)
	header_text = f"Available tasks from {source}:" if source is not None else "Tasks:"
	lines = [maybe_color(header_text, CAMAS_VIOLET, color)]
	for name, node in items:
		body = (
			node.help
			if node.help is not None
			else colorize_summary(task_summary(node, names), color)
		)
		annotation = (
			""
			if isinstance(node, Task) or node.matrix is None
			else f"  [matrix: {' '.join(format_axis(k, v) for k, v in node.matrix.items())}]"
		)
		est = timings.estimate(node, observed)
		timing = timing_note(est, color) if est is not None else ""
		marked = name == default_task_name
		name_cell = maybe_color(name, BOLD_CYAN, color) + (
			maybe_color(marker, GREY, color) if marked else ""
		)
		pad = " " * (width + 1 - len(name) - (len(marker) if marked else 0))
		annotated = maybe_color(annotation, GREY, color) if annotation else ""
		lines.append(f"  {name_cell}{pad} {body}{annotated}{f'  {timing}' if timing else ''}")
	return "\n".join(lines)


def timing_note(estimate: Estimate, color: bool) -> str:
	"""A task's estimated-duration annotation, ``~Ns`` composed from observed leaf times.

	>>> from camas.core.timings import Estimate
	>>> timing_note(Estimate(32.0, 3, "test", 31.9), color=False)
	'~32.00s'
	"""
	return maybe_color(f"~{estimate.elapsed_s:.2f}s", GREY, color)


def print_task_summary_listing(
	tasks: Mapping[str, TaskNode],
	source: Path | None,
	default_task_name: str | None = None,
	camas_dir: Path | None = None,
) -> None:
	print(
		format_task_summary_listing(
			tasks,
			source,
			color=color_on(),
			default_task_name=default_task_name,
			camas_dir=camas_dir,
		)
	)


def print_task_trees(
	tasks: Mapping[str, TaskNode], source: Path | None, camas_dir: Path | None = None
) -> None:
	"""Print every defined task's tree with commands expanded — verbose ``--tree`` output."""
	if not tasks:
		print_task_summary_listing(tasks, source)
		return
	header = f"Available tasks from {source}:" if source is not None else "Tasks:"
	print(maybe_color(header, CAMAS_VIOLET, color_on()))
	print()
	observed = timings.load(camas_dir) if camas_dir is not None else {}
	for _, task in sorted(tasks.items()):
		print_tree(task, show_cmd=True)
		est = timings.estimate(task, observed)
		if est is not None:
			print(f"  {timing_note(est, color_on())}")
		print()


def format_matrix_axes_help(axes: Mapping[str, tuple[str, ...]], color: bool) -> str:
	"""Render the ``Matrix axes:`` block for per-task ``--help``.

	>>> "Matrix axes" in format_matrix_axes_help({"PY": ("3.13", "3.14")}, color=False)
	True
	>>> "--PY" in format_matrix_axes_help({"PY": ("3.13",)}, color=False)
	True
	"""
	width = max(len(name) for name in axes)
	lines = [maybe_color("Matrix axes (override with --AXIS VAL[,VAL...]):", CAMAS_VIOLET, color)]
	for name, values in axes.items():
		flag = f"--{name}".ljust(width + 2)
		current = ", ".join(values)
		lines.append(
			f"  {maybe_color(flag, BOLD_CYAN, color)}  {maybe_color(current, GREY, color)}"
		)
	return "\n".join(lines)


def print_task_help(name: str, task: TaskNode) -> None:
	"""Print subcommand help for a single task: its expanded tree and any
	matrix axes the user can override from the CLI.
	"""
	axes = matrix_axes(task)
	axis_flags = "".join(f" [--{k} VAL[,VAL...]]" for k in axes)
	preview = "[--dry-run | --github-matrix]" if axes else "[--dry-run]"
	print(f"usage: camas {name} [-h] {preview} [--effects EFFECTS]{axis_flags}")
	if task.help is not None:
		print()
		print(task.help)
	print()
	print(f"runs the {name!r} task:")
	print_tree(task, show_cmd=True)
	if axes:
		print()
		print(format_matrix_axes_help(axes, color_on()))


def format_annotation(annotation: Any, color: bool) -> str:
	"""Render a type annotation for help output, stripping module qualifiers."""
	if annotation is Any:
		text = "Any"
	elif isinstance(annotation, type):
		text = annotation.__name__
	else:
		text = re.sub(r"(?:[\w]+\.)+(\w+)", r"\1", str(annotation))
	return wrap_ansi(text, CAMAS_LIGHT_PINK) if color else text


def format_default(default: Any, color: bool) -> str:
	"""Render a constructor default value compactly."""
	if default is MISSING:
		return ""
	rendered = repr(default) if isinstance(default, str) else str(default)
	return f" = {wrap_ansi(rendered, GREY) if color else rendered}"


def format_signature(cls: Any, indent: str, color: bool) -> list[str]:
	"""Render ``cls(field: Type = default, …)`` plus nested signatures of any
	camas.effect classes appearing in the field annotations.
	"""
	import inspect

	fields = signature_fields(cls)
	cls_name = wrap_ansi(cls.__name__, BOLD_CYAN) if color else cls.__name__
	if not fields:
		return [f"{indent}{cls_name}()"]
	parts: list[str] = []
	nested: list[Any] = []
	for name, kind, annotation, default in fields:
		ann = format_annotation(annotation, color)
		prefix = (
			"*"
			if kind is inspect.Parameter.VAR_POSITIONAL
			else "**"
			if kind is inspect.Parameter.VAR_KEYWORD
			else ""
		)
		parts.append(f"{prefix}{name}: {ann}{format_default(default, color)}")
		for leaf in flatten_annotation(annotation):
			if (
				isinstance(leaf, type)
				and getattr(leaf, "__module__", "").startswith("camas.effect")
				and leaf is not cls
				and leaf not in nested
			):
				nested.append(leaf)
	lines = [f"{indent}{cls_name}({', '.join(parts)})"]
	for child in nested:
		lines.extend(format_signature(child, indent + "  ", color))
	return lines


def format_available_effects(
	color: bool | None = None,
	scope_effects: Mapping[str, type[Effect[Any]]] = {},
	default_effect_names: frozenset[str] = frozenset(),
) -> str:
	"""Render each discovered Effect with its full constructor signature and
	the signatures of every parameter type it transitively references.

	``scope_effects`` adds user-defined Effect classes to the listing;
	``default_effect_names`` marks the environment's default effect(s) with ``(default)``.
	"""
	_, effects = available_effects(scope_effects)
	if not effects:
		return ""
	on = color_on() if color is None else color
	lines = [maybe_color("Available Effects:", CAMAS_VIOLET, on)]
	for i, (name, cls) in enumerate(sorted(effects)):
		if i > 0:
			lines.append("")
		doc = first_line_doc(cls)
		name_str = maybe_color(name, BOLD_CYAN, on)
		default_str = maybe_color(" (default)", GREY, on) if name in default_effect_names else ""
		doc_str = f"  — {doc}" if doc else ""
		lines.append(f"  {name_str}{default_str}{doc_str}")
		lines.extend(format_signature(cls, "    ", on))
	return "\n".join(lines)


def format_try_hint(color: bool) -> str:
	"""Three-line table teaching ``( )`` (Sequential), ``{ }`` (Parallel), and ``--effects``."""
	rows = (
		('camas \'("echo Hello", "echo world!")\'', "( ) → Sequential"),
		('camas \'{"echo Hello", "echo world!"}\'', "{ } → Parallel"),
		(
			"camas --effects '(Summary(term_width=Fixed(60)),)' <task>",
			"post-run summary instead of live tree",
		),
	)
	width = max(len(cmd) for cmd, _ in rows)
	header = maybe_color("Try:", CAMAS_VIOLET, color)
	body = "\n".join(
		f"  {cmd.ljust(width)}  {maybe_color(f'# {note}', GREY, color)}" for cmd, note in rows
	)
	return f"{header}\n{body}"


def format_reference(color: bool) -> str:
	"""``Reference:`` block — local source path, remote examples, PyPI."""
	# Source is the package's install path — a local directory openable without a
	# network round-trip; examples ship only on GitHub, not in the wheel.
	entries = (
		("Source (docs live here)", str(Path(__file__).parent.parent)),
		("Examples", "https://github.com/JPHutchins/camas/tree/main/examples"),
		("PyPI", "https://pypi.org/project/camas/"),
	)
	width = max(len(label) for label, _ in entries) + 1
	header = maybe_color("Reference:", CAMAS_VIOLET, color)
	body = "\n".join(
		f"  {maybe_color((label + ':').ljust(width), BOLD_CYAN, color)}  "
		f"{maybe_color(value, GREY, color)}"
		for label, value in entries
	)
	return f"{header}\n{body}"


def print_available_effects(
	scope_effects: Mapping[str, type[Effect[Any]]] = {},
	default_effect_names: frozenset[str] = frozenset(),
) -> None:
	print(
		format_available_effects(
			scope_effects=scope_effects, default_effect_names=default_effect_names
		)
	)


def first_line_doc(obj: Any) -> str:
	doc = getattr(obj, "__doc__", None) or ""
	return next((stripped for line in doc.splitlines() if (stripped := line.strip())), "")
