# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Matrix expansion: bind axis values, specialize subtrees, and apply CLI overrides."""

from __future__ import annotations

import functools
import itertools
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from .task import MatrixBinding, Parallel, Sequential, Task, TaskNode, VarBinding, task_label

if TYPE_CHECKING:
	from collections.abc import Mapping


def resolve_cmd(cmd: str | tuple[str, ...]) -> tuple[str, ...]:
	"""Resolve a command to a tuple of argv tokens, splitting shell strings with shlex.

	>>> resolve_cmd(("echo", "hi"))
	('echo', 'hi')
	>>> resolve_cmd("echo hi")
	('echo', 'hi')
	"""
	match cmd:
		case str():
			return tuple(shlex.split(cmd))
		case tuple():
			return cmd
		case _:
			assert_never(cmd)


def substitute_in_str(text: str, binding: MatrixBinding) -> str:
	"""Replace {name} placeholders in a string with binding values.

	>>> substitute_in_str("test --python {PY}", (VarBinding("PY", "3.14"),))
	'test --python 3.14'
	>>> substitute_in_str("no placeholders", (VarBinding("X", "1"),))
	'no placeholders'
	"""
	return functools.reduce(
		lambda acc, vb: acc.replace(f"{{{vb.name}}}", vb.value),
		binding,
		text,
	)


def substitute_in_tuple(parts: tuple[str, ...], binding: MatrixBinding) -> tuple[str, ...]:
	"""Replace {name} placeholders in each element of a tuple.

	>>> substitute_in_tuple(("test", "--python", "{PY}"), (VarBinding("PY", "3.14"),))
	('test', '--python', '3.14')
	"""
	return functools.reduce(
		lambda acc, vb: tuple(p.replace(f"{{{vb.name}}}", vb.value) for p in acc),
		binding,
		parts,
	)


def substitute_cwd(cwd: Path | None, binding: MatrixBinding) -> Path | None:
	return Path(substitute_in_str(str(cwd), binding)) if cwd is not None else None


def specialize_task(task: Task, binding: MatrixBinding, suffix: str) -> Task:
	"""Specialize a leaf Task with concrete variable values from a matrix binding.

	>>> specialize_task(Task("test {PY}"), (VarBinding("PY", "3.14"),), "[PY=3.14]")
	Task(cmd='test 3.14', name='test 3.14 [PY=3.14]', env={'PY': '3.14'}, cwd=None)
	>>> specialize_task(Task("go", env={"VENV": ".venv-{PY}"}), (VarBinding("PY", "3.14"),), "[PY=3.14]").env
	{'VENV': '.venv-3.14', 'PY': '3.14'}
	"""
	match task.cmd:
		case str():
			new_cmd: str | tuple[str, ...] = substitute_in_str(task.cmd, binding)
		case tuple():
			new_cmd = substitute_in_tuple(task.cmd, binding)
		case _:
			assert_never(task.cmd)
	return Task(
		cmd=new_cmd,
		name=f"{substitute_in_str(task_label(task), binding)} {suffix}",
		env={k: substitute_in_str(v, binding) for k, v in task.env.items()} | dict(binding),
		cwd=substitute_cwd(task.cwd, binding),
	)


def specialize_node(task: TaskNode, binding: MatrixBinding, suffix: str) -> TaskNode:
	"""Recursively specialize an entire task tree with concrete variable values.

	>>> specialize_node(Task("test {X}"), (VarBinding("X", "1"),), "[X=1]")
	Task(cmd='test 1', name='test 1 [X=1]', env={'X': '1'}, cwd=None)
	"""
	match task:
		case Task():
			return specialize_task(task, binding, suffix)
		case Sequential(tasks=tasks, name=name, cwd=cwd):
			return Sequential(
				*(specialize_node(t, binding, suffix) for t in tasks),
				name=f"{name} {suffix}" if name is not None else None,
				cwd=substitute_cwd(cwd, binding),
			)
		case Parallel(tasks=tasks, name=name, cwd=cwd):
			return Parallel(
				*(specialize_node(t, binding, suffix) for t in tasks),
				name=f"{name} {suffix}" if name is not None else None,
				cwd=substitute_cwd(cwd, binding),
			)
		case _:
			assert_never(task)


def matrix_axes(task: TaskNode) -> dict[str, tuple[str, ...]]:
	"""Walk a task tree and collect every matrix axis with its values.

	When the same axis name appears in multiple matrices, the outermost (closest
	to the root) wins — that's the value the user sees in ``--list`` and the one
	overrides should target. Inner duplicates with the same key but different
	values are unusual; users almost never write them.

	>>> matrix_axes(Task("hi"))
	{}
	>>> matrix_axes(Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")}))
	{'PY': ('3.12', '3.13')}
	>>> matrix_axes(Sequential(Parallel(Task("t"), matrix={"OS": ("a", "b")}), matrix={"PY": ("3.12",)}))
	{'PY': ('3.12',), 'OS': ('a', 'b')}
	"""
	match task:
		case Task():
			return {}
		case Sequential(tasks=tasks, matrix=matrix) | Parallel(tasks=tasks, matrix=matrix):
			result: dict[str, tuple[str, ...]] = dict(matrix) if matrix else {}
			for child in tasks:
				for k, v in matrix_axes(child).items():
					result.setdefault(k, v)
			return result
		case _:
			assert_never(task)


def override_matrix(task: TaskNode, overrides: Mapping[str, tuple[str, ...]]) -> TaskNode:
	"""Return a tree with each ``matrix[k]`` replaced by ``overrides[k]`` everywhere
	the key appears. Strict on keys: every override must match an axis present in
	the tree. Permissive on values: any tuple is accepted.

	Raises:
		ValueError: if an override key matches no matrix axis in the tree.

	>>> override_matrix(Parallel(Task("t"), matrix={"PY": ("3.12", "3.13")}), {"PY": ("3.13",)}).matrix  # type: ignore[union-attr]
	{'PY': ('3.13',)}
	>>> override_matrix(Task("hi"), {})
	Task(cmd='hi', name=None, env={}, cwd=None)
	>>> override_matrix(Parallel(Task("t"), matrix={"PY": ("3.12",)}), {"XX": ("a",)})
	Traceback (most recent call last):
	    ...
	ValueError: unknown matrix axis 'XX' (known: PY)
	"""
	if not overrides:
		return task
	axes: Final = matrix_axes(task)
	for key in overrides:
		if key not in axes:
			known = ", ".join(sorted(axes)) or "none"
			raise ValueError(f"unknown matrix axis {key!r} (known: {known})")
	return apply_overrides(task, overrides)


def apply_overrides(task: TaskNode, overrides: Mapping[str, tuple[str, ...]]) -> TaskNode:
	def applied(matrix: dict[str, tuple[str, ...]] | None) -> dict[str, tuple[str, ...]] | None:
		if matrix is None:
			return None
		return {k: overrides.get(k, v) for k, v in matrix.items()}

	match task:
		case Task():
			return task
		case Sequential(tasks=tasks, name=name, matrix=matrix, env=env, cwd=cwd, help=help):
			return Sequential(
				*(apply_overrides(t, overrides) for t in tasks),
				name=name,
				matrix=applied(matrix),
				env=env,
				cwd=cwd,
				help=help,
			)
		case Parallel(tasks=tasks, name=name, matrix=matrix, env=env, cwd=cwd, help=help):
			return Parallel(
				*(apply_overrides(t, overrides) for t in tasks),
				name=name,
				matrix=applied(matrix),
				env=env,
				cwd=cwd,
				help=help,
			)
		case _:
			assert_never(task)


def matrix_bindings(matrix: dict[str, tuple[str, ...]]) -> tuple[MatrixBinding, ...]:
	"""Generate all cartesian product bindings from a matrix definition.

	>>> matrix_bindings({"PY": ("3.12", "3.13")})
	((VarBinding(name='PY', value='3.12'),), (VarBinding(name='PY', value='3.13'),))
	>>> len(matrix_bindings({"A": ("1", "2"), "B": ("x", "y")}))
	4
	"""
	keys: Final = tuple(matrix.keys())
	return tuple(
		tuple(VarBinding(k, v) for k, v in zip(keys, vals, strict=True))
		for vals in itertools.product(*matrix.values())
	)


def binding_suffix(binding: MatrixBinding) -> str:
	"""Format a matrix binding as a bracketed suffix.

	>>> binding_suffix((VarBinding("PY", "3.14"),))
	'[PY=3.14]'
	>>> binding_suffix((VarBinding("OS", "linux"), VarBinding("PY", "3.14")))
	'[OS=linux, PY=3.14]'
	"""
	return "[" + ", ".join(f"{vb.name}={vb.value}" for vb in binding) + "]"


def expand_sequential_matrix(
	children: tuple[TaskNode, ...],
	matrix: dict[str, tuple[str, ...]],
	name: str | None,
	container_env: dict[str, str],
	container_cwd: Path | None,
) -> Parallel:
	"""Expand a Sequential's matrix into a Parallel of cloned Sequentials.

	Each per-binding Sequential carries the binding-scope env (container env with
	matrix values substituted, plus the binding itself) so the display can show
	it once at the group header instead of on every leaf.

	>>> result = expand_sequential_matrix((Task("build"), Task("test")), {"X": ("1", "2")}, "ci", {}, None)
	>>> len(result.tasks)
	2
	>>> all(isinstance(t, Sequential) for t in result.tasks)
	True
	>>> result.tasks[0].env  # type: ignore[union-attr]
	{'X': '1'}
	"""
	return Parallel(
		*(
			Sequential(
				*(specialize_node(child, binding, binding_suffix(binding)) for child in children),
				name=(f"{name} {binding_suffix(binding)}" if name is not None else None),
				env={k: substitute_in_str(v, binding) for k, v in container_env.items()}
				| dict(binding),
				cwd=substitute_cwd(container_cwd, binding),
			)
			for binding in matrix_bindings(matrix)
		),
		name=name,
	)


def expand_parallel_matrix(
	children: tuple[TaskNode, ...],
	matrix: dict[str, tuple[str, ...]],
	name: str | None,
	container_env: dict[str, str],
	container_cwd: Path | None,
) -> Parallel:
	"""Expand a Parallel's matrix into a flat Parallel of all binding × child products.

	The shared ``container_env`` is kept on the outer Parallel only when it has the
	same value across every binding; per-binding pieces land on the individual
	specialized leaves via ``specialize_node``.

	>>> result = expand_parallel_matrix((Task("test"),), {"PY": ("3.12", "3.13")}, None, {}, None)
	>>> len(result.tasks)
	2
	"""
	return Parallel(
		*(
			specialize_node(child, binding, binding_suffix(binding))
			for binding in matrix_bindings(matrix)
			for child in children
		),
		name=name,
		env={k: v for k, v in container_env.items() if "{" not in v},
		cwd=container_cwd if container_cwd is not None and "{" not in str(container_cwd) else None,
	)


def expand_matrix(
	task: TaskNode,
	ancestor_env: Mapping[str, str] | None = None,
	ancestor_cwd: Path | None = None,
) -> TaskNode:
	"""Recursively expand all matrix parameters and propagate container env/cwd into leaves.

	Container env and cwd are also retained on the expanded group nodes (for display
	purposes); execution still reads the accumulated values from leaves. A child's
	``cwd`` takes precedence over an ancestor's.

	>>> expand_matrix(Task("echo hi"))
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> result = expand_matrix(Parallel(Task("test"), matrix={"X": ("1", "2")}))
	>>> len(result.tasks)  # type: ignore[union-attr]
	2
	>>> expand_matrix(Parallel(Task("hi"), env={"K": "v"})).tasks[0].env  # type: ignore[union-attr]
	{'K': 'v'}
	>>> expand_matrix(Parallel(Task("hi"), cwd=Path("w"))).tasks[0].cwd == Path("w")  # type: ignore[union-attr]
	True
	"""
	parent_env: Final = dict(ancestor_env) if ancestor_env else {}
	match task:
		case Task():
			return Task(
				cmd=task.cmd,
				name=task.name,
				env={**parent_env, **task.env},
				cwd=task.cwd if task.cwd is not None else ancestor_cwd,
			)
		case Sequential(tasks=tasks, matrix=matrix, env=env, cwd=cwd):
			seq_env: Final = parent_env | env
			seq_cwd: Final = cwd if cwd is not None else ancestor_cwd
			seq_expanded: Final = tuple(expand_matrix(t, seq_env, seq_cwd) for t in tasks)
			if matrix is None:
				return Sequential(*seq_expanded, name=task.name, env=env, cwd=cwd)
			return expand_sequential_matrix(seq_expanded, matrix, task.name, env, cwd)
		case Parallel(tasks=tasks, matrix=matrix, env=env, cwd=cwd):
			par_env: Final = parent_env | env
			par_cwd: Final = cwd if cwd is not None else ancestor_cwd
			par_expanded: Final = tuple(expand_matrix(t, par_env, par_cwd) for t in tasks)
			if matrix is None:
				return Parallel(*par_expanded, name=task.name, env=env, cwd=cwd)
			return expand_parallel_matrix(par_expanded, matrix, task.name, env, cwd)
		case _:
			assert_never(task)
