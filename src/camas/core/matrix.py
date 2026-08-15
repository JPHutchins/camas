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

from ..v0.task import Group, Parallel, Sequential, Task, TaskNode, rebuilt
from .task import MatrixBinding, VarBinding, node_label, task_label

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import PathScope, WhenPredicate


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


def substitute_help(help: str | None, binding: MatrixBinding) -> str | None:
	return substitute_in_str(help, binding) if help is not None else None


def substitute_paths(
	paths: str | PathScope | None, binding: MatrixBinding
) -> str | PathScope | None:
	return substitute_in_str(paths, binding) if isinstance(paths, str) else paths


def substitute_when(
	when: str | tuple[str, ...] | WhenPredicate | None, binding: MatrixBinding
) -> str | tuple[str, ...] | WhenPredicate | None:
	"""Substitute matrix bindings into a ``when`` predicate: a prefix string or tuple of
	prefixes is templated like any other field; a callable passes through unchanged.

	>>> substitute_when("pkg-{PY}", (VarBinding("PY", "3.14"),))
	'pkg-3.14'
	>>> substitute_when(("pkg-{PY}", "other"), (VarBinding("PY", "3.14"),))
	('pkg-3.14', 'other')
	"""
	match when:
		case None:
			return None
		case str():
			return substitute_in_str(when, binding)
		case tuple():
			return tuple(  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
				substitute_in_str(w, binding)  # ty: ignore[invalid-argument-type]
				for w in when
			)
		case _:
			return when


def specialize_task(task: Task, binding: MatrixBinding, suffix: str) -> Task:
	"""Specialize a leaf Task with concrete variable values from a matrix binding.

	>>> specialize_task(Task("test {PY}"), (VarBinding("PY", "3.14"),), "[PY=3.14]")
	Task(cmd='test 3.14', name='test 3.14 [PY=3.14]', env={'PY': '3.14'}, cwd=None)
	>>> specialize_task(Task("go", env={"VENV": ".venv-{PY}"}), (VarBinding("PY", "3.14"),), "[PY=3.14]").env
	{'VENV': '.venv-3.14', 'PY': '3.14'}
	>>> specialize_task(Task("t {paths}", paths="pkg-{PY}"), (VarBinding("PY", "3.14"),), "[PY=3.14]").paths
	'pkg-3.14'
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
		help=substitute_help(task.help, binding),
		mutates=task.mutates,
		paths=substitute_paths(task.paths, binding),
		when=substitute_when(task.when, binding),
		agent_format=task.agent_format,
	)


def specialize_node(task: TaskNode, binding: MatrixBinding, suffix: str) -> TaskNode:
	"""Recursively specialize an entire task tree with concrete variable values.

	Runs on a subtree :func:`expand_matrix` has already expanded, where no node carries a ``matrix``
	or ``variants`` any more — so the rebuilt groups carry none (:func:`rebuilt` passes every field
	verbatim, unexpanded trees included).

	>>> specialize_node(Task("test {X}"), (VarBinding("X", "1"),), "[X=1]")
	Task(cmd='test 1', name='test 1 [X=1]', env={'X': '1'}, cwd=None)
	"""
	match task:
		case Task():
			return specialize_task(task, binding, suffix)
		case Group() as group:
			return rebuilt(
				group,
				tuple(specialize_node(t, binding, suffix) for t in group.tasks),
				name=f"{group.name} {suffix}" if group.name is not None else None,
				env={k: substitute_in_str(v, binding) for k, v in group.env.items()},
				cwd=substitute_cwd(group.cwd, binding),
				help=substitute_help(group.help, binding),
				paths=substitute_paths(group.paths, binding),
				when=substitute_when(group.when, binding),
			)
		case _:
			assert_never(task)


def matrix_axes(task: TaskNode) -> dict[str, tuple[str, ...]]:
	"""Walk a task tree and collect every matrix axis with its values.

	On a name collision the outermost matrix (closest to the root) wins — the value
	``--list`` shows and the one overrides target.

	>>> matrix_axes(Task("hi"))
	{}
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


def variant_axes(task: TaskNode) -> dict[str, tuple[str, ...]]:
	"""Walk a task tree and collect every ``variants`` key with the distinct values it binds,
	in first-seen order. On a name collision the outermost declaration wins, as in
	:func:`matrix_axes`.

	>>> variant_axes(Task("hi"))
	{}
	>>> variant_axes(Parallel(Task("t"), variants=({"b": "x", "t": "1"}, {"b": "y", "t": "1"})))
	{'b': ('x', 'y'), 't': ('1',)}
	"""
	match task:
		case Task():
			return {}
		case Sequential(tasks=tasks, variants=variants) | Parallel(tasks=tasks, variants=variants):
			result: dict[str, tuple[str, ...]] = (
				{key: tuple(dict.fromkeys(v[key] for v in variants)) for key in variants[0]}
				if variants
				else {}
			)
			for child in tasks:
				for k, v in variant_axes(child).items():
					result.setdefault(k, v)
			return result
		case _:
			assert_never(task)


def declared_variants(task: TaskNode) -> tuple[dict[str, str], ...] | None:
	"""The outermost ``variants`` declaration in ``task``'s tree, or ``None`` when it declares
	none — the coupled bundles a catalog shows beside the axes, so a list of axis values is not
	mistaken for a cross product.

	>>> declared_variants(Task("hi")) is None
	True
	>>> declared_variants(Sequential(Parallel(Task("t"), variants=({"b": "x"},))))
	({'b': 'x'},)
	"""
	match task:
		case Task():
			return None
		case (
			Sequential(tasks=children, variants=variants)
			| Parallel(tasks=children, variants=variants)
		):
			if variants is not None:
				return variants
			return next(
				(found for child in children if (found := declared_variants(child)) is not None),
				None,
			)
		case _:
			assert_never(task)


def overridable_axes(task: TaskNode) -> dict[str, tuple[str, ...]]:
	"""Every axis a CLI override can target: the ``matrix`` axes — whose values an override
	*replaces* — then the ``variants`` keys, which an override *filters* to the bundles binding it.

	>>> overridable_axes(Parallel(Task("t"), matrix={"P": ("d",)}, variants=({"b": "x"},)))
	{'P': ('d',), 'b': ('x',)}
	"""
	axes: Final = matrix_axes(task)
	return {**axes, **{k: v for k, v in variant_axes(task).items() if k not in axes}}


def empty_variant_labels(task: TaskNode) -> tuple[str, ...]:
	"""The label of every node in ``task``'s tree declaring ``variants=()``, in first-seen order.

	Zero variants is the coupled counterpart of an empty matrix axis: the node expands to no
	leaves, so that branch silently does nothing — realistically reached by a variant tuple built
	from a comprehension that filtered everything out.

	>>> empty_variant_labels(Task("hi"))
	()
	>>> empty_variant_labels(Parallel(Task("t"), variants=(), name="qemu"))
	('qemu',)
	>>> empty_variant_labels(Sequential(Parallel(Task("t"), variants=())))
	('<unnamed group>',)
	"""
	match task:
		case Task():
			return ()
		case Sequential(tasks=tasks, variants=variants) | Parallel(tasks=tasks, variants=variants):
			here: Final = (node_label(task),) if variants is not None and not variants else ()
			return (*here, *(label for child in tasks for label in empty_variant_labels(child)))
		case _:
			assert_never(task)


def unfilled_required_axes(task: TaskNode) -> tuple[str, ...]:
	"""Every matrix axis declared with no values anywhere in ``task``'s tree, in first-seen order.

	An empty-value axis (``matrix={"version": ()}``) is a required input: its cartesian product is
	empty, so the node it sits on expands to zero leaves and that branch silently does nothing.
	Every node's own matrix is inspected — unlike :func:`matrix_axes`, whose outermost-wins merge
	would mask an inner empty axis behind an outer filled one of the same name — so callers can
	reject a run before it expands to nothing.

	>>> unfilled_required_axes(Task("hi"))
	()
	>>> unfilled_required_axes(Parallel(Task("t"), matrix={"version": ()}))
	('version',)
	>>> unfilled_required_axes(Sequential(Task("t"), matrix={"PY": ("3.13",), "version": ()}))
	('version',)
	>>> unfilled_required_axes(Sequential(Parallel(Task("t"), matrix={"PY": ()}), matrix={"PY": ("3.13",)}))
	('PY',)
	"""
	match task:
		case Task():
			return ()
		case Sequential(tasks=tasks, matrix=matrix) | Parallel(tasks=tasks, matrix=matrix):
			here = tuple(name for name, values in (matrix or {}).items() if not values)
			nested = tuple(a for child in tasks for a in unfilled_required_axes(child))
			return tuple(dict.fromkeys((*here, *nested)))
		case _:
			assert_never(task)


def override_matrix(task: TaskNode, overrides: Mapping[str, tuple[str, ...]]) -> TaskNode:
	"""Replace each ``matrix[k]`` with ``overrides[k]``, and keep only the ``variants`` binding
	one of ``overrides[k]``, wherever the key appears.

	Strict on keys (every override must match an axis in the tree), permissive on
	values for a ``matrix`` axis (any tuple is accepted) — a ``variants`` key instead selects
	among the declared bundles, since replacing one key of a coupled bundle would fabricate a
	combination the project never declared.

	Raises:
		ValueError: if an override key matches no axis in the tree, or if it leaves a node with
			no variant.

	>>> override_matrix(Parallel(Task("t"), matrix={"PY": ("3.12", "3.13")}), {"PY": ("3.13",)}).matrix  # type: ignore[union-attr]
	{'PY': ('3.13',)}
	>>> override_matrix(Parallel(Task("t"), matrix={"PY": ("3.12",)}), {"XX": ("a",)})
	Traceback (most recent call last):
	    ...
	ValueError: unknown matrix axis 'XX' (known: PY)
	>>> node = Parallel(Task("t"), variants=({"b": "x", "t": "1"}, {"b": "y", "t": "2"}))
	>>> override_matrix(node, {"b": ("y",)}).variants  # type: ignore[union-attr]
	({'b': 'y', 't': '2'},)
	>>> override_matrix(node, {"b": ("z",)})
	Traceback (most recent call last):
	    ...
	ValueError: no variant matches b=z; the declared variants bind b: x, y
	"""
	if not overrides:
		return task
	axes: Final = overridable_axes(task)
	for key in overrides:
		if key not in axes:
			known = ", ".join(sorted(axes)) or "none"
			raise ValueError(f"unknown matrix axis {key!r} (known: {known})")
	return apply_overrides(task, overrides)


def _no_variant_message(
	variants: tuple[dict[str, str], ...], overrides: Mapping[str, tuple[str, ...]]
) -> str:
	"""The error for an override that filters a node's variants down to nothing.

	>>> _no_variant_message(({"b": "x"}, {"b": "y"}), {"b": ("z",)})
	'no variant matches b=z; the declared variants bind b: x, y'
	"""
	relevant: Final = {k: v for k, v in overrides.items() if k in variants[0]}
	pinned: Final = ", ".join(f"{k}={'|'.join(v)}" for k, v in relevant.items())
	declared: Final = "; ".join(
		f"{k}: {', '.join(dict.fromkeys(v[k] for v in variants))}" for k in relevant
	)
	return f"no variant matches {pinned}; the declared variants bind {declared}"


def apply_overrides(task: TaskNode, overrides: Mapping[str, tuple[str, ...]]) -> TaskNode:
	def applied(matrix: dict[str, tuple[str, ...]] | None) -> dict[str, tuple[str, ...]] | None:
		if matrix is None:
			return None
		return {k: overrides.get(k, v) for k, v in matrix.items()}

	def kept(variants: tuple[dict[str, str], ...] | None) -> tuple[dict[str, str], ...] | None:
		if not variants:
			return variants
		matched = tuple(
			v for v in variants if all(v[k] in vals for k, vals in overrides.items() if k in v)
		)
		if not matched:
			raise ValueError(_no_variant_message(variants, overrides))
		return matched

	match task:
		case Task():
			return task
		case Group() as group:
			return rebuilt(
				group,
				tuple(apply_overrides(t, overrides) for t in group.tasks),
				matrix=applied(group.matrix),
				variants=kept(group.variants),
			)
		case _:
			assert_never(task)


def matrix_bindings(matrix: dict[str, tuple[str, ...]]) -> tuple[MatrixBinding, ...]:
	"""Generate all cartesian-product bindings from a matrix definition.

	>>> matrix_bindings({"PY": ("3.12", "3.13")})
	((VarBinding(name='PY', value='3.12'),), (VarBinding(name='PY', value='3.13'),))
	"""
	keys: Final = tuple(matrix.keys())
	return tuple(
		tuple(VarBinding(k, v) for k, v in zip(keys, vals, strict=True))
		for vals in itertools.product(*matrix.values())
	)


def variant_bindings(variants: tuple[dict[str, str], ...]) -> tuple[MatrixBinding, ...]:
	"""Each coupled variant bundle as one binding.

	>>> variant_bindings(({"backend": "thumbv7", "target": "thumbv7m-none-eabi"},))
	((VarBinding(name='backend', value='thumbv7'), VarBinding(name='target', value='thumbv7m-none-eabi')),)
	"""
	return tuple(tuple(VarBinding(k, v) for k, v in variant.items()) for variant in variants)


def node_bindings(
	matrix: dict[str, tuple[str, ...]] | None, variants: tuple[dict[str, str], ...] | None
) -> tuple[MatrixBinding, ...]:
	"""Every binding a node fans out over: its ``matrix`` cross product × its ``variants``.

	An absent declaration is the identity — one empty binding — while a declared-but-empty one
	(``matrix={"PY": ()}``, ``variants=()``) yields no bindings at all, so the node expands to
	nothing; the required-axis and empty-variant checks reject that before a run.

	>>> node_bindings({"PY": ("3.13",)}, None) == matrix_bindings({"PY": ("3.13",)})
	True
	>>> node_bindings(None, ({"b": "x"}, {"b": "y"}))
	((VarBinding(name='b', value='x'),), (VarBinding(name='b', value='y'),))
	>>> node_bindings({"P": ("d", "r")}, ({"b": "x"},))
	((VarBinding(name='P', value='d'), VarBinding(name='b', value='x')), (VarBinding(name='P', value='r'), VarBinding(name='b', value='x')))
	>>> node_bindings(None, ()), node_bindings({"PY": ()}, None)
	((), ())
	"""
	axes: Final = matrix_bindings(matrix) if matrix is not None else ((),)
	bundles: Final = variant_bindings(variants) if variants is not None else ((),)
	return tuple((*a, *b) for a in axes for b in bundles)


def binding_suffix(binding: MatrixBinding) -> str:
	"""Format a matrix binding as a bracketed ``[name=value, ...]`` suffix."""
	return "[" + ", ".join(f"{vb.name}={vb.value}" for vb in binding) + "]"


def expand_sequential_matrix(
	children: tuple[TaskNode, ...],
	bindings: tuple[MatrixBinding, ...],
	name: str | None,
	container_env: dict[str, str],
	container_cwd: Path | None,
	help: str | None,
) -> Parallel:
	"""Expand a Sequential's matrix into a Parallel of cloned Sequentials.

	Each per-binding Sequential carries the binding-scope env (container env with
	matrix values substituted, plus the binding itself) so the display can show
	it once at the group header instead of on every leaf.

	>>> bindings = node_bindings({"X": ("1", "2")}, None)
	>>> result = expand_sequential_matrix((Task("build"), Task("test")), bindings, "ci", {}, None, None)
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
				help=substitute_help(help, binding),
			)
			for binding in bindings
		),
		name=name,
		help=help,
	)


def expand_parallel_matrix(
	children: tuple[TaskNode, ...],
	bindings: tuple[MatrixBinding, ...],
	name: str | None,
	container_env: dict[str, str],
	container_cwd: Path | None,
	help: str | None,
) -> Parallel:
	"""Expand a Parallel's matrix into a flat Parallel of all binding × child products.

	The shared ``container_env`` is kept on the outer Parallel only when it has the
	same value across every binding; per-binding pieces land on the individual
	specialized leaves via ``specialize_node``.

	>>> bindings = node_bindings({"PY": ("3.12", "3.13")}, None)
	>>> result = expand_parallel_matrix((Task("test"),), bindings, None, {}, None, None)
	>>> len(result.tasks)
	2
	"""
	return Parallel(
		*(
			specialize_node(child, binding, binding_suffix(binding))
			for binding in bindings
			for child in children
		),
		name=name,
		env={k: v for k, v in container_env.items() if "{" not in v},
		cwd=container_cwd if container_cwd is not None and "{" not in str(container_cwd) else None,
		help=help,
	)


def _when_from_cwd(cwd: Path | None) -> str | None:
	"""The ``when`` prefix a leaf falls back to when it sets none: a relative ``cwd`` as its own
	POSIX prefix, so a task rooted in a subdir runs only when that subdir changed. An absolute
	``cwd`` — which could never match the repo-relative changed set — has no fallback.

	>>> _when_from_cwd(Path("code-gen"))
	'code-gen'
	>>> _when_from_cwd(None) is None
	True
	"""
	return cwd.as_posix() if cwd is not None and not cwd.is_absolute() else None


def expand_matrix(
	task: TaskNode,
	ancestor_env: Mapping[str, str] | None = None,
	ancestor_cwd: Path | None = None,
	ancestor_paths: str | PathScope | None = None,
	ancestor_when: str | tuple[str, ...] | WhenPredicate | None = None,
) -> TaskNode:
	"""Recursively expand all matrix parameters and propagate container env/cwd/paths/when into
	leaves.

	Execution and scoping read the accumulated values from leaves. A group rebuilt without a
	matrix retains its own env/cwd/paths/when for display; a matrix expansion's synthesized
	wrapper nodes carry env/cwd only — paths and when live on the specialized leaves. A child's
	own ``cwd``/``paths``/``when`` takes precedence over an ancestor's. A leaf that ends up with a
	relative ``cwd`` but no ``when`` falls back to gating on its ``cwd`` directory
	(:func:`_when_from_cwd`); set ``when="."`` to opt back into always-run.

	>>> expand_matrix(Task("echo hi"))
	Task(cmd='echo hi', name=None, env={}, cwd=None)
	>>> result = expand_matrix(Parallel(Task("test"), matrix={"X": ("1", "2")}))
	>>> len(result.tasks)  # type: ignore[union-attr]
	2
	>>> coupled = expand_matrix(Parallel(Task("go {b}"), variants=({"b": "x"}, {"b": "y"})))
	>>> [t.cmd for t in coupled.tasks]  # type: ignore[union-attr]
	['go x', 'go y']
	>>> expand_matrix(Parallel(Task("hi"), env={"K": "v"})).tasks[0].env  # type: ignore[union-attr]
	{'K': 'v'}
	>>> expand_matrix(Parallel(Task("hi"), cwd=Path("w"))).tasks[0].cwd == Path("w")  # type: ignore[union-attr]
	True
	>>> expand_matrix(Parallel(Task("ruff {paths}"), paths=".")).tasks[0].paths  # type: ignore[union-attr]
	'.'
	>>> expand_matrix(Parallel(Task("cargo build"), when="src")).tasks[0].when  # type: ignore[union-attr]
	'src'
	>>> expand_matrix(Task("cargo build", cwd="code-gen")).when
	'code-gen'
	>>> expand_matrix(Task("cargo build", cwd="code-gen", when="only")).when
	'only'
	"""
	parent_env: Final = dict(ancestor_env) if ancestor_env else {}
	match task:
		case Task():
			leaf_cwd: Final = task.cwd if task.cwd is not None else ancestor_cwd
			leaf_when: Final = task.when if task.when is not None else ancestor_when
			return Task(
				cmd=task.cmd,
				name=task.name,
				env={**parent_env, **task.env},
				cwd=leaf_cwd,
				help=task.help,
				mutates=task.mutates,
				paths=task.paths if task.paths is not None else ancestor_paths,
				when=leaf_when if leaf_when is not None else _when_from_cwd(leaf_cwd),
				agent_format=task.agent_format,
			)
		case Sequential(
			tasks=tasks, matrix=matrix, variants=variants, env=env, cwd=cwd, paths=paths, when=when
		):
			seq_env: Final = parent_env | env
			seq_cwd: Final = cwd if cwd is not None else ancestor_cwd
			seq_paths: Final = paths if paths is not None else ancestor_paths
			seq_when: Final = when if when is not None else ancestor_when
			seq_expanded: Final = tuple(
				expand_matrix(t, seq_env, seq_cwd, seq_paths, seq_when) for t in tasks
			)
			if matrix is None and variants is None:
				return Sequential(
					*seq_expanded,
					name=task.name,
					env=env,
					cwd=cwd,
					help=task.help,
					paths=paths,
					when=when,
				)
			return expand_sequential_matrix(
				seq_expanded, node_bindings(matrix, variants), task.name, env, cwd, task.help
			)
		case Parallel(
			tasks=tasks, matrix=matrix, variants=variants, env=env, cwd=cwd, paths=paths, when=when
		):
			par_env: Final = parent_env | env
			par_cwd: Final = cwd if cwd is not None else ancestor_cwd
			par_paths: Final = paths if paths is not None else ancestor_paths
			par_when: Final = when if when is not None else ancestor_when
			par_expanded: Final = tuple(
				expand_matrix(t, par_env, par_cwd, par_paths, par_when) for t in tasks
			)
			if matrix is None and variants is None:
				return Parallel(
					*par_expanded,
					name=task.name,
					env=env,
					cwd=cwd,
					help=task.help,
					paths=paths,
					when=when,
				)
			return expand_parallel_matrix(
				par_expanded, node_bindings(matrix, variants), task.name, env, cwd, task.help
			)
		case _:
			assert_never(task)
