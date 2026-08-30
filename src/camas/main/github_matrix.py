# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""``--github-matrix``: emit a task's fan-out as GitHub Actions ``strategy.matrix`` JSON.

Three shapes, one per way a ``tasks.py`` fans out — picked automatically, or pinned with
``--github-matrix=SHAPE`` so a refactor that would change the emitted *shape* fails the
``discover`` job instead of handing the workflow a differently-shaped object:

``axes``
	A ``matrix=`` cross product as object-of-arrays — GHA's own matrix form::

	    {"PY": ["3.10", "3.11"], "PROFILE": ["debug", "release"]}

``variants``
	A coupled ``variants=`` fan-out, fully enumerated as GHA ``include:`` entries::

	    {"include": [{"backend": "thumbv7", "target": "thumbv7m-none-eabi"}, …]}

	No base axis is emitted beside ``include``: an ``include`` entry merges into every base
	combination rather than multiplying it, so a base axis would silently change the job set.

``tasks``
	One job per child of a ``Parallel`` — the common CI shape with no version matrix at all::

	    {"task": ["build", "lint", "test"]}

	Values are the children's *binding* names, the same names ``camas <task>`` dispatches on,
	so every emitted value resolves to a real task — a ``Project()`` child included, which emits
	its own binding (``web``) when that binding runs what the parent composed, else the mounted
	binding of the node it did compose (``web.build``).

Consume the whole object as the matrix::

    matrix: ${{ fromJSON(needs.discover.outputs.matrix) }}

or, for ``axes`` and ``tasks``, one array at a time, composed with the YAML-side axes a shell
command can't set from inside a job (a runner's ``os``)::

    matrix:
      os: [ubuntu-latest, macos-latest]
      PY: ${{ fromJSON(needs.discover.outputs.matrix).PY }}

Whatever the shape, the emitted object is checked against the task's real run-set
(:func:`unfaithful`): each emitted job must select exactly its own cell's leaves, and the jobs
together must run every leaf exactly once. A fan-out with no faithful projection — independent
fan-outs in one tree, heterogeneous nested matrices, a plain leaf beside matrixed siblings — is
rejected rather than silently widened or narrowed. CLI overrides (``--PY 3.13``) flow through
the same ``override_matrix`` pipeline the runner uses, so the emitted JSON reflects exactly what
the run would have executed.

Output is TTY-aware: indented for interactive preview, compact one-line for pipes — so
``camas matrix --github-matrix`` reads cleanly in a shell *and*
``$(camas matrix --github-matrix)`` works directly with ``$GITHUB_OUTPUT``.
"""

from __future__ import annotations

import functools
import json
import math
import sys
from collections import Counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple, TypeAlias, cast

if sys.version_info >= (3, 11):
	from typing import assert_never
else:  # pragma: no cover
	from typing_extensions import assert_never

from ..core.matrix import (
	empty_variant_labels,
	expand_matrix,
	matrix_axes,
	node_bindings,
	override_matrix,
	unfilled_required_axes,
	variant_axes,
)
from ..core.task import node_label, task_label
from ..core.traversal import flatten_leaves
from ..v0.task import Parallel, Pipe, Sequential, Task
from .format import format_empty_variants_error

if TYPE_CHECKING:
	from collections.abc import Mapping

	from ..v0.task import TaskNode


ShapeName: TypeAlias = Literal["auto", "axes", "variants", "tasks"]
"""``--github-matrix=SHAPE``: the projection to emit, or ``auto`` to derive it from the tree."""

SHAPE_NAMES: Final = ("auto", "axes", "variants", "tasks")
"""Every ``--github-matrix`` shape, for argparse choices and help text."""

JOB_AXIS: Final = "task"
"""The axis name the ``tasks`` shape emits, consumed as ``${{ matrix.task }}``."""

NO_TASKS: Final[Mapping[str, TaskNode]] = MappingProxyType({})
"""Read-only stand-in for "no task mapping given" — the shapes that don't resolve bindings pass
nothing. Immutable (as :data:`camas.v0.task._EMPTY_ENV` is) so a shared default can't be mutated
into cross-call state.
"""

JOB_STATE_FIELDS: Final = ("matrix", "variants", "env", "cwd", "paths", "when")
"""Every ``Group`` field a per-child ``camas <child>`` job would *not* inherit, so the ``tasks``
shape refuses to emit past one that is set. ``name``/``help`` are display-only and don't count; a
new ``Group`` field has to be classified one way or the other, which a drift test enforces.
"""


class Fanout(NamedTuple):
	"""A subtree's declared fan-out: the distinct cells it runs — each a binding of axis and
	variant keys to one value — plus the labels of any leaves that run under no cell at all.
	"""

	cells: tuple[dict[str, str], ...]
	uncovered: tuple[str, ...]


class Axes(NamedTuple):
	"""The cross-product projection: one array per axis."""

	axes: dict[str, list[str]]


class Variants(NamedTuple):
	"""The enumerated projection: one GHA ``include:`` entry per cell, with no base axes."""

	cells: tuple[dict[str, str], ...]


class Jobs(NamedTuple):
	"""The per-child projection: one job per child, named as ``camas`` dispatches it."""

	names: tuple[str, ...]


Emission: TypeAlias = Axes | Variants | Jobs


def merge_cells(
	left: tuple[dict[str, str], ...], right: tuple[dict[str, str], ...]
) -> tuple[dict[str, str], ...]:
	"""``left`` and ``right`` as one run of distinct cells, in first-seen order — deduplicated
	*within* each side too, so a node whose own bindings collapse onto one cell (an axis shadowed
	by an outer one of the same name) doesn't leave a repeat behind.

	>>> merge_cells(({"PY": "3.13"},), ({"PY": "3.13"}, {"PY": "3.14"}))
	({'PY': '3.13'}, {'PY': '3.14'})
	>>> merge_cells((), ({"PY": "3.13"}, {"PY": "3.13"}))
	({'PY': '3.13'},)
	"""
	return tuple({tuple(sorted(c.items())): c for c in (*left, *right)}.values())


def bound_cells(
	bindings: tuple[tuple[tuple[str, str], ...], ...], inner: tuple[dict[str, str], ...]
) -> tuple[dict[str, str], ...]:
	"""Each of a node's own ``bindings`` crossed with the cells its children declare — the
	binding's keys first, and winning a name collision, matching
	:func:`camas.core.matrix.matrix_axes`' outermost-wins merge. A node whose children declare no
	cells of their own contributes one cell per binding.

	>>> bound_cells(((("PY", "3.13"),),), ({"PROFILE": "debug"},))
	({'PY': '3.13', 'PROFILE': 'debug'},)
	>>> bound_cells(((("PY", "3.13"),), (("PY", "3.14"),)), ())
	({'PY': '3.13'}, {'PY': '3.14'})
	"""

	def crossed(binding: tuple[tuple[str, str], ...]) -> tuple[dict[str, str], ...]:
		outer = dict(binding)
		if not inner:
			return (outer,)
		return tuple({**outer, **{k: v for k, v in c.items() if k not in outer}} for c in inner)

	return functools.reduce(merge_cells, map(crossed, bindings), ())


def merge_fanouts(left: Fanout, right: Fanout) -> Fanout:
	"""Two subtrees' fan-outs as one: their cells deduplicated, their uncovered leaves appended.

	>>> merge_fanouts(Fanout(({"PY": "3.13"},), ()), Fanout(({"PY": "3.13"},), ("lint",)))
	Fanout(cells=({'PY': '3.13'},), uncovered=('lint',))
	"""
	return Fanout(merge_cells(left.cells, right.cells), (*left.uncovered, *right.uncovered))


def declared_cells(task: TaskNode) -> Fanout:
	"""``task``'s fan-out as the cells it declares — never inferred from a leaf's ``env``, so a
	user ``env=`` key that happens to share an axis name cannot masquerade as a matrix value.

	>>> from camas import Parallel, Sequential, Task
	>>> declared_cells(Parallel(Task("t"), matrix={"PY": ("3.12", "3.13")})).cells
	({'PY': '3.12'}, {'PY': '3.13'})
	>>> declared_cells(Task("lint"))
	Fanout(cells=(), uncovered=('lint',))
	>>> declared_cells(Parallel(Task("t"), variants=({"b": "x"}, {"b": "y"}))).cells
	({'b': 'x'}, {'b': 'y'})
	>>> declared_cells(Sequential(Parallel(Task("t"), matrix={"P": ("d",)}), Task("lint")))
	Fanout(cells=({'P': 'd'},), uncovered=('lint',))
	>>> declared_cells(Parallel(Task("t"), matrix={"PY": ("3.13",)}, variants=({"b": "x"},))).cells
	({'PY': '3.13', 'b': 'x'},)
	"""
	match task:
		case Task():
			return Fanout((), (task_label(task),))
		case (
			Sequential(tasks=children, matrix=matrix, variants=variants)
			| Parallel(tasks=children, matrix=matrix, variants=variants)
			| Pipe(tasks=children, matrix=matrix, variants=variants)
		):
			inner: Final = functools.reduce(
				merge_fanouts, map(declared_cells, children), Fanout((), ())
			)
			if matrix is None and variants is None:
				return inner
			return Fanout(bound_cells(node_bindings(matrix, variants), inner.cells), ())
		case _:
			assert_never(task)


def is_cross_product(cells: tuple[dict[str, str], ...], axes: tuple[str, ...]) -> bool:
	"""Whether ``cells`` is exactly the cartesian product of each axis's distinct values —
	i.e. representable as object-of-arrays without adding or dropping a single job.

	>>> is_cross_product(({'PY': '3.12'}, {'PY': '3.13'}), ('PY',))
	True
	>>> is_cross_product(
	...     ({'PROFILE': 'debug', 'PY': '3.12'}, {'PROFILE': 'debug', 'PY': '3.13'},
	...      {'PROFILE': 'release', 'PY': '3.12'}, {'PROFILE': 'release', 'PY': '3.13'}),
	...     ('PROFILE', 'PY'))
	True
	>>> is_cross_product(
	...     ({'PROFILE': 'debug', 'PY': '3.12'}, {'PROFILE': 'release', 'PY': '3.13'}),
	...     ('PROFILE', 'PY'))
	False
	>>> is_cross_product(({'PY': '3.12'}, {'PROFILE': 'debug'}), ('PY', 'PROFILE'))
	False
	"""
	if not cells or any(tuple(c) != axes for c in cells):
		return False
	sizes: Final = tuple(len(dict.fromkeys(c[a] for c in cells)) for a in axes)
	return math.prod(sizes) == len(cells)


def leaf_counts(node: TaskNode) -> Counter[Task]:
	"""How many times ``node``'s expansion runs each distinct leaf."""
	return Counter(info.task for info in flatten_leaves(expand_matrix(node)))


def unfaithful(task: TaskNode, selected: tuple[TaskNode, ...]) -> str | None:
	"""Why the jobs ``selected`` are not a faithful fan-out of ``task`` — or ``None`` when they
	are: every leaf ``task`` runs is run by exactly one job, and no job runs anything else.

	``selected`` is what ``camas`` would run in each emitted job, so this compares the emitted
	job graph against the run-set rather than trusting the projection that produced it.
	"""
	full: Final = leaf_counts(task)
	emitted: Final = sum(map(leaf_counts, selected), Counter[Task]())
	if emitted == full:
		return None
	jobs: Final = len(selected)
	over: Final = tuple((leaf, n) for leaf, n in emitted.items() if n > full[leaf])
	if over:
		leaf, count = over[0]
		ran = (
			"the task never runs it"
			if full[leaf] == 0
			else f"the task runs it {full[leaf]} time(s)"
		)
		return f"{task_label(leaf)!r} would run in {count} of the {jobs} emitted job(s), but {ran}"
	leaf, count = next((leaf, n) for leaf, n in full.items() if emitted[leaf] < n)
	return (
		f"{task_label(leaf)!r} would run in {emitted[leaf]} of the {jobs} emitted job(s), but the "
		f"task runs it {count} time(s)"
	)


def cell_selection(task: TaskNode, cells: tuple[dict[str, str], ...]) -> tuple[TaskNode, ...]:
	"""What ``camas <task>`` runs in each emitted job: ``task`` with that cell pinned — the same
	``override_matrix`` a ``camas <task> --AXIS VALUE`` job command goes through.

	Raises:
		ValueError: when a cell isn't selectable on its own — pinning it leaves another node with
			no variant, so no job could run just that cell.
	"""
	try:
		return tuple(override_matrix(task, {k: (v,) for k, v in cell.items()}) for cell in cells)
	except ValueError as e:
		raise ValueError(
			f"matrix is not a faithful fan-out: a cell cannot be pinned on its own ({e}) — "
			"declare the coupled keys with variants= on one node, so one job is one cell"
		) from e


def reject_unfaithful(task: TaskNode, selected: tuple[TaskNode, ...], cause: str) -> None:
	"""Raise when ``selected`` doesn't reproduce ``task``'s run-set, appending ``cause`` — the
	shapes this fan-out could have taken instead.

	Raises:
		ValueError: naming the leaf that would run too often, or not at all.
	"""
	diagnosis: Final = unfaithful(task, selected)
	if diagnosis is not None:
		raise ValueError(f"matrix is not a faithful fan-out: {diagnosis} — {cause}")


def reject_uncovered(fanout: Fanout) -> None:
	"""Raise when a leaf runs under no cell — a plain leaf beside matrixed siblings, which no
	GHA matrix can represent (it would run in every job, or in none).

	Raises:
		ValueError: naming each uncovered leaf.
	"""
	if fanout.uncovered:
		raise ValueError(
			f"matrix does not cover every leaf ({', '.join(fanout.uncovered)}): a leaf that runs "
			"under no matrix axis cannot be represented in a GitHub Actions matrix — mixing "
			"matrixed and plain leaves under one --github-matrix task is unsupported"
		)


def reject_unsatisfiable(task: TaskNode) -> None:
	"""Raise when a declaration leaves nothing to expand: an axis with no values (a required
	input) or an empty ``variants``.

	Raises:
		ValueError: naming the axis or the node.
	"""
	required: Final = unfilled_required_axes(task)
	if required:
		raise ValueError(f"matrix axis {required[0]!r} has no values")
	empty: Final = empty_variant_labels(task)
	if empty:
		raise ValueError(format_empty_variants_error(empty))


def axes_emission(task: TaskNode) -> Axes:
	"""``task``'s ``matrix`` cross product as object-of-arrays.

	Raises:
		ValueError: when ``task`` declares no axes, an axis has no values, a leaf runs under no
			axis, or the run-set is not a clean cross product.
	"""
	axes_map: Final = matrix_axes(task)
	if not axes_map:
		raise ValueError(
			"task has no matrix axes to emit as a GitHub Actions job matrix"
			+ (
				" — it declares coupled variants=, which emit as include: entries: "
				"--github-matrix=variants"
				if variant_axes(task)
				else "; --github-matrix=tasks emits one job per child of a Parallel instead"
			)
		)
	reject_unsatisfiable(task)
	repeated: Final = tuple(a for a, v in axes_map.items() if len(dict.fromkeys(v)) != len(v))
	if repeated:
		raise ValueError(
			f"matrix axis {repeated[0]!r} repeats a value, so it runs that cell more than once "
			"but emits it as one job — deduplicate the axis (a version file with a duplicated "
			"line is the usual cause)"
		)
	fanout: Final = declared_cells(task)
	reject_uncovered(fanout)
	axes: Final = tuple(axes_map)
	if not is_cross_product(fanout.cells, axes):
		raise ValueError(
			"matrix is not a clean cross-product; object-of-arrays cannot represent a "
			"heterogeneous fan-out (independent fan-outs in one tree, or nested matrices that "
			"disagree) — declare the coupled cells with variants= to emit them as include:"
		)
	reject_unfaithful(
		task,
		cell_selection(task, fanout.cells),
		"pinning one cell's axes would run another cell's leaves; split the tree, or declare the "
		"coupled cells with variants=",
	)
	return Axes({a: list(dict.fromkeys(c[a] for c in fanout.cells)) for a in axes})


def variants_emission(task: TaskNode) -> Variants:
	"""``task``'s fan-out enumerated as GHA ``include:`` entries.

	Raises:
		ValueError: when ``task`` declares no fan-out at all, a declaration is unsatisfiable, a
			leaf runs under no cell, or the enumerated cells don't reproduce the run-set.
	"""
	reject_unsatisfiable(task)
	fanout: Final = declared_cells(task)
	if not fanout.cells:
		raise ValueError(
			"task has no matrix axes or variants to emit as a GitHub Actions job matrix; "
			"--github-matrix=tasks emits one job per child of a Parallel instead"
		)
	reject_uncovered(fanout)
	reject_unfaithful(
		task,
		cell_selection(task, fanout.cells),
		"pinning one cell would run another cell's leaves; declare the coupled keys with "
		"variants= on one node instead of as separate matrix axes",
	)
	return Variants(fanout.cells)


def dispatch_name(node: TaskNode, tasks: Mapping[str, TaskNode]) -> str | None:
	"""The name ``camas`` dispatches ``node`` by — the binding it is bound to, matched by
	identity first so a value-equal twin can't claim another's name — or ``None`` when ``node``
	is not a task in its own right.

	>>> from camas import Parallel, Task
	>>> lint = Task("ruff check .")
	>>> dispatch_name(lint, {"lint": lint, "ci": Parallel(lint)})
	'lint'
	>>> dispatch_name(Task("other"), {"lint": lint}) is None
	True
	"""
	for name, candidate in tasks.items():
		if candidate is node:
			return name
	for name, candidate in tasks.items():
		if candidate == node:
			return name
	return None


def jobs_emission(task: TaskNode, tasks: Mapping[str, TaskNode]) -> Jobs:
	"""``task``'s children as one job each, dispatched by binding name.

	Raises:
		ValueError: when ``task`` is a leaf or a ``Sequential``, when it declares a fan-out or
			state a per-child job would not inherit, or when a child is not a dispatchable task.
	"""
	match task:
		case Task():
			raise ValueError(
				"task has no matrix axes to emit as a GitHub Actions job matrix, and it is a "
				"single leaf — there are no children to fan out as one job each"
			)
		case Sequential():
			raise ValueError(
				"a Sequential's children run in order, but GitHub Actions matrix jobs run "
				"in parallel — express the ordering with needs: in the workflow, and emit the "
				"parallel step of the pipeline"
			)
		case Pipe():
			raise ValueError(
				"a Pipe's stages are fd-wired into one process pipeline, which GitHub Actions "
				"matrix jobs cannot split — emit the pipeline as a single job, or express the "
				"stages as a Sequential with needs: ordering in the workflow"
			)
		case Parallel(tasks=children) as group:
			blocking = tuple(
				field for field in JOB_STATE_FIELDS if getattr(group, field) not in (None, {})
			)
			if blocking:
				raise ValueError(
					f"{node_label(group)} declares {', '.join(blocking)}, which a per-child "
					"`camas <child>` job would not inherit — move it onto the children, or emit "
					"the axes shape"
				)
			names = tuple(dispatch_name(child, tasks) for child in children)
			missing = tuple(
				node_label(child)
				for child, name in zip(children, names, strict=True)
				if name is None
			)
			if missing:
				raise ValueError(
					f"{', '.join(missing)}: not reachable as `camas <name>`, so a job cannot run "
					"it — bind every child at module scope in tasks.py (build = Task(...); "
					"ci = Parallel(build, ...)). A Project() child composed into a Config field "
					"resolves to that project's node for that field, so name it in that project's "
					"tasks.py to give the job something to dispatch"
				)
			named: Final = cast("tuple[str, ...]", names)
			resolved = tuple(tasks[name] for name in named)
			reject_unfaithful(
				task,
				resolved,
				"a child runs something different on its own than it does under this task",
			)
			return Jobs(named)
		case _:
			assert_never(task)


def auto_shape(task: TaskNode) -> ShapeName:
	"""The shape ``task``'s own declarations call for: ``variants`` when it declares any coupled
	bundle, ``axes`` when it declares a cross product, else ``tasks``.

	>>> from camas import Parallel, Task
	>>> auto_shape(Parallel(Task("t"), matrix={"PY": ("3.13",)}))
	'axes'
	>>> auto_shape(Parallel(Task("t"), variants=({"b": "x"},)))
	'variants'
	>>> auto_shape(Parallel(Task("build"), Task("lint")))
	'tasks'
	"""
	if variant_axes(task) or empty_variant_labels(task):
		return "variants"
	if matrix_axes(task):
		return "axes"
	return "tasks"


def resolve_emission(
	task: TaskNode, tasks: Mapping[str, TaskNode], shape: ShapeName = "auto"
) -> Emission:
	"""``task``'s fan-out projected into the ``shape`` GHA consumes, verified against the run-set;
	the projection raises ``ValueError`` when the fan-out has none.
	"""
	match shape:
		case "auto":
			return resolve_emission(task, tasks, auto_shape(task))
		case "axes":
			return axes_emission(task)
		case "variants":
			return variants_emission(task)
		case "tasks":
			return jobs_emission(task, tasks)
		case _:
			assert_never(shape)


def to_json_object(emission: Emission) -> dict[str, Any]:
	"""``emission`` as the JSON object ``strategy.matrix`` consumes.

	>>> to_json_object(Axes({"PY": ["3.13"]}))
	{'PY': ['3.13']}
	>>> to_json_object(Variants(({"b": "x"},)))
	{'include': [{'b': 'x'}]}
	>>> to_json_object(Jobs(("build", "lint")))
	{'task': ['build', 'lint']}
	"""
	match emission:
		case Axes(axes=axes):
			return dict(axes)
		case Variants(cells=cells):
			return {"include": [dict(cell) for cell in cells]}
		case Jobs(names=names):
			return {JOB_AXIS: list(names)}
		case _:
			assert_never(emission)


def to_matrix_object(
	task: TaskNode, tasks: Mapping[str, TaskNode] = NO_TASKS, shape: ShapeName = "auto"
) -> dict[str, Any]:
	"""Project ``task``'s fan-out into the object ``strategy.matrix`` consumes.

	``tasks`` is the project's task mapping, which the ``tasks`` shape resolves child bindings
	against; the ``axes`` and ``variants`` shapes don't need it. A fan-out with no faithful
	projection raises ``ValueError`` (see :func:`resolve_emission`).

	>>> from camas import Parallel, Task
	>>> to_matrix_object(Parallel(Task("test"), matrix={"PY": ("3.12", "3.13")}))
	{'PY': ['3.12', '3.13']}
	>>> to_matrix_object(Parallel(Task("t"), matrix={"PY": ("3.13",), "PROFILE": ("release",)}))
	{'PY': ['3.13'], 'PROFILE': ['release']}
	>>> to_matrix_object(Parallel(Task("t {b}"), variants=({"b": "x"}, {"b": "y"})))
	{'include': [{'b': 'x'}, {'b': 'y'}]}
	>>> build, lint = Task("make"), Task("ruff check .")
	>>> to_matrix_object(Parallel(build, lint), {"build": build, "lint": lint})
	{'task': ['build', 'lint']}
	>>> to_matrix_object(Task("hi"))
	Traceback (most recent call last):
	    ...
	ValueError: task has no matrix axes to emit as a GitHub Actions job matrix, and it is a single leaf — there are no children to fan out as one job each
	>>> to_matrix_object(Parallel(Task("t"), matrix={"PY": ()}))
	Traceback (most recent call last):
	    ...
	ValueError: matrix axis 'PY' has no values
	>>> to_matrix_object(Parallel(
	...     Parallel(Task("t"), matrix={"PROFILE": ("release",), "PY": ("3.13",)}),
	...     Parallel(Task("t"), matrix={"PROFILE": ("debug",), "PY": ("3.12", "3.13")}),
	... ))
	Traceback (most recent call last):
	    ...
	ValueError: matrix is not a clean cross-product; object-of-arrays cannot represent a heterogeneous fan-out (independent fan-outs in one tree, or nested matrices that disagree) — declare the coupled cells with variants= to emit them as include:
	>>> to_matrix_object(Parallel(
	...     Parallel(Task("echo {X}"), matrix={"X": ("a", "b")}, name="matrixed"),
	...     Task("echo plain", name="plain"),
	... ))
	Traceback (most recent call last):
	    ...
	ValueError: matrix does not cover every leaf (plain): a leaf that runs under no matrix axis cannot be represented in a GitHub Actions matrix — mixing matrixed and plain leaves under one --github-matrix task is unsupported
	"""
	return to_json_object(resolve_emission(task, tasks, shape))


def format_matrix_json(matrix: Mapping[str, Any], *, pretty: bool) -> str:
	"""Serialize the matrix object to JSON.

	Compact (no spaces, single line) when ``pretty`` is False — the canonical shape for
	``echo "matrix=$(...)" >> $GITHUB_OUTPUT``. Indented two spaces when ``pretty`` is True —
	readable preview for interactive use.

	>>> format_matrix_json({"PY": ["3.12", "3.13"]}, pretty=False)
	'{"PY":["3.12","3.13"]}'
	>>> print(format_matrix_json({"PY": ["3.12"]}, pretty=True))
	{
	  "PY": [
	    "3.12"
	  ]
	}
	"""
	if pretty:
		return json.dumps(matrix, indent=2)
	return json.dumps(matrix, separators=(",", ":"))


def emit(
	task: TaskNode,
	tasks: Mapping[str, TaskNode] = NO_TASKS,
	shape: ShapeName = "auto",
	*,
	pretty: bool,
) -> str:
	"""Compose :func:`to_matrix_object` and :func:`format_matrix_json`.

	>>> from camas import Parallel, Task
	>>> emit(Parallel(Task("t"), matrix={"PY": ("3.12",)}), pretty=False)
	'{"PY":["3.12"]}'
	>>> emit(Parallel(Task("t {b}"), variants=({"b": "x"},)), pretty=False)
	'{"include":[{"b":"x"}]}'
	"""
	return format_matrix_json(to_matrix_object(task, tasks, shape), pretty=pretty)
