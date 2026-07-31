# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Whether git will commit what ``camas mcp init`` just wrote, and the ``.gitignore`` lines that
would make it: a generated file a repo silently excludes is not a shared one, and the exclude rule
decides which re-inclusion actually works.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
	from collections.abc import Mapping, Sequence


class Excluded(NamedTuple):
	"""A generated path git excludes, and the exclude rule responsible for it."""

	path: str
	source: str
	"""The file holding the rule: a per-directory ``.gitignore``, ``.git/info/exclude``, or the
	absolute ``core.excludesFile`` path. Which one decides both who is affected — the repo or only
	this clone — and which re-inclusion actually works, since a ``.gitignore`` outranks the other
	two."""
	line: str
	pattern: str


def parse_check_ignore(stdout: str) -> tuple[Excluded, ...]:
	r"""The excluded paths in ``git check-ignore -v -z --stdin`` output: NUL-delimited
	``source, line, pattern, path`` records, an unterminated trailing one dropped — the field
	slices are uneven by construction, since git's output ends with a NUL, which is why the ``zip``
	is not ``strict``. ``-v`` also reports a path matched by a negated pattern, which means it is
	*not* excluded, so those go too.

	>>> parse_check_ignore("\x00".join((".gitignore", "12", ".claude/", ".claude", "")))
	(Excluded(path='.claude', source='.gitignore', line='12', pattern='.claude/'),)
	>>> parse_check_ignore("\x00".join((".gitignore", "2", "!.mcp.json", ".mcp.json", "")))
	()
	>>> parse_check_ignore("")
	()
	"""
	fields = stdout.split("\0")
	return tuple(
		Excluded(path=path, source=source, line=line, pattern=pattern)
		for source, line, pattern, path in zip(
			fields[0::4], fields[1::4], fields[2::4], fields[3::4], strict=False
		)
		if not pattern.startswith("!")
	)


def check_ignore(paths: Sequence[str]) -> tuple[Excluded, ...]:
	"""Which of ``paths`` — repo-relative, forward-slashed, the spelling git itself speaks — git
	excludes, and why. Empty when git is not on PATH, this is not a repository, or nothing is
	excluded. A tracked path is never reported: ``check-ignore`` consults the index, so a force-added
	file already counts as committable. A directory-only pattern (``.claude/``) matches a directory
	only once it exists, which is why the check runs after the files are written rather than before.

	Output is decoded leniently because none of the matching depends on those bytes: git echoes each
	pathname back exactly as given, and camas only ever asks about its own ASCII literals. A
	non-UTF-8 ``.gitignore`` pattern — or an excludes file under a non-UTF-8 directory — would
	otherwise take down an advisory warning over characters that only ever get printed.
	"""
	git = shutil.which("git")
	if git is None:
		return ()
	try:
		proc = subprocess.run(
			[git, "check-ignore", "-v", "-z", "--stdin"],
			input="".join(f"{path}\0" for path in paths),
			capture_output=True,
			text=True,
			encoding="utf-8",
			errors="replace",
			check=False,
		)
	except OSError:
		return ()
	return parse_check_ignore(proc.stdout)


def ancestry(path: str) -> tuple[str, ...]:
	"""``path`` and every directory above it, outermost first — the chain along which a single
	exclude rule anywhere is enough to make ``path`` uncommittable.

	>>> ancestry(".claude/agents/camas-test-fixer.md")
	('.claude', '.claude/agents', '.claude/agents/camas-test-fixer.md')
	>>> ancestry(".mcp.json")
	('.mcp.json',)
	"""
	parts = PurePosixPath(path).parts
	return tuple("/".join(parts[: depth + 1]) for depth in range(len(parts)))


def outermost(path: str, found: Mapping[str, Excluded]) -> Excluded | None:
	"""The shallowest exclusion on ``path``'s chain — the one to report and to undo, since
	re-including anything below it is what git refuses. ``None`` when nothing on the chain is
	excluded.

	>>> claude = Excluded(".claude", ".gitignore", "1", ".claude/")
	>>> outermost(".claude/agents/x.md", {".claude": claude, ".claude/agents": claude})
	Excluded(path='.claude', source='.gitignore', line='1', pattern='.claude/')
	>>> outermost(".mcp.json", {".claude": claude}) is None
	True
	"""
	return next((found[step] for step in ancestry(path) if step in found), None)


def excluded_roots(paths: Sequence[str]) -> tuple[Excluded, ...]:
	"""The outermost excluded path behind each of ``paths``, deduplicated — one entry for an
	excluded ``.claude`` rather than one per generated file underneath it.
	"""
	chains = tuple(dict.fromkeys(step for path in paths for step in ancestry(path)))
	found = {excluded.path: excluded for excluded in check_ignore(chains)}
	return tuple(dict.fromkeys(filter(None, (outermost(path, found) for path in paths))))


def child(root: str, path: str) -> str:
	"""The step just below ``root`` on the way to ``path``, with a trailing ``/`` when it is a
	directory — the granularity a ``!`` line has to name to re-include ``path``.

	>>> child(".claude", ".claude/agents/camas-test-fixer.md")
	'.claude/agents/'
	>>> child(".claude", ".claude/settings.json")
	'.claude/settings.json'
	"""
	depth = len(PurePosixPath(root).parts) + 1
	parts = PurePosixPath(path).parts
	return "/".join(parts[:depth]) + ("/" if depth < len(parts) else "")


class Unignore(NamedTuple):
	"""How to make one excluded root committable: what has to change, and the ``.gitignore`` lines
	that do it.
	"""

	advice: str
	lines: tuple[str, ...]


def in_repo_gitignore(source: str) -> bool:
	r"""Whether the excluding rule lives in a per-directory ``.gitignore`` — a file the repo commits,
	and the highest-precedence source, so the fix is edited in place. The alternatives are local to
	one clone and are outranked by any ``.gitignore``: ``.git/info/exclude``, and a
	``core.excludesFile``, which git reports as an absolute path. Absoluteness is tested in both
	path flavors rather than the host's: a leading ``/`` is absolute to git and to POSIX but only
	drive-relative to Windows, so ``Path`` alone reads a POSIX excludes file as repo-relative when
	camas runs on Windows.

	>>> in_repo_gitignore(".gitignore"), in_repo_gitignore("sub/dir/.gitignore")
	(True, True)
	>>> in_repo_gitignore(".git/info/exclude"), in_repo_gitignore("/home/dev/.gitignore")
	(False, False)
	>>> in_repo_gitignore("C:/Users/dev/.gitignore"), in_repo_gitignore(r"C:\Users\dev\.gitignore")
	(False, False)
	"""
	return PurePosixPath(source).name == ".gitignore" and not (
		PurePosixPath(source).is_absolute() or PureWindowsPath(source).is_absolute()
	)


def unignore(root: Excluded, paths: Sequence[str]) -> Unignore:
	"""How to re-include the ``paths`` that ``root`` excludes: a generated file that is itself
	excluded only needs the rule negated, while an excluded *directory* has to stop being excluded
	first — git never descends into it, so no ``!`` line underneath can bring anything back. A rule
	from outside a ``.gitignore`` cannot be edited into shape at all — it is local to the clone, and
	the ``.gitignore`` that outranks it has to re-include the directory itself before narrowing back
	down, or nothing under it is reachable.

	>>> unignore(Excluded(".mcp.json", ".gitignore", "7", "*.json"), (".mcp.json",)).lines
	('!.mcp.json',)
	>>> unignore(
	...     Excluded(".claude", ".gitignore", "12", ".claude/"),
	...     (".claude/settings.json", ".claude/agents/camas-test-fixer.md"),
	... ).lines
	('.claude/*', '!.claude/settings.json', '!.claude/agents/')
	>>> unignore(
	...     Excluded(".claude", ".git/info/exclude", "1", ".claude/"),
	...     (".claude/settings.json",),
	... ).lines
	('!.claude/', '.claude/*', '!.claude/settings.json')
	"""
	if root.path in paths:
		return Unignore(
			advice=(
				"drop that rule, or negate it below"
				if in_repo_gitignore(root.source)
				else "that file is local to this clone, so negate it in a .gitignore, which "
				"outranks it"
			),
			lines=(f"!{root.path}",),
		)
	narrowed = (
		f"{root.path}/*",
		*dict.fromkeys(
			f"!{child(root.path, path)}" for path in paths if path.startswith(f"{root.path}/")
		),
	)
	if in_repo_gitignore(root.source):
		return Unignore(
			advice=(
				"git cannot re-include a path whose parent directory is excluded, so that rule has "
				f"to stop matching {root.path} itself, e.g."
			),
			lines=narrowed,
		)
	return Unignore(
		advice=(
			"that file is local to this clone, so re-include "
			f"{root.path} in a .gitignore, which outranks it, then narrow back down"
		),
		lines=(f"!{root.path}/", *narrowed),
	)


def _root_report(root: Excluded, paths: Sequence[str]) -> tuple[str, ...]:
	"""One excluded root's lines of the warning: the rule that excludes it, then the fix."""
	fix = unignore(root, paths)
	return (
		f"  {root.path} is excluded by {root.source}:{root.line} (`{root.pattern}`) — {fix.advice}:",
		*(f"      {line}" for line in fix.lines),
	)


def gitignore_warning(paths: Sequence[str], *, consequence: str) -> str | None:
	"""The warning for generated files git will not commit, or ``None`` when every path is
	committable — or when git cannot say, since a repo-less or git-less checkout has no sharing
	story to break.
	"""
	roots = excluded_roots(paths)
	if not roots:
		return None
	return "\n".join(
		(
			f"warning: git will not commit what camas just wrote — {consequence}.",
			*(line for root in roots for line in _root_report(root, paths)),
		)
	)


def warn_uncommittable(paths: Sequence[str], *, consequence: str) -> None:
	"""Report to stderr any of ``paths`` git excludes, so a silently-uncommittable artifact does not
	pass for a shared one. Flushes stdout first, so the warning still lands under the report it
	qualifies when the two streams are captured together and stdout is block-buffered.

	Neither write may raise, whatever is wrong with the stream: this runs after the files are
	already on disk, so failing here turns a completed init into an exception over an advisory
	message. Whole categories are suppressed on purpose, not the two failures that happen to have
	been seen — ``OSError`` for a reader that walked away (``BrokenPipeError``, from
	``camas mcp init | head``, or ``2>&1 |`` with both streams on the one pipe) and any other
	OS-level write failure, ``ValueError`` for a stream someone closed outright. A report on
	finished work has nothing useful to do with any of them. The two guards stay separate because a
	dead stdout must not cost the warning a stderr that is still being read.
	"""
	warning = gitignore_warning(paths, consequence=consequence)
	if warning is not None:
		with suppress(OSError, ValueError):
			sys.stdout.flush()
		with suppress(OSError, ValueError):
			print(warning, file=sys.stderr)
