# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

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
	path: str
	source: str
	line: str
	pattern: str


def parse_check_ignore(stdout: str) -> tuple[Excluded, ...]:
	r"""
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
	"""
	>>> ancestry(".claude/agents/camas-test-fixer.md")
	('.claude', '.claude/agents', '.claude/agents/camas-test-fixer.md')
	>>> ancestry(".mcp.json")
	('.mcp.json',)
	"""
	parts = PurePosixPath(path).parts
	return tuple("/".join(parts[: depth + 1]) for depth in range(len(parts)))


def outermost(path: str, found: Mapping[str, Excluded]) -> Excluded | None:
	"""
	>>> claude = Excluded(".claude", ".gitignore", "1", ".claude/")
	>>> outermost(".claude/agents/x.md", {".claude": claude, ".claude/agents": claude})
	Excluded(path='.claude', source='.gitignore', line='1', pattern='.claude/')
	>>> outermost(".mcp.json", {".claude": claude}) is None
	True
	"""
	return next((found[step] for step in ancestry(path) if step in found), None)


def excluded_roots(paths: Sequence[str]) -> tuple[Excluded, ...]:
	chains = tuple(dict.fromkeys(step for path in paths for step in ancestry(path)))
	found = {excluded.path: excluded for excluded in check_ignore(chains)}
	return tuple(dict.fromkeys(filter(None, (outermost(path, found) for path in paths))))


def child(root: str, path: str) -> str:
	"""
	>>> child(".claude", ".claude/agents/camas-test-fixer.md")
	'.claude/agents/'
	>>> child(".claude", ".claude/settings.json")
	'.claude/settings.json'
	"""
	depth = len(PurePosixPath(root).parts) + 1
	parts = PurePosixPath(path).parts
	return "/".join(parts[:depth]) + ("/" if depth < len(parts) else "")


class Unignore(NamedTuple):
	advice: str
	lines: tuple[str, ...]


def in_repo_gitignore(source: str) -> bool:
	r"""
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
	"""
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
	fix = unignore(root, paths)
	return (
		f"  {root.path} is excluded by {root.source}:{root.line} (`{root.pattern}`) — {fix.advice}:",
		*(f"      {line}" for line in fix.lines),
	)


def gitignore_warning(paths: Sequence[str], *, consequence: str) -> str | None:
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
	warning = gitignore_warning(paths, consequence=consequence)
	if warning is not None:
		with suppress(OSError, ValueError):
			sys.stdout.flush()
		with suppress(OSError, ValueError):
			print(warning, file=sys.stderr)
