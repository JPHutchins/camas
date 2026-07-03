# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Parse the camas requirement from a PEP 723 inline-script-metadata block in a ``tasks.py``."""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
	from pathlib import Path

if sys.version_info >= (3, 11):
	import tomllib
else:  # pragma: no cover
	import tomli as tomllib

_CAMAS_REQ_RE: Final = re.compile(
	r"^(?i:camas)"
	r"(?:\[(?P<extras>[^\]]*)\])?"
	r"(?P<spec>(?:~=|==|!=|<=|>=|<|>|===).*)?$"
)
"""A PEP 508 requirement string whose distribution name is exactly ``camas``.

The regex anchors on ``camas`` (case-insensitive), then optional ``[extras]``,
then an optional PEP 440 version specifier. It rejects names like ``camas-foo``
or ``my-camas``: those would leave trailing chars that fail to match the
version-op guard or ``$``.
"""


def parse_camas_requirement(source: str) -> str | None:
	"""The camas PEP 508 requirement string from a PEP 723 ``# /// script`` block in ``source``, or None.

	>>> parse_camas_requirement('''# /// script
	... # dependencies = ["camas>=0.1.8"]
	... # ///''')
	'camas>=0.1.8'

	>>> parse_camas_requirement('''# /// script
	... # dependencies = ["other-pkg"]
	... # ///''') is None
	True

	>>> parse_camas_requirement('no block here') is None
	True

	>>> parse_camas_requirement('''# /// script
	... # dependencies = ["camas[mcp]==0.1.18", "pytest"]
	... # ///''')
	'camas[mcp]==0.1.18'

	>>> parse_camas_requirement('''# /// script
	... # dependencies = ["camas"]
	... # ///''')
	'camas'
	"""
	in_block = False
	toml_lines: list[str] = []

	for line in source.splitlines():
		stripped = line.rstrip("\n\r")
		if not in_block:
			if stripped.strip() == "# /// script":
				in_block = True
			continue
		if stripped.strip() == "# ///":
			break
		if stripped.startswith("# "):
			toml_lines.append(stripped[2:])
		elif stripped.startswith("#"):
			toml_lines.append(stripped[1:])

	if not toml_lines:
		return None

	try:
		data: dict[str, object] = tomllib.loads("\n".join(toml_lines))
	except tomllib.TOMLDecodeError:
		return None

	deps = data.get("dependencies")
	if not isinstance(deps, list):
		return None

	for dep in cast("list[object]", deps):
		if isinstance(dep, str) and _CAMAS_REQ_RE.match(dep):
			return dep

	return None


def camas_requirement_from(path: Path) -> str | None:
	"""Read ``path``, return its camas requirement (or None)."""
	try:
		source = path.read_text(encoding="utf-8")
	except OSError:
		return None
	return parse_camas_requirement(source)


def with_mcp_extra(req: str) -> str:
	"""``req`` with ``mcp`` ensured in its extras.

	>>> with_mcp_extra('camas>=0.1.8')
	'camas[mcp]>=0.1.8'

	>>> with_mcp_extra('camas[mcp]==1')
	'camas[mcp]==1'

	>>> with_mcp_extra('camas[test]==1')
	'camas[test,mcp]==1'

	>>> with_mcp_extra('camas')
	'camas[mcp]'

	>>> with_mcp_extra('camas[check,test]>=0.1')
	'camas[check,test,mcp]>=0.1'

	Raises:
	    ValueError: Not a camas requirement.
	"""
	m = _CAMAS_REQ_RE.match(req)
	if m is None:
		raise ValueError(f"Not a camas requirement: {req!r}")
	extras_str = m.group("extras")
	spec = m.group("spec") or ""
	if extras_str is None:
		return f"camas[mcp]{spec}"
	extras = [e.strip() for e in extras_str.split(",")]
	if "mcp" not in extras:
		extras.append("mcp")
	return f"camas[{','.join(extras)}]{spec}"


def version_specifier(req: str) -> str | None:
	"""The trailing PEP 440 specifier from ``req``, or None if unpinned.

	>>> version_specifier('camas[mcp]>=0.1.8')
	'>=0.1.8'

	>>> version_specifier('camas[mcp]==0.1.18')
	'==0.1.18'

	>>> version_specifier('camas') is None
	True
	>>> version_specifier('camas[test]') is None
	True

	Raises:
	    ValueError: Not a camas requirement.
	"""
	m = _CAMAS_REQ_RE.match(req)
	if m is None:
		raise ValueError(f"Not a camas requirement: {req!r}")
	spec = m.group("spec")
	return spec or None
