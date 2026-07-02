# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""Sync the plugin.json version from the package version (SSOT: git tags via setuptools-scm)."""

from __future__ import annotations

import json
from pathlib import Path

from setuptools_scm import get_version

PLUGIN_JSON = Path(__file__).resolve().parent / ".claude-plugin" / "plugin.json"


def main() -> None:
	version = get_version()
	manifest = json.loads(PLUGIN_JSON.read_text(encoding="utf-8"))
	manifest["version"] = version
	PLUGIN_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
	print(f"plugin.json -> {version}")


if __name__ == "__main__":
	main()
