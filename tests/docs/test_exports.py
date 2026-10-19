# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 JP Hutchins

"""The `exports <module>` directive: the module's `import X as X` re-exports, in source order."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable


def test_re_exports_render_as_one_import_including_type_checking_ones(
	generated: Callable[[str], tuple[str, ...]],
) -> None:
	assert generated("exports demo.api") == (
		"```python",
		"from demo.api import (",
		"    Options,",
		"    Thing,",
		"    bare,",
		")",
		"```",
	)
