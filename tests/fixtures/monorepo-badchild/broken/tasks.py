"""A camas tasks file that raises during evaluation."""

import camas

raise RuntimeError(f"boom in broken child (camas {camas.__name__})")
