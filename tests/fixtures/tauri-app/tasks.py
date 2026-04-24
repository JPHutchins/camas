"""camas task definitions for the tauri-app fixture."""

from pathlib import Path

from camas import Parallel, Sequential, Task

src_tauri = Path("src-tauri")
bin = Path("node_modules/.bin")

check = Parallel(
	tasks=(
		Task(f"{bin}/prettier --check ."),
		Task(f"{bin}/tsc --noEmit"),
		Task(f"{bin}/eslint src/"),
		Task(f"{bin}/vitest run"),
		Task("cargo fmt --all -- --check", cwd=src_tauri),
		Task("cargo clippy --all-targets --locked -- -D warnings", cwd=src_tauri),
		Task("cargo test --all-targets --locked", cwd=src_tauri),
	),
)

format = Parallel(
	tasks=(
		Task(f"{bin}/prettier --write ."),
		Task("cargo fmt --all", cwd=src_tauri),
	),
)

all = Sequential(tasks=(format, check))

build = Parallel(
	tasks=(Task("npm run tauri build {FLAG}"),),
	matrix={"FLAG": ("-- --debug", "")},
)
