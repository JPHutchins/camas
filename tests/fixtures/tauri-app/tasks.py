"""camas task definitions for the tauri-app fixture."""

from pathlib import Path

from camas import Parallel, Sequential, Task

src_tauri = Path("src-tauri")
python_sdk = Path("python-sdk")
node = Path("node_modules/.bin")

ts = Sequential(
	(
		Task(f"{node}/prettier --write ."),
		Parallel(
			(
				Task(f"{node}/eslint src/"),
				Task(f"{node}/tsc --noEmit"),
				Task(f"{node}/vitest run"),
			),
		),
	),
)

rust = Sequential(
	(
		Task("cargo fmt --all", cwd=src_tauri),
		Parallel(
			(
				Task("cargo clippy --all-targets --locked -- -D warnings", cwd=src_tauri),
				Task("cargo test --all-targets --locked", cwd=src_tauri),
			),
		),
	),
)

python = Sequential(
	(
		Task("uv run ruff check --fix .", cwd=python_sdk),
		Task("uv run ruff format .", cwd=python_sdk),
		Parallel(
			(
				Task("uv run mypy .", cwd=python_sdk),
				Task("uv run pytest", cwd=python_sdk),
			),
		),
	),
)

all = Parallel((ts, rust, python))

fix = Parallel(
	(
		Task(f"{node}/prettier --write ."),
		Task("cargo fmt --all", cwd=src_tauri),
		Sequential(
			(
				Task("uv run ruff check --fix .", cwd=python_sdk),
				Task("uv run ruff format .", cwd=python_sdk),
			),
		),
	),
)

build = Parallel(
	(Task("npm run tauri build {FLAG}"),),
	matrix={"FLAG": ("-- --debug", "")},
)
