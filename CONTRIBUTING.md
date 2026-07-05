# Camas

Contributions and bug reports are welcome!

> [!IMPORTANT]
> ### First time setup
>
> - Install [uv](https://github.com/astral-sh/uv)
> - Initialize the environment:
>   ```
>   uv sync --locked --all-extras --all-packages --dev
>   ```

## Linting & Testing

Use `uv run camas --help` to see a list of available checks.

## Releasing

Releases are cut from `main`; the checked-in `VERSION` file and the git tag must agree — CI's `version-gate` job blocks publish otherwise. Pushing the tag is what publishes to PyPI.

```
uv run camas release -- 0.1.22   # asserts clean, synced main; bumps VERSION; commits `release: 0.1.22`; tags
git push origin main 0.1.22
```

The nix flake reports the bare `VERSION` on a clean checkout (a tagged fetch included) and `VERSION+<rev>` on a dirty tree.
