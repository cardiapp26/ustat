# Project Setup — Local Reference

> This project follows the centralized dependency stores configured on this
> machine. Full rationale lives in `~/.zcode/AGENTS.md` ("Project Setup" section).
> This file is the per-project reminder.

## Toolchain (use these, not their legacy equivalents)

| Language       | Use            | Never             | Why                                                        |
|----------------|----------------|-------------------|------------------------------------------------------------|
| JavaScript/TS  | `pnpm`         | `npm`, `yarn`     | Global store hardlinks; per-project `node_modules` is near-free |
| Python         | `uv`           | `pip` (directly)  | APFS clone shares bytes between venvs and cache            |
| Rust / Tauri   | `cargo`/`tauri`| (nothing changes) | Shared `~/.cargo-target` via `~/.cargo/config.toml`        |

## Common commands

```bash
# JS — install (uses global store, hardlinks to packages)
pnpm install
pnpm add <pkg>
pnpm remove <pkg>

# Python — create / update venv
uv sync
uv add <pkg>
uv pip install <pkg>      # drop-in pip inside the active venv

# Rust — normal; target is already redirected to the shared dir
cargo build
cargo test
```

## If you (the agent) are scaffolding a new sub-project here

- Initialize JS with `pnpm init`, not `npm init`.
- Initialize Python with `uv init`, not `python -m venv`.
- Do not run `npm install` or `pip install` as a fallback — they duplicate bytes that pnpm/uv already share.

## Migration notes (if this project was converted from npm/pip)

- The original npm lockfile is backed up as `package-lock.json.npm-bak`.
- `pnpm-lock.yaml` is the source of truth now.
