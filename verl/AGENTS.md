# Repository Guidelines

## Project Structure & Module Organization

- `verl/`: core library (trainers, workers, rollout backends, utilities).
- `tests/`: `pytest` suites (many are CPU-friendly; some require GPU/cluster features).
- `examples/`: runnable training scripts and configs (PPO/GRPO/SFT, etc.).
- `docs/`: Sphinx/ReadTheDocs sources and build artifacts.
- `scripts/`: developer utilities and pre-commit hook entrypoints (e.g. config generation).
- `recipe/`: git submodule with end-to-end recipes; initialize with `git submodule update --init --recursive recipe`.

## Build, Test, and Development Commands

- Install for local development (Python ≥3.10): `pip install -e ".[test]"`.
  - Backend-specific extras: `pip install -e ".[test,vllm]"` or `pip install -e ".[test,sglang]"`.
- Run unit tests: `pytest -q` (target a subset with `pytest tests/single_controller -q`).
- Run lint/format/type checks: `pre-commit run --all-files --show-diff-on-failure --color=always`.
- Build docs:
  - `cd docs && pip install -r requirements-docs.txt`
  - `make html && python -m http.server -d _build/html/`

## Coding Style & Naming Conventions

- Formatting/linting is enforced by `pre-commit` (`ruff`, `ruff-format`, `mypy`).
- Line length: 120 (see `pyproject.toml`); use 4-space indentation.
- Naming: `snake_case` for functions/files, `PascalCase` for classes, `UPPER_CASE` for constants.
- Do not hand-edit generated trainer configs: `verl/trainer/config/_generated_*.yaml` is produced by `scripts/generate_trainer_config.sh` (also run by the `autogen-trainer-cfg` hook).

## Testing Guidelines

- Use `pytest`; add tests under `tests/` and name files `test_*.py`.
- Prefer fast, deterministic tests; if CI can’t cover a change (e.g., new model/algorithm), include experiment evidence in the PR.
- When adding a new feature, consider updating the relevant `.github/workflows/*.yml` `paths` filters so CI exercises it.

## Commit & Pull Request Guidelines

- PR title must match: `[{modules}] {type}: {description}` (example: `[fsdp, megatron] feat: dynamic batching`); use `[BREAKING]` when changing APIs.
- Keep commits focused, run `pre-commit`, and update docs for user-facing changes.
- Fill out `.github/PULL_REQUEST_TEMPLATE.md`: link issues/PRs, describe tests/results, and include usage examples when APIs change.
