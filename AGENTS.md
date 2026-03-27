# AGENTS.md

This file defines required behavior for coding agents working in this repository.
These instructions apply to the entire repo tree.

## 1) Branch and Change Scope

- Follow the repository's current GitFlow-style branch roles:
  - `main`: stable history.
  - `dev`: integration branch for upcoming work.
  - `feature/*`: branch from `dev`, merge back into `dev`.
  - `release/*`: branch from `dev`, merge into `main`.
  - `hotfix/*`: branch from `main`, merge into `main`.
- Never mix unrelated changes in one branch, pull request, or commit when they can be separated cleanly.
- Keep commits scoped to one logical change whenever possible.
- Prefer pull requests over direct pushes to long-lived branches.

## 1.1) Commit Messages

- Use semantic commit messages when creating commits, for example:
  - `feat: ...`
  - `fix: ...`
  - `docs: ...`
  - `refactor: ...`
  - `test: ...`
  - `chore: ...`

## 1.2) Versioning

- Follow Semantic Versioning (`MAJOR.MINOR.PATCH`) for release versions.
- Use:
  - `MAJOR` for breaking API or behavior changes
  - `MINOR` for backward-compatible feature additions
  - `PATCH` for backward-compatible bug fixes or documentation-only maintenance tied to a release

## 2) Repository State

- The public package name is `statesmoother`.
- Source code lives under `src/state_smoother/`.
- Tests live under `tests/`.
- The demonstration notebook is [`demonstration.ipynb`](./demonstration.ipynb).
- `uv.lock` is developer-local and should not be tracked in git for this repository.
- A pytest GitHub Actions workflow exists at [`.github/workflows/pytest.yaml`](./.github/workflows/pytest.yaml).
- There is no documented package publishing workflow in this repository at this time. Do not claim one exists unless it is added.

## 3) Documentation and README Alignment

- Keep [`README.md`](./README.md) aligned with the code that actually exists in this repository.
- Do not document features, package indexes, badges, workflows, or APIs that are not implemented here.
- When adding or changing public behavior, update README and tests in the same change when appropriate.
- Keep examples runnable against the current package layout:
  - `from state_smoother import Smoother`

## 4) Testing and Verification

- Prefer `uv run pytest -q` for the test suite.
- If a change affects library behavior, add or update pytest coverage when practical.
- If tests cannot be run in the current environment, state that clearly in the final summary.

## 5) Safety and Transparency

- Before making changes, state what you intend to change.
- After making changes, summarize exactly what changed and where.
- If a requested action conflicts with repository policy or the current repository state, call out the conflict explicitly.

## 6) AI Usage Guidance

- AI tooling may assist with documentation, tests, refactors, and development workflow support.
- Do not invent implementation details, release processes, or support guarantees that are not present in the repository.
- Functional correctness remains more important than stylistic cleanup.
