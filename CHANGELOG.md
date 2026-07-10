# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Relicensed the project from AGPL-3.0 to the MIT License.
- Restructured the repository to a `src/` layout: the package now lives in
  `src/duq/` and the test suite in `tests/`.
- Modernized the development toolchain:
  - a single root `pyproject.toml` that serves as both the packaging manifest
    (PEP 621/639, [hatchling](https://hatch.pypa.io) build backend) and the
    [pixi](https://pixi.sh) workspace manifest;
  - [Ruff](https://docs.astral.sh/ruff/) for linting and formatting;
  - [mypy](https://mypy-lang.org/) for static type checking;
  - a consolidated GitHub Actions CI workflow driven by pixi;
  - a `.pre-commit-config.yaml` with Ruff and standard hygiene hooks.
