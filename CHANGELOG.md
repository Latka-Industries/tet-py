# Changelog

## [Unreleased]

### Added

- `preview=N` on `query_execute`, `execute`, and reduction helpers (`mean`, `sum`, …); `QueryResult.preview` / `preview_ndarray()` for capped `execution.*_preview` samples (parity with `tet query --preview`) ([#7](https://github.com/Latka-Industries/tet-py/issues/7))

### Planned

- `read_numpy` memory budget preflight ([#9](https://github.com/Latka-Industries/tet-py/issues/9))
- Integer `write_dataset` dtypes ([#8](https://github.com/Latka-Industries/tet-py/issues/8))
- `tet.convert` optional extras ([#10](https://github.com/Latka-Industries/tet-py/issues/10))
- Zero-copy mmap → NumPy ([#11](https://github.com/Latka-Industries/tet-py/issues/11))

## [0.1.0] - 2026-06-12

First public release on [PyPI](https://pypi.org/project/tet-py/0.1.0/). Links **tetration 0.1.9** from [crates.io](https://crates.io/crates/tetration/0.1.9).

### Added

- **Read / query:** `tet.open`, catalog (`Dataset`, `f["name"]`), reductions (`mean`, `sum`, `quantile`, `histogram`, …), `QueryResult`, `build_query`, `axis_slice`, `selection_slices`, typed `TetError` / `CatalogError`
- **NumPy read — ram:** `read_numpy`, `Dataset.to_numpy`, `transform.to_numpy.*`
- **NumPy read — spill:** `read_spill`, `transform.to_spill.*`, `SpillReadResult.to_numpy`, `SpillTransformResult.to_numpy`
- **NumPy read — sidecar:** `transform.to_sidecar.*`, `SidecarTransformResult.to_numpy` / `.open`
- **NumPy write:** `TetWriter`, `write_dataset` (`float32` / `float64`)
- **Packaging:** abi3 wheels (`cp311+`) for Linux (x86_64, aarch64), macOS universal2, Windows; `numpy>=2.0` dependency
- **CI:** GitHub Actions on Linux, macOS, Windows; vendored fixtures in `tests/fixtures/`
- **Publish workflow:** tag `v*` → maturin build → PyPI upload

### Changed

- Package layout: `python/tet/` subpackages (`_core`, `_query`, `_io`, `_transform`); public `import tet` API unchanged

### Notes

- Do not `pip install tetration` — unrelated PyPI package. Use **`tet-py`** / **`import tet`**.
- `tet.core_version()` reports the linked tetration crate (e.g. `0.1.9`); `tet.__version__` is the tet-py release.

[0.1.0]: https://github.com/Latka-Industries/tet-py/releases/tag/v0.1.0
