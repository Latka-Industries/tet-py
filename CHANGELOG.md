# Changelog

## [Unreleased]

### Planned

- Full `tet.convert` extras implementations ([THI-22](https://linear.app/thicclatka/issue/THI-22))
- Zero-copy mmap → NumPy ([THI-21](https://linear.app/thicclatka/issue/THI-21))

## [0.1.2] - 2026-10-15

### Added

- `memory_budget_bytes=` / `memory_budget_percent=` on `execute`, `query_execute`, reductions, and transform sinks (THI-454)
- Query strings accept TOML as well as JSON (CLI Auto parity via `parse_query_text`) (THI-456)
- `tet.convert` scaffolding: `.npy` / single-array `.npz` → `.tet`; optional extras stubs for h5/netcdf/zarr/csv/parquet (THI-22)
- `write_dataset` / `TetWriter` accept layout-v1 integer and `float16` NumPy dtypes via `TetDatasetWrite::row_major` (THI-26)

### Changed

- CI runs `mypy python/tet` after `maturin develop`; workflow triggers on `dev` as well as `main` (THI-453)
- Pin tetration **0.1.12** from crates.io (THI-457): `read_numpy` memory-budget preflight (THI-23), DenseBuffer **f16/u32/u64** → NumPy (THI-15), writer uses `f64_row_major`
- CI `paths-ignore` for docs / markdown so documentation-only pushes skip the test matrix

## [0.1.1] - 2026-06-12

### Added

- `preview=N` on `query_execute`, `execute`, and reduction helpers (`mean`, `sum`, …); `QueryResult.preview` (`numpy.ndarray`, 1-D) for capped `execution.*_preview` samples (parity with `tet query --preview`) ([#7](https://github.com/Latka-Industries/tet-py/issues/7) / THI-24)

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
- Rust edition **2024**, MSRV **1.95**, PyO3 **0.28** (abi3-py311)

### Notes

- `tet.core_version()` reports the linked tetration crate (e.g. `0.1.9`); `tet.__version__` is the tet-py release.
- GPU / native HDF5-NetCDF convert codecs are not in the wheel (`default-features = false` on the tetration dependency).
