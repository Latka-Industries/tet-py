# Changelog

## [Unreleased]

## [0.3.2] - 2026-10-17

### Changed

- Pin tetration **0.3.2** from crates.io (cross-host `publishing/<G>` claim) and align package version (**0.3.2**) ([THI-514](https://linear.app/thicclatka/issue/THI-514))

## [0.3.1] - 2026-10-17

### Added

- `S3Store.connect` + ingest / publish / pin open against an S3-compatible endpoint ([THI-506](https://linear.app/thicclatka/issue/THI-506))
- Store ingest / publish / seal / CAS helpers on `MemoryStore` (same path as S3)

### Changed

- Pin tetration **0.3.1** from crates.io (`tetration-s3` feature) and align package version (**0.3.1**)

## [0.3.0] - 2026-10-17

### Added

- Store URI open + generation pin: `tet.open` accepts `file://` / `?generation=N`; package opens pin `CURRENT` once (`TetFile.generation`) ([THI-504](https://linear.app/thicclatka/issue/THI-504))
- `MemoryStore.from_package` + `memory://` open for in-memory pin tests (no S3)

### Changed

- Pin tetration **0.3.0** from crates.io and align package version (**0.3.0**)

## [0.2.3] - 2026-10-16

### Added

- `tet.export` / `TetFile.export`: write HDF5, NetCDF, or Zarr via optional extras (`tet-py[h5]`, `[netcdf]`, `[zarr]`, or `[export]`) ([THI-494](https://linear.app/thicclatka/issue/THI-494))

### Changed

- Pin tetration **0.2.3** from crates.io and align package version (**0.2.3**)

## [0.2.2] - 2026-10-16

### Added

- `tet.convert` multi-array `.npz`, core `.csv`/`.tsv`, and optional HDF5 / NetCDF / Zarr / Parquet loaders ([THI-484](https://linear.app/thicclatka/issue/THI-484))
- `tet.open` on `.tet.d` packages plus `tet.ingest` / `tet.publish` ([THI-482](https://linear.app/thicclatka/issue/THI-482))
- Zero-copy `read_numpy` for full-width multi-chunk raw slabs ([THI-483](https://linear.app/thicclatka/issue/THI-483))

### Changed

- Pin tetration **0.2.2** from crates.io and align package version (**0.2.2**)

## [0.2.1] - 2026-10-15

### Changed

- Pin tetration **0.2.1** from crates.io (package verify hardening + package fuzz; no Python API change)
- Align package version with tetration (**0.2.1**)

## [0.2.0] - 2026-10-15

### Added

- Zero-copy `read_numpy`: a single raw (uncompressed) chunk whose payload equals the logical selection returns a read-only `numpy.memmap`; multi-chunk / zstd / partial-tile / strided plans still copy ([THI-21](https://linear.app/thicclatka/issue/THI-21) / [#11](https://github.com/Latka-Industries/tet-py/issues/11))

### Changed

- Pin tetration **0.2.0** from crates.io (writer protocol v2 packages + prior materialize work)
- Align package version with tetration (**0.2.0**)

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
