# Changelog

## [Unreleased]

## [0.6.6] - 2026-10-21

### Changed

- Pin tetration **0.6.6** from crates.io and align the package version (**0.6.6**). Optional `coord_label_postings` + closed `label_filter` on the crate; package-root INDEX pushdown + OOC filter correctness; no new Python API ([THI-614](https://linear.app/thicclatka/issue/THI-614))

## [0.6.5] - 2026-10-21

### Changed

- Pin tetration **0.6.5** from crates.io and align the package version (**0.6.5**). Optional `sorted_axis_range` INDEX pushdown + fold dense-slab speedups on the crate; no new Python API ([THI-613](https://linear.app/thicclatka/issue/THI-613))

## [0.6.4] - 2026-10-21

### Changed

- Pin tetration **0.6.4** from crates.io and align the package version (**0.6.4**). Crate hygiene + broader benches + ephemeral `write: switch` spill cleanup; no new Python API ([THI-607](https://linear.app/thicclatka/issue/THI-607) / [THI-608](https://linear.app/thicclatka/issue/THI-608))

## [0.6.3] - 2026-10-21

### Changed

- Pin tetration **0.6.3** from crates.io and align the package version (**0.6.3**). Window `pad` / `step` / `align` / `min_count` + multi-dtype folds on the crate; no new Python API ([THI-606](https://linear.app/thicclatka/issue/THI-606))

## [0.6.2] - 2026-10-21

### Changed

- Pin tetration **0.6.2** from crates.io and align the package version (**0.6.2**). Label-value `group_by` (`by: labels`) on the crate; no new Python API ([THI-596](https://linear.app/thicclatka/issue/THI-596))

## [0.6.1] - 2026-10-21

### Changed

- Pin tetration **0.6.1** from crates.io and align the package version (**0.6.1**). Multi-dataset `align` checks + `window.mode` `bins`/`rolling` on the crate; no new Python API ([THI-594](https://linear.app/thicclatka/issue/THI-594))

## [0.6.0] - 2026-10-21

### Changed

- Pin tetration **0.6.0** from crates.io and align the package version (**0.6.0**). Windowed stats (`window` wire) on the crate; no new Python API ([THI-595](https://linear.app/thicclatka/issue/THI-595))

## [0.5.4] - 2026-10-20

### Changed

- Pin tetration **0.5.4** from crates.io and align the package version (**0.5.4**). Crate hygiene + query-wire fuzz; no new Python API ([THI-604](https://linear.app/thicclatka/issue/THI-604))

## [0.5.3] - 2026-10-20

### Changed

- Pin tetration **0.5.3** from crates.io and align the package version (**0.5.3**). Conformance pack grow on the crate; no new Python API ([THI-605](https://linear.app/thicclatka/issue/THI-605))

## [0.5.2] - 2026-10-20

### Changed

- Pin tetration **0.5.2** from crates.io and align the package version (**0.5.2**). SPEC/conformance milestone on the crate; no new Python API ([THI-591](https://linear.app/thicclatka/issue/THI-591))

## [0.5.1] - 2026-10-20

### Changed

- Pin tetration **0.5.1** from crates.io and align the package version (**0.5.1**). Declared science/media attrs remain via footer metadata / `tet semantics` on the crate; no new Python metadata ORM ([THI-585](https://linear.app/thicclatka/issue/THI-585))

## [0.5.0] - 2026-10-20

### Changed

- Pin tetration **0.5.0** from crates.io and align the package version (**0.5.0**). Distributed fold stays on the Rust `query::distributed` API; no new Python wrapper ([THI-581](https://linear.app/thicclatka/issue/THI-581))

## [0.4.5] - 2026-10-20

### Changed

- Pin tetration **0.4.5** from crates.io and align the package version (**0.4.5**). Optional value-posting skips for exact `filter.eq` come through existing plan JSON (`index_filter_skipped_chunks`); no new Python API ([THI-574](https://linear.app/thicclatka/issue/THI-574))

## [0.4.4] - 2026-10-19

### Changed

- Pin tetration **0.4.4** from crates.io and align the package version (**0.4.4**). Closed `filter` now applies per cell inside kept chunks via the crate; no new Python API ([THI-569](https://linear.app/thicclatka/issue/THI-569))

## [0.4.3] - 2026-10-19

### Changed

- Pin tetration **0.4.3** from crates.io and align the package version (**0.4.3**). Store prune stays on `tet gc` (including `s3://`); no new Python API ([THI-560](https://linear.app/thicclatka/issue/THI-560))

## [0.4.2] - 2026-10-19

### Changed

- Pin tetration **0.4.2** from crates.io and align the package version (**0.4.2**). Closed `group_by` and `pipeline` wire is available via existing JSON/TOML query documents passed to `execute`; no new Python API ([THI-556](https://linear.app/thicclatka/issue/THI-556))

## [0.4.1] - 2026-10-19

### Changed

- Pin tetration **0.4.1** from crates.io and align the package version (**0.4.1**). Closed query `filter` wire (predicate pushdown / zone-map chunk skip) is available via existing JSON/TOML query documents; no new Python API ([THI-550](https://linear.app/thicclatka/issue/THI-550))

## [0.4.0] - 2026-10-18

### Changed

- Pin tetration **0.4.0** from crates.io and align the package version (**0.4.0**). Native NumPy export uses `ReadPlan.logical_element_count` (THI-540) ([THI-542](https://linear.app/thicclatka/issue/THI-542))

## [0.3.8] - 2026-10-18

### Changed

- Pin tetration **0.3.8** from crates.io and align the package version (**0.3.8**) ([THI-536](https://linear.app/thicclatka/issue/THI-536))

## [0.3.7] - 2026-10-18

### Added

- `tet.publish(..., dry_run=False, replace=None)`: claim/schema dry-run and opt-in dataset replace vs `CURRENT` ([THI-525](https://linear.app/thicclatka/issue/THI-525))

### Changed

- Pin tetration **0.3.7** from crates.io and align the package version (**0.3.7**) ([THI-525](https://linear.app/thicclatka/issue/THI-525))

## [0.3.6] - 2026-10-18

### Added

- `tet.pack(package, dest, *, generation=None, remove_package=False)` copies a sealed generation to a standalone `.tet` (optional package delete after verify) ([THI-523](https://linear.app/thicclatka/issue/THI-523))

### Changed

- Pin tetration **0.3.6** from crates.io and align the package version (**0.3.6**) ([THI-523](https://linear.app/thicclatka/issue/THI-523))

## [0.3.5] - 2026-10-18

### Added

- `tet.convert` reads a Zarr v2 or v3 prefix on `s3://` through the zarr library, fetching each chunk as needed. The `.tet` matches a local-directory convert of the same store ([THI-516](https://linear.app/thicclatka/issue/THI-516))

### Changed

- Align the package version with tetration **0.3.5** ([THI-516](https://linear.app/thicclatka/issue/THI-516))

## [0.3.4] - 2026-10-17

### Added

- `tet.open("s3://bucket/prefix")` pins a package (`?generation=N` optional) and range-reads on query. Credentials stay in `TETRATION_S3_*`. ([THI-517](https://linear.app/thicclatka/issue/THI-517))

### Changed

- Align the package version with tetration **0.3.4** ([THI-517](https://linear.app/thicclatka/issue/THI-517))

## [0.3.3] - 2026-10-17

### Added

- `tet.convert` accepts `s3://bucket/object.ext`: full download, then the existing importer. Zarr-on-S3 is not included ([THI-507](https://linear.app/thicclatka/issue/THI-507))

### Changed

- Pin tetration **0.3.3** from crates.io and align the package version (**0.3.3**) ([THI-515](https://linear.app/thicclatka/issue/THI-515))

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
