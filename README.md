# tet-py

[![PyPI](https://img.shields.io/pypi/v/tet-py.svg)](https://pypi.org/project/tet-py/)
[![Docs](https://img.shields.io/badge/docs-docs.tetration.sh-blue)](https://docs.tetration.sh/python/)
[![CI](https://github.com/Latka-Industries/tet-py/actions/workflows/ci.yml/badge.svg)](https://github.com/Latka-Industries/tet-py/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)

Python bindings for [**Tetration**](https://github.com/Latka-Industries/tetration) — mmap-friendly `.tet` tensors and the JSON/TOML query engine.

| | |
| --- | --- |
| **Install** | [`pip install tet-py`](https://pypi.org/project/tet-py/) → `import tet` |
| **Docs** | [docs.tetration.sh/python](https://docs.tetration.sh/python/) |
| **Rust core** | [`tetration`](https://crates.io/crates/tetration) (linked at build time) |
| **CLI** | [`tet`](https://github.com/Latka-Industries/tetration) binary from the main repo |


## Quick start

```bash
pip install tet-py
python -c "import tet; print(tet.__version__, tet.core_version())"
```

```python
import tet

with tet.open("data.tet") as f:
    print(f.mean("temperature"), f.quantile("temperature", 0.5))
    arr = f.read_numpy("temperature")                 # ram (memmap for full-width raw slabs)
    z = f.transform.to_numpy.zscore("temperature")  # transform → ndarray
```

More examples: [Python quick start](https://docs.tetration.sh/python/quick-start) · [operations](https://docs.tetration.sh/python/operations) · [NumPy](https://docs.tetration.sh/python/numpy)

## Convert

`tet.convert` writes one `.tet` from a foreign file. `.npy`, multi-member `.npz`, and `.csv` / `.tsv` are in the core wheel. Other formats are optional extras: `tet-py[h5]`, `[netcdf]`, `[zarr]`, `[parquet]`, or `[convert]` for all of them.

`src` may be `s3://bucket/object.ext` (full download, then the same importer) or an `s3://` Zarr v2/v3 prefix (each chunk is fetched as the zarr library reads it). Credentials stay in `TETRATION_S3_*`.

| Extra | What it imports |
| --- | --- |
| *(core)* | `.npy` (one dataset), `.npz` (every member), `.csv` / `.tsv` (numeric + category text columns) |
| `h5` | HDF5 datasets, groups as `parent/child` names |
| `netcdf` | NetCDF variables, including groups |
| `zarr` | Zarr v2 and v3 directory stores |
| `parquet` | Parquet numeric columns and UTF-8 category columns |

Compared with Rust `tet convert`: Python does not apply CF scale/offset, chunks the whole array unless you pass `chunk_shape`, and can read codecs the Rust importer skips when the extra library can decode them. Columns that are not layout-v1 numeric (or category text) are omitted.

## Export

`tet.export` (and `TetFile.export`) writes a `.tet` to HDF5, NetCDF, or Zarr. Same optional extras as convert (`[h5]`, `[netcdf]`, `[zarr]`, or `[export]` for those three). Dense numeric datasets only; carries `dim_names` and basic string attrs when present. Values are densified via `read_numpy` — prefer Rust `tet export` when you need native HDF5/NetCDF or chunk-preserving Zarr.

## Packages

`tet.open("pkg.tet.d")` reads the generation named by `CURRENT` (`packed.tet`). `tet.ingest` copies a sealed `.tet` into staging, and `tet.publish` seals those parts and updates `CURRENT`. `tet.pack(package, "out.tet")` copies that sealed image to a standalone file; pass `remove_package=True` to delete the `.tet.d` after a verified copy.

For an S3-compatible backend:

```python
from tet import S3Store

store = S3Store.connect(endpoint="http://127.0.0.1:9000", bucket="…",
                        access_key_id="…", secret_access_key="…", prefix="demo")
store.ingest("part.tet", session_id="s0")
store.publish(["part"])
f = store.open("current")  # pin; later publish does not retarget f
# or, with TETRATION_S3_* set: tet.open("s3://bucket/demo")
```

Endpoint and credentials are arguments or `TETRATION_S3_*` (not in the URI). `tet.open("s3://bucket/prefix")` range-reads the pinned package. `S3Store.open` still downloads `packed.tet`. See [`docs/operations.md`](docs/operations.md#packages).

## Documentation

| Topic | Where |
| ----- | ----- |
| Install, API, NumPy sinks | [docs.tetration.sh/python](https://docs.tetration.sh/python/) |
| Query ops (local mirror) | [`docs/operations.md`](docs/operations.md) |
| Release notes | [`CHANGELOG.md`](CHANGELOG.md) |
| Format + query wire | [docs.tetration.sh](https://docs.tetration.sh) · [tetration repo](https://github.com/Latka-Industries/tetration) |
| Roadmap | [docs.tetration.sh/roadmap](https://docs.tetration.sh/roadmap) |

## Develop

```bash
uv sync --extra dev
mise run develop          # or: uv run maturin develop
uv run pytest -q
uv run mypy python/tet
```

Requires Python **3.11+** and Rust **1.95+** ([`.mise.toml`](.mise.toml)). Fixtures: `tests/fixtures/`.

## License

**MIT OR Apache-2.0**, same as [tetration](https://github.com/Latka-Industries/tetration).
