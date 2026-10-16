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

Do **not** `pip install tetration` — that PyPI name is unrelated math code.

## Quick start

```bash
pip install tet-py
python -c "import tet; print(tet.__version__, tet.core_version())"
```

```python
import tet

with tet.open("data.tet") as f:
    print(f.mean("temperature"), f.quantile("temperature", 0.5))
    arr = f.read_numpy("temperature")                 # ram (memmap when single-chunk raw)
    z = f.transform.to_numpy.zscore("temperature")  # transform → ndarray
```

More examples: [Python quick start](https://docs.tetration.sh/python/quick-start) · [operations](https://docs.tetration.sh/python/operations) · [NumPy](https://docs.tetration.sh/python/numpy)

## Convert

`tet.convert` writes one `.tet` from a foreign file. `.npy` and multi-member `.npz` are in the core wheel. Other formats are optional extras: `tet-py[h5]`, `[netcdf]`, `[zarr]`, `[csv]`, `[parquet]`, or `[convert]` for all of them.

| Extra | What it imports |
| --- | --- |
| *(core)* | `.npy` (one dataset), `.npz` (every member) |
| `h5` | HDF5 datasets, groups as `parent/child` names |
| `netcdf` | NetCDF variables, including groups |
| `zarr` | Zarr v2 and v3 directory stores |
| `csv` | `.csv` / `.tsv` numeric columns, plus text columns as `int64` category codes (`tetration_categories`) |
| `parquet` | Parquet numeric columns and UTF-8 category columns |

Compared with Rust `tet convert`: Python does not apply CF scale/offset, chunks the whole array unless you pass `chunk_shape`, and can read codecs the Rust importer skips when the extra library can decode them. Columns that are not layout-v1 numeric (or category text) are omitted.

## Packages

`tet.open("pkg.tet.d")` reads the generation named by `CURRENT` (`packed.tet`). `tet.ingest` copies a sealed `.tet` into staging, and `tet.publish` seals those parts and updates `CURRENT`. See [`docs/operations.md`](docs/operations.md#packages).

## Documentation

| Topic | Where |
| ----- | ----- |
| Install, API, NumPy sinks | [docs.tetration.sh/python](https://docs.tetration.sh/python/) |
| Query ops (local mirror) | [`docs/operations.md`](docs/operations.md) |
| Release notes | [`CHANGELOG.md`](CHANGELOG.md) |
| Dev / agent notes | [`docs/HANDOFF.md`](docs/HANDOFF.md) |
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
