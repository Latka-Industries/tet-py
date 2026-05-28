# tet-py

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)
[![CI](https://github.com/Latka-Industries/tet-py/actions/workflows/ci.yml/badge.svg)](https://github.com/Latka-Industries/tet-py/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)

Python bindings for [**Tetration**](https://github.com/Latka-Industries/tetration) — mmap-friendly `.tet` tensor files and the JSON/TOML query engine.

| Install (PyPI) | `pip install tet-py` _(when published)_ |
| Import | `import tet` |
| Rust core | [`tetration`](https://crates.io/crates/tetration) on crates.io |
| CLI (no Python) | [`tet`](https://github.com/Latka-Industries/tetration) binary from the main repo |

**Status:** read/query API with catalog access, reduction helpers, selection builders, and file-specific errors. NumPy read/write and convert extras are planned ([Phase 11](https://github.com/Latka-Industries/tetration/blob/main/GETTING_STARTED.md#phase-11--bindings-python--c-abi)).

Do not `pip install tetration` — that PyPI name is an unrelated math package. Use **`tet-py`** / **`import tet`**.

## Quick start

### Prerequisites

- Python **3.11+**
- Rust **1.95+** ([`.mise.toml`](.mise.toml) pins `rust = "1.95"`, or install matching [rustup](https://rustup.rs/) toolchain; `native/Cargo.toml` sets `rust-version = "1.95"`)
- [maturin](https://www.maturin.rs/) (`pip install maturin` or `uv tool install maturin`)

### Development (local `tetration` checkout)

This repo expects the Rust library next to it by default:

```text
~/Code/tetration   # main crate
~/Code/tet-py      # this repo (`native/Cargo.toml` path dependency)
```

```bash
cd ~/Code/tet-py
uv sync --extra dev
mise run develop   # or: uv run maturin develop
uv run python -c "import tet; print(tet.__version__, tet.core_version())"
uv run pytest -q
uv run mypy python/tet
```

(`uv sync` runs an editable build via maturin; `maturin develop` refreshes the native extension after Rust changes.)

### Example

```python
import tet

with tet.open("../tetration/fixtures/small/tet/large.tet") as f:
    print(list(f), f.dataset("a").shape)   # ['a'], (34, 64)
    print(f.mean("a"), f.quantile("a", 0.5))
```

**Operations reference** (every op with examples): [**docs/operations.md**](docs/operations.md)

| Topic                                     | Where                                                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `mean`, `min`, `quantile`, `histogram`, … | [docs/operations.md](docs/operations.md)                                                               |
| `build_query`, `selection_slices`         | [docs/operations.md#selection-and-build_query](docs/operations.md#selection-and-build_query)           |
| Wire schema / CLI                         | [tetration query engine](https://github.com/Latka-Industries/tetration/blob/main/docs/query_engine.md) |
| All docs                                  | [docs/README.md](docs/README.md)                                                                       |

### `info()` / `summary()`

Both return the full **`tet info --json`** dict (superblock, datasets, **all chunk rows**, metadata). For exploration, prefer `list(f)`, `f.dataset(name)`, or `info["datasets"]` — not printing the whole dict in the REPL.

## Project layout

```text
tet-py/
  pyproject.toml      # PEP 621 + maturin
  python/tet/         # facade (_file, _catalog, _query, _query_doc, _errors)
  native/             # PyO3 extension (links tetration)
    src/lib.rs
  tests/
  docs/               # operations.md, HANDOFF.md
```

## Roadmap

- [x] Scaffold: maturin, `tet.open`, `datasets`, `query` (JSON document)
- [x] `query()` / `execute(raw=False)` → `QueryResult`
- [x] `summary()` / `info()` — dict parity with `tet info --json`
- [x] Reduction helpers (`mean`, `sum`, `min`, `max`, …)
- [x] `quantile`, `histogram`, `covariance`, `correlation`
- [x] `build_query`, `axis_slice`, `selection_slices`
- [x] `Dataset`, `iter_datasets()`, `f[0]` / `f["name"]`, axis index or `dim_names`
- [x] `UnknownDatasetError` / `UnknownAxisError`; optional `typing_stub()`
- [x] mypy + `_native.pyi`
- [ ] NumPy read (`read_numpy`) / write (`write_dataset`)
- [ ] Optional convert extras: `h5py`, `netCDF4`, `zarr`, `pandas`, `pyarrow`
- [ ] Wheels on PyPI; pin `tetration = "x.y.z"` from crates.io for release builds

## Related

- [docs/operations.md](docs/operations.md) — query ops (`mean`, `quantile`, …) with examples
- [docs/README.md](docs/README.md) — doc index
- [HANDOFF.md](docs/HANDOFF.md) — phases, dev commands, agent notes
- [Layout v1](https://github.com/Latka-Industries/tetration/blob/main/docs/layout_v1.md)
- [Query engine](https://github.com/Latka-Industries/tetration/blob/main/docs/query_engine.md)
- [AGENTS.md](https://github.com/Latka-Industries/tetration/blob/main/AGENTS.md) — ops and phase status

## License

Dual-licensed under **MIT OR Apache-2.0**, same as [tetration](https://github.com/Latka-Industries/tetration).
