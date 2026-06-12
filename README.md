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

**Status:** read/query API; NumPy interchange on all three tetration dense sinks (**ram** `read_numpy` / `to_numpy`, **spill** `read_spill` / `to_spill`, **sidecar** `to_sidecar`); write via `TetWriter` / `write_dataset`. Convert extras and PyPI wheels are next — see [docs/HANDOFF.md](docs/HANDOFF.md#github-tracking-tet-py).

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
    print(f.mean("a"), f.quantile("a", 0.5))
    arr = f.read_numpy("a")                         # ram
    z = f.transform.to_numpy.zscore("a")            # transform → ram
    spill = f.transform.to_spill.zscore("a", path="a_zscore.bin")
    same = spill.to_numpy()                         # transform → spill → ndarray
```

**Operations reference** (every op with examples): [**docs/operations.md**](docs/operations.md)

| Topic                                     | Where                                                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `mean`, `min`, `quantile`, `histogram`, … | [docs/operations.md](docs/operations.md)                                                               |
| NumPy ram / spill / sidecar               | [docs/operations.md#read_numpy](docs/operations.md#read_numpy)                                         |
| `build_query`, `selection_slices`         | [docs/operations.md#selection-and-build_query](docs/operations.md#selection-and-build_query)           |
| Wire schema / CLI                         | [tetration query engine](https://github.com/Latka-Industries/tetration/blob/main/docs/query_engine.md) |
| All docs                                  | [docs/README.md](docs/README.md)                                                                       |

### `info()` / `summary()`

Both return the full **`tet info --json`** dict (superblock, datasets, **all chunk rows**, metadata). For exploration, prefer `list(f)`, `f.dataset(name)`, or `info["datasets"]` — not printing the whole dict in the REPL.

## Project layout

```text
tet-py/
  pyproject.toml      # PEP 621 + maturin
  python/tet/         # facade (_file, _numpy, _spill, _transform, _query, …)
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
- [x] NumPy read — ram (`read_numpy`, `Dataset.to_numpy`, `transform.to_numpy`)
- [x] NumPy read — spill (`read_spill`, `transform.to_spill`, `.to_numpy()` loaders)
- [x] NumPy read — sidecar (`transform.to_sidecar`, `SidecarTransformResult.to_numpy`)
- [x] NumPy write (`TetWriter`, `write_dataset`)
- [ ] Optional convert extras: `h5py`, `netCDF4`, `zarr`, `pandas`, `pyarrow` — [#10](https://github.com/Latka-Industries/tet-py/issues/10)
- [ ] Wheels on PyPI; pin `tetration = "x.y.z"` from crates.io for release builds — [#5](https://github.com/Latka-Industries/tet-py/issues/5)

## Related

- [docs/operations.md](docs/operations.md) — query ops (`mean`, `quantile`, …) with examples
- [docs/README.md](docs/README.md) — doc index
- [HANDOFF.md](docs/HANDOFF.md) — phases, dev commands, agent notes
- [Layout v1](https://github.com/Latka-Industries/tetration/blob/main/docs/layout_v1.md)
- [Query engine](https://github.com/Latka-Industries/tetration/blob/main/docs/query_engine.md)
- [AGENTS.md](https://github.com/Latka-Industries/tetration/blob/main/AGENTS.md) — ops and phase status

## License

Dual-licensed under **MIT OR Apache-2.0**, same as [tetration](https://github.com/Latka-Industries/tetration).
