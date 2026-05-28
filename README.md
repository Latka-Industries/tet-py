# tet-py

[![uv]]

[![CI](https://github.com/Latka-Industries/tet-py/actions/workflows/ci.yml/badge.svg)](https://github.com/Latka-Industries/tet-py/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)

Python bindings for [**Tetration**](https://github.com/Latka-Industries/tetration) — mmap-friendly `.tet` tensor files and the JSON/TOML query engine.

| Install (PyPI) | `pip install tet-py` *(when published)* |
| Import | `import tet` |
| Rust core | [`tetration`](https://crates.io/crates/tetration) on crates.io |
| CLI (no Python) | [`tet`](https://github.com/Latka-Industries/tetration) binary from the main repo |

**Status:** early scaffold — `open`, catalog summary, and `query` via JSON documents. Write path, NumPy views, and ecosystem convert extras are planned ([Phase 11](https://github.com/Latka-Industries/tetration/blob/main/GETTING_STARTED.md#phase-11--bindings-python--c-abi)).

Do not `pip install tetration` — that PyPI name is an unrelated math package. Use **`tet-py`** / **`import tet`**.

## Quick start

### Prerequisites

- Python **3.11+**
- Rust **1.95+** ([`rust-toolchain.toml`](rust-toolchain.toml) or [mise](https://mise.jdx.dev/))
- [maturin](https://www.maturin.rs/) (`pip install maturin` or `uv tool install maturin`)

### Development (local `tetration` checkout)

This repo expects the Rust library next to it by default:

```text
~/Code/tetration   # main crate
~/Code/tet-py      # this repo (path dependency in Cargo.toml)
```

```bash
cd ~/Code/tet-py
uv sync --extra dev
mise run develop   # or: uv run maturin develop
uv run python -c "import tet; print(tet.__version__, tet.core_version())"
uv run pytest -q
```

(`uv sync` runs an editable build via maturin; `maturin develop` refreshes the native extension after Rust changes.)

### Example

```python
import tet

f = tet.open("data.tet")
print(f.datasets())  # catalog dataset names

summary = f.summary()  # or f.info() — same as `tet info --json`
print(summary["datasets"][0]["name"])

print(f.mean("temperature"))  # helpers build the query document

doc = {"dataset": "temperature", "mean": []}
plan = f.plan_only(doc)       # plan without -x execution
out = f.query(doc)            # full execute
print(out["execution"]["operation_mean"])
```

`query` accepts a **dict** or a JSON string — same schema as `tet query` / [`fixtures/queries/`](https://github.com/Latka-Industries/tetration/tree/main/fixtures/queries).

## Project layout

```text
tet-py/
  Cargo.toml          # PyO3 extension (links tetration)
  pyproject.toml      # PEP 621 + maturin
  src/lib.rs          # native module tet._native
  python/tet/         # pure-Python facade
  tests/
```

## Roadmap

- [x] Scaffold: maturin, `tet.open`, `datasets`, `query` (JSON document)
- [x] `query()` returns `dict` (no `json.loads`)
- [x] `summary()` / `info()` — dict parity with `tet info --json`
- [x] `TetError` / `CatalogError` exceptions (not bare `RuntimeError`)
- [x] `plan_only()`, `mean()`, `sum()` helpers
- [ ] Typed query helpers (`QueryDocument` builders)
- [ ] Write path: NumPy → chunk tiles (`TetWriterSession`)
- [ ] Optional convert extras: `h5py`, `netCDF4`, `zarr`, `pandas` (CSV), `pyarrow` (Parquet)
- [ ] Wheels on PyPI; pin `tetration = "x.y.z"` from crates.io (drop path dep for release)

## Related

- [Layout v1](https://github.com/Latka-Industries/tetration/blob/main/docs/layout_v1.md)
- [Query engine](https://github.com/Latka-Industries/tetration/blob/main/docs/query_engine.md)
- [AGENTS.md](https://github.com/Latka-Industries/tetration/blob/main/AGENTS.md) — ops and phase status

## License

Dual-licensed under **MIT OR Apache-2.0**, same as [tetration](https://github.com/Latka-Industries/tetration).
