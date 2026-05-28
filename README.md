# tet-py

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)
[![CI](https://github.com/Latka-Industries/tet-py/actions/workflows/ci.yml/badge.svg)](https://github.com/Latka-Industries/tet-py/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)

Python bindings for [**Tetration**](https://github.com/Latka-Industries/tetration) — mmap-friendly `.tet` tensor files and the JSON/TOML query engine.

| Install (PyPI) | `pip install tet-py` *(when published)* |
| Import | `import tet` |
| Rust core | [`tetration`](https://crates.io/crates/tetration) on crates.io |
| CLI (no Python) | [`tet`](https://github.com/Latka-Industries/tetration) binary from the main repo |

**Status:** read/query API with catalog access, reduction helpers, selection builders, and file-specific errors. NumPy read/write and convert extras are planned ([Phase 11](https://github.com/Latka-Industries/tetration/blob/main/GETTING_STARTED.md#phase-11--bindings-python--c-abi)).

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

Paths below use the sibling **tetration** fixture `fixtures/small/tet/large.tet` (dataset **`a`**, shape `[34, 64]`). Adjust the path for your machine.

```python
import tet
from tet import axis_slice, build_query, selection_slices

path = "../tetration/fixtures/small/tet/large.tet"

with tet.open(path) as f:
    print(list(f))                    # ['a']
    print(f.dataset("a").shape)       # (34, 64)

    print(f.mean("a"))                # scalar over all axes
    print(f.min("a"))
    print(f.max("a", axis=0))         # partial reduction → QueryResult

    r = f.execute({"dataset": "a", "mean": []})
    print(r.scalar)                   # same as f.mean("a"); raw=False default

    f.execute({"dataset": "a"}, plan=True)  # plan only (no op keys)

    # Subregion + op
    sel = selection_slices(axis_slice(0, 4), axis_slice(0, 4))
    f.execute(build_query("a", selection=sel, mean=[]))

    f.quantile("a", 0.5)
    f.histogram("a", bins=8)
```

For rank-2 covariance/correlation, use a 2-D dataset (e.g. `sample.tet` / `"temperature"`):

```python
f = tet.open("../tetration/fixtures/small/tet/sample.tet")
r = f.covariance("temperature", axis=1)
print(r.matrix_order, r.matrix)
```

### Query API

**Prefer op helpers** — `mean`, `sum`, `min`, `max`, `std`, `var`, `count`, `product`, `norm_l1`, `norm_l2`, `median`, `all_finite`, `any_nan`, `arg_min`, `arg_max`, plus `quantile`, `histogram`, `covariance`, `correlation`.

Use **`f.execute(doc)`** with `raw=False` (default) for a [`QueryResult`](python/tet/_query.py) (`.scalar`, `.reduced`, `.matrix`, …). Use **`f.query(..., raw=True)`** for the full wire dict (same as `tet query -x` JSON).

**Build documents in Python:**

```python
from tet import axis_slice, build_query, selection_slices

doc = build_query(
    "a",
    selection=selection_slices(axis_slice(0, 2), axis_slice(0, 2)),
    mean=[],
)
f.execute(doc)
```

`query`, `plan_only`, `query_execute`, and `execute` accept a **dict** or JSON string — same schema as the [`tet query`](https://github.com/Latka-Industries/tetration/blob/main/docs/query_engine.md) CLI.

| Topic | Notes |
| ----- | ----- |
| Fixtures | [`tetration/fixtures/queries/`](https://github.com/Latka-Industries/tetration/tree/main/fixtures/queries) |
| `execute(..., plan=True)` | Plan only; omit `mean` / `sum` / other op keys |
| `execute(..., device="cpu")` | Sets `execution.device` before execute |
| `raw=True` | Full `QueryResponse` dict |
| Errors | `UnknownDatasetError` / `UnknownAxisError` list names valid for **this** file |
| IDE names | `tet.typing_stub(path)` → save `.pyi` with `Literal` dataset names (optional) |

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
  docs/HANDOFF.md     # phases and agent notes
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

- [HANDOFF.md](docs/HANDOFF.md) — phases, dev commands, agent notes
- [Layout v1](https://github.com/Latka-Industries/tetration/blob/main/docs/layout_v1.md)
- [Query engine](https://github.com/Latka-Industries/tetration/blob/main/docs/query_engine.md)
- [AGENTS.md](https://github.com/Latka-Industries/tetration/blob/main/AGENTS.md) — ops and phase status

## License

Dual-licensed under **MIT OR Apache-2.0**, same as [tetration](https://github.com/Latka-Industries/tetration).
