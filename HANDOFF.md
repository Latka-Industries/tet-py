# tet-py — handoff, phases, and outlook

Agent/onboarding doc for **`~/Code/tet-py`**. Parent project: **[tetration](https://github.com/thicclatka/tetration)** (Rust crate + `tet` CLI). Phase 11 in main repo: [`GETTING_STARTED.md#phase-11`](https://github.com/thicclatka/tetration/blob/main/GETTING_STARTED.md#phase-11--bindings-python--c-abi).

## Naming (decided)

| Role             | Value                                                                                    |
| ---------------- | ---------------------------------------------------------------------------------------- |
| GitHub / PyPI    | **`tet-py`**                                                                             |
| `import`         | **`tet`**                                                                                |
| Rust dependency  | **`tetration`** on crates.io (PyPI name `tetration` is unrelated math code — do not use) |
| Native extension | `tet._native` (`.so` / `.pyd`)                                                           |

**C ABI** stays in the **tetration** repo later (Julia/R/Go), not in tet-py. Python uses **PyO3 → `tetration` rlib**, not the C layer.

## Current state (May 2026)

### Done / working\*\*

- [x] Repo scaffold: `pyproject.toml` (maturin), `Cargo.toml`, `build.rs` (`TETRATION_VERSION` from `../tetration`)
- [x] Path dep: `tetration = { path = "../tetration", default-features = false }` (lean wheels; no HDF5/NetCDF in extension)
- [x] PyO3 **0.28**, `uv sync --extra dev`, `uv run maturin develop`
- [x] API: `tet.open(path)` (`str` / `Path`), `TetFile.path`, `datasets()`, `summary_json()`, `query(dict|str)` → JSON `QueryResponse`
- [x] Smoke tests vs `../tetration/fixtures/small/tet/sample.tet` (mean = 3.5)
- [x] `tet.__version__` (extension 0.1.0), `tet.core_version()` (linked tetration, e.g. 0.1.5)

### Not done\*\*

- No PyPI publish; git remote on **Latka-Industries/tet-py** (private)
- No typed Python query objects, NumPy buffers, write path, convert extras
- `query()` returns **`dict`** (parsed `QueryResponse` JSON)
- No `TetFile.open` classmethod (use `tet.open` only)

## Dev commands (uv)

```bash
cd ~/Code/tet-py
uv sync --extra dev
uv run maturin develop          # after Rust changes
uv run pytest -q
uv run python -c "import tet; print(tet.__version__, tet.core_version())"
```

Layout: `python/tet/` = facade; `src/lib.rs` = PyO3; tests expect sibling `~/Code/tetration`.

---

## Phases (recommended order)

### Phase 0 — Repo hygiene (P0)

- [x] Push to `github.com/Latka-Industries/tet-py`
- [x] CI: `.github/workflows/ci.yml` (checkout sibling **tetration**, `uv sync` → `maturin develop` → `pytest`)
- [ ] Register **`tet-py`** on PyPI (empty/meta release optional) to hold the name
- [x] `LICENSE-MIT` + `LICENSE-APACHE` (dual license, match tetration)
- [ ] Link README ↔ tetration Phase 11 when milestones land

### Phase 1 — Read / query UX (P0)

Goal: parity with common `tet query -t … -x` embedder paths without hand-rolled JSON everywhere.

- [x] `query()` → return **`dict`** (parsed from Rust JSON in Python facade)
- [ ] Optional: `query_execute(doc, preview=0, device=None)` mapping `ExecuteQueryOptions` + `execution.device`
- [x] `info()` / `summary()` → **`dict`** (parsed `summary_json()`; parity with `tet info --json`)
- [x] `plan_only(doc)` → plan without execution (`ExecuteQueryOptions::plan_only`)
- [x] `mean(dataset, axes=...)`, `sum(...)` — thin helpers over `query()`
- [ ] More ops / selection slices — build `QueryDocument` in Python or thin Rust exports
- [ ] Document query schema → link `tetration/fixtures/queries/`
- [x] Errors: `tet.TetError`, `tet.CatalogError`; `OSError` on missing file

### Phase 2 — Write path (P1)

Goal: create/append `.tet` from NumPy (no Rust HDF5 in wheels).

- [ ] Expose `TetWriterSession` / `TetFile` commit path from tetration catalog API
- [ ] `write_dataset(name, array: numpy.ndarray, chunk_shape=..., attrs=..., coords=...)`
- [ ] Footer metadata + history events on commit
- [ ] Tests: roundtrip write → read → query sum/mean vs golden

### Phase 3 — Ecosystem convert (P1, Python-only)

Goal: long-tail formats via Python stack → tiles → Rust writer (not `tetration-hdf5` / `tetration-netcdf` in wheels).

Optional extras in `pyproject.toml`:

| Extra     | Libraries        | Notes                         |
| --------- | ---------------- | ----------------------------- |
| `h5`      | h5py             | HDF5 → NumPy tiles            |
| `netcdf`  | netCDF4 / xarray | NetCDF                        |
| `zarr`    | zarr, xarray     | Zarr v3 dirs                  |
| `csv`     | pandas           | Needs shape/dtype conventions |
| `parquet` | pyarrow          | Table → tensor rules          |
| `npy`     | numpy            | Smoke / trivial path          |

- [ ] `tet.convert(src, dst.tet, ...)` orchestration
- [ ] Defer **sparse MTX** unless explicit densify + size limits

Rust CLI **`tet convert`** remains the fast path for HDF5/NetCDF/Zarr on machines with native libs.

### Phase 4 — Release engineering (P1)

- [ ] Release: pin `tetration = "x.y.z"` from crates.io; remove `path = "../tetration"` in published `Cargo.toml` (or maturin/source dist policy)
- [ ] Linux / macOS / Windows wheels via maturin-action
- [ ] Version policy: `tet-py` version tracks compatible `tetration` minor (document in README)
- [ ] `numpy` as dependency when buffer APIs land

### Phase 5 — Nice-to-have (P2)

- [ ] Context manager: `with tet.open(path) as f:`
- [ ] `tet.TetFile.open` classmethod if desired (`PyType` import in PyO3)
- [ ] Submodule `tet.convert`, `tet.query` for large API surface
- [ ] Jupyter / xarray accessor sketch (`ds.tet.write(...)` out of scope unless requested)
- [ ] Shared fixtures: git submodule `tetration` or vendor `fixtures/small/` for CI without sibling path

---

## Architecture reminders

```text
User code  →  import tet  →  python/tet/__init__.py
                              →  tet._native (PyO3)
                              →  tetration crate (rlib)
                              →  .tet mmap / query engine
```

- **Single query contract:** `QueryDocument` (JSON/TOML on wire; dict in Python). Same as CLI `tet query`.
- **Do not fork** op semantics in Python-only code; add sugar that still calls `execute_query_document`.
- **default-features = false** on `tetration` for wheels; GPU features only if explicitly requested later.

---

## Outlook

| Horizon    | Target                                                                                         |
| ---------- | ---------------------------------------------------------------------------------------------- |
| **Short**  | Stable read/query API, CI, PyPI alpha `0.1.x`, docs + examples                                 |
| **Medium** | Write + convert extras; embedders replace subprocess `tet`                                     |
| **Long**   | Versioned wheels per platform; optional alignment with tetration **C ABI** for non-Python only |

**Success:** `pip install tet-py` → `import tet` → open `.tet` → query with dict or helpers → optional convert from CSV/Parquet/HDF5 via extras — without requiring Rust on the end user machine.

**Non-goals for tet-py v0.x:** reimplement layout/query in Python; sparse native format; GPU in Python wheel by default; duplicate Rust `tet convert` codecs inside the extension.

---

## Open questions for maintainers

1. Pin **one** Python (3.11 vs 3.13) for CI wheels?
2. Submodule **tetration** vs path sibling for CI?
3. Publish **0.1.0** before write path, or after Phase 2 smoke?
4. Exception hierarchy depth vs plain `RuntimeError` from PyO3 today?

---

## Quick file map

| File                     | Purpose                                               |
| ------------------------ | ----------------------------------------------------- |
| `src/lib.rs`             | `PyTetFile`, `open`, `core_version`, `_native` module |
| `python/tet/__init__.py` | Public exports, `__version__`                         |
| `build.rs`               | `TETRATION_VERSION` from sibling crate                |
| `tests/test_smoke.py`    | `sample.tet` mean query                               |
| `README.md`              | User-facing quick start                               |

When behavior changes, update **README.md** and this file; link Phase 11 checklist in **tetration** `GETTING_STARTED.md` when milestones complete.
