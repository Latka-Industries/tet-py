# tet-py — handoff, phases, and outlook

Agent/onboarding doc for **`~/Code/tet-py`**. Parent project: **[tetration](https://github.com/Latka-Industries/tetration)** (Rust crate + `tet` CLI). Python bindings: this repo; C ABI / Phase 11: [`tetration/docs/ffi.md`](https://github.com/Latka-Industries/tetration/blob/main/docs/ffi.md).

## Naming (decided)

| Role             | Value                                                                                    |
| ---------------- | ---------------------------------------------------------------------------------------- |
| GitHub / PyPI    | **`tet-py`**                                                                             |
| `import`         | **`tet`**                                                                                |
| Rust dependency  | **`tetration`** on crates.io (PyPI name `tetration` is unrelated math code — do not use) |
| Native extension | `tet._native` (`.so` / `.pyd`)                                                           |

**C ABI** stays in the **tetration** repo later (Julia/R/Go), not in tet-py. Python uses **PyO3 → `tetration` rlib**, not the C layer.

## Current state (June 2026)

### Done / working

- [x] Phase 1 read/query UX: reductions, `QueryResult`, catalog, `build_query`, typed errors
- [x] Phase 2 NumPy **ram** path: `read_numpy`, `Dataset.to_numpy`, `transform.to_numpy.*`, `TetWriter`, `write_dataset`
- [x] Phase 2 NumPy **spill** path: `read_spill`, `transform.to_spill.*`, `SpillReadResult.to_numpy`, `SpillTransformResult.to_numpy`
- [x] Phase 2 NumPy **sidecar** path: `transform.to_sidecar.*`, `SidecarTransformResult.to_numpy`
- [x] `uv sync --extra dev`, PyO3 **0.28**, path dep `tetration = { path = "../tetration" }`, smoke + interchange tests (55)
- [x] `tet.__version__` (0.1.0), `tet.core_version()` (linked tetration, e.g. 0.1.9)

### Not done

- No PyPI publish — [#5](https://github.com/Latka-Industries/tet-py/issues/5)
- Phase 3 **convert** extras (`tet.convert`) — [#10](https://github.com/Latka-Industries/tet-py/issues/10)
- CI sibling **tetration** checkout — [#6](https://github.com/Latka-Industries/tet-py/issues/6)
- Phase 2 tail: preview ndarray ([#7](https://github.com/Latka-Industries/tet-py/issues/7)), `read_numpy` preflight ([#9](https://github.com/Latka-Industries/tet-py/issues/9)), integer write dtypes ([#8](https://github.com/Latka-Industries/tet-py/issues/8))

## GitHub tracking (tet-py)

| Issue | Topic |
| ----- | ----- |
| [#4](https://github.com/Latka-Industries/tet-py/issues/4) | Docs hygiene (this file + README) |
| [#5](https://github.com/Latka-Industries/tet-py/issues/5) | PyPI + cross-platform wheels (Phase 4) |
| [#6](https://github.com/Latka-Industries/tet-py/issues/6) | CI fixtures without sibling checkout |
| [#7](https://github.com/Latka-Industries/tet-py/issues/7) | Preview ndarray from `query_execute` |
| [#8](https://github.com/Latka-Industries/tet-py/issues/8) | `write_dataset` integer dtypes |
| [#9](https://github.com/Latka-Industries/tet-py/issues/9) | `read_numpy` memory budget preflight |
| [#10](https://github.com/Latka-Industries/tet-py/issues/10) | `tet.convert` (Phase 3) |
| [#11](https://github.com/Latka-Industries/tet-py/issues/11) | Zero-copy mmap → NumPy (P2) |

**tetration (upstream):** [#18](https://github.com/Latka-Industries/tetration/issues/18) docs link to tet-py · [#19](https://github.com/Latka-Industries/tetration/issues/19) selection preflight · [#20](https://github.com/Latka-Industries/tetration/issues/20) f16/u32/u64 export · [#21](https://github.com/Latka-Industries/tetration/issues/21) crates.io 0.1.9+ for wheels.

## Dev commands (uv)

```bash
cd ~/Code/LatkaIndustries/tet-py   # or ~/Code/tet-py
uv sync --extra dev
uv run maturin develop          # after Rust changes
uv run pytest -q
uv run mypy python/tet
uv run python -c "import tet; print(tet.__version__, tet.core_version())"
```

Layout: `python/tet/` = facade (`_numpy`, `_spill`, `_transform`, …); `native/src/lib.rs` = PyO3; tests expect sibling `tetration` fixtures.

---

## Phases (recommended order)

### Phase 0 — Repo hygiene (P0)

- [x] Push to `github.com/Latka-Industries/tet-py`
- [x] CI: `.github/workflows/ci.yml` (checkout sibling **tetration**, `uv sync` → `maturin develop` → `pytest`)
- [ ] Register **`tet-py`** on PyPI (empty/meta release optional) to hold the name — [#5](https://github.com/Latka-Industries/tet-py/issues/5)
- [x] `LICENSE-MIT` + `LICENSE-APACHE` (dual license, match tetration)
- [x] README links tetration Phase 11, query fixtures, query_engine docs

### Phase 1 — Read / query UX (P0)

Goal: parity with common `tet query -t … -x` embedder paths without hand-rolled JSON everywhere.

- [x] `query()` → return **`dict`** (parsed from Rust JSON in Python facade)
- [x] `query_execute(doc, device=...)` sets `execution.device` (preview still via raw doc / future Rust knob)
- [x] `info()` / `summary()` → **`dict`** (parsed `summary_json()`; parity with `tet info --json`)
- [x] `plan_only(doc)` → plan without execution (`ExecuteQueryOptions::plan_only`)
- [x] `mean` / `sum` / `min` / `max` / `std` / `var` / `count` / `product` / `norm_l1` / `norm_l2` / `median` / `all_finite` / `any_nan` / `arg_min` / `arg_max` — helpers over `query()`; `axis=` by index or `dim_names`
- [x] `execute(..., raw=False)` default → [`QueryResult`](../python/tet/_query.py) (`.scalar` / `.reduced`); `raw=True` for full wire dict
- [x] `Dataset`, `iter_datasets()`, `dataset(0)` / `f["name"]` catalog access (Phase 1 hardening)
- [x] `quantile`, `histogram`, `covariance`, `correlation` helpers (object-shaped wire ops)
- [x] Selection slices — [`build_query`](../python/tet/_query_doc.py), `axis_slice`, `selection_slices`
- [x] Document query schema → README links `tetration/fixtures/queries/`
- [x] Errors: `tet.TetError`, `tet.CatalogError`; `OSError` on missing file

### Phase 2 — NumPy interchange (P1)

Goal: make **NumPy** the primary Python array surface for `.tet` (read and write). Add `numpy` as a dependency when these APIs land. Decode/materialize stays in Rust (`materialize_read_plan_*`, `TetWriterSession`); Python gets `ndarray` views or copies.

**Read (`.tet` → NumPy)** — dense export via PyO3 + tetration `embed_materialize`.

- [x] `read_numpy(dataset, selection=...)` (or `Dataset.to_numpy()`) → `numpy.ndarray` with catalog shape / dtype
- [x] `transform.to_numpy.*` → full transformed `numpy.ndarray` (wire `write: ram`; budget preflight)
- [x] `transform.to_spill.*` → spill `.bin` + :meth:`~tet.SpillTransformResult.to_numpy`
- [x] `transform.to_sidecar.*` → sidecar `.tet` + :meth:`~tet.SidecarTransformResult.to_numpy`
- [x] `read_spill` selection export + :meth:`~tet.SpillReadResult.to_numpy`
- [x] PyO3 wrapper over `materialize_read_plan_*` / `materialize_query_transform_ram`; copy into NumPy (v1)
- [ ] Optional: `query_execute(..., preview=N)` → `ndarray` for capped `execution.*_preview` samples — [#7](https://github.com/Latka-Industries/tet-py/issues/7)
- [x] Document RAM budget: `to_numpy` vs `to_spill`, `read_numpy` slice/spill notes ([`operations.md`](operations.md#memory-budget))
- [ ] `read_numpy` budget preflight (blocked on [tetration#19](https://github.com/Latka-Industries/tetration/issues/19)) — [#9](https://github.com/Latka-Industries/tet-py/issues/9)
- [ ] Integer `write_dataset` dtypes beyond f32/f64 — [#8](https://github.com/Latka-Industries/tet-py/issues/8)
- [ ] Defer zero-copy mmap → NumPy until copy path is stable (P2) — [#11](https://github.com/Latka-Industries/tet-py/issues/11)

**Write (NumPy → `.tet`)** — no Rust HDF5/NetCDF in wheels.

- [x] Expose `TetWriterSession` / commit path from tetration catalog API
- [x] `write_dataset(name, array: numpy.ndarray, chunk_shape=..., attrs=..., coords=...)`
- [x] Footer metadata + history events on commit

**Tests**

- [x] Roundtrip: `write_dataset` → `read_numpy` → `mean`/`sum` vs golden
- [x] Small fixture only in CI; skip/multi-GB guarded locally

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

- [ ] `tet.convert(src, dst.tet, ...)` orchestration — [#10](https://github.com/Latka-Industries/tet-py/issues/10) (formats: [tetration#17](https://github.com/Latka-Industries/tetration/issues/17))
- [ ] Defer **sparse MTX** unless explicit densify + size limits

Rust CLI **`tet convert`** remains the fast path for HDF5/NetCDF/Zarr on machines with native libs.

### Phase 4 — Release engineering (P1)

- [ ] Release: pin `tetration = "x.y.z"` from crates.io; remove `path = "../tetration"` in published `Cargo.toml` — [#5](https://github.com/Latka-Industries/tet-py/issues/5), [tetration#21](https://github.com/Latka-Industries/tetration/issues/21)
- [ ] Linux / macOS / Windows wheels via maturin-action — [#5](https://github.com/Latka-Industries/tet-py/issues/5)
- [ ] Version policy: `tet-py` version tracks compatible `tetration` minor (document in README) — [#5](https://github.com/Latka-Industries/tet-py/issues/5)
- [x] Plan: `numpy` dependency when Phase 2 read/write APIs land (see Phase 2)

### Phase 5 — Nice-to-have (P2)

- [x] Context manager: `with tet.open(path) as f:`
- [x] `tet.TetFile.open` classmethod
- [ ] Submodule `tet.convert`, `tet.query` for large API surface
- [ ] Jupyter / xarray accessor sketch (`ds.tet.write(...)` out of scope unless requested)
- [ ] Shared fixtures: git submodule `tetration` or vendor `fixtures/small/` for CI without sibling path — [#6](https://github.com/Latka-Industries/tet-py/issues/6)

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
| **Medium** | NumPy read/write + convert extras; embedders replace subprocess `tet`                          |
| **Long**   | Versioned wheels per platform; optional alignment with tetration **C ABI** for non-Python only |

**Success:** `pip install tet-py` → `import tet` → open `.tet` → query, **`read_numpy`** / **`read_spill`**, transform sinks (**`to_numpy`** / **`to_spill`** / **`to_sidecar`**), optional **`write_dataset`** / convert via extras — without Rust on the end-user machine.

**Non-goals for tet-py v0.x:** reimplement layout/query in Python; sparse native format; GPU in Python wheel by default; duplicate Rust `tet convert` codecs inside the extension.

---

## Open questions for maintainers

1. Pin **one** Python (3.11 vs 3.13) for release wheels? (CI already matrix-tests 3.11+)
2. Submodule **tetration** vs path sibling vs vendored fixtures — [#6](https://github.com/Latka-Industries/tet-py/issues/6)
3. Exception hierarchy depth vs plain `RuntimeError` from PyO3 today?

Resolved: Phase 2 three-sink NumPy interchange (ram/spill/sidecar) shipped on branch; PyPI timing tracked in [#5](https://github.com/Latka-Industries/tet-py/issues/5). Docs hygiene ([#4](https://github.com/Latka-Industries/tet-py/issues/4)) synced in June 2026 pass.

---

## Quick file map

| File                     | Purpose                                               |
| ------------------------ | ----------------------------------------------------- |
| `native/src/lib.rs`      | `PyTetFile`, `read_numpy`, `transform_to_numpy`, writer |
| `python/tet/_numpy.py`   | `read_numpy_array`, `read_spill_array`                |
| `python/tet/_spill.py`   | spill path resolve, `load_spill_array`, `SpillReadResult` |
| `python/tet/_transform.py` | `TransformOps` sinks (`to_numpy` / `to_spill` / `to_sidecar`) |
| `python/tet/__init__.py` | Public exports, `__version__`                         |
| `tests/test_numpy.py`    | ram / spill / sidecar interchange tests               |
| `README.md`              | User-facing quick start                               |

When behavior changes, update **README.md** and this file; keep **GitHub tracking** issue links in sync. Cross-repo docs: [tetration#18](https://github.com/Latka-Industries/tetration/issues/18).
