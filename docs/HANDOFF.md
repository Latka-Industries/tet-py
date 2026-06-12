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
- [x] Phase 2 NumPy **sidecar** path: `transform.to_sidecar.*`, `SidecarTransformResult.to_numpy` / `.open`
- [x] Phase 4 **PyPI** (0.1.0 shipped; **0.1.1** preview API on `main`, tag pending): abi3 wheels, `publish.yml`, `tetration = "0.1.9"` from crates.io only
- [x] CI fixtures vendored in `tests/fixtures/` (no sibling tetration checkout)
- [x] Package subpackages: `_core`, `_query`, `_io`, `_transform`; public `import tet` unchanged
- [x] `uv sync --extra dev`, PyO3 **0.28**, **50** pytest tests, mypy on `python/tet`
- [x] `tet.__version__` (0.1.1 on `main`), `tet.core_version()` (linked tetration 0.1.9)
- [x] Query preview: `preview=N` on reducers / `query_execute`; `QueryResult.preview` (THI-24)
- [x] Docs: [tetration-docs/python](https://latka-industries.github.io/tetration-docs/python/), [CHANGELOG.md](../CHANGELOG.md)

### Not done

- Phase 3 **convert** extras (`tet.convert`) — [#10](https://github.com/Latka-Industries/tet-py/issues/10) / THI-22
- Phase 2 tail: `read_numpy` preflight ([#9](https://github.com/Latka-Industries/tet-py/issues/9) / THI-23), integer write dtypes ([#8](https://github.com/Latka-Industries/tet-py/issues/8) / THI-26)
- Zero-copy mmap → NumPy ([#11](https://github.com/Latka-Industries/tet-py/issues/11) / THI-21)

## Issue tracking

**Linear** is canonical for Latka-Industries repos. GitHub issues #4–#11 were closed as duplicates; use Linear for status.

| Linear | Topic                                                          |
| ------ | -------------------------------------------------------------- |
| THI-22 | `tet.convert` orchestration (Phase 3)                          |
| THI-23 | `read_numpy` memory budget preflight                           |
| THI-26 | `write_dataset` integer dtypes                                 |
| THI-21 | Zero-copy mmap → NumPy                                         |
| THI-60 | Upstream: publish `f64_row_major` / `row_major` in catalog API |
| THI-61 | Upstream: unify transform sidecar with `TetWriterSession`      |

**Shipped (Linear done):** THI-20 (PyPI), THI-24 (preview ndarray), THI-25 (fixtures), THI-27 (docs).

**tetration (upstream):** [#19](https://github.com/Latka-Industries/tetration/issues/19) selection preflight · [#20](https://github.com/Latka-Industries/tetration/issues/20) f16/u32/u64 export

## Dev commands (uv)

```bash
cd ~/Code/LatkaIndustries/tet-py   # or ~/Code/tet-py
uv sync --extra dev
uv run maturin develop          # after Rust changes
uv run pytest -q
uv run mypy python/tet
uv run python -c "import tet; print(tet.__version__, tet.core_version())"
```

Layout: `python/tet/` subpackages (`_core`, `_query`, `_io`, `_transform`); `native/src/lib.rs` = PyO3; tests use `tests/fixtures/*.tet`.

---

## Phases (recommended order)

### Phase 0 — Repo hygiene (P0)

- [x] Push to `github.com/Latka-Industries/tet-py`
- [x] CI: `.github/workflows/ci.yml` (checkout sibling **tetration**, `uv sync` → `maturin develop` → `pytest`)
- [x] **`tet-py`** on [PyPI](https://pypi.org/project/tet-py/) (0.1.0)
- [x] `LICENSE-MIT` + `LICENSE-APACHE` (dual license, match tetration)
- [x] README links tetration Phase 11, query fixtures, query_engine docs

### Phase 1 — Read / query UX (P0)

Goal: parity with common `tet query -t … -x` embedder paths without hand-rolled JSON everywhere.

- [x] `query()` → return **`dict`** (parsed from Rust JSON in Python facade)
- [x] `query_execute(doc, device=..., preview=N)` sets `execution.device` and preview cap; `QueryResult.preview`
- [x] `info()` / `summary()` → **`dict`** (parsed `summary_json()`; parity with `tet info --json`)
- [x] `plan_only(doc)` → plan without execution (`ExecuteQueryOptions::plan_only`)
- [x] `mean` / `sum` / `min` / `max` / `std` / `var` / `count` / `product` / `norm_l1` / `norm_l2` / `median` / `all_finite` / `any_nan` / `arg_min` / `arg_max` — helpers over `query()`; `axis=` by index or `dim_names`
- [x] `execute(..., raw=False)` default → [`QueryResult`](../python/tet/_query/result.py) (`.scalar` / `.reduced`); `raw=True` for full wire dict
- [x] `Dataset`, `iter_datasets()`, `dataset(0)` / `f["name"]` catalog access (Phase 1 hardening)
- [x] `quantile`, `histogram`, `covariance`, `correlation` helpers (object-shaped wire ops)
- [x] Selection slices — [`build_query`](../python/tet/_query/doc.py), `axis_slice`, `selection_slices`
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
- [x] `query_execute(..., preview=N)` → `QueryResult.preview` for capped `execution.*_preview` samples — [#7](https://github.com/Latka-Industries/tet-py/issues/7) / THI-24
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

- [x] Pin `tetration = "0.1.9"` from crates.io in published `native/Cargo.toml` (no path dep)
- [x] Linux / macOS / Windows wheels via maturin-action + `.github/workflows/publish.yml`
- [x] Version policy: `tet.__version__` (tet-py) vs `tet.core_version()` (tetration crate) — see README / CHANGELOG
- [x] `numpy>=2.0` dependency

### Phase 5 — Nice-to-have (P2)

- [x] Context manager: `with tet.open(path) as f:`
- [x] `tet.TetFile.open` classmethod
- [ ] Submodule `tet.convert`, `tet.query` for large API surface
- [ ] Jupyter / xarray accessor sketch (`ds.tet.write(...)` out of scope unless requested)
- [x] Vendored fixtures in `tests/fixtures/` for CI without sibling tetration checkout

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

| Horizon    | Target                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------- |
| **Short**  | PyPI 0.1.1 (preview) when tagged; Phase 2 tail (preflight), docs hygiene                  |
| **Medium** | `tet.convert` extras; upstream THI-60/61 writer/sidecar cleanup                             |
| **Long**   | Zero-copy read; object-store paths (tetration Phase 12); optional C ABI for non-Python only |

**Success (0.1.x):** `pip install tet-py` → `import tet` → open `.tet` → query (optional **`preview=N`**), **`read_numpy`** / **`read_spill`**, transform sinks (**`to_numpy`** / **`to_spill`** / **`to_sidecar`**), **`write_dataset`** — without Rust on the end-user machine. Convert extras still planned.

**Non-goals for tet-py v0.x:** reimplement layout/query in Python; sparse native format; GPU in Python wheel by default; duplicate Rust `tet convert` codecs inside the extension.

---

## Open questions for maintainers

1. Pin **one** Python (3.11 vs 3.13) for release wheels? → **abi3 `cp311+`** wheels ship; CI tests 3.11+.
2. Fixtures — **resolved:** vendored `tests/fixtures/`.
3. Exception hierarchy depth vs plain `RuntimeError` from PyO3 today?

Resolved: Phase 2 three-sink NumPy interchange; PyPI 0.1.0 (THI-20); preview API 0.1.1 (THI-24); docs site + CHANGELOG (June 2026).

---

## Quick file map

| File                           | Purpose                                                       |
| ------------------------------ | ------------------------------------------------------------- |
| `native/src/lib.rs`            | `PyTetFile`, materialize, `transform_to_numpy`, writer PyO3   |
| `python/tet/file.py`           | `TetFile` facade                                              |
| `python/tet/_io/numpy.py`      | `read_numpy_array`, `read_spill_array`                        |
| `python/tet/_io/spill.py`      | spill path resolve, `load_spill_array`, `SpillReadResult`     |
| `python/tet/_io/writer.py`     | `TetWriter`, `write_dataset`                                  |
| `python/tet/_transform/ops.py` | `TransformOps` sinks (`to_numpy` / `to_spill` / `to_sidecar`) |
| `python/tet/__init__.py`       | Public exports, `__version__`                                 |
| `python/tet/_query/preview.py` | `execution.*_preview` → NumPy                                   |
| `tests/test_preview.py`        | Preview cap + `QueryResult.preview`                             |
| `tests/test_numpy.py`          | ram / spill / sidecar interchange tests                       |
| `CHANGELOG.md`                 | Release notes                                                 |
| `README.md`                    | User-facing quick start                                       |

When behavior changes, update **README.md**, **CHANGELOG.md**, and this file. User docs: [tetration-docs/python](https://latka-industries.github.io/tetration-docs/python/).
