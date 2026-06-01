# Query operations

`tet` exposes one operation per query document. Prefer **methods on** [`TetFile`](../python/tet/_file.py) (`f.mean(...)`, `f.quantile(...)`, `f.transform.to_numpy.zscore(...)`, …); use [`build_query`](../python/tet/_query_doc.py) when you need the wire dict explicitly.

Fixtures in examples (sibling **tetration** repo):

| File                            | Dataset         | Shape      | Notes                             |
| ------------------------------- | --------------- | ---------- | --------------------------------- |
| `fixtures/small/tet/large.tet`  | `"a"`           | `(34, 64)` | Reductions, quantile, histogram   |
| `fixtures/small/tet/sample.tet` | `"temperature"` | `(2, 3)`   | Covariance / correlation (rank 2) |

```python
import tet
from tet import axis_slice, build_query, selection_slices

path = "../tetration/fixtures/small/tet/large.tet"
f = tet.open(path)
```

## Axis rules

**Every list-style reduction** (`mean`, `sum`, `min`, `max`, `count`, `std`, `var`, `product`, `norm_l1`, `norm_l2`, `median`, `all_finite`, `any_nan`, `any_inf`, `arg_min`, `arg_max`, `nan_count`, `inf_count`, `null_count`, `nan_mean`, `nan_std`) uses the same `axis` / `axes` arguments. Examples below use `mean`; swap in `std`, `count`, etc. as needed.

| Call                       | Effect                                                                   |
| -------------------------- | ------------------------------------------------------------------------ |
| `f.mean("a")`              | Reduce **all** axes → scalar                                             |
| `f.mean("a", axis=0)`      | Reduce axis 0 → [`QueryResult`](../python/tet/_query.py) with `.reduced` |
| `f.mean("a", axes=[0, 1])` | Reduce multiple axes                                                     |
| `f.mean("a", axis="time")` | Resolve name from footer `dim_names` when present                        |

**`axis` vs `axes`**

| Parameter | Type                           | Use when                                       |
| --------- | ------------------------------ | ---------------------------------------------- |
| `axis`    | `int` or `str` (one dimension) | Reducing a **single** axis — convenience alias |
| `axes`    | sequence of `int` or `str`     | Reducing **one or more** axes (pass a list)    |

You must pass **only one** of them, not both. Internally, `axis=1` becomes `axes=[1]` on the wire.

```python
f.std("a", axis=1)                    # same as f.std("a", axes=[1])
f.std("a", axes=[0, 1])               # reduce two axes at once — needs axes=
```

Partial reduction example (works for `std` / `var` / `count` the same way):

```python
# shape (34, 64) — reduce axis 0, leave axis 1
r = f.std("a", axis=0)
r.reduced                             # length 64, one std per column
```

## Return shape: `raw` and `QueryResult`

| `raw`             | Full reduction                                 | Partial reduction (one or more axes left)               |
| ----------------- | ---------------------------------------------- | ------------------------------------------------------- |
| `False` (default) | Python `float` / `int` / `bool`                | `QueryResult` — use `.scalar`, `.reduced`, `.matrix`, … |
| `True`            | Full wire `dict` (same as `tet query -x` JSON) | Full wire `dict`                                        |

```python
scalar = f.mean("a")                    # float
partial = f.max("a", axis=0)            # QueryResult
partial.reduced                         # list along remaining axes

wire = f.mean("a", raw=True)            # dict with execution + operation_* fields
```

Use **`f.execute(doc)`** (or `f.query(doc)`) for generic documents; default is `QueryResult` when `raw=False`.

```python
r = f.execute({"dataset": "a", "mean": []})
r.scalar                                # same as f.mean("a")

f.execute({"dataset": "a"}, plan=True)  # plan only — no op keys
f.execute(doc, device="cpu")            # sets execution.device
```

---

## List-style reductions

Wire form: `"<op>": [<axis indices>]` — `[]` means reduce all axes.

| Method       | Result (full reduce) | Notes                                                   |
| ------------ | -------------------- | ------------------------------------------------------- |
| `mean`       | `float`              | Arithmetic mean                                         |
| `sum`        | `float`              | Sum                                                     |
| `min`        | `float`              | Minimum                                                 |
| `max`        | `float`              | Maximum                                                 |
| `count`      | `int`                | Element count (wire op); see `numel` below              |
| `numel`      | `int`                | Same as `count` — Python alias (wire still `"count"`)   |
| `std`        | `float`              | Population std, `ddof=0`; supports `axis` / `axes`      |
| `var`        | `float`              | Population variance, `ddof=0`; supports `axis` / `axes` |
| `product`    | `float`              | Product of elements                                     |
| `norm_l1`    | `float`              | L1 norm                                                 |
| `norm_l2`    | `float`              | √(sum of squares)                                       |
| `median`     | `float`              | Median (materializes selection)                         |
| `all_finite` | `bool`               | All elements finite                                     |
| `any_nan`    | `bool`               | Any NaN present                                         |
| `arg_min`    | `int`                | Flat index of minimum                                   |
| `arg_max`    | `int`                | Flat index of maximum                                   |

### `mean`, `sum`, `min`, `max`

```python
f.mean("a")
f.sum("a")
f.min("a")
f.max("a")

f.max("a", axis=0)                      # partial → QueryResult
f.min("a", axis=1)                      # reduce axis 1 only
```

Wire equivalents:

```python
f.execute({"dataset": "a", "mean": []})
f.execute({"dataset": "a", "max": [0]})   # reduce axis 0
```

### `count`, `std`, `var`, `product`

#### `count` / `numel` — element count, not axis length

`count` is the wire / CLI op name. **`numel`** is a Python alias with the same behavior (NumPy `size` semantics).

`count` / `numel` is how many **values** are aggregated in the selection — like NumPy **`arr.size`**, not **`len(arr)`** (first-axis length only).

| Call                             | Result (shape `(34, 64)`, full dataset)                                 |
| -------------------------------- | ----------------------------------------------------------------------- |
| `f.numel("a")` or `f.count("a")` | `2176` (= 34 × 64, all axes reduced)                                    |
| `f.numel("a", axis=0)`           | length-`64` vector; each entry is `34` (elements combined along axis 0) |

Counts **every** stored element (including NaN/inf). For NaN / inf / fill-missing tallies use :meth:`~tet.TetFile.nan_count`, :meth:`~tet.TetFile.inf_count`, and :meth:`~tet.TetFile.null_count`.

```python
n = f.numel("a")                      # total elements in selection
n = f.numel("a", axis=0)              # partial → QueryResult.reduced

build_query("a", count=[])            # wire name is still "count"
```

#### `std` and `var` — same axes as `mean`

`std` and `var` follow the same **`axis` / `axes`** rules as `mean` and `sum`. They are **population** statistics (**`ddof=0`**), matching the tetration engine — not NumPy’s default sample `std`/`var` (`ddof=1`).

```python
s = f.std("a")                        # scalar over all axes
v = f.var("a", axis=0)                # partial variance along remaining axes
r = f.std("a", axis=1)                # or axes=[1] — equivalent
r.reduced
```

#### `product`

```python
p = f.product("a")
p = f.product("a", axis=0)
```

### `norm_l1`, `norm_l2`

```python
f.norm_l1("a")
f.norm_l2("a")
```

### `median`

```python
f.median("a")
f.median("a", axis=0)                   # partial median vector
```

### `all_finite`, `any_nan`

```python
ok = f.all_finite("a")
has_nan = f.any_nan("a")
```

### `arg_min`, `arg_max`

Flat index over the **selected** region (all axes reduced):

```python
i = f.arg_min("a")
j = f.arg_max("a")
```

---

## `quantile`

Wire: `"quantile": { "q": <float>, optional "axis" / "axes" }`.

```python
f.quantile("a", 0.5)                    # median
f.quantile("a", 0.25, axis=0)

build_query("a", quantile={"q": 0.9, "axis": 1})
f.execute(build_query("a", quantile={"q": 0.5}))
```

With a subregion:

```python
sel = selection_slices(axis_slice(0, 8), axis_slice(0, 8))
f.quantile("a", 0.5, selection=sel)
```

---

## `histogram`

Wire: `"histogram": { "bins": <int>, optional "min", "max", "axis" / "axes" }`.

```python
r = f.histogram("a", bins=8)
r.histogram_counts
r.histogram_edges

f.histogram("a", bins=16, min=0.0, max=100.0, axis=0)
```

```python
doc = build_query("a", histogram={"bins": 4, "axis": 0})
r = f.execute(doc)
```

---

## `covariance` and `correlation`

Rank-**2** datasets only. **`axis`** is the **observation** dimension; variables lie on the other axis.

Use `sample.tet` / `"temperature"` in examples:

```python
f = tet.open("../tetration/fixtures/small/tet/sample.tet")

r = f.covariance("temperature", axis=1)
r.matrix_order                        # e.g. 2
r.matrix                              # row-major order×order

r = f.correlation("temperature", axis=1)
r.matrix
```

Wire:

```python
f.execute({"dataset": "temperature", "covariance": 1})
f.execute({"dataset": "temperature", "correlation": {"axis": 1}})
```

---

## QC counts and NaN-skipping stats

Wire form matches list-style reductions unless noted.

| Method       | Result (full reduce) | Notes                                    |
| ------------ | -------------------- | ---------------------------------------- |
| `nan_count`  | `int` / `float`      | Count of NaN elements                    |
| `inf_count`  | `int` / `float`      | Count of ±infinity elements              |
| `null_count` | `int` / `float`      | Fill-missing count; optional `fill=`     |
| `any_inf`    | `bool`               | True when any ±infinity present          |
| `nan_mean`   | `float`              | Mean skipping NaNs (same axes as `mean`) |
| `nan_std`    | `float`              | Population std skipping NaNs (`ddof=0`)  |

```python
f.nan_count("a")
f.inf_count("a")
f.null_count("a")                     # fill from footer attrs when present
f.null_count("a", fill=99.0)

f.any_inf("a")
f.nan_mean("a")
f.nan_std("a", axis=0)
```

On clean finite data, `nan_mean` / `nan_std` match `mean` / `std`.

---

## `transform`

Sink-first API only: **`f.transform.to_<sink>.<method>(dataset, ...)`**. For arbitrary wire docs use :func:`build_query` + :meth:`~tet.TetFile.execute`.

| Sink         | Example                                       | Returns                                          |
| ------------ | --------------------------------------------- | ------------------------------------------------ |
| `to_numpy`   | `f.transform.to_numpy.zscore("a")`            | `numpy.ndarray` (full logical selection)         |
| `to_spill`   | `f.transform.to_spill.softmax("a", path="…")` | :class:`~tet.SpillTransformResult` (`.path`)     |
| `to_sidecar` | `f.transform.to_sidecar.center("a")`          | :class:`~tet.SidecarTransformResult` (`.open()`) |

Methods on each sink: `zscore`, `minmax`, `l1`, `l2`, `center`, `scale`, `log1p`, `sqrt`, `softmax`.

```python
arr = f.transform.to_numpy.zscore("a")   # numpy.ndarray; fails if selection > memory_budget_bytes
arr.mean()

s = f.transform.to_spill.softmax("a", path="/tmp/out.bin")  # when RAM export would fail
s.path                               # Path to raw LE tensor bytes
```

See [Memory budget](#memory-budget) for `to_numpy` vs `to_spill`.

```python
# low-level wire (any write token, including switch):
f.execute(build_query("a", transform={"method": "zscore"}))
```

Returns :class:`~tet.QueryResult` (or wire `dict` when `raw=True`) for generic execute; not a single scalar.

---

## `read_numpy`

Materialize a dataset selection (no reduction) into RAM as `numpy.ndarray`.

```python
arr = f.read_numpy("temperature")
arr = f.dataset("temperature").to_numpy(f)

# sub-selection (preferred for large tensors)
arr = f.read_numpy("a", selection=tet.selection_slices(tet.axis_slice(0, 100)))
```

Integer dtypes (`i32`, `u8`, …) are supported where the engine materializes them; `f16` / `u32` / `u64` are not yet exported.

**Memory:** `read_numpy` decodes the **full logical selection** into a new array (copy). There is no `memory_budget_bytes` preflight on this path today — if the selection is huge, slice it or use the wire **`spill`** export via :meth:`~tet.TetFile.execute` (see [Memory budget](#memory-budget)). See also [tetration query engine — memory budget](https://github.com/Latka-Industries/tetration/blob/main/docs/query_engine.md#memory-budget-and-execution-strategies).

---

## Memory budget

The query engine resolves a dense-RAM cap (**`memory_budget_bytes`**) the same way as `tet query -x`. Precedence (first match wins):

1. Query `execution.memory_budget_bytes`
2. Per-file chunk-index header `memory_budget_bytes` (footer / `catalog.file_execution`)
3. Query `execution.memory_budget_percent` (or `memory_budget_percent_bps` on the wire)
4. Per-file header percent (**0** → engine default **25%** of detected host RAM)

Resolved values appear on execution responses as `execution.memory_budget_bytes`, `execution.logical_selection_bytes`, and related fields when you :meth:`~tet.TetFile.execute` with `raw=True`.

### `transform.to_numpy` (wire `write: ram`)

**Preflight:** if the logical selection size exceeds the resolved budget, the engine returns :class:`~tet.TetError` (validation) — same rule as CLI `write: ram`. Message shape:

```text
logical selection (N elements, B bytes) exceeds memory_budget_bytes (M);
use `write`: `switch` or `spill`, or raise execution.memory_budget_bytes
```

In Python, use **`to_spill`** instead of `switch`:

```python
# in-RAM (fails when selection > budget)
arr = f.transform.to_numpy.zscore("a")

# dense row-major LE bytes on disk (no RAM cap on output size; path allowlist applies)
s = f.transform.to_spill.zscore("a", path="/tmp/a_zscore.f32.bin")
arr = np.fromfile(s.path, dtype=np.float32).reshape(s.shape)
```

`to_numpy` only supports **`f32` / `f64`** transforms. Spill files are dtype-native little-endian; use `s.shape` and the source dataset dtype tag to pick `np.float32` vs `np.float64`.

High-level transform methods accept `device=` only. To raise the budget for one call, build the wire document yourself (`execution.memory_budget_bytes` or `memory_budget_percent`) and use the native materialize path, or execute with `raw=True` to inspect `execution.memory_budget_bytes` after a plan.

### `read_numpy`

No budget preflight — the full selection is materialized. For tensors larger than RAM, **slice** the selection, or spill at the wire layer (top-level `"spill": "path"` on a selection-only query — no `transform` key) and load with NumPy from the file.

### Reductions vs dense export

| API                               | Over budget                                                    |
| --------------------------------- | -------------------------------------------------------------- |
| `mean`, `sum`, streaming folds    | Chunk streaming / GPU fold — no full dense buffer              |
| `median`, `quantile`, `histogram` | Engine `temp_spill_materialize` or refuse — tier-C materialize |
| `transform.to_numpy`              | **Refuse** (use `to_spill`)                                    |
| `read_numpy`                      | **No preflight** (slice or wire spill)                         |

---

## `write_dataset` / :class:`~tet.TetWriter`

Write row-major **`float32` / `float64`** NumPy arrays to a new or appended `.tet` file via tetration :class:`TetWriterSession`.

```python
import numpy as np
import tet

arr = np.arange(6, dtype=np.float32).reshape(2, 3)

# one-shot
tet.write_dataset("out.tet", "temperature", arr, chunk_shape=(2, 3))

# session (multiple datasets, history, footer metadata)
w = tet.TetWriter.create("out.tet")
w.push_history_event("write", "notebook")
w.write_dataset(
    "temperature",
    arr,
    chunk_shape=(2, 3),
    attrs={"units": "K"},
    dim_names=("row", "col"),
    coords={"row": ("r0", "r1")},
)
w.commit()

# append another dataset to an existing file
w = tet.TetWriter.open_append("out.tet")
w.write_dataset("humidity", arr)
w.commit()
```

| Parameter     | Notes                                                   |
| ------------- | ------------------------------------------------------- |
| `chunk_shape` | Tile geometry; defaults to `array.shape` (single chunk) |
| `attrs`       | `str → str` footer CF-style attributes                  |
| `dim_names`   | One name per axis (optional)                            |
| `coords`      | Axis name → coordinate label list (optional)            |

On :meth:`TetWriter.commit`, footer `tool` is set to `tet-py` and `library_version` to the extension version. Default history row `write` is added when none were pushed (same as Rust session).

Roundtrip: :func:`write_dataset` → :meth:`TetFile.read_numpy` → reductions.

---

## Selection and `build_query`

Slice one dimension with [`axis_slice`](../python/tet/_query_doc.py) (`start` inclusive, `stop` exclusive):

```python
axis_slice(0, 4)                        # rows 0..3
axis_slice(0, 4, 2)                     # step 2
axis_slice(start_label="2020", stop_label="2024")  # when footer coords exist

sel = selection_slices(
    axis_slice(0, 4),
    axis_slice(0, 4),
)
```

Subregion + reduction:

```python
doc = build_query("a", selection=sel, mean=[])
f.execute(doc)                          # scalar over 4×4 corner

f.execute(build_query("a", selection=sel, sum=[]))
```

List op values are axis indices to reduce (after selection is applied):

```python
build_query("a", mean=[])               # all axes
build_query("a", mean=[0])              # axis 0 only
build_query("a", selection=sel, min=[1])
```

---

## Entry points (same schema as CLI)

| Python                                 | CLI parity                              |
| -------------------------------------- | --------------------------------------- |
| `f.query(doc)`                         | `tet query -x` (execute, JSON string)   |
| `f.plan_only(doc)`                     | `tet query` without `-x`                |
| `f.query_execute(doc, device=...)`     | execute with `execution.device`         |
| `f.execute(doc, plan=True)`            | plan only                               |
| `f.read_numpy(dataset, selection=...)` | Materialize selection → `numpy.ndarray` |
| `tet.write_dataset(path, name, array)` | Create one-dataset `.tet` from NumPy    |
| `tet.TetWriter.create(path)`           | Buffered multi-dataset writer           |

`doc` may be a **`dict`** or **JSON string**.

```python
import json
f.query('{"dataset": "a", "mean": []}')
f.query({"dataset": "a", "mean": []})
```

---

## Errors and IDE hints

| Exception             | When                                           |
| --------------------- | ---------------------------------------------- |
| `UnknownDatasetError` | Bad dataset name on `f.dataset()` / reductions |
| `UnknownAxisError`    | Bad index or `dim_names` label                 |
| `TetError`            | Query parse, validation, or execution          |
| `CatalogError`        | File layout / catalog read / write validation  |

Optional per-file stubs:

```python
stub = tet.typing_stub(path)            # or f.typing_stub() on an open file
# save as mydata_tet.pyi for Literal dataset names in the IDE
```

---

## Quick reference: all `TetFile` op methods

```text
mean, sum, min, max, count, numel, std, var, product,
norm_l1, norm_l2, median, all_finite, any_nan, any_inf,
arg_min, arg_max,
nan_count, inf_count, null_count, nan_mean, nan_std,
quantile, histogram, covariance, correlation,
transform.to_numpy.*, transform.to_spill.*, transform.to_sidecar.*,
read_numpy
```

Write: `tet.TetWriter`, `tet.write_dataset`.

Plus: `execute`, `query`, `query_execute`, `plan_only`, `dataset`, `summary`, `info`.

---

## Planned improvements (tracking)

| Topic | Issue |
| ----- | ----- |
| `read_numpy` memory budget preflight | [tet-py#9](https://github.com/Latka-Industries/tet-py/issues/9), [tetration#19](https://github.com/Latka-Industries/tetration/issues/19) |
| `write_dataset` integer dtypes | [tet-py#8](https://github.com/Latka-Industries/tet-py/issues/8) |
| `f16` / `u32` / `u64` read export | [tetration#20](https://github.com/Latka-Industries/tetration/issues/20) |
| Preview ndarray API | [tet-py#7](https://github.com/Latka-Industries/tet-py/issues/7) |
| Full index | [HANDOFF.md — GitHub tracking](HANDOFF.md#github-tracking-tet-py) |
