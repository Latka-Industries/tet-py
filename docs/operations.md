# Query operations

`tet` exposes one operation per query document. Prefer **methods on** [`TetFile`](../python/tet/_file.py) (`f.mean(...)`, `f.quantile(...)`, …); use [`build_query`](../python/tet/_query_doc.py) when you need the wire dict explicitly.

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

**Every list-style reduction** (`mean`, `sum`, `min`, `max`, `count`, `std`, `var`, `product`, `norm_l1`, `norm_l2`, `median`, `all_finite`, `any_nan`, `arg_min`, `arg_max`) uses the same `axis` / `axes` arguments. Examples below use `mean`; swap in `std`, `count`, etc. as needed.

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

| Method       | Result (full reduce) | Notes                                                     |
| ------------ | -------------------- | --------------------------------------------------------- |
| `mean`       | `float`              | Arithmetic mean                                           |
| `sum`        | `float`              | Sum                                                       |
| `min`        | `float`              | Minimum                                                   |
| `max`        | `float`              | Maximum                                                   |
| `count`      | `int`                | Number of elements (see below — not `len` along one axis) |
| `std`        | `float`              | Population std, `ddof=0`; supports `axis` / `axes`        |
| `var`        | `float`              | Population variance, `ddof=0`; supports `axis` / `axes`   |
| `product`    | `float`              | Product of elements                                       |
| `norm_l1`    | `float`              | L1 norm                                                   |
| `norm_l2`    | `float`              | √(sum of squares)                                         |
| `median`     | `float`              | Median (materializes selection)                           |
| `all_finite` | `bool`               | All elements finite                                       |
| `any_nan`    | `bool`               | Any NaN present                                           |
| `arg_min`    | `int`                | Flat index of minimum                                     |
| `arg_max`    | `int`                | Flat index of maximum                                     |

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

#### `count` — element count, not axis length

`count` is how many **values** are aggregated in the selection — like NumPy **`arr.size`**, not **`len(arr)`** (first-axis length only).

| Call                   | Result (shape `(34, 64)`, full dataset)                                 |
| ---------------------- | ----------------------------------------------------------------------- |
| `f.count("a")`         | `2176` (= 34 × 64, all axes reduced)                                    |
| `f.count("a", axis=0)` | length-`64` vector; each entry is `34` (elements combined along axis 0) |

Counts **every** stored element (including NaN/inf). For “how many NaNs?” use wire op `nan_count` (not a `f.*` helper yet).

```python
n = f.count("a")                      # total elements in selection
n = f.count("a", axis=0)              # partial → QueryResult.reduced
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

| Python                             | CLI parity                            |
| ---------------------------------- | ------------------------------------- |
| `f.query(doc)`                     | `tet query -x` (execute, JSON string) |
| `f.plan_only(doc)`                 | `tet query` without `-x`              |
| `f.query_execute(doc, device=...)` | execute with `execution.device`       |
| `f.execute(doc, plan=True)`        | plan only                             |
| `f.execute(doc, raw=True)`         | full wire dict                        |

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
| `CatalogError`        | File layout / catalog read                     |

Optional per-file stubs:

```python
stub = tet.typing_stub(path)            # or f.typing_stub() on an open file
# save as mydata_tet.pyi for Literal dataset names in the IDE
```

---

## Quick reference: all `TetFile` op methods

```text
mean, sum, min, max, count, std, var, product,
norm_l1, norm_l2, median, all_finite, any_nan, arg_min, arg_max,
quantile, histogram, covariance, correlation
```

Plus: `execute`, `query`, `query_execute`, `plan_only`, `dataset`, `summary`, `info`.
