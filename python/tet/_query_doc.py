"""Build query documents: selection slices and wire op shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from tet._catalog import Dataset, axes_for_query, axes_wire
from tet._query import REDUCTION_OPS, reduction_doc

# Single-op keys accepted on the query document wire (subset; see tetration query_engine.md).
QUERY_OP_KEYS: frozenset[str] = frozenset(REDUCTION_OPS) | frozenset(
    ("quantile", "histogram", "covariance", "correlation")
)

_QUERY_DOC_META = frozenset(
    {"dataset", "layout_version", "selection", "execution", "output"}
)


def axis_slice(
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    *,
    start_label: str | None = None,
    stop_label: str | None = None,
) -> dict[str, Any]:
    """One per-axis half-open slice for ``selection`` (``start`` inclusive, ``stop`` exclusive)."""
    out: dict[str, Any] = {}
    if start is not None:
        out["start"] = start
    if stop is not None:
        out["stop"] = stop
    if step is not None:
        out["step"] = step
    if start_label is not None:
        out["start_label"] = start_label
    if stop_label is not None:
        out["stop_label"] = stop_label
    return out


def selection_slices(*slices: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build a ``selection`` array (one slice dict per dimension, in order)."""
    return [dict(s) for s in slices]


def full_selection(ndim: int) -> list[dict[str, Any]]:
    """Full extent on every axis (omit bounds → engine uses dataset shape)."""
    return [axis_slice() for _ in range(ndim)]


def wire_axis_fields(
    dataset: Dataset,
    axes: Sequence[int | str] | None = None,
    *,
    axis: int | str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """``axis`` / ``axes`` fields for object-shaped ops (``quantile``, ``histogram``, …)."""
    if axis is not None and axes is not None:
        raise TypeError("pass only one of axes= or axis=")
    merged = axes_for_query(axes if axis is None else [axis])
    if not merged:
        return {}
    wire = axes_wire(dataset, merged, path=path)
    if len(wire) == 1:
        return {"axis": wire[0]}
    return {"axes": wire}


def build_query(
    dataset: str,
    *,
    selection: Sequence[Mapping[str, Any]] | None = None,
    **op: Any,
) -> dict[str, Any]:
    """Build a one-op query document.

    List ops: ``build_query("a", mean=[])`` or ``build_query("a", mean=[0])``.

    Object ops: ``build_query("a", quantile={"q": 0.5, "axis": 0})``.

    With selection: ``build_query("a", selection=selection_slices(...), sum=[])``.
    """
    if len(op) != 1:
        raise ValueError("exactly one operation key required")
    key, value = next(iter(op.items()))
    if key not in QUERY_OP_KEYS:
        raise ValueError(f"unknown query op {key!r}")
    doc: dict[str, Any] = {"dataset": dataset, key: value}
    if selection is not None:
        doc["selection"] = [dict(s) for s in selection]
    return doc


def quantile_op(
    dataset: Dataset,
    q: float,
    axes: Sequence[int | str] | None = None,
    *,
    axis: int | str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"q": q}
    body.update(wire_axis_fields(dataset, axes, axis=axis, path=path))
    return body


def histogram_op(
    dataset: Dataset,
    bins: int,
    axes: Sequence[int | str] | None = None,
    *,
    axis: int | str | None = None,
    min: float | None = None,
    max: float | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"bins": bins}
    body.update(wire_axis_fields(dataset, axes, axis=axis, path=path))
    if min is not None:
        body["min"] = min
    if max is not None:
        body["max"] = max
    return body


def covariance_op(
    dataset: Dataset,
    axis: int | str,
    *,
    path: str | None = None,
) -> int | str | dict[str, Any]:
    """Observation axis for rank-2 covariance (wire: int axis or ``{"axis": …}``)."""
    fields = wire_axis_fields(dataset, None, axis=axis, path=path)
    if "axis" in fields:
        return cast(int | str, fields["axis"])
    if "axes" in fields:
        return fields
    raise ValueError("covariance requires axis=")


def correlation_op(
    dataset: Dataset,
    axis: int | str,
    *,
    path: str | None = None,
) -> dict[str, Any]:
    """Observation axis for rank-2 correlation."""
    fields = wire_axis_fields(dataset, None, axis=axis, path=path)
    if not fields:
        raise ValueError("correlation requires axis=")
    return fields


def op_key_from_doc(doc: dict[str, Any]) -> str | None:
    """Single operation key on the document, if any."""
    ops = [k for k in doc if k in QUERY_OP_KEYS and k not in _QUERY_DOC_META]
    if len(ops) == 1:
        return ops[0]
    return None
