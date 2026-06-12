"""Construct query JSON documents (selection + one operation).

Wire shape matches the ``tet query`` CLI; see tetration ``docs/query_engine.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from tet._catalog import Dataset, axes_for_query, axes_wire
from tet._query import REDUCTION_OPS, reduction_doc

# Single-op keys accepted on the query document wire (subset; see tetration query_engine.md).
QUERY_OP_KEYS: frozenset[str] = frozenset(REDUCTION_OPS) | frozenset(
    ("quantile", "histogram", "covariance", "correlation", "transform")
)

TRANSFORM_METHODS: frozenset[str] = frozenset(
    ("zscore", "minmax", "l1", "l2", "center", "scale", "log1p", "sqrt", "softmax")
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
    """Build one per-axis slice dict for a query ``selection`` array.

    Parameters
    ----------
    start, stop, step : int, optional
        Half-open interval ``[start, stop)`` on this axis; ``step`` defaults to 1 in the engine.
    start_label, stop_label : str, optional
        Coordinate labels resolved at plan time when footer coords exist.

    Returns
    -------
    dict
        Slice object for one dimension (omit keys for “full extent” on that axis).
    """
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
    """Build a ``selection`` array for a query document.

    Parameters
    ----------
    *slices : mapping
        One slice dict per dimension (use :func:`axis_slice`); order is axis 0, 1, …

    Returns
    -------
    list[dict]
        Wire ``selection`` list passed to :func:`build_query`.
    """
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


def build_selection_query(
    dataset: str,
    *,
    selection: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a selection-only query document (materialize / ``read_numpy``).

    Parameters
    ----------
    dataset : str
        Target dataset name.
    selection : sequence of mapping, optional
        Per-axis slices; omit for the full tensor.

    Returns
    -------
    dict
        Wire document with ``"dataset"`` and optional ``"selection"`` only.
    """
    doc: dict[str, Any] = {"dataset": dataset}
    if selection is not None:
        doc["selection"] = [dict(s) for s in selection]
    return doc


def build_query(
    dataset: str,
    *,
    selection: Sequence[Mapping[str, Any]] | None = None,
    **op: Any,
) -> dict[str, Any]:
    """Build a one-operation query document.

    Parameters
    ----------
    dataset : str
        Target dataset name (not validated until execute).
    selection : sequence of mapping, optional
        Per-axis slices; see :func:`selection_slices`.
    **op
        Exactly one operation keyword argument, e.g. ``mean=[]``, ``quantile={...}``.

    Returns
    -------
    dict
        Wire document with ``"dataset"``, optional ``"selection"``, and one op key.

    Raises
    ------
    ValueError
        If zero or more than one op keyword is passed, or op name is unknown.
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
    """Body for ``"quantile": { "q": …, "axis": … }``."""
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
    """Body for ``"histogram": { "bins": …, optional "min"/"max", "axis": … }``."""
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


def null_count_op(
    dataset: Dataset,
    axes: Sequence[int | str] | None = None,
    *,
    axis: int | str | None = None,
    fill: float | None = None,
    path: str | None = None,
) -> list[int | str] | dict[str, Any]:
    """Body for ``"null_count": []`` or ``{ "fill": …, "axis": … }``."""
    if axis is not None and axes is not None:
        raise TypeError("pass only one of axes= or axis=")
    merged = axes_for_query(axes if axis is None else [axis])
    if fill is None and not merged:
        return []
    body: dict[str, Any] = {}
    if fill is not None:
        body["fill"] = fill
    body.update(wire_axis_fields(dataset, merged, path=path))
    return body


def transform_op(
    dataset: Dataset,
    method: str,
    axes: Sequence[int | str] | None = None,
    *,
    axis: int | str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Body for ``"transform": { "method": …, optional "axis" / "axes" }``."""
    if method not in TRANSFORM_METHODS:
        raise ValueError(
            f"unknown transform method {method!r} "
            f"(expected one of {sorted(TRANSFORM_METHODS)})"
        )
    body: dict[str, Any] = {"method": method}
    body.update(wire_axis_fields(dataset, axes, axis=axis, path=path))
    return body


def op_key_from_doc(doc: dict[str, Any]) -> str | None:
    """Return the sole operation key on ``doc``, or ``None`` if zero or many."""
    ops = [k for k in doc if k in QUERY_OP_KEYS and k not in _QUERY_DOC_META]
    if len(ops) == 1:
        return ops[0]
    return None
