"""Query execution results and mapping from wire op keys to ``execution`` fields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tet._core.errors import check_query_response
from tet._native import TetError
from tet._query.preview import preview_lists_from_execution, preview_ndarray_from_response

def _op_fields(
    name: str,
    *,
    scalar: str | None = None,
    reduced: str | None = None,
) -> tuple[str, str]:
    """Map wire op key → ``(operation_<scalar>, operation_reduced_<reduced>)``."""
    return (
        f"operation_{scalar if scalar is not None else name}",
        f"operation_reduced_{reduced if reduced is not None else name}",
    )


# List-style ops: wire key ``mean: [0]`` → operation_mean / operation_reduced_mean.
_REDUCTION_WIRE_KEYS = (
    "sum",
    "mean",
    "min",
    "max",
    "var",
    "std",
    "product",
    "norm_l1",
    "norm_l2",
    "all_finite",
    "any_nan",
    "median",
    "nan_count",
    "inf_count",
    "null_count",
    "nan_mean",
    "nan_std",
    "any_inf",
)

# Public map used by tests and advanced callers.
REDUCTION_OPS: dict[str, tuple[str, str]] = {
    key: _op_fields(key) for key in _REDUCTION_WIRE_KEYS
} | {
    "count": _op_fields("count", scalar="element_count", reduced="count"),
    "arg_min": _op_fields("arg_min", scalar="argmin_index", reduced="argmin"),
    "arg_max": _op_fields("arg_max", scalar="argmax_index", reduced="argmax"),
}

def reduction_doc(
    dataset: str, op: str, axes: Sequence[int | str]
) -> dict[str, Any]:
    """Build a minimal list-style reduction query document.

    Parameters
    ----------
    dataset : str
        Target dataset name.
    op : str
        Wire op key (e.g. ``"mean"``, ``"sum"``).
    axes : sequence of int or str
        Axis indices or names to reduce.

    Returns
    -------
    dict
        ``{"dataset": ..., op: [axis indices]}``.

    Raises
    ------
    ValueError
        If ``op`` is not in :data:`REDUCTION_OPS`.
    """
    if op not in REDUCTION_OPS:
        raise ValueError(f"unknown reduction op: {op!r}")
    return {"dataset": dataset, op: list(axes)}


# Scalar fields for ops not in REDUCTION_OPS map.
_EXTRA_SCALAR_FIELDS: dict[str, str] = {
    "quantile": "operation_quantile",
    "median": "operation_median",
}

_MATRIX_FIELDS: dict[str, tuple[str, str]] = {
    "covariance": ("operation_covariance", "operation_covariance_order"),
    "correlation": ("operation_correlation", "operation_correlation_order"),
}


@dataclass(frozen=True)
class QueryResult:
    """Parsed query response from :meth:`~tet.TetFile.execute` (``raw=False``).

    Attributes
    ----------
    scalar : float, int, bool, or None
        Full reduction result when all requested axes were reduced.
    reduced : list or None
        Partial reduction vector along remaining axes.
    matrix : list of list or None
        Covariance/correlation matrix when applicable.
    matrix_order : list of str or None
        Axis labels for matrix rows/columns.
    histogram_counts, histogram_edges : list or None
        Histogram op outputs.
    raw : dict
        Full wire JSON from the engine.

    See Also
    --------
    value : ``scalar`` if set, else ``reduced``.
    """

    accepted: bool
    message: str | None
    scalar: float | int | bool | None
    reduced: list[float] | list[int] | list[bool] | None
    reduced_shape: list[int] | None
    plan: dict[str, Any] | None
    execution: dict[str, Any] | None
    raw: dict[str, Any]
    matrix: list[float] | None = None  # row-major, order×order for cov/corr
    matrix_order: int | None = None
    histogram_counts: list[float] | None = None
    histogram_edges: list[float] | None = None

    @property
    def value(self) -> float | int | bool | list[float] | list[int] | list[bool]:
        """Primary result: ``.scalar`` if set, else ``.reduced``.

        Returns
        -------
        float, int, bool, or list
            The aggregate or reduced vector.

        Raises
        ------
        TetError
            If neither ``scalar`` nor ``reduced`` is set.
        """
        if self.scalar is not None:
            return self.scalar
        if self.reduced is not None:
            return self.reduced
        raise TetError("query result has no scalar or reduced value")

    @property
    def preview_samples(self) -> list[float] | None:
        """Capped ``execution.*_preview`` list when the engine included samples."""
        values, _ = preview_lists_from_execution(self.execution)
        return values

    @property
    def preview_truncated(self) -> bool:
        """Whether ``preview_samples`` was truncated to the preview cap."""
        _, truncated = preview_lists_from_execution(self.execution)
        return truncated

    def preview_ndarray(self) -> np.ndarray | None:
        """First N logical row-major values as a 1-D ``numpy.ndarray``, or ``None``."""
        return preview_ndarray_from_response(self.raw)

    @classmethod
    def from_response(
        cls,
        raw: dict[str, Any],
        *,
        op: str | None = None,
        path: str | None = None,
        require_execution: bool = False,
    ) -> QueryResult:
        """Build from a wire ``QueryResponse`` dict; pass ``op`` to fill typed fields."""
        check_query_response(
            raw, path=path, require_execution=require_execution
        )

        execution = raw.get("execution")
        if not isinstance(execution, dict):
            return cls(
                accepted=True,
                message=raw.get("message"),
                scalar=None,
                reduced=None,
                reduced_shape=None,
                plan=raw.get("plan") if isinstance(raw.get("plan"), dict) else None,
                execution=None,
                raw=raw,
            )

        scalar: float | int | bool | None = None
        reduced: list[float] | list[int] | list[bool] | None = None
        reduced_shape = execution.get("operation_reduced_shape")
        if isinstance(reduced_shape, list):
            reduced_shape = [int(x) for x in reduced_shape]
        else:
            reduced_shape = None

        matrix: list[float] | None = None
        matrix_order: int | None = None
        histogram_counts: list[float] | None = None
        histogram_edges: list[float] | None = None

        if op and op in REDUCTION_OPS:
            scalar_key, reduced_key = REDUCTION_OPS[op]
            if execution.get(scalar_key) is not None:
                scalar = execution[scalar_key]
            reduced_val = execution.get(reduced_key)
            if isinstance(reduced_val, list):
                reduced = reduced_val
        elif op and op in _EXTRA_SCALAR_FIELDS:
            key = _EXTRA_SCALAR_FIELDS[op]
            if execution.get(key) is not None:
                scalar = execution[key]
        elif op == "histogram":
            counts = execution.get("operation_histogram_counts")
            edges = execution.get("operation_histogram_edges")
            if isinstance(counts, list):
                histogram_counts = counts
            if isinstance(edges, list):
                histogram_edges = edges
            reduced_val = execution.get("operation_reduced_histogram_counts")
            if isinstance(reduced_val, list):
                reduced = reduced_val
        elif op and op in _MATRIX_FIELDS:
            mat_key, order_key = _MATRIX_FIELDS[op]
            mat = execution.get(mat_key)
            if isinstance(mat, list):
                matrix = mat
            order = execution.get(order_key)
            if order is not None:
                matrix_order = int(order)

        return cls(
            accepted=True,
            message=raw.get("message"),
            scalar=scalar,
            reduced=reduced,
            reduced_shape=reduced_shape,
            plan=raw.get("plan") if isinstance(raw.get("plan"), dict) else None,
            execution=execution,
            raw=raw,
            matrix=matrix,
            matrix_order=matrix_order,
            histogram_counts=histogram_counts,
            histogram_edges=histogram_edges,
        )


def scalar_from_response(
    raw: dict[str, Any], op: str
) -> float | int | bool:
    """Extract a scalar reduction from a full response dict."""
    result = QueryResult.from_response(raw, op=op)
    if result.scalar is None:
        raise TetError(f"missing scalar for op {op!r} (partial reduction?)")
    return result.scalar
