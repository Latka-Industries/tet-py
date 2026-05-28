"""Python bindings for Tetration `.tet` files and the query engine."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import tet._native as _native
from tet._native import CatalogError, TetError, core_version

__version__: str = _native.__version__


def _parse_json_response(raw: str) -> dict[str, Any]:
    return json.loads(raw)


def _mean_sum_doc(dataset: str, op: str, axes: Sequence[int] | None) -> dict[str, Any]:
    return {"dataset": dataset, op: list(axes) if axes is not None else []}


class TetFile:
    """Opened `.tet` file; `query` returns a parsed response dict."""

    def __init__(self, inner: _native.TetFile) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def summary(self) -> dict[str, Any]:
        """Catalog + footer summary (`TetFileSummaryV1`), same JSON as `tet info --json`."""
        return _parse_json_response(self._inner.summary_json())

    def info(self) -> dict[str, Any]:
        """Alias for [`summary()`] (parity with `tet info --json`)."""
        return self.summary()

    def query(self, query: Any) -> dict[str, Any]:
        """Execute a query document (same as `tet query -t … -x`)."""
        return _parse_json_response(self._inner.query(query))

    def plan_only(self, query: Any) -> dict[str, Any]:
        """Plan without execution (same as `tet query` without `-x`).

        Query documents with `mean`/`sum`/etc. require [`query()`] instead.
        """
        return _parse_json_response(self._inner.plan_only(query))

    def mean(self, dataset: str, axes: Sequence[int] | None = None) -> float:
        """Mean over `dataset`; `axes` selects reduction axes (default: all)."""
        out = self.query(_mean_sum_doc(dataset, "mean", axes))
        return _scalar_from_execution(out, "operation_mean")

    def sum(self, dataset: str, axes: Sequence[int] | None = None) -> float:
        """Sum over `dataset`; `axes` selects reduction axes (default: all)."""
        out = self.query(_mean_sum_doc(dataset, "sum", axes))
        return _scalar_from_execution(out, "operation_sum")


def _scalar_from_execution(out: dict[str, Any], field: str) -> float:
    if not out.get("accepted"):
        raise TetError(out.get("message") or "query not accepted")
    execution = out.get("execution")
    if not execution:
        raise TetError("query has no execution block")
    value = execution.get(field)
    if value is None:
        raise TetError(f"missing {field} in execution")
    return float(value)


def open(path: str | Any) -> TetFile:
    return TetFile(_native.open(path))


__all__ = [
    "CatalogError",
    "TetError",
    "TetFile",
    "__version__",
    "core_version",
    "open",
]
