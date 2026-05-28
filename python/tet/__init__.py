"""Python bindings for Tetration `.tet` files and the query engine."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import tet._native as _native
from tet._catalog import Dataset, axes_for_query, axes_wire, dataset_from_summary
from tet._native import CatalogError, TetError, core_version

__version__: str = _native.__version__


def _parse_json_response(raw: str) -> dict[str, Any]:
    return json.loads(raw)


def _mean_sum_doc(dataset: str, op: str, axes: Sequence[int | str]) -> dict[str, Any]:
    return {"dataset": dataset, op: list(axes)}


def _resolve_path(path: str | PathLike[str]) -> Path:
    """Expand `~` and normalize before passing paths to Rust."""
    return Path(path).expanduser()


def _coerce_query_doc(query: Any) -> dict[str, Any]:
    if isinstance(query, dict):
        return query
    if isinstance(query, str):
        return json.loads(query)
    raise TypeError("query must be a dict or JSON string")


class TetFile:
    """Opened `.tet` file; `query` returns a parsed response dict."""

    def __init__(self, inner: _native.TetFile) -> None:
        self._inner = inner
        self._summary_cache: dict[str, Any] | None = None

    @classmethod
    def open(cls, path: str | PathLike[str]) -> TetFile:
        """Open `.tet` read-only (same as [`tet.open`])."""
        return cls(_native.open(_resolve_path(path)))

    def __enter__(self) -> TetFile:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self) -> Iterator[str]:
        """Iterate dataset names (same order as the catalog)."""
        yield from self.datasets()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _summary_dict(self) -> dict[str, Any]:
        if self._summary_cache is None:
            self._summary_cache = _parse_json_response(self._inner.summary_json())
        return self._summary_cache

    def _metadata_for_dataset(self, name: str) -> dict[str, Any]:
        meta = self._summary_dict().get("metadata") or {}
        datasets = meta.get("datasets") or {}
        entry = datasets.get(name)
        return entry if isinstance(entry, dict) else {}

    def iter_datasets(self) -> Iterator[Dataset]:
        """Iterate [`Dataset`] records (name, shape, optional dim_names)."""
        for record in self._summary_dict()["datasets"]:
            name = str(record["name"])
            dim_names = self._metadata_for_dataset(name).get("dim_names")
            yield dataset_from_summary(record, dim_names)

    def dataset(self, key: str | int) -> Dataset:
        """Look up a dataset by catalog index (0, 1, …) or name."""
        if isinstance(key, int):
            datasets = list(self.iter_datasets())
            try:
                return datasets[key]
            except IndexError as exc:
                raise IndexError(
                    f"dataset index {key} out of range (count={len(datasets)})"
                ) from exc
        for ds in self.iter_datasets():
            if ds.name == key:
                return ds
        raise KeyError(key)

    def __getitem__(self, key: str | int) -> Dataset:
        """``f[0]`` or ``f['temperature']`` → [`Dataset`]."""
        return self.dataset(key)

    def summary(self) -> dict[str, Any]:
        """Catalog + footer summary (`TetFileSummaryV1`), same JSON as `tet info --json`."""
        return dict(self._summary_dict())

    def info(self) -> dict[str, Any]:
        """Alias for [`summary()`] (parity with `tet info --json`)."""
        return self.summary()

    def query(self, query: Any) -> dict[str, Any]:
        """Execute a query document (same as `tet query -t … -x`)."""
        return _parse_json_response(self._inner.query(query))

    def query_execute(
        self,
        query: Any,
        *,
        device: str | None = None,
    ) -> dict[str, Any]:
        """Execute with optional `execution.device` (e.g. ``cpu``, ``auto``, ``cuda:0``)."""
        doc = _coerce_query_doc(query)
        if device is not None:
            execution = dict(doc.get("execution") or {})
            execution["device"] = device
            doc = {**doc, "execution": execution}
        return self.query(doc)

    def plan_only(self, query: Any) -> dict[str, Any]:
        """Plan without execution (same as `tet query` without `-x`).

        Query documents with `mean`/`sum`/etc. require [`query()`] instead.
        """
        return _parse_json_response(self._inner.plan_only(query))

    def _resolve_reduction_axes(
        self,
        dataset: str,
        axes: Sequence[int | str] | None,
        *,
        axis: int | str | None,
    ) -> list[int | str]:
        if axis is not None and axes is not None:
            raise TypeError("pass only one of axes= or axis=")
        merged = axes_for_query(axes if axis is None else [axis])
        if not merged:
            return []
        return axes_wire(self.dataset(dataset), merged)

    def mean(
        self,
        dataset: str,
        axes: Sequence[int | str] | None = None,
        *,
        axis: int | str | None = None,
    ) -> float:
        """Mean over `dataset`.

        Use integer axis indices (0, 1, …) or names from footer `dim_names` when present.
        Omit `axes` / `axis` to reduce over all dimensions.
        """
        wire_axes = self._resolve_reduction_axes(dataset, axes, axis=axis)
        out = self.query(_mean_sum_doc(dataset, "mean", wire_axes))
        return _scalar_from_execution(out, "operation_mean")

    def sum(
        self,
        dataset: str,
        axes: Sequence[int | str] | None = None,
        *,
        axis: int | str | None = None,
    ) -> float:
        """Sum over `dataset` (same axis rules as [`mean()`])."""
        wire_axes = self._resolve_reduction_axes(dataset, axes, axis=axis)
        out = self.query(_mean_sum_doc(dataset, "sum", wire_axes))
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


def open(path: str | PathLike[str]) -> TetFile:
    return TetFile.open(path)


__all__ = [
    "CatalogError",
    "Dataset",
    "TetError",
    "TetFile",
    "__version__",
    "core_version",
    "open",
]
