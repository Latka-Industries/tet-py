"""Errors and validation for a specific open ``.tet`` file.

Messages list dataset names and axis labels from that file's catalog, not generic hints.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from tet._native import TetError


class UnknownDatasetError(KeyError, TetError):
    """``dataset`` name is missing from this file's catalog (subclass of :class:`KeyError`)."""

    def __init__(
        self,
        name: str,
        *,
        path: Path | str | None,
        available: Sequence[str],
    ) -> None:
        self.name = name
        self.path = Path(path) if path is not None else None
        self.available = tuple(available)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        where = f" in {self.path}" if self.path is not None else ""
        lines = [
            f"unknown dataset {self.name!r}{where}",
            f"available datasets ({len(self.available)}):",
        ]
        lines.extend(f"  - {n!r}" for n in self.available)
        suggestions = difflib.get_close_matches(
            self.name, list(self.available), n=3, cutoff=0.5
        )
        if suggestions:
            lines.append(f"did you mean: {', '.join(repr(s) for s in suggestions)}?")
        return "\n".join(lines)


class UnknownAxisError(ValueError, TetError):
    """Axis index or name is invalid for a dataset in this file (subclass of :class:`ValueError`)."""

    def __init__(
        self,
        axis: int | str,
        *,
        dataset: str,
        path: Path | str | None,
        ndim: int,
        shape: tuple[int, ...],
        dim_names: tuple[str, ...] | None,
    ) -> None:
        self.axis = axis
        self.dataset = dataset
        self.path = Path(path) if path is not None else None
        self.ndim = ndim
        self.shape = shape
        self.dim_names = dim_names
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        where = f" in {self.path}" if self.path is not None else ""
        lines = [
            f"invalid axis {self.axis!r} for dataset {self.dataset!r}{where}",
            f"ndim={self.ndim}, shape={self.shape}",
        ]
        if isinstance(self.axis, str):
            if self.dim_names is None:
                lines.append(
                    "this dataset has no dim_names in footer metadata; "
                    f"use integer axis indices 0..{self.ndim - 1}"
                )
            else:
                lines.append("dim_names (index → name):")
                for i, n in enumerate(self.dim_names):
                    lines.append(f"  {i} → {n!r}")
                suggestions = difflib.get_close_matches(
                    self.axis, list(self.dim_names), n=3, cutoff=0.5
                )
                if suggestions:
                    lines.append(
                        f"did you mean: {', '.join(repr(s) for s in suggestions)}?"
                    )
        else:
            lines.append(f"valid integer axis indices: 0..{self.ndim - 1} (or negative indexing)")
            if self.dim_names is not None:
                lines.append("dim_names (index → name):")
                for i, n in enumerate(self.dim_names):
                    lines.append(f"  {i} → {n!r}")
        return "\n".join(lines)


def format_dataset_list(names: Sequence[str]) -> str:
    """Comma-separated quoted names for error messages."""
    if not names:
        return "(empty catalog)"
    return ", ".join(repr(n) for n in names)


def check_query_response(
    raw: dict[str, Any],
    *,
    path: Path | str | None,
    require_execution: bool = False,
) -> None:
    """Raise when the engine rejects the query or the dataset is not in this file.

    Call after parsing a ``QueryResponse`` dict. Checks ``catalog.matched`` and
    ``accepted`` before optional ``execution`` presence.
    """
    catalog = raw.get("catalog")
    if isinstance(catalog, dict) and catalog.get("matched") is False:
        requested = str(raw.get("dataset", ""))
        available = catalog.get("available_datasets")
        if not isinstance(available, list):
            available = []
        names = [str(n) for n in available]
        raise UnknownDatasetError(requested, path=path, available=names)

    if not raw.get("accepted"):
        raise TetError(raw.get("message") or "query not accepted")

    if require_execution and not isinstance(raw.get("execution"), dict):
        msg = raw.get("message") or "query has no execution block"
        raise TetError(msg)


def coerce_query_doc(query: Any) -> dict[str, Any]:
    """Parse a query document from Python or JSON text.

    Parameters
    ----------
    query : dict or str
        Query document mapping, or JSON string.

    Returns
    -------
    dict
        Query document ready for the Rust engine.

    Raises
    ------
    TetError
        If a string is not valid JSON.
    TypeError
        If ``query`` is neither dict nor str.
    """
    if isinstance(query, dict):
        return query
    if isinstance(query, str):
        try:
            return cast(dict[str, Any], json.loads(query))
        except json.JSONDecodeError as exc:
            raise TetError(f"query string is not valid JSON: {exc}") from exc
    raise TypeError("query must be a dict or JSON string")
