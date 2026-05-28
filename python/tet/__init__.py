"""Python bindings for Tetration `.tet` files and the query engine."""

from __future__ import annotations

import json
from typing import Any

import tet._native as _native
from tet._native import core_version

__version__: str = _native.__version__


class TetFile:
    """Opened `.tet` file; `query` returns a parsed response dict."""

    def __init__(self, inner: _native.TetFile) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def query(self, query: Any) -> dict[str, Any]:
        return json.loads(self._inner.query(query))


def open(path: str | Any) -> TetFile:
    return TetFile(_native.open(path))


__all__ = ["TetFile", "__version__", "core_version", "open"]
