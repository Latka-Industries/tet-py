"""Python bindings for Tetration `.tet` files and the query engine."""

from __future__ import annotations

import tet._native as _native
from tet._native import TetFile, core_version, open

__version__: str = _native.__version__

__all__ = ["TetFile", "__version__", "core_version", "open"]
