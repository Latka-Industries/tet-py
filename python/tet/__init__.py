"""Python bindings for Tetration `.tet` files and the query engine."""

from __future__ import annotations

from os import PathLike

import tet._native as _native
from tet._catalog import Dataset
from tet._errors import UnknownAxisError, UnknownDatasetError
from tet._file import TetFile
from tet._native import CatalogError, TetError, core_version
from tet._query import REDUCTION_OPS, QueryResult

__version__: str = _native.__version__


def typing_stub(path: str | PathLike[str]) -> str:
    """Return a ``.pyi`` snippet with ``Literal`` dataset (and axis) names for one file.

    Save the output next to your code (e.g. ``mydata_tet.pyi``) so Pyright/Pylance can
    autocomplete names for that specific ``.tet`` path. Names are not known statically
    without opening the file.
    """
    with TetFile.open(path) as f:
        return f.typing_stub()


def open(path: str | PathLike[str]) -> TetFile:
    return TetFile.open(path)


__all__ = [
    "CatalogError",
    "Dataset",
    "QueryResult",
    "REDUCTION_OPS",
    "TetError",
    "TetFile",
    "UnknownAxisError",
    "UnknownDatasetError",
    "__version__",
    "core_version",
    "open",
    "typing_stub",
]
