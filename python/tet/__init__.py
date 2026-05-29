"""Public API for Tetration ``.tet`` files and the query engine.

Import :func:`open` to get a :class:`TetFile`. Reduction helpers (``mean``, ``sum``,
``quantile``, …), :class:`QueryResult`, and query builders live on the file object
or as module-level exports.
"""

from __future__ import annotations

from os import PathLike

import numpy as np

import tet._native as _native
from tet._catalog import Dataset
from tet._errors import UnknownAxisError, UnknownDatasetError
from tet._file import TetFile
from tet._native import CatalogError, TetError, core_version
from tet._query import REDUCTION_OPS, QueryResult
from tet._query_doc import (
    QUERY_OP_KEYS,
    axis_slice,
    build_query,
    full_selection,
    selection_slices,
)
from tet._transform import TransformWrite
from tet._transform_result import (
    NumpyTransformResult,
    SidecarTransformResult,
    SpillTransformResult,
    TransformResult,
    TransformStats,
)

__version__: str = _native.__version__


def typing_stub(path: str | PathLike[str]) -> str:
    """Build a ``.pyi`` snippet with ``Literal`` names for one ``.tet`` file.

    Parameters
    ----------
    path : str or path-like
        Path to the ``.tet`` file to introspect (opens read-only, then closes).

    Returns
    -------
    str
        Python source for optional IDE stubs (dataset names; axis ``Literal``s when
        footer ``dim_names`` exist). Save to e.g. ``mydata_tet.pyi`` in your project.

    Raises
    ------
    OSError, CatalogError
        Same as :func:`open`.
    """
    with TetFile.open(path) as f:
        return f.typing_stub()


def open(path: str | PathLike[str]) -> TetFile:
    """Open a ``.tet`` file read-only (mmap).

    Parameters
    ----------
    path : str or path-like
        Filesystem path; ``~`` expanded.

    Returns
    -------
    TetFile
        See :meth:`TetFile.open`.

    Raises
    ------
    OSError, CatalogError
        See :meth:`TetFile.open`.
    """
    return TetFile.open(path)


__all__ = [
    "CatalogError",
    "Dataset",
    "NumpyTransformResult",
    "QUERY_OP_KEYS",
    "QueryResult",
    "REDUCTION_OPS",
    "SidecarTransformResult",
    "SpillTransformResult",
    "TransformResult",
    "TransformStats",
    "TransformWrite",
    "axis_slice",
    "build_query",
    "full_selection",
    "selection_slices",
    "TetError",
    "TetFile",
    "UnknownAxisError",
    "UnknownDatasetError",
    "__version__",
    "core_version",
    "open",
    "typing_stub",
]
