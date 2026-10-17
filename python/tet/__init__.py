"""Public API for Tetration ``.tet`` files and the query engine.

Import :func:`open` to get a :class:`TetFile`. Reduction helpers (``mean``, ``sum``,
``quantile``, …), :class:`QueryResult`, and query builders live on the file object
or as module-level exports.
"""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path

import numpy as np

import tet._native as _native
from tet.catalog import Dataset
from tet._core.errors import UnknownAxisError, UnknownDatasetError
from tet.file import TetFile
from tet._io import SpillReadResult, TetWriter, write_dataset
from tet.convert import convert
from tet.export import export
from tet.package import MemoryStore, S3Store, create, ingest, is_package, publish
from tet._native import CatalogError, TetError, core_version
from tet._query import (
    QUERY_OP_KEYS,
    REDUCTION_OPS,
    QueryResult,
    axis_slice,
    build_query,
    full_selection,
    selection_slices,
)
from tet._transform import (
    NumpyTransformResult,
    SidecarTransformResult,
    SpillTransformResult,
    TransformResult,
    TransformStats,
    TransformWrite,
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
    """Open a ``.tet`` file, package, or tetration URI.

    Parameters
    ----------
    path : str or path-like
        Filesystem path (``~`` expanded) or URI. A directory whose name ends in
        ``.tet.d`` (or that contains ``PROTOCOL``) pins ``CURRENT`` once.
        URI forms: ``file://…``, ``path?generation=N``, ``s3://bucket/prefix``
        (optional ``?generation=N``). ``s3://`` pins the package with
        ``TETRATION_S3_*`` and range-reads on query; credentials are not in the URI.
        Use :class:`MemoryStore` for ``memory://`` and :class:`S3Store` for ingest /
        publish. ``S3Store.open`` still downloads the whole ``packed.tet``.
        ``tet.convert("s3://bucket/object.ext")`` is a separate full download of one
        foreign object.

    Returns
    -------
    TetFile
        See :meth:`TetFile.open` and :meth:`TetFile.open_package`. Package opens
        expose :attr:`TetFile.generation`.

    Raises
    ------
    OSError, CatalogError, ValueError
        See :meth:`TetFile.open`. A package with no published generation raises
        :class:`CatalogError`.
    """
    text = os.fspath(path)
    if "://" in text or "?generation=" in text:
        return TetFile.from_native(_native.open_uri(text))
    resolved = Path(text).expanduser()
    if is_package(resolved):
        return TetFile.open_package(resolved)
    return TetFile.open(resolved)


__all__ = [
    "CatalogError",
    "Dataset",
    "MemoryStore",
    "S3Store",
    "NumpyTransformResult",
    "QUERY_OP_KEYS",
    "QueryResult",
    "REDUCTION_OPS",
    "SidecarTransformResult",
    "SpillReadResult",
    "SpillTransformResult",
    "TransformResult",
    "TransformStats",
    "TetWriter",
    "TransformWrite",
    "axis_slice",
    "build_query",
    "convert",
    "export",
    "create",
    "ingest",
    "is_package",
    "publish",
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
    "write_dataset",
]
