"""NumPy interchange: materialize logical selections and transforms.

Three export sinks from tetration map to Python as:

- **ram** — :func:`read_numpy_array` / native ``transform_to_numpy`` (in-process)
- **spill** — :func:`read_spill_array` (``.bin`` row-major LE)
- **sidecar** — :class:`~tet.SidecarTransformResult` (derived ``.tet``; transform only)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any

import numpy as np

from tet._core.docstrings import numpy_read_doc, spill_read_doc
from tet._query.doc import build_selection_query
from tet._io.spill import (
    SpillReadResult,
    infer_spill_dtype_tag,
    logical_shape_from_raw,
    resolve_spill_path,
    spill_path_from_execution,
)

if TYPE_CHECKING:
    from tet.file import TetFile


def read_numpy_array(
    file: TetFile,
    dataset: str,
    selection: Sequence[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    file._require_dataset(dataset)
    doc = build_selection_query(dataset, selection=selection)
    arr = file._inner.read_numpy(doc)
    return np.asarray(arr)


read_numpy_array.__doc__ = numpy_read_doc(
    "Materialize a catalog selection into a dense ``numpy.ndarray`` (wire ``ram``)."
)


def read_spill_array(
    file: TetFile,
    dataset: str,
    path: str | PathLike[str] | None = None,
    *,
    selection: Sequence[Mapping[str, Any]] | None = None,
) -> SpillReadResult:
    ds = file.dataset(dataset)
    doc = build_selection_query(dataset, selection=selection)
    if path is not None:
        doc["spill"] = str(resolve_spill_path(path, file.path))
    raw = file.execute(doc, raw=True)
    assert isinstance(raw, dict)
    execution = raw.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("read spill response missing execution block")
    spill_path, byte_len = spill_path_from_execution(execution)
    dtype_tag = infer_spill_dtype_tag(execution, fallback=ds.dtype)
    return SpillReadResult(
        path=spill_path,
        shape=logical_shape_from_raw(raw) or ds.shape,
        dtype_tag=dtype_tag,
        byte_len=byte_len,
        memory_strategy=(
            str(execution["memory_strategy"])
            if execution.get("memory_strategy") is not None
            else None
        ),
        raw=raw,
    )


read_spill_array.__doc__ = spill_read_doc(
    "Spill a selection-only query to a dense row-major LE file (wire ``spill``)."
)
