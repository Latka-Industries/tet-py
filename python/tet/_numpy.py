"""NumPy interchange: materialize logical selections and transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from tet._query_doc import build_selection_query

if TYPE_CHECKING:
    from tet._file import TetFile


def read_numpy_array(
    file: TetFile,
    dataset: str,
    selection: Sequence[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    """Materialize a catalog selection into a dense ``numpy.ndarray``."""
    file._require_dataset(dataset)
    doc = build_selection_query(dataset, selection=selection)
    arr = file._inner.read_numpy(doc)
    return np.asarray(arr)
