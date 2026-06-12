"""Shared primitives: errors, wire dtype tags, docstring helpers."""

from tet._core.dtype import WIRE_DTYPE_TAG_V1, numpy_dtype_from_tag
from tet._core.errors import UnknownAxisError, UnknownDatasetError, check_query_response, coerce_query_doc
from tet._core.docstrings import numpy_read_doc, spill_read_doc

__all__ = [
    "WIRE_DTYPE_TAG_V1",
    "UnknownAxisError",
    "UnknownDatasetError",
    "check_query_response",
    "coerce_query_doc",
    "numpy_dtype_from_tag",
    "numpy_read_doc",
    "spill_read_doc",
]
