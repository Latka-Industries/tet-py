"""Query wire builders and :class:`QueryResult`."""

from tet._query.doc import (
    QUERY_OP_KEYS,
    apply_execution_hints,
    axis_slice,
    build_query,
    build_selection_query,
    full_selection,
    selection_slices,
    transform_op,
)
from tet._query.result import REDUCTION_OPS, QueryResult, reduction_doc

__all__ = [
    "QUERY_OP_KEYS",
    "REDUCTION_OPS",
    "QueryResult",
    "apply_execution_hints",
    "axis_slice",
    "build_query",
    "build_selection_query",
    "full_selection",
    "reduction_doc",
    "selection_slices",
    "transform_op",
]
