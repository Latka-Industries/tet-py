"""Transform ops (zscore, …) and typed sink results."""

from tet._transform.ops import TransformOps, TransformWrite, write_to_wire
from tet._transform.result import (
    NumpyTransformResult,
    SidecarTransformResult,
    SpillTransformResult,
    TransformResult,
    TransformStats,
)

__all__ = [
    "NumpyTransformResult",
    "SidecarTransformResult",
    "SpillTransformResult",
    "TransformOps",
    "TransformResult",
    "TransformStats",
    "TransformWrite",
    "write_to_wire",
]
