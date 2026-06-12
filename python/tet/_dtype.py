"""Wire dtype tags (tetration ``DATASET_DTYPE_TAG_V1``) → NumPy dtypes.

Spill files and ``read_numpy`` materialize row-major **little-endian** bytes.
This module maps catalog ``dtype`` integers to matching NumPy dtypes for
:func:`~tet._spill.load_spill_array` and type checking.
"""

from __future__ import annotations

import numpy as np

# Matches tetration `catalog::DATASET_DTYPE_TAG_V1` (layout v1, docs/layout_v1.md).
WIRE_DTYPE_TAG_V1: dict[str, int] = {
    "f32": 1,
    "f64": 2,
    "i32": 3,
    "i64": 4,
    "u8": 5,
    "u16": 6,
    "i16": 7,
    "u32": 8,
    "f16": 9,
    "u64": 10,
}

# Little-endian NumPy dtypes for each supported wire tag (spill / materialize).
_TAG_TO_NUMPY: dict[int, np.dtype] = {
    WIRE_DTYPE_TAG_V1["f32"]: np.dtype("<f4"),
    WIRE_DTYPE_TAG_V1["f64"]: np.dtype("<f8"),
    WIRE_DTYPE_TAG_V1["i32"]: np.dtype("<i4"),
    WIRE_DTYPE_TAG_V1["i64"]: np.dtype("<i8"),
    WIRE_DTYPE_TAG_V1["u8"]: np.dtype("u1"),
    WIRE_DTYPE_TAG_V1["u16"]: np.dtype("<u2"),
    WIRE_DTYPE_TAG_V1["i16"]: np.dtype("<i2"),
    WIRE_DTYPE_TAG_V1["u32"]: np.dtype("<u4"),
    WIRE_DTYPE_TAG_V1["u64"]: np.dtype("<u8"),
    # f16 is tagged on wire but not exported by tet-py materialize paths yet.
    WIRE_DTYPE_TAG_V1["f16"]: np.dtype("<f2"),
}


def numpy_dtype_from_tag(dtype_tag: int) -> np.dtype:
    """Map a catalog wire ``dtype`` tag to a little-endian NumPy dtype.

    Parameters
    ----------
    dtype_tag : int
        Value from ``Dataset.dtype`` or ``catalog.dtype`` on the wire.

    Returns
    -------
    numpy.dtype
        Dtype suitable for ``numpy.fromfile`` on spill exports (row-major LE).

    Raises
    ------
    ValueError
        If ``dtype_tag`` is not in :data:`WIRE_DTYPE_TAG_V1`.
    """
    try:
        return _TAG_TO_NUMPY[dtype_tag]
    except KeyError as exc:
        supported = ", ".join(str(v) for v in sorted(_TAG_TO_NUMPY))
        raise ValueError(
            f"unsupported dataset dtype tag {dtype_tag} "
            f"(supported tags: {supported})"
        ) from exc
