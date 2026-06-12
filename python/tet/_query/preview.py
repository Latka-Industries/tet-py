"""Capped execution preview samples → NumPy."""

from __future__ import annotations

from typing import Any

import numpy as np

from tet._core.dtype import numpy_dtype_from_tag


def preview_lists_from_execution(
    execution: dict[str, Any] | None,
) -> tuple[list[float] | None, bool]:
    """Extract capped ``*_preview`` lists from an ``execution`` block."""
    if not execution:
        return None, False
    for key, trunc_key in (
        ("f32_preview", "f32_preview_truncated"),
        ("f64_preview", "f64_preview_truncated"),
        ("i32_preview", "i32_preview_truncated"),
        ("i64_preview", "i64_preview_truncated"),
        ("u8_preview", "u8_preview_truncated"),
        ("u16_preview", "u16_preview_truncated"),
        ("i16_preview", "i16_preview_truncated"),
        ("u32_preview", "u32_preview_truncated"),
        ("u64_preview", "u64_preview_truncated"),
        ("f16_preview", "f16_preview_truncated"),
    ):
        values = execution.get(key)
        if isinstance(values, list) and values:
            truncated = bool(execution.get(trunc_key, False))
            return [float(x) for x in values], truncated
    return None, False


def preview_from_response(raw: dict[str, Any]) -> np.ndarray | None:
    """Build a 1-D preview array from ``execution.*_preview`` when present.

    Returns
    -------
    numpy.ndarray or None
        1-D array (dtype from ``catalog.dtype`` when available); ``None`` if no samples.
    """
    execution = raw.get("execution")
    if not isinstance(execution, dict):
        return None
    values, _ = preview_lists_from_execution(execution)
    if values is None:
        return None
    catalog = raw.get("catalog")
    dtype_tag: int | None = None
    if isinstance(catalog, dict) and catalog.get("dtype") is not None:
        dtype_tag = int(catalog["dtype"])
    if dtype_tag is None:
        return np.asarray(values, dtype=np.float64)
    return np.asarray(values, dtype=numpy_dtype_from_tag(dtype_tag))
