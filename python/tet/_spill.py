"""Spill-file helpers: path resolution and dense row-major LE → ``numpy.ndarray``.

Tetration can export dense tensors to spill files (``write: spill`` on transforms,
top-level ``spill`` on selection-only reads). Files are row-major little-endian
raw bytes — not ``.tet`` layout. Use :func:`load_spill_array` or result
``.to_numpy()`` helpers to reload into NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from tet._dtype import WIRE_DTYPE_TAG_V1, numpy_dtype_from_tag

# execution keys checked in order (transform/read spill may use f32 or f64 path).
_SPILL_PATH_KEYS: tuple[tuple[str, str | None], ...] = (
    ("spill_f32_path", "spill_f32_bytes"),
    ("spill_f64_path", "spill_f64_bytes"),
)


def normalize_path(path: str | PathLike[str]) -> Path:
    """Resolve engine-returned paths to a stable absolute form.

    Tetration on Windows may return extended-length paths (``\\\\?\\...`` or
    ``//?/...``). Callers expect ordinary ``Path.resolve()`` paths without
    the prefix.
    """
    return Path(path).expanduser().resolve()


def resolve_spill_path(path: str | PathLike[str], tet_path: Path | str) -> Path:
    """Resolve a spill or sidecar path for tetration allowlist rules.

    Tetration only writes under the source ``.tet`` directory, platform cache,
    or explicit allowlist roots. Relative paths must be resolved beside the
    source file — not the process CWD — or validation fails.

    Parameters
    ----------
    path : str or path-like
        User-supplied output path (may be relative).
    tet_path : path-like
        Open ``.tet`` file used as the allowlist anchor.

    Returns
    -------
    pathlib.Path
        Absolute, normalized path passed on the query wire.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path(tet_path).resolve().parent / resolved).resolve()
    return resolved


def spill_path_from_execution(execution: dict[str, Any]) -> tuple[Path, int | None]:
    """Extract spill file location from a query ``execution`` block.

    Parameters
    ----------
    execution : dict
        ``response["execution"]`` from :meth:`~tet.TetFile.execute`.

    Returns
    -------
    tuple of (pathlib.Path, int or None)
        Spill file path and optional ``spill_f32_bytes`` / ``spill_f64_bytes``.

    Raises
    ------
    ValueError
        If neither ``spill_f32_path`` nor ``spill_f64_path`` is set.
    """
    for path_key, bytes_key in _SPILL_PATH_KEYS:
        raw_path = execution.get(path_key)
        if isinstance(raw_path, str) and raw_path:
            byte_len = execution.get(bytes_key) if bytes_key else None
            return (
                normalize_path(raw_path),
                int(byte_len) if byte_len is not None else None,
            )
    raise ValueError("execution missing spill_f32_path or spill_f64_path")


def logical_shape_from_raw(raw: dict[str, Any] | None) -> tuple[int, ...] | None:
    """Read logical output shape from a wire response ``catalog`` block.

    After execute, ``catalog.shape`` reflects the logical selection shape
    (sub-slices included), which may differ from the on-disk dataset shape.
    """
    if not raw:
        return None
    catalog = raw.get("catalog")
    if not isinstance(catalog, dict):
        return None
    shape = catalog.get("shape")
    if isinstance(shape, list):
        return tuple(int(x) for x in shape)
    return None


def load_spill_array(
    path: Path | str,
    *,
    shape: tuple[int, ...] | None,
    dtype_tag: int,
    byte_len: int | None = None,
    raw: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load a dense row-major little-endian spill file into a NumPy array.

    Parameters
    ----------
    path : path-like
        Spill file on disk (from ``execution.spill_f32_path`` or similar).
    shape : tuple of int, optional
        Logical tensor shape. When omitted, uses ``catalog.shape`` from ``raw``
        or returns a 1-D array sized from ``byte_len`` / file size.
    dtype_tag : int
        Catalog wire dtype tag (see :data:`~tet._dtype.WIRE_DTYPE_TAG_V1`).
    byte_len : int, optional
        Expected byte length from execution metadata (sanity check for 1-D load).
    raw : dict, optional
        Full wire response; used to infer ``shape`` when not passed explicitly.

    Returns
    -------
    numpy.ndarray
        Shaped row-major array (copy from file; not mmap-backed).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        Unsupported dtype tag, byte length mismatch, or element count mismatch.
    """
    dtype = numpy_dtype_from_tag(dtype_tag)
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"spill file not found: {file_path}")

    resolved_shape = shape or logical_shape_from_raw(raw)
    if resolved_shape is None:
        # No shape metadata: load flat buffer (caller may reshape).
        if byte_len is None:
            byte_len = file_path.stat().st_size
        itemsize = dtype.itemsize
        if byte_len % itemsize != 0:
            raise ValueError(
                f"spill byte length {byte_len} is not a multiple of "
                f"{dtype.name} item size {itemsize}"
            )
        return np.fromfile(file_path, dtype=dtype, count=byte_len // itemsize)

    expected = int(np.prod(resolved_shape, dtype=np.int64))
    arr = np.fromfile(file_path, dtype=dtype, count=expected)
    if arr.size != expected:
        raise ValueError(
            f"spill file {file_path} has {arr.size} elements, "
            f"expected {expected} for shape {resolved_shape}"
        )
    return arr.reshape(resolved_shape)


def infer_spill_dtype_tag(
    execution: dict[str, Any],
    *,
    fallback: int,
) -> int:
    """Pick f32 vs f64 spill dtype from execution metadata.

    Transform spill uses separate ``spill_f32_path`` / ``spill_f64_path`` keys.
    Selection read spill currently exports f32 paths; ``fallback`` is the
    catalog dtype when no spill key is present.
    """
    if execution.get("spill_f64_path"):
        return WIRE_DTYPE_TAG_V1["f64"]
    if execution.get("spill_f32_path"):
        return WIRE_DTYPE_TAG_V1["f32"]
    return fallback


@dataclass(frozen=True, slots=True)
class SpillReadResult:
    """Selection-only dense export spilled to a row-major LE file (``mmap_spill``).

    Attributes
    ----------
    path : pathlib.Path
        Spill file written by the engine.
    shape : tuple of int or None
        Logical selection shape when known from ``catalog.shape``.
    dtype_tag : int
        Wire dtype tag for :func:`load_spill_array`.
    byte_len : int or None
        ``spill_f32_bytes`` / ``spill_f64_bytes`` from execution when set.
    memory_strategy : str or None
        Typically ``"mmap_spill"`` for selection-only export.
    raw : dict
        Full wire response from :meth:`~tet.TetFile.execute`.
    """

    path: Path
    shape: tuple[int, ...] | None
    dtype_tag: int
    byte_len: int | None
    memory_strategy: str | None
    raw: dict[str, Any]

    def to_numpy(self) -> np.ndarray:
        """Load the spill file as a shaped ``numpy.ndarray``.

        Returns
        -------
        numpy.ndarray
            Same logical tensor as :meth:`~tet.TetFile.read_numpy` would return
            for the same selection, without holding the full buffer in RAM
            during export.

        Raises
        ------
        FileNotFoundError, ValueError
            See :func:`load_spill_array`.
        """
        return load_spill_array(
            self.path,
            shape=self.shape,
            dtype_tag=self.dtype_tag,
            byte_len=self.byte_len,
            raw=self.raw,
        )
