"""Write NumPy arrays to a new or appended ``.tet`` file."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

import tet._native as _native


def _resolve_path(path: str | PathLike[str]) -> Path:
    return Path(path).expanduser()


class TetWriter:
    """Buffered writer: queue datasets, then :meth:`commit` to one ``.tet`` file."""

    def __init__(self, inner: _native.TetWriterSession) -> None:
        self._inner = inner

    @classmethod
    def create(cls, path: str | PathLike[str]) -> TetWriter:
        """Create a new ``.tet`` at ``path`` (truncates on commit)."""
        return cls(_native.TetWriterSession.create(_resolve_path(path)))

    @classmethod
    def open_append(cls, path: str | PathLike[str]) -> TetWriter:
        """Open an existing ``.tet`` and queue additional datasets."""
        return cls(_native.TetWriterSession.open_append(_resolve_path(path)))

    @property
    def path(self) -> Path:
        """Target ``.tet`` path."""
        return Path(self._inner.path)

    @property
    def dataset_count(self) -> int:
        """Number of datasets queued (not yet committed)."""
        return int(self._inner.dataset_count)

    def write_dataset(
        self,
        name: str,
        array: np.ndarray,
        *,
        chunk_shape: Sequence[int] | None = None,
        attrs: Mapping[str, str] | None = None,
        dim_names: Sequence[str] | None = None,
        coords: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Queue one row-major C-contiguous array (``float32`` / ``float64``).

        Parameters
        ----------
        name : str
            Catalog dataset name.
        array : numpy.ndarray
            Row-major tensor; copied into the file on :meth:`commit`.
        chunk_shape : sequence of int, optional
            Tile shape; defaults to ``array.shape`` (single chunk).
        attrs : mapping of str to str, optional
            CF-style footer attributes.
        dim_names : sequence of str, optional
            Dimension names (length must match ``array.ndim``).
        coords : mapping, optional
            Axis name → coordinate label list for the footer.
        """
        wire_chunk = tuple(int(x) for x in chunk_shape) if chunk_shape is not None else None
        opts: _native.WriteDatasetOptions | None = None
        if wire_chunk is not None or attrs is not None or dim_names is not None or coords is not None:
            opts = _native.WriteDatasetOptions()
            if wire_chunk is not None:
                opts.chunk_shape = wire_chunk
            if attrs is not None:
                opts.attrs = dict(attrs)
            if dim_names is not None:
                opts.dim_names = list(dim_names)
            if coords is not None:
                opts.coords = {k: list(v) for k, v in coords.items()}
        self._inner.write_dataset(name, array, opts)

    def push_history_event(self, op: str, source: str) -> None:
        """Append a footer history row (flushed on :meth:`commit`)."""
        self._inner.push_history_event(op, source)

    def commit(self) -> Path:
        """Write the ``.tet`` file and return its path."""
        return Path(self._inner.commit())


def write_dataset(
    path: str | PathLike[str],
    name: str,
    array: np.ndarray,
    *,
    chunk_shape: Sequence[int] | None = None,
    attrs: Mapping[str, str] | None = None,
    dim_names: Sequence[str] | None = None,
    coords: Mapping[str, Sequence[str]] | None = None,
    history_op: str | None = None,
    history_source: str | None = None,
) -> Path:
    """Create a one-dataset ``.tet`` file from a NumPy array.

    Convenience wrapper around :class:`TetWriter` for the common single-array case.
    """
    writer = TetWriter.create(path)
    if history_op is not None:
        writer.push_history_event(
            history_op,
            history_source if history_source is not None else str(writer.path),
        )
    writer.write_dataset(
        name,
        array,
        chunk_shape=chunk_shape,
        attrs=attrs,
        dim_names=dim_names,
        coords=coords,
    )
    return writer.commit()
