"""Catalog view: :class:`Dataset` records and axis name/index resolution."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tet._errors import UnknownAxisError


@dataclass(frozen=True, slots=True)
class Dataset:
    """One dataset from the file catalog (shape, dtype tag, optional ``dim_names``)."""

    name: str
    shape: tuple[int, ...]
    dtype: int  # wire tag; see tetration ``DATASET_DTYPE_TAG_V1``
    chunk_shape: tuple[int, ...]
    dim_names: tuple[str, ...] | None = None  # from footer metadata when present

    @property
    def ndim(self) -> int:
        """Number of dimensions (``len(shape)``)."""
        return len(self.shape)

    def axis_index(
        self,
        axis: int | str,
        *,
        path: Path | str | None = None,
    ) -> int:
        """Resolve an axis label to a 0-based dimension index.

        Parameters
        ----------
        axis : int or str
            Non-negative or negative index, or a name from ``dim_names``.
        path : path-like, optional
            Shown in :class:`~tet.UnknownAxisError` messages.

        Returns
        -------
        int
            Axis index in ``0 .. ndim-1``.

        Raises
        ------
        UnknownAxisError
            Index out of range or unknown name (or no ``dim_names`` on this dataset).
        """
        if isinstance(axis, int):
            idx = axis + self.ndim if axis < 0 else axis
            if idx < 0 or idx >= self.ndim:
                raise UnknownAxisError(
                    axis,
                    dataset=self.name,
                    path=path,
                    ndim=self.ndim,
                    shape=self.shape,
                    dim_names=self.dim_names,
                )
            return idx
        if self.dim_names is None:
            raise UnknownAxisError(
                axis,
                dataset=self.name,
                path=path,
                ndim=self.ndim,
                shape=self.shape,
                dim_names=None,
            )
        try:
            return self.dim_names.index(axis)
        except ValueError as exc:
            raise UnknownAxisError(
                axis,
                dataset=self.name,
                path=path,
                ndim=self.ndim,
                shape=self.shape,
                dim_names=self.dim_names,
            ) from exc


def dataset_from_summary(
    record: Mapping[str, Any],
    dim_names: Sequence[str] | None = None,
) -> Dataset:
    """Build a :class:`Dataset` from a catalog JSON record.

    Parameters
    ----------
    record : mapping
        One entry from ``summary["datasets"]`` (name, shape, dtype, chunk_shape).
    dim_names : sequence of str, optional
        Footer ``metadata.datasets[name].dim_names`` when present.

    Returns
    -------
    Dataset
    """
    shape = tuple(int(x) for x in record["shape"])
    chunk_shape = tuple(int(x) for x in record["chunk_shape"])
    names = tuple(dim_names) if dim_names else None
    return Dataset(
        name=str(record["name"]),
        shape=shape,
        dtype=int(record["dtype"]),
        chunk_shape=chunk_shape,
        dim_names=names,
    )


def axes_for_query(axes: Sequence[int | str] | None) -> list[int | str]:
    """Normalize `axes` / `axis` for query wire (`[]` means reduce all)."""
    if axes is None:
        return []
    return list(axes)


def axes_wire(
    dataset: Dataset,
    axes: Sequence[int | str] | None,
    *,
    path: Path | str | None = None,
) -> list[int | str]:
    """Map user axes to query wire form (decimal index strings or resolved ints)."""
    if not axes:
        return []
    return [dataset.axis_index(a, path=path) for a in axes]
