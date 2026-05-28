"""Catalog helpers: dataset records, axis indices, iteration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tet._errors import UnknownAxisError


@dataclass(frozen=True, slots=True)
class Dataset:
    """One dataset entry from the `.tet` catalog."""

    name: str
    shape: tuple[int, ...]
    dtype: int
    chunk_shape: tuple[int, ...]
    dim_names: tuple[str, ...] | None = None

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def axis_index(
        self,
        axis: int | str,
        *,
        path: Path | str | None = None,
    ) -> int:
        """Resolve axis by dimension index (0, 1, …) or `dim_names` label."""
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
    """Map user axes to query wire form (ints or string indices)."""
    if not axes:
        return []
    return [dataset.axis_index(a, path=path) for a in axes]
