"""Shape-preserving transforms: ``f.transform.to_numpy.zscore(...)``, etc."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from os import PathLike
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

from tet._io.spill import resolve_spill_path
from tet._query.doc import build_query, transform_op
from tet._query.result import QueryResult
from tet._transform.result import (
    SidecarTransformResult,
    SpillTransformResult,
    parse_transform_result,
)

if TYPE_CHECKING:
    from tet.file import TetFile

TransformResult = SpillTransformResult | SidecarTransformResult
NumpyTransformFn = Callable[..., np.ndarray | dict[str, Any]]
PathTransformFn = Callable[..., TransformResult | dict[str, Any]]


class TransformWrite(StrEnum):
    """Where transform pass-2 writes its dense output (tetration ``write`` wire key).

    Use :attr:`TransformOps.to_numpy`, :attr:`TransformOps.to_spill`, or
    :attr:`TransformOps.to_sidecar` — not this enum directly.
    """

    NUMPY = "ram"
    SPILL = "spill"
    SIDECAR = "sidecar"


def write_to_wire(
    write: TransformWrite,
    path: str | PathLike[str] | None = None,
) -> str | dict[str, Any]:
    """Build the query document ``write`` field from :class:`TransformWrite`.

    Parameters
    ----------
    write : TransformWrite
        ``ram``, ``spill``, or ``sidecar`` sink token.
    path : str or path-like, optional
        Required for explicit spill/sidecar destinations (already resolved
        beside the source ``.tet`` when relative).

    Returns
    -------
    str or dict
        Wire value: ``"ram"`` / ``"sidecar"``, or ``{"target": "spill", "path": ...}``.
    """
    if path is None:
        return write.value
    return {"target": write.value, "path": str(path)}


_TRANSFORM_IO_BASE = """\
Parameters
----------
dataset : str
    Catalog dataset name.
axes, axis : optional
    Axes to fold per-cell stats; omit for one global stat set.
{extra_params}selection, device, raw : optional
    Same as :meth:`~tet.TetFile.quantile`. ``raw=True`` returns the wire ``dict``.

Returns
-------
numpy.ndarray, SpillTransformResult, SidecarTransformResult, or dict
    ``to_numpy`` returns a dense array; spill/sidecar return sink metadata; wire JSON when ``raw=True``.

Raises
------
UnknownDatasetError, UnknownAxisError, TetError
    See :meth:`~tet.TetFile.query`.
"""

_NUMPY_EXTRA = """\
Returns ``numpy.ndarray`` with the full transformed logical selection (``write: ram``).

"""

_SPILL_EXTRA = """\
path : str or path-like, optional
    Spill file path beside the source ``.tet`` or absolute under allowlist;
    omit for an engine temp file.

Returns :class:`~tet.SpillTransformResult`; call ``.to_numpy()`` to load.

"""

_SIDECAR_EXTRA = """\
path : str or path-like, optional
    Sidecar ``.tet`` path; default beside the source file when omitted.

Returns :class:`~tet.SidecarTransformResult` (``.open()`` / ``.to_numpy()``).

"""


def _transform_doc(summary: str, *, extra_params: str = "") -> str:
    """Compose a transform method docstring (sink-specific ``extra_params``)."""
    body = _TRANSFORM_IO_BASE.format(extra_params=extra_params)
    return f"{summary.strip()}\n\n{body}"


class TransformOps:
    """Transform namespace on :class:`~tet.TetFile` (sink-first API only).

    Use ``f.transform.to_numpy.zscore(...)``, ``f.transform.to_spill.softmax(...)``,
    or ``f.transform.to_sidecar.center(...)``.
    """

    __slots__ = ("_file", "to_numpy", "to_spill", "to_sidecar")

    def __init__(self, file: TetFile) -> None:
        self._file = file
        self.to_numpy = _NumpySink(self)
        self.to_spill = _SpillSink(self)
        self.to_sidecar = _SidecarSink(self)

    def _run(
        self,
        method: str,
        dataset: str,
        axes: Sequence[int | str] | None = None,
        *,
        axis: int | str | None = None,
        write: TransformWrite = TransformWrite.NUMPY,
        path: str | PathLike[str] | None = None,
        selection: Sequence[Mapping[str, Any]] | None = None,
        device: str | None = None,
        raw: bool = False,
    ) -> np.ndarray | TransformResult | dict[str, Any]:
        """Run one transform method routed to ram, spill, or sidecar sink.

        ``write=ram`` uses native ``transform_to_numpy`` (full array in-process).
        ``spill`` / ``sidecar`` use :meth:`~tet.TetFile.execute` and parse
        sink-specific result types with ``.to_numpy()`` loaders.
        """
        ds = self._file.dataset(dataset)
        doc = build_query(
            dataset,
            selection=selection,
            transform=transform_op(
                ds, method, axes, axis=axis, path=str(self._file.path)
            ),
        )
        doc["write"] = write_to_wire(
            write,
            resolve_spill_path(path, self._file.path) if path is not None else None,
        )
        if device is not None:
            execution = dict(doc.get("execution") or {})
            execution["device"] = device
            doc = {**doc, "execution": execution}
        if write == TransformWrite.NUMPY:
            # Fast path: Rust materialize → ndarray (write: ram).
            if raw:
                out = self._file._execute_doc(
                    doc, device=device, raw=True, scalar_op="transform"
                )
                assert isinstance(out, dict)
                return out
            arr = self._file._inner.transform_to_numpy(doc)
            return np.asarray(arr)
        out = self._file._execute_doc(
            doc, device=device, raw=raw, scalar_op="transform"
        )
        if raw and isinstance(out, dict):
            return out
        if not isinstance(out, QueryResult):
            raise TypeError("transform execute expected QueryResult when raw=False")
        logical_shape = _logical_shape_from_query(out, fallback=ds.shape)
        return cast(
            TransformResult,
            parse_transform_result(
                out,
                method=method,
                sink=write.value,
                shape=logical_shape,
                dtype_tag=ds.dtype,
                raw=raw,
            ),
        )


def _logical_shape_from_query(
    out: QueryResult,
    *,
    fallback: tuple[int, ...],
) -> tuple[int, ...]:
    """Prefer ``catalog.shape`` from execute response over catalog record shape."""
    from tet._io.spill import logical_shape_from_raw

    return logical_shape_from_raw(out.raw) or fallback


class _TransformSink:
    """Shared base for fixed routing sinks."""

    __slots__ = ("_ops",)

    def __init__(self, ops: TransformOps) -> None:
        self._ops = ops


class _NumpySink(_TransformSink):
    """``f.transform.to_numpy`` — in-process dense export (wire ``write: ram``)."""


class _SpillSink(_TransformSink):
    """``f.transform.to_spill`` — row-major LE ``.bin`` (wire ``write: spill``)."""


class _SidecarSink(_TransformSink):
    """``f.transform.to_sidecar`` — derived ``.tet`` (wire ``write: sidecar``)."""


_TRANSFORM_SUMMARIES: dict[str, str] = {
    "zscore": "Z-score normalize (mean 0, std 1 per fold cell).",
    "minmax": "Scale to [0, 1] using min and max per fold cell.",
    "l1": "L1-normalize (divide by sum of absolute values per fold cell).",
    "l2": "L2-normalize (divide by Euclidean norm per fold cell).",
    "center": "Subtract mean per fold cell.",
    "scale": "Divide by population std (ddof=0) per fold cell.",
    "log1p": "Element-wise ``log1p``.",
    "sqrt": "Element-wise square root.",
    "softmax": "Softmax along the folded axes.",
}


def _make_numpy_method(method: str, summary: str) -> NumpyTransformFn:
    """Bind one transform name onto :class:`_NumpySink` (ram sink)."""

    def transform_fn(
        self: _NumpySink,
        dataset: str,
        axes: Sequence[int | str] | None = None,
        *,
        axis: int | str | None = None,
        selection: Sequence[Mapping[str, Any]] | None = None,
        device: str | None = None,
        raw: bool = False,
    ) -> np.ndarray | dict[str, Any]:
        return cast(
            np.ndarray | dict[str, Any],
            self._ops._run(
                method,
                dataset,
                axes,
                axis=axis,
                write=TransformWrite.NUMPY,
                selection=selection,
                device=device,
                raw=raw,
            ),
        )

    transform_fn.__name__ = method
    transform_fn.__doc__ = _transform_doc(summary, extra_params=_NUMPY_EXTRA)
    return transform_fn


def _make_path_method(
    method: str,
    summary: str,
    *,
    write: TransformWrite,
    extra_params: str,
) -> PathTransformFn:
    """Bind one transform name onto a spill or sidecar sink class."""

    def transform_fn(
        self: _SpillSink | _SidecarSink,
        dataset: str,
        axes: Sequence[int | str] | None = None,
        *,
        axis: int | str | None = None,
        path: str | PathLike[str] | None = None,
        selection: Sequence[Mapping[str, Any]] | None = None,
        device: str | None = None,
        raw: bool = False,
    ) -> TransformResult | dict[str, Any]:
        return cast(
            TransformResult | dict[str, Any],
            self._ops._run(
                method,
                dataset,
                axes,
                axis=axis,
                write=write,
                path=path,
                selection=selection,
                device=device,
                raw=raw,
            ),
        )

    transform_fn.__name__ = method
    transform_fn.__doc__ = _transform_doc(summary, extra_params=extra_params)
    return transform_fn  # type: ignore[return-value]


for _name, _summary in _TRANSFORM_SUMMARIES.items():
    setattr(_NumpySink, _name, _make_numpy_method(_name, _summary))
    setattr(
        _SpillSink,
        _name,
        _make_path_method(
            _name,
            _summary,
            write=TransformWrite.SPILL,
            extra_params=_SPILL_EXTRA,
        ),
    )
    setattr(
        _SidecarSink,
        _name,
        _make_path_method(
            _name,
            _summary,
            write=TransformWrite.SIDECAR,
            extra_params=_SIDECAR_EXTRA,
        ),
    )
