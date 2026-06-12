"""Typed results for :class:`~tet.TransformOps` sink methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tet._query import QueryResult
from tet._dtype import WIRE_DTYPE_TAG_V1
from tet._spill import (
    infer_spill_dtype_tag,
    load_spill_array,
    logical_shape_from_raw,
    normalize_path,
)

if TYPE_CHECKING:
    from tet._file import TetFile


@dataclass(frozen=True, slots=True)
class TransformStats:
    """Pass-1 fold statistics from transform execution."""

    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    var: float | None = None

    @classmethod
    def from_execution(cls, execution: dict[str, Any] | None) -> TransformStats:
        """Build pass-1 fold stats from ``execution.operation_*`` fields."""
        if not execution:
            return cls()
        return cls(
            mean=_float_or_none(execution.get("operation_mean")),
            std=_float_or_none(execution.get("operation_std")),
            min=_float_or_none(execution.get("operation_min")),
            max=_float_or_none(execution.get("operation_max")),
            var=_float_or_none(execution.get("operation_var")),
        )


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Base transform response (method, pass-1 stats, shape, wire JSON)."""

    method: str
    stats: TransformStats
    shape: tuple[int, ...] | None
    dtype_tag: int | None
    accepted: bool
    message: str | None
    memory_strategy: str | None
    raw: dict[str, Any]

    @classmethod
    def _base_fields(
        cls,
        query: QueryResult,
        *,
        method: str,
        shape: tuple[int, ...] | None,
        dtype_tag: int | None,
    ) -> dict[str, Any]:
        """Shared fields for sink-specific transform result dataclasses."""
        execution = query.execution if isinstance(query.execution, dict) else None
        return {
            "method": method,
            "stats": TransformStats.from_execution(execution),
            "shape": shape or _shape_from_raw(query.raw),
            "dtype_tag": dtype_tag or _dtype_tag_from_raw(query.raw),
            "accepted": query.accepted,
            "message": query.message,
            "memory_strategy": (
                str(execution["memory_strategy"])
                if execution and execution.get("memory_strategy") is not None
                else None
            ),
            "raw": query.raw,
        }


@dataclass(frozen=True, slots=True)
class NumpyTransformResult(TransformResult):
    """``to_numpy`` execute path (wire ``ram``) — capped preview lists from JSON.

    Full dense arrays are returned directly by :meth:`~tet.TransformOps.to_numpy`
    via native materialize; this type is used when parsing ``raw=False`` execute
    responses that include ``execution.f32_preview`` samples.
    """

    preview: list[float] | None
    preview_truncated: bool

    @classmethod
    def from_query(
        cls,
        query: QueryResult,
        *,
        method: str,
        shape: tuple[int, ...] | None = None,
        dtype_tag: int | None = None,
    ) -> NumpyTransformResult:
        """Parse a transform execute response for the ``ram`` sink."""
        execution = query.execution if isinstance(query.execution, dict) else {}
        preview, truncated = _preview_from_execution(execution)
        return cls(
            preview=preview,
            preview_truncated=truncated,
            **TransformResult._base_fields(
                query, method=method, shape=shape, dtype_tag=dtype_tag
            ),
        )


@dataclass(frozen=True, slots=True)
class SpillTransformResult(TransformResult):
    """``to_spill`` — dense row-major LE spill file (wire ``spill``, not ``.tet``).

    Attributes
    ----------
    path : pathlib.Path
        Spill file path from ``execution.spill_f32_path`` or ``spill_f64_path``.
    byte_len : int or None
        Byte length metadata from execution when present.
    """

    path: Path
    byte_len: int | None

    def to_numpy(self) -> np.ndarray:
        """Load the spill file as a shaped ``numpy.ndarray``.

        Returns
        -------
        numpy.ndarray
            Full transformed logical selection (same values as
            ``f.transform.to_numpy.<method>(...)`` when RAM export fits budget).

        Raises
        ------
        ValueError
            Missing ``dtype_tag`` or shape / byte length mismatch.
        FileNotFoundError
            Spill path does not exist.
        """
        if self.dtype_tag is None:
            raise ValueError("transform spill result missing dtype tag")
        return load_spill_array(
            self.path,
            shape=self.shape,
            dtype_tag=self.dtype_tag,
            byte_len=self.byte_len,
            raw=self.raw,
        )

    @classmethod
    def from_query(
        cls,
        query: QueryResult,
        *,
        method: str,
        shape: tuple[int, ...] | None = None,
        dtype_tag: int | None = None,
    ) -> SpillTransformResult:
        """Parse a transform execute response for the ``spill`` sink."""
        execution = query.execution if isinstance(query.execution, dict) else {}
        spill_path = execution.get("spill_f32_path")
        if not isinstance(spill_path, str) or not spill_path:
            spill_path = execution.get("spill_f64_path")
        if not isinstance(spill_path, str) or not spill_path:
            raise ValueError("transform spill result missing spill path in execution")
        byte_len = execution.get("spill_f32_bytes")
        if byte_len is None:
            byte_len = execution.get("spill_f64_bytes")
        resolved_tag = infer_spill_dtype_tag(
            execution,
            fallback=dtype_tag
            if dtype_tag is not None
            else WIRE_DTYPE_TAG_V1["f32"],
        )
        return cls(
            path=normalize_path(spill_path),
            byte_len=int(byte_len) if byte_len is not None else None,
            **TransformResult._base_fields(
                query,
                method=method,
                shape=shape or logical_shape_from_raw(query.raw),
                dtype_tag=resolved_tag,
            ),
        )


@dataclass(frozen=True, slots=True)
class SidecarTransformResult(TransformResult):
    """``to_sidecar`` — derived one-chunk ``.tet`` published beside the source.

    Sidecar pass-2 writes a draft ``.tet`` in platform cache and publishes to
    ``path`` (often ``{stem}.{method}.tet``). Dataset name is typically
    ``{source}-{method}``.

    Attributes
    ----------
    path : pathlib.Path or None
        Published sidecar path (engine may use ``spill_f32_path`` on wire).
    """

    path: Path | None

    def to_numpy(
        self,
        source: TetFile | Path | str,
        *,
        dataset: str | None = None,
    ) -> np.ndarray:
        """Open the sidecar and materialize one dataset as ``numpy.ndarray``.

        Parameters
        ----------
        source : TetFile or path-like
            Original ``.tet``; used to resolve a relative ``path`` when needed.
        dataset : str, optional
            Sidecar dataset name. When omitted, uses the sole catalog entry or
            the unique name ending in ``"-{method}"``.

        Returns
        -------
        numpy.ndarray
            Dense row-major array from :meth:`~tet.TetFile.read_numpy`.

        Raises
        ------
        ValueError
            Sidecar path missing or ambiguous dataset name.
        OSError, CatalogError, TetError
            See :func:`tet.open` and :meth:`~tet.TetFile.read_numpy`.
        """
        sidecar = self.open(source)
        name = dataset or _infer_sidecar_dataset(sidecar, self.method)
        return sidecar.read_numpy(name)

    def open(self, source: TetFile | Path | str) -> TetFile:
        """Open the sidecar ``.tet`` read-only.

        Parameters
        ----------
        source : TetFile or path-like
            Used only to resolve a relative ``path`` when the engine omits it.

        Raises
        ------
        ValueError
            If no sidecar path is available.
        OSError, CatalogError
            See :func:`tet.open`.
        """
        from tet._file import TetFile

        if self.path is None:
            raise ValueError("sidecar path not set in transform execution")
        path = self.path
        if not path.is_absolute():
            base = source.path if isinstance(source, TetFile) else Path(source)
            path = (base.parent / path).resolve()
        return TetFile.open(path)

    @classmethod
    def from_query(
        cls,
        query: QueryResult,
        *,
        method: str,
        shape: tuple[int, ...] | None = None,
        dtype_tag: int | None = None,
    ) -> SidecarTransformResult:
        """Parse a transform execute response for the ``sidecar`` sink."""
        execution = query.execution if isinstance(query.execution, dict) else {}
        # Engine publishes sidecar path as sidecar_path or spill_f32_path on wire.
        sidecar = execution.get("sidecar_path") or execution.get("spill_f32_path")
        path = normalize_path(sidecar) if isinstance(sidecar, str) and sidecar else None
        return cls(
            path=path,
            **TransformResult._base_fields(
                query, method=method, shape=shape, dtype_tag=dtype_tag
            ),
        )


def parse_transform_result(
    out: QueryResult | dict[str, Any],
    *,
    method: str,
    sink: str,
    shape: tuple[int, ...] | None,
    dtype_tag: int | None,
    raw: bool,
) -> (
    NumpyTransformResult
    | SpillTransformResult
    | SidecarTransformResult
    | dict[str, Any]
):
    """Build a sink-specific result from execute output.

    Parameters
    ----------
    sink
        Tetration wire write token: ``"ram"``, ``"spill"``, or ``"sidecar"``.
    """
    if raw:
        return out if isinstance(out, dict) else out.raw
    if not isinstance(out, QueryResult):
        raise TypeError("expected QueryResult when raw=False")
    if sink == "ram":
        return NumpyTransformResult.from_query(
            out, method=method, shape=shape, dtype_tag=dtype_tag
        )
    if sink == "spill":
        return SpillTransformResult.from_query(
            out, method=method, shape=shape, dtype_tag=dtype_tag
        )
    return SidecarTransformResult.from_query(
        out, method=method, shape=shape, dtype_tag=dtype_tag
    )


def _float_or_none(value: object) -> float | None:
    """Coerce JSON numbers to ``float``; reject bools."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _shape_from_raw(raw: dict[str, Any]) -> tuple[int, ...] | None:
    """Fallback shape from ``catalog.shape`` on a wire response."""
    catalog = raw.get("catalog")
    if isinstance(catalog, dict):
        shape = catalog.get("shape")
        if isinstance(shape, list):
            return tuple(int(x) for x in shape)
    return None


def _dtype_tag_from_raw(raw: dict[str, Any]) -> int | None:
    """Fallback dtype tag from ``catalog.dtype`` on a wire response."""
    catalog = raw.get("catalog")
    if isinstance(catalog, dict) and catalog.get("dtype") is not None:
        return int(catalog["dtype"])
    return None


def _infer_sidecar_dataset(sidecar: TetFile, method: str) -> str:
    """Resolve sidecar dataset name (``{source}-{method}`` convention)."""
    names = list(sidecar)
    if len(names) == 1:
        return names[0]
    suffix = f"-{method}"
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        "sidecar has multiple datasets; pass dataset= explicitly "
        f"(available: {names!r})"
    )


def _preview_from_execution(
    execution: dict[str, Any],
) -> tuple[list[float] | None, bool]:
    """Extract capped ``f32_preview`` / ``f64_preview`` lists from execution."""
    for key, trunc_key in (
        ("f32_preview", "f32_preview_truncated"),
        ("f64_preview", "f64_preview_truncated"),
    ):
        values = execution.get(key)
        if isinstance(values, list):
            truncated = bool(execution.get(trunc_key, False))
            return [float(x) for x in values], truncated
    return None, False
