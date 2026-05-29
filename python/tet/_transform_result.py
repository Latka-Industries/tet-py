"""Typed results for :class:`~tet.TransformOps` sink methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tet._query import QueryResult

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
    """``to_numpy`` — in-process preview (wire ``ram``)."""

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
    """``to_spill`` — dense row-major LE spill file (not a ``.tet``)."""

    path: Path
    byte_len: int | None

    @classmethod
    def from_query(
        cls,
        query: QueryResult,
        *,
        method: str,
        shape: tuple[int, ...] | None = None,
        dtype_tag: int | None = None,
    ) -> SpillTransformResult:
        execution = query.execution if isinstance(query.execution, dict) else {}
        spill_path = execution.get("spill_f32_path")
        if not isinstance(spill_path, str) or not spill_path:
            spill_path = execution.get("spill_f64_path")
        if not isinstance(spill_path, str) or not spill_path:
            raise ValueError("transform spill result missing spill path in execution")
        byte_len = execution.get("spill_f32_bytes")
        if byte_len is None:
            byte_len = execution.get("spill_f64_bytes")
        return cls(
            path=Path(spill_path),
            byte_len=int(byte_len) if byte_len is not None else None,
            **TransformResult._base_fields(
                query, method=method, shape=shape, dtype_tag=dtype_tag
            ),
        )


@dataclass(frozen=True, slots=True)
class SidecarTransformResult(TransformResult):
    """``to_sidecar`` — ``.tet`` beside source (engine support pending)."""

    path: Path | None

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
        execution = query.execution if isinstance(query.execution, dict) else {}
        sidecar = execution.get("sidecar_path") or execution.get("spill_f32_path")
        path = Path(sidecar) if isinstance(sidecar, str) and sidecar else None
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
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _shape_from_raw(raw: dict[str, Any]) -> tuple[int, ...] | None:
    catalog = raw.get("catalog")
    if isinstance(catalog, dict):
        shape = catalog.get("shape")
        if isinstance(shape, list):
            return tuple(int(x) for x in shape)
    return None


def _dtype_tag_from_raw(raw: dict[str, Any]) -> int | None:
    catalog = raw.get("catalog")
    if isinstance(catalog, dict) and catalog.get("dtype") is not None:
        return int(catalog["dtype"])
    return None


def _preview_from_execution(
    execution: dict[str, Any],
) -> tuple[list[float] | None, bool]:
    for key, trunc_key in (
        ("f32_preview", "f32_preview_truncated"),
        ("f64_preview", "f64_preview_truncated"),
    ):
        values = execution.get(key)
        if isinstance(values, list):
            truncated = bool(execution.get(trunc_key, False))
            return [float(x) for x in values], truncated
    return None, False
