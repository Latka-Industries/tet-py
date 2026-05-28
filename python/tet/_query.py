"""Query helpers: op wrappers, [`QueryResult`], and execution field mapping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tet._errors import check_query_response
from tet._native import TetError

def _op_fields(
    name: str,
    *,
    scalar: str | None = None,
    reduced: str | None = None,
) -> tuple[str, str]:
    """Map wire op key → ``(operation_<scalar>, operation_reduced_<reduced>)``."""
    return (
        f"operation_{scalar if scalar is not None else name}",
        f"operation_reduced_{reduced if reduced is not None else name}",
    )


# Wire key → (scalar execution field, partial-reduction execution field).
_REDUCTION_WIRE_KEYS = (
    "sum",
    "mean",
    "min",
    "max",
    "var",
    "std",
    "product",
    "norm_l1",
    "norm_l2",
    "all_finite",
    "any_nan",
    "median",
    "nan_count",
    "inf_count",
    "null_count",
)

REDUCTION_OPS: dict[str, tuple[str, str]] = {
    key: _op_fields(key) for key in _REDUCTION_WIRE_KEYS
} | {
    "count": _op_fields("count", scalar="element_count", reduced="count"),
    "arg_min": _op_fields("arg_min", scalar="argmin_index", reduced="argmin"),
    "arg_max": _op_fields("arg_max", scalar="argmax_index", reduced="argmax"),
}

_QUERY_DOC_META = frozenset(
    {"dataset", "layout_version", "selection", "execution", "output"}
)


def reduction_doc(
    dataset: str, op: str, axes: Sequence[int | str]
) -> dict[str, Any]:
    if op not in REDUCTION_OPS:
        raise ValueError(f"unknown reduction op: {op!r}")
    return {"dataset": dataset, op: list(axes)}


def op_key_from_doc(doc: dict[str, Any]) -> str | None:
    """Single list-style reduction key on the document, if any."""
    ops = [k for k in doc if k in REDUCTION_OPS and k not in _QUERY_DOC_META]
    if len(ops) == 1:
        return ops[0]
    return None


@dataclass(frozen=True)
class QueryResult:
    """Parsed query response (use ``TetFile.query(..., raw=False)`` or op helpers)."""

    accepted: bool
    message: str | None
    scalar: float | int | bool | None
    reduced: list[float] | list[int] | list[bool] | None
    reduced_shape: list[int] | None
    plan: dict[str, Any] | None
    execution: dict[str, Any] | None
    raw: dict[str, Any]

    @property
    def value(self) -> float | int | bool | list[float] | list[int] | list[bool]:
        """Scalar aggregate, or partial-reduction vector when no scalar."""
        if self.scalar is not None:
            return self.scalar
        if self.reduced is not None:
            return self.reduced
        raise TetError("query result has no scalar or reduced value")

    @classmethod
    def from_response(
        cls,
        raw: dict[str, Any],
        *,
        op: str | None = None,
        path: str | None = None,
        require_execution: bool = False,
    ) -> QueryResult:
        check_query_response(
            raw, path=path, require_execution=require_execution
        )

        execution = raw.get("execution")
        if not isinstance(execution, dict):
            return cls(
                accepted=True,
                message=raw.get("message"),
                scalar=None,
                reduced=None,
                reduced_shape=None,
                plan=raw.get("plan") if isinstance(raw.get("plan"), dict) else None,
                execution=None,
                raw=raw,
            )

        scalar: float | int | bool | None = None
        reduced: list[float] | list[int] | list[bool] | None = None
        reduced_shape = execution.get("operation_reduced_shape")
        if isinstance(reduced_shape, list):
            reduced_shape = [int(x) for x in reduced_shape]
        else:
            reduced_shape = None

        if op and op in REDUCTION_OPS:
            scalar_key, reduced_key = REDUCTION_OPS[op]
            if execution.get(scalar_key) is not None:
                scalar = execution[scalar_key]
            reduced_val = execution.get(reduced_key)
            if isinstance(reduced_val, list):
                reduced = reduced_val

        return cls(
            accepted=True,
            message=raw.get("message"),
            scalar=scalar,
            reduced=reduced,
            reduced_shape=reduced_shape,
            plan=raw.get("plan") if isinstance(raw.get("plan"), dict) else None,
            execution=execution,
            raw=raw,
        )


def scalar_from_response(
    raw: dict[str, Any], op: str
) -> float | int | bool:
    """Extract a scalar reduction from a full response dict."""
    result = QueryResult.from_response(raw, op=op)
    if result.scalar is None:
        raise TetError(f"missing scalar for op {op!r} (partial reduction?)")
    return result.scalar
