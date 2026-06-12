"""QC ops (nan_count, nan_mean, transform, …)."""

from __future__ import annotations

from pathlib import Path

import pytest

import tet
from tet import QueryResult, TransformWrite, build_query
from tet._query import transform_op
from tet._transform import write_to_wire


def test_clean_sample_has_no_non_finite(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert f.nan_count("temperature") == 0.0
    assert f.inf_count("temperature") == 0.0
    assert f.any_inf("temperature") is False


def test_nan_reducers_match_on_clean_sample(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert f.nan_mean("temperature") == pytest.approx(f.mean("temperature"))
    assert f.nan_std("temperature") == pytest.approx(f.std("temperature"))


def test_build_query_nan_mean(sample_path: Path) -> None:
    f = tet.open(sample_path)
    r = f.execute(build_query("temperature", nan_mean=[]))
    assert isinstance(r, QueryResult)
    assert r.scalar == pytest.approx(3.5)


def test_transform_write_to_wire() -> None:
    assert write_to_wire(TransformWrite.NUMPY) == "ram"
    assert write_to_wire(TransformWrite.SPILL, "/tmp/out.bin") == {
        "target": "spill",
        "path": "/tmp/out.bin",
    }
    assert write_to_wire(TransformWrite.SIDECAR) == "sidecar"


def test_transform_zscore_numpy_wire(large_path: Path) -> None:
    f = tet.open(large_path)
    ds = f.dataset("a")
    doc = build_query(
        "a",
        transform=transform_op(ds, "zscore"),
    )
    doc["write"] = "ram"
    r = f.execute(doc)
    assert isinstance(r, QueryResult)
    ex = r.execution or {}
    assert ex.get("transform_method") == "zscore"
    assert ex.get("memory_strategy") == "transform_ram"
    assert ex.get("operation_mean") is not None


def test_transform_to_numpy_raw(large_path: Path) -> None:
    f = tet.open(large_path)
    raw = f.transform.to_numpy.zscore("a", raw=True)
    assert isinstance(raw, dict)
    assert raw["execution"]["transform_method"] == "zscore"
