"""QC ops (nan_count, nan_mean, transform, …) vs tetration fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

import tet
from tet import QueryResult, build_query
from tet._query_doc import transform_op

TETRATION_ROOT = Path(__file__).resolve().parents[2] / "tetration"
SAMPLE_TET = TETRATION_ROOT / "fixtures" / "small" / "tet" / "sample.tet"
LARGE_TET = TETRATION_ROOT / "fixtures" / "small" / "tet" / "large.tet"


@pytest.fixture(scope="module")
def sample_path() -> Path:
    if not SAMPLE_TET.is_file():
        pytest.skip(f"missing fixture: {SAMPLE_TET}")
    return SAMPLE_TET


@pytest.fixture(scope="module")
def large_path() -> Path:
    if not LARGE_TET.is_file():
        pytest.skip(f"missing fixture: {LARGE_TET}")
    return LARGE_TET


def test_nan_count_zero_on_clean_sample(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert f.nan_count("temperature") == 0.0


def test_inf_count_zero_on_clean_sample(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert f.inf_count("temperature") == 0.0


def test_any_inf_false_on_clean_sample(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert f.any_inf("temperature") is False


def test_nan_mean_matches_mean_on_clean_sample(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert f.nan_mean("temperature") == pytest.approx(f.mean("temperature"))


def test_nan_std_matches_std_on_clean_sample(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert f.nan_std("temperature") == pytest.approx(f.std("temperature"))


def test_build_query_nan_mean(sample_path: Path) -> None:
    f = tet.open(sample_path)
    r = f.execute(build_query("temperature", nan_mean=[]))
    assert isinstance(r, QueryResult)
    assert r.scalar == pytest.approx(3.5)


def test_transform_zscore_ram(large_path: Path) -> None:
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


def test_transform_method_on_file(large_path: Path) -> None:
    f = tet.open(large_path)
    r = f.transform("a", "zscore", write="ram")
    assert isinstance(r, QueryResult)
    assert r.execution is not None
    assert r.execution.get("transform_method") == "zscore"
