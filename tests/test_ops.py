"""Reduction op helpers and execute(raw=)."""

from __future__ import annotations

from pathlib import Path

import pytest

import tet
from tet import QueryResult


def test_min_max_on_sample(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert f.min("temperature") == 1.0
    assert f.max("temperature") == 6.0


def test_execute_returns_query_result(sample_path: Path) -> None:
    f = tet.open(sample_path)
    r = f.execute({"dataset": "temperature", "mean": []})
    assert isinstance(r, QueryResult)
    assert r.scalar == pytest.approx(3.5)
    assert r.value == pytest.approx(3.5)


def test_query_raw_false_matches_helper(sample_path: Path) -> None:
    f = tet.open(sample_path)
    r = f.query({"dataset": "temperature", "sum": []}, raw=False)
    assert isinstance(r, QueryResult)
    assert r.scalar == pytest.approx(21.0)


def test_query_raw_true_is_full_dict(sample_path: Path) -> None:
    f = tet.open(sample_path)
    out = f.query({"dataset": "temperature", "mean": []}, raw=True)
    assert isinstance(out, dict)
    assert out["execution"]["operation_mean"] == pytest.approx(3.5)


def test_execute_device_cpu(sample_path: Path) -> None:
    f = tet.open(sample_path)
    r = f.execute(
        {"dataset": "temperature", "mean": []},
        device="cpu",
    )
    assert r.scalar == pytest.approx(3.5)


def test_plan_execute(sample_path: Path) -> None:
    f = tet.open(sample_path)
    r = f.execute({"dataset": "temperature"}, plan=True)
    assert r.execution is None
    assert r.accepted is True
