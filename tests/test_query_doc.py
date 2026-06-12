"""Selection builders and quantile / histogram / cov / corr helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import tet
from tet import axis_slice, build_query, selection_slices


def test_build_query_selection() -> None:
    doc = build_query(
        "a",
        selection=selection_slices(
            axis_slice(0, 2),
            axis_slice(0, 2),
        ),
        mean=[],
    )
    assert doc["dataset"] == "a"
    assert doc["selection"] == [{"start": 0, "stop": 2}, {"start": 0, "stop": 2}]
    assert doc["mean"] == []


def test_mean_with_selection_subset(large_path: Path) -> None:
    f = tet.open(large_path)
    sel = selection_slices(axis_slice(0, 2), axis_slice(0, 2))
    doc = build_query("a", selection=sel, mean=[])
    r = f.execute(doc)
    assert r.scalar is not None


def test_quantile_on_large(large_path: Path) -> None:
    f = tet.open(large_path)
    q = f.quantile("a", 0.5)
    assert isinstance(q, float)


def test_histogram_on_large(large_path: Path) -> None:
    f = tet.open(large_path)
    r = f.histogram("a", bins=4)
    assert r.histogram_counts is not None
    assert len(r.histogram_counts) == 4


def test_covariance_on_sample(sample_path: Path) -> None:
    f = tet.open(sample_path)
    # temperature is 2×3; observation axis 1 → 2×2 covariance order
    r = f.covariance("temperature", axis=1)
    assert r.matrix_order == 2
    assert r.matrix is not None
    assert len(r.matrix) == 4
