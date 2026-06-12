"""Preview sample arrays from query_execute."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tet


def test_query_execute_preview(large_path: Path) -> None:
    f = tet.open(large_path)
    r = f.query_execute({"dataset": "a", "mean": []}, preview=32, raw=False)
    arr = r.preview
    assert arr is not None
    assert arr.ndim == 1
    assert arr.size == 32
    assert arr.dtype == np.float32
    assert r.preview_truncated


def test_mean_preview_interchangeable(large_path: Path) -> None:
    f = tet.open(large_path)
    r = f.mean("a", preview=32)
    assert isinstance(r, tet.QueryResult)
    assert r.preview is not None
    assert r.preview.size == 32
    assert r.scalar is not None


def test_query_execute_no_preview_by_default(large_path: Path) -> None:
    f = tet.open(large_path)
    r = f.query_execute({"dataset": "a", "mean": []}, raw=False)
    assert r.preview is None
    assert r.preview_samples is None
