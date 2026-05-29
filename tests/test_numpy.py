"""Phase 2: NumPy read and transform materialize."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tet

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


def test_read_numpy_full_dataset(sample_path: Path) -> None:
    f = tet.open(sample_path)
    arr = f.read_numpy("temperature")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)
    assert arr.dtype == np.float32
    assert arr.mean() == pytest.approx(f.mean("temperature"))


def test_dataset_to_numpy(sample_path: Path) -> None:
    f = tet.open(sample_path)
    ds = f.dataset("temperature")
    arr = ds.to_numpy(f)
    assert arr.shape == ds.shape
    np.testing.assert_allclose(arr.ravel(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_transform_to_numpy_returns_array(large_path: Path) -> None:
    f = tet.open(large_path)
    arr = f.transform.to_numpy.zscore("a")
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (34, 64)
    assert arr.dtype == np.float32
    assert float(np.nanmean(arr)) == pytest.approx(0.0, abs=1e-4)
