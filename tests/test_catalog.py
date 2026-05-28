"""Dataset iteration and axis helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import tet

TETRATION_ROOT = Path(__file__).resolve().parents[2] / "tetration"
SAMPLE_TET = TETRATION_ROOT / "fixtures" / "small" / "tet" / "sample.tet"
LARGE_TET = TETRATION_ROOT / "fixtures" / "small" / "tet" / "large.tet"


@pytest.fixture(scope="module")
def sample_path() -> Path:
    if not SAMPLE_TET.is_file():
        pytest.skip(f"missing fixture (clone tetration next to tet-py): {SAMPLE_TET}")
    return SAMPLE_TET


def test_iter_dataset_names(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert list(f) == ["temperature"]
    assert list(f.iter_datasets())[0].name == "temperature"


def test_dataset_lookup(sample_path: Path) -> None:
    f = tet.open(sample_path)
    ds = f.dataset("temperature")
    assert ds.shape == (2, 3)
    assert ds.axis_index(0) == 0
    assert ds.axis_index(-1) == 1
    assert f.dataset(0).name == "temperature"
    assert f[0].name == "temperature"
    assert f["temperature"] == f.dataset("temperature")


def test_mean_by_axis_index() -> None:
    if not LARGE_TET.is_file():
        pytest.skip("missing large.tet")
    f = tet.open(LARGE_TET)
    out = f.query({"dataset": "a", "mean": [0]})
    assert out["accepted"] is True
    # partial reduction → vector mean, not a single operation_mean scalar
    assert out["execution"]["operation_reduced_shape"] == [64]


def test_dim_name_requires_metadata(sample_path: Path) -> None:
    ds = tet.open(sample_path).dataset("temperature")
    with pytest.raises(ValueError, match="dim_names"):
        ds.axis_index("time")
