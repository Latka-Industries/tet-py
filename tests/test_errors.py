"""Exception mapping and file-specific error messages."""

from __future__ import annotations

from pathlib import Path

import pytest

import tet
from tet import UnknownAxisError, UnknownDatasetError

TETRATION_ROOT = Path(__file__).resolve().parents[2] / "tetration"
SAMPLE_TET = TETRATION_ROOT / "fixtures" / "small" / "tet" / "sample.tet"


@pytest.fixture(scope="module")
def sample_path() -> Path:
    if not SAMPLE_TET.is_file():
        pytest.skip(f"missing fixture (clone tetration next to tet-py): {SAMPLE_TET}")
    return SAMPLE_TET


def test_missing_file_raises_os_error() -> None:
    with pytest.raises(OSError):
        tet.open("/nonexistent/does-not-exist.tet")


def test_invalid_query_json_raises_tet_error(sample_path: Path) -> None:
    f = tet.open(sample_path)
    with pytest.raises(tet.TetError, match="valid JSON"):
        f.query("not json")


def test_validation_error_raises_tet_error(sample_path: Path) -> None:
    f = tet.open(sample_path)
    with pytest.raises(tet.TetError, match="dataset"):
        f.query({})


def test_unknown_dataset_on_lookup(sample_path: Path) -> None:
    f = tet.open(sample_path)
    with pytest.raises(UnknownDatasetError, match="temperature") as exc:
        f.dataset("temprature")
    assert exc.value.available == ("temperature",)
    assert "did you mean" in str(exc.value).lower()


def test_unknown_dataset_on_mean(sample_path: Path) -> None:
    f = tet.open(sample_path)
    with pytest.raises(UnknownDatasetError, match="available datasets"):
        f.mean("nope")


def test_unknown_dataset_on_query_doc(sample_path: Path) -> None:
    f = tet.open(sample_path)
    with pytest.raises(UnknownDatasetError):
        f.query({"dataset": "nope", "mean": []})


def test_unknown_axis_name(sample_path: Path) -> None:
    ds = tet.open(sample_path).dataset("temperature")
    with pytest.raises(UnknownAxisError, match="dim_names"):
        ds.axis_index("time", path=sample_path)


def test_unknown_axis_index(sample_path: Path) -> None:
    ds = tet.open(sample_path).dataset("temperature")
    with pytest.raises(UnknownAxisError, match="ndim=2"):
        ds.axis_index(9, path=sample_path)


def test_typing_stub_lists_dataset(sample_path: Path) -> None:
    stub = tet.typing_stub(sample_path)
    assert "temperature" in stub and "Literal" in stub
    assert "DatasetName" in stub
