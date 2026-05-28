"""Exception mapping from Rust errors."""

from __future__ import annotations

from pathlib import Path

import pytest

import tet

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
    with pytest.raises(tet.TetError):
        f.query("not json")


def test_validation_error_raises_tet_error(sample_path: Path) -> None:
    f = tet.open(sample_path)
    with pytest.raises(tet.TetError, match="dataset"):
        f.query({})
