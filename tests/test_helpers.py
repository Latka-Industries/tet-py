"""plan_only and mean/sum helpers."""

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


def test_plan_only_has_no_execution(sample_path: Path) -> None:
    f = tet.open(sample_path)
    # Operations require execute; plan-only uses dataset selection without ops.
    out = f.plan_only({"dataset": "temperature"})
    assert out["accepted"] is True
    assert out.get("execution") is None


def test_mean_helper(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert abs(f.mean("temperature") - 3.5) < 1e-9


def test_sum_helper(sample_path: Path) -> None:
    f = tet.open(sample_path)
    # sample.tet temperature is 2×3, values 1..6 -> sum = 21
    assert abs(f.sum("temperature") - 21.0) < 1e-5


def test_open_expands_tilde(sample_path: Path) -> None:
    home = Path.home()
    if not str(sample_path).startswith(str(home)):
        pytest.skip("fixture not under home directory")
    tilde_path = "~" + str(sample_path)[len(str(home)) :]
    with tet.open(tilde_path) as f:
        assert "temperature" in f.datasets()


def test_context_manager_and_class_open(sample_path: Path) -> None:
    with tet.TetFile.open(sample_path) as f:
        assert abs(f.mean("temperature") - 3.5) < 1e-9


def test_query_execute_device_cpu(sample_path: Path) -> None:
    f = tet.open(sample_path)
    out = f.query_execute({"dataset": "temperature", "mean": []}, device="cpu")
    assert out["accepted"] is True
    assert abs(out["execution"]["operation_mean"] - 3.5) < 1e-9
