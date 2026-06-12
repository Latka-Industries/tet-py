"""plan_only and mean/sum helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import tet


def test_plan_only_has_no_execution(sample_path: Path) -> None:
    f = tet.open(sample_path)
    # Operations require execute; plan-only uses dataset selection without ops.
    out = f.plan_only({"dataset": "temperature"})
    assert out["accepted"] is True
    assert out.get("execution") is None


def test_sum_helper(sample_path: Path) -> None:
    f = tet.open(sample_path)
    # sample.tet temperature is 2×3, values 1..6 -> sum = 21
    assert abs(f.sum("temperature") - 21.0) < 1e-5


def test_numel_matches_count(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert f.numel("temperature") == f.count("temperature")


def test_open_expands_tilde(sample_path: Path) -> None:
    home = Path.home()
    if not str(sample_path).startswith(str(home)):
        pytest.skip("fixture not under home directory")
    tilde_path = "~" + str(sample_path)[len(str(home)) :]
    with tet.open(tilde_path) as f:
        assert "temperature" in f.datasets()


def test_context_manager_and_class_open(sample_path: Path) -> None:
    with tet.TetFile.open(sample_path) as f:
        assert "temperature" in f.datasets()
