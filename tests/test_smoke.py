"""Smoke tests against vendored small fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

import tet


def test_versions() -> None:
    assert tet.__version__
    assert tet.core_version()


def test_summary_and_info(sample_path: Path) -> None:
    f = tet.open(sample_path)
    summary = f.summary()
    assert f.info() == summary
    names = [d["name"] for d in summary["datasets"]]
    assert "temperature" in names
    assert "superblock" in summary


def test_open_and_mean(sample_path: Path) -> None:
    f = tet.open(sample_path)
    assert f.path.name == "sample.tet"
    assert "temperature" in f.datasets()

    assert abs(f.mean("temperature") - 3.5) < 1e-9
