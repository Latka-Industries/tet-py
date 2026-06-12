"""Shared test fixtures (from tetration 0.1.9 fixtures/small/tet)."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_TET = FIXTURES / "sample.tet"
LARGE_TET = FIXTURES / "large.tet"


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
