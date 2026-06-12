"""Phase 2: NumPy read and transform materialize (ram, spill, sidecar)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tet
from tet import SpillTransformResult


def _beside(path: Path, name: str) -> Path:
    return path.parent / name


def test_normalize_path_strips_windows_extended_prefix() -> None:
    import os

    from tet._io.spill import normalize_path

    if os.name != "nt":
        pytest.skip("Windows only")
    expected = Path("D:/a/tet-py/tet_py_read_spill.bin").resolve()
    assert normalize_path("//?/D:/a/tet-py/tet_py_read_spill.bin") == expected


def test_read_numpy(sample_path: Path) -> None:
    f = tet.open(sample_path)
    ds = f.dataset("temperature")
    arr = f.read_numpy("temperature")
    assert arr.shape == ds.shape == (2, 3)
    assert arr.dtype == np.float32
    assert arr.mean() == pytest.approx(f.mean("temperature"))
    np.testing.assert_allclose(ds.to_numpy(f).ravel(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_transform_ram_zscore(large_path: Path) -> None:
    f = tet.open(large_path)
    arr = f.transform.to_numpy.zscore("a")
    assert arr.shape == (34, 64)
    assert arr.dtype == np.float32
    assert float(np.nanmean(arr)) == pytest.approx(0.0, abs=1e-4)


def test_read_spill_matches_ram(large_path: Path) -> None:
    spill_path = _beside(large_path, "tet_py_read_spill.bin")
    spill_path.unlink(missing_ok=True)
    try:
        f = tet.open(large_path)
        result = f.read_spill("a", path=spill_path)
        assert result.memory_strategy == "mmap_spill"
        assert result.path == spill_path.resolve()
        np.testing.assert_allclose(result.to_numpy(), f.read_numpy("a"))
    finally:
        spill_path.unlink(missing_ok=True)


def test_transform_spill_matches_ram(large_path: Path) -> None:
    f = tet.open(large_path)
    ram = f.transform.to_numpy.zscore("a")

    cases: tuple[tuple[Path | str, Path], ...] = (
        (_beside(large_path, "tet_py_transform_spill.bin"), _beside(large_path, "tet_py_transform_spill.bin")),
        ("tet_py_relative_spill.bin", _beside(large_path, "tet_py_relative_spill.bin")),
    )
    for path_arg, spill_path in cases:
        spill_path.unlink(missing_ok=True)
        try:
            result = f.transform.to_spill.zscore("a", path=path_arg)
            assert isinstance(result, SpillTransformResult)
            assert result.memory_strategy == "transform_spill"
            assert result.path == spill_path.resolve()
            np.testing.assert_allclose(result.to_numpy(), ram)
        finally:
            spill_path.unlink(missing_ok=True)


def test_transform_sidecar_matches_ram(large_path: Path) -> None:
    sidecar_path = _beside(large_path, "tet_py_sidecar.tet")
    sidecar_path.unlink(missing_ok=True)
    try:
        f = tet.open(large_path)
        result = f.transform.to_sidecar.zscore("a", path=sidecar_path)
        assert result.memory_strategy == "transform_sidecar"
        np.testing.assert_allclose(result.to_numpy(f), f.transform.to_numpy.zscore("a"))
    finally:
        sidecar_path.unlink(missing_ok=True)
