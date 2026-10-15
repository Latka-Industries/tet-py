"""tet.convert scaffolding (npy/npz + optional-extra stubs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tet


def test_convert_npy_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "a.npy"
    dst = tmp_path / "a.tet"
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(src, arr)
    out = tet.convert(src, dst, dataset="a")
    assert out == dst
    with tet.open(dst) as f:
        got = f.read_numpy("a")
    np.testing.assert_array_equal(got, arr)


def test_convert_npz_single_array(tmp_path: Path) -> None:
    src = tmp_path / "a.npz"
    dst = tmp_path / "a.tet"
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    np.savez(src, data=arr)
    tet.convert(src, dst, dataset="data")
    with tet.open(dst) as f:
        np.testing.assert_array_equal(f.read_numpy("data"), arr)


def test_convert_npz_multi_raises(tmp_path: Path) -> None:
    src = tmp_path / "multi.npz"
    np.savez(src, a=np.ones(2, dtype=np.float32), b=np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="exactly one"):
        tet.convert(src, tmp_path / "out.tet")


def test_convert_missing_extra_h5(tmp_path: Path) -> None:
    src = tmp_path / "x.h5"
    src.write_bytes(b"not a real hdf5")
    with pytest.raises(ImportError, match=r"tet-py\[h5\]"):
        tet.convert(src, tmp_path / "out.tet")


def test_convert_unknown_suffix(tmp_path: Path) -> None:
    src = tmp_path / "x.bin"
    src.write_bytes(b"x")
    with pytest.raises(ValueError, match="infer"):
        tet.convert(src, tmp_path / "out.tet")
