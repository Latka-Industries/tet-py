"""tet.export: HDF5 / NetCDF / Zarr via optional extras."""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

import tet


def _write_sample_tet(tmp_path: Path) -> Path:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    nested = np.array([1, 2, 3], dtype=np.int32)
    dst = tmp_path / "sample.tet"
    writer = tet.TetWriter.create(dst)
    writer.write_dataset(
        "temp",
        values,
        dim_names=["y", "x"],
        attrs={"units": "K", "long_name": "temperature"},
    )
    writer.write_dataset("nested/bias", nested, attrs={"units": "1"})
    return writer.commit()


def test_export_missing_extra_h5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = _write_sample_tet(tmp_path)
    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "h5py":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"tet-py\[h5\]"):
        tet.export(src, tmp_path / "out.h5")


def test_export_unknown_suffix(tmp_path: Path) -> None:
    src = _write_sample_tet(tmp_path)
    with pytest.raises(ValueError, match="infer"):
        tet.export(src, tmp_path / "out.bin")


def test_export_dst_exists(tmp_path: Path) -> None:
    src = _write_sample_tet(tmp_path)
    dst = tmp_path / "out.h5"
    dst.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        tet.export(src, dst, format="h5")


def test_export_h5_roundtrip(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    src = _write_sample_tet(tmp_path)
    dst = tmp_path / "out.h5"
    out = tet.export(src, dst)
    assert out == dst
    with h5py.File(dst, "r") as handle:
        np.testing.assert_array_equal(handle["temp"][:], np.arange(12, dtype=np.float32).reshape(3, 4))
        assert handle["temp"].attrs["units"] == "K"
        assert handle["temp"].attrs["dim_names"] == "y,x"
        np.testing.assert_array_equal(handle["nested/bias"][:], np.array([1, 2, 3], dtype=np.int32))


def test_export_netcdf_roundtrip(tmp_path: Path) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    src = _write_sample_tet(tmp_path)
    dst = tmp_path / "out.nc"
    tet.export(src, dst)
    with netCDF4.Dataset(dst, "r") as handle:
        np.testing.assert_array_equal(handle.variables["temp"][:], np.arange(12, dtype=np.float32).reshape(3, 4))
        assert handle.variables["temp"].units == "K"
        assert tuple(handle.variables["temp"].dimensions) == ("y", "x")
        nested = handle.groups["nested"]
        np.testing.assert_array_equal(nested.variables["bias"][:], np.array([1, 2, 3], dtype=np.int32))


def test_export_zarr_roundtrip(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    src = _write_sample_tet(tmp_path)
    dst = tmp_path / "out.zarr"
    tet.export(src, dst)
    root = zarr.open_group(dst, mode="r")
    np.testing.assert_array_equal(root["temp"][:], np.arange(12, dtype=np.float32).reshape(3, 4))
    assert root["temp"].attrs["units"] == "K"
    assert list(root["temp"].attrs["dim_names"]) == ["y", "x"]
    np.testing.assert_array_equal(root["nested"]["bias"][:], np.array([1, 2, 3], dtype=np.int32))


def test_tetfile_export_method(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    src = _write_sample_tet(tmp_path)
    dst = tmp_path / "via_method.h5"
    with tet.open(src) as handle:
        handle.export(dst)
    assert dst.is_file()
