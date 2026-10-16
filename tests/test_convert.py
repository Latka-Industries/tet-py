"""tet.convert: npy/npz in the core wheel, extras when installed."""

from __future__ import annotations

import builtins
import importlib.util
import json
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
    tet.convert(src, dst)
    with tet.open(dst) as f:
        np.testing.assert_array_equal(f.read_numpy("data"), arr)


def test_convert_npz_multi_array(tmp_path: Path) -> None:
    src = tmp_path / "multi.npz"
    dst = tmp_path / "out.tet"
    a = np.ones(2, dtype=np.float32)
    b = np.zeros((2, 2), dtype=np.int32)
    np.savez(src, a=a, b=b)
    tet.convert(src, dst)
    with tet.open(dst) as f:
        np.testing.assert_array_equal(f.read_numpy("a"), a)
        np.testing.assert_array_equal(f.read_numpy("b"), b)


def test_convert_missing_extra_h5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "x.h5"
    src.write_bytes(b"not a real hdf5")
    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "h5py":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"tet-py\[h5\]"):
        tet.convert(src, tmp_path / "out.tet")


def test_convert_unknown_suffix(tmp_path: Path) -> None:
    src = tmp_path / "x.bin"
    src.write_bytes(b"x")
    with pytest.raises(ValueError, match="infer"):
        tet.convert(src, tmp_path / "out.tet")


def test_convert_h5_group_roundtrip(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    src = tmp_path / "sample.h5"
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    with h5py.File(src, "w") as handle:
        handle.create_dataset("temp", data=values)
        group = handle.create_group("nested")
        group.create_dataset("bias", data=np.array([1, 2], dtype=np.int32))
        handle.create_dataset("label", data=np.array(["a", "b"], dtype="S1"))
    dst = tmp_path / "out.tet"
    tet.convert(src, dst)
    with tet.open(dst) as f:
        np.testing.assert_array_equal(f.read_numpy("temp"), values)
        np.testing.assert_array_equal(f.read_numpy("nested/bias"), np.array([1, 2], dtype=np.int32))
        assert "label" not in {ds.name for ds in f.iter_datasets()}


def test_convert_netcdf_roundtrip(tmp_path: Path) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    src = tmp_path / "sample.nc"
    values = np.arange(4, dtype=np.float64)
    with netCDF4.Dataset(src, "w") as handle:
        handle.createDimension("n", 4)
        var = handle.createVariable("temp", "f8", ("n",))
        var[:] = values
    dst = tmp_path / "out.tet"
    tet.convert(src, dst)
    with tet.open(dst) as f:
        np.testing.assert_array_equal(f.read_numpy("temp"), values)


def test_convert_zarr_v2_and_v3(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    values = np.arange(4, dtype=np.float32)
    for version, name in ((2, "v2.zarr"), (3, "v3.zarr")):
        store = tmp_path / name
        array = zarr.create_array(
            store,
            name="temp",
            shape=values.shape,
            dtype=values.dtype,
            zarr_format=version,
        )
        array[:] = values
        dst = tmp_path / f"{name}.tet"
        tet.convert(store, dst)
        with tet.open(dst) as f:
            np.testing.assert_array_equal(f.read_numpy("temp"), values)


def test_convert_csv_numeric_and_category(tmp_path: Path) -> None:
    if importlib.util.find_spec("pandas") is None:
        pytest.skip("pandas is not installed")
    src = tmp_path / "table.csv"
    src.write_text("temp,label\n1,red\n2,blue\n3,red\n", encoding="utf-8")
    dst = tmp_path / "out.tet"
    tet.convert(src, dst)
    with tet.open(dst) as f:
        np.testing.assert_array_equal(f.read_numpy("temp"), np.array([1, 2, 3], dtype=np.int64))
        np.testing.assert_array_equal(f.read_numpy("label"), np.array([0, 1, 0], dtype=np.int64))
        attrs = f.summary()["metadata"]["datasets"]["label"]["attrs"]
    note = json.loads(attrs["tetration_categories"])
    assert note["missing"] == -1
    assert note["codes"] == {"blue": 1, "red": 0}


def test_convert_parquet_numeric_and_category(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    src = tmp_path / "table.parquet"
    table = pa.table(
        {
            "temp": pa.array([1.5, 2.5], type=pa.float32()),
            "label": pa.array(["red", "blue"]),
            "flag": pa.array([True, False]),
        }
    )
    pq.write_table(table, src)
    dst = tmp_path / "out.tet"
    tet.convert(src, dst)
    with tet.open(dst) as f:
        np.testing.assert_array_equal(f.read_numpy("temp"), np.array([1.5, 2.5], dtype=np.float32))
        np.testing.assert_array_equal(f.read_numpy("label"), np.array([0, 1], dtype=np.int64))
        names = {ds.name for ds in f.iter_datasets()}
    assert "flag" not in names
