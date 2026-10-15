"""Phase 2: NumPy write, read roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tet
from tet import TetWriter, write_dataset

# 0.1.9 write surface is float32 / float64 only (integer writes land after 0.1.10 pin).
_UNSUPPORTED_WRITE_DTYPES = (
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.int16,
    np.float16,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["f32", "f64"])
def test_write_read_roundtrip_float(tmp_path: Path, dtype: np.dtype) -> None:
    path = tmp_path / f"roundtrip_{np.dtype(dtype).name}.tet"
    arr = np.arange(6, dtype=dtype).reshape(2, 3)
    write_dataset(path, "temperature", arr, chunk_shape=(2, 3))
    f = tet.open(path)
    back = f.read_numpy("temperature")
    np.testing.assert_array_equal(back, arr)
    assert back.dtype == np.dtype(dtype)
    assert f.mean("temperature") == pytest.approx(float(arr.mean()))


def test_writer_session_metadata(tmp_path: Path) -> None:
    path = tmp_path / "meta.tet"
    arr = np.ones((2, 3), dtype=np.float32)
    w = TetWriter.create(path)
    w.push_history_event("write", "pytest")
    w.write_dataset(
        "temperature",
        arr,
        chunk_shape=(2, 3),
        attrs={"units": "K"},
        dim_names=("row", "col"),
        coords={"row": ("r0", "r1")},
    )
    out = w.commit()
    assert out == path
    summary = tet.open(path).summary()
    assert summary["history"][-1]["op"] == "write"
    ds_meta = summary["metadata"]["datasets"]["temperature"]
    assert ds_meta["attrs"]["units"] == "K"
    assert ds_meta["dim_names"] == ["row", "col"]


def test_write_f64_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "f64.tet"
    arr = np.linspace(0.0, 1.0, 6, dtype=np.float64).reshape(2, 3)
    write_dataset(path, "values", arr)
    back = tet.open(path).read_numpy("values")
    assert back.dtype == np.float64
    np.testing.assert_allclose(back, arr)


@pytest.mark.parametrize("dtype", _UNSUPPORTED_WRITE_DTYPES, ids=lambda d: np.dtype(d).name)
def test_write_rejects_unsupported_dtype(tmp_path: Path, dtype: np.dtype) -> None:
    path = tmp_path / f"reject_{np.dtype(dtype).name}.tet"
    arr = np.arange(6, dtype=dtype).reshape(2, 3)
    with pytest.raises(TypeError, match=r"unsupported numpy dtype .*supported: float32, float64"):
        write_dataset(path, "values", arr)
    assert not path.exists()


def test_writer_session_rejects_unsupported_dtype(tmp_path: Path) -> None:
    path = tmp_path / "reject_session.tet"
    w = TetWriter.create(path)
    with pytest.raises(TypeError, match=r"unsupported numpy dtype"):
        w.write_dataset("values", np.ones((2, 3), dtype=np.int32))
    assert w.dataset_count == 0


def test_append_dataset(tmp_path: Path, sample_path: Path) -> None:
    path = tmp_path / "appended.tet"
    import shutil

    shutil.copy(sample_path, path)
    extra = np.full((2, 3), 9.0, dtype=np.float32)
    w = TetWriter.open_append(path)
    w.write_dataset("humidity", extra, chunk_shape=(2, 3))
    w.commit()
    f = tet.open(path)
    assert "humidity" in f.dataset_names
    np.testing.assert_array_equal(f.read_numpy("humidity"), extra)
