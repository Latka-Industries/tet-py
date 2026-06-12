"""Phase 2: NumPy write, read roundtrip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tet
from tet import TetWriter, write_dataset


def test_write_read_roundtrip_f32(tmp_path: Path) -> None:
    path = tmp_path / "roundtrip.tet"
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    write_dataset(path, "temperature", arr, chunk_shape=(2, 3))
    f = tet.open(path)
    back = f.read_numpy("temperature")
    np.testing.assert_array_equal(back, arr)
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
