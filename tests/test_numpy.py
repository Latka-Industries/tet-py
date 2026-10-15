"""Phase 2: NumPy read and transform materialize (ram, spill, sidecar)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tet
from tet import SpillTransformResult
from tet._core.dtype import WIRE_DTYPE_TAG_V1

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_EXPECTED_INTEGER = np.arange(1, 7).reshape(2, 3)
_INTEGER_READ_CASES = (
    ("multichunk_u8.tet", np.uint8, WIRE_DTYPE_TAG_V1["u8"]),
    ("multichunk_u16.tet", np.uint16, WIRE_DTYPE_TAG_V1["u16"]),
    ("multichunk_u32.tet", np.uint32, WIRE_DTYPE_TAG_V1["u32"]),
    ("multichunk_i16.tet", np.int16, WIRE_DTYPE_TAG_V1["i16"]),
    ("multichunk_i32.tet", np.int32, WIRE_DTYPE_TAG_V1["i32"]),
    ("multichunk_i64.tet", np.int64, WIRE_DTYPE_TAG_V1["i64"]),
)


def _beside(path: Path, name: str) -> Path:
    return path.parent / name


def test_read_numpy(sample_path: Path) -> None:
    f = tet.open(sample_path)
    ds = f.dataset("temperature")
    arr = f.read_numpy("temperature")
    assert arr.shape == ds.shape == (2, 3)
    assert arr.dtype == np.float32
    assert arr.mean() == pytest.approx(f.mean("temperature"))
    np.testing.assert_allclose(ds.to_numpy(f).ravel(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_read_numpy_zero_copy_contiguous_raw(tmp_path: Path) -> None:
    """Single-chunk raw payloads return a read-only memmap (THI-21)."""
    path = tmp_path / "zc_raw.tet"
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    tet.write_dataset(path, "a", expected, chunk_shape=(2, 3))
    f = tet.open(path)
    arr = f.read_numpy("a")
    np.testing.assert_array_equal(arr, expected)
    assert isinstance(arr, np.memmap)
    assert not arr.flags.writeable
    # View stays valid after closing the TetFile (separate memmap).
    del f
    np.testing.assert_array_equal(arr, expected)


def test_read_numpy_copy_fallback_multichunk(tmp_path: Path) -> None:
    """Multi-chunk layouts copy — tile concatenation ≠ dense row-major in general."""
    path = tmp_path / "zc_multi.tet"
    expected = np.arange(12, dtype=np.float32).reshape(4, 3)
    tet.write_dataset(path, "a", expected, chunk_shape=(2, 3))
    arr = tet.open(path).read_numpy("a")
    np.testing.assert_array_equal(arr, expected)
    assert not isinstance(arr, np.memmap)


def test_read_numpy_copy_fallback_partial_selection(tmp_path: Path) -> None:
    """Selection that only partially covers a tile must copy (not memmap)."""
    path = tmp_path / "zc_partial.tet"
    full = np.arange(12, dtype=np.float32).reshape(4, 3)
    tet.write_dataset(path, "a", full, chunk_shape=(4, 3))
    arr = tet.open(path).read_numpy(
        "a",
        selection=tet.selection_slices(tet.axis_slice(1, 3), tet.axis_slice()),
    )
    np.testing.assert_array_equal(arr, full[1:3])
    assert not isinstance(arr, np.memmap)


@pytest.mark.parametrize(
    ("fixture_name", "numpy_dtype", "wire_tag"),
    _INTEGER_READ_CASES,
    ids=[Path(c[0]).stem.removeprefix("multichunk_") for c in _INTEGER_READ_CASES],
)
def test_read_numpy_integer_roundtrip(
    fixture_name: str, numpy_dtype: np.dtype, wire_tag: int
) -> None:
    path = _FIXTURES / fixture_name
    if not path.is_file():
        pytest.skip(f"missing fixture: {path}")
    f = tet.open(path)
    ds = f.dataset("a")
    assert ds.dtype == wire_tag
    assert ds.shape == (2, 3)
    arr = f.read_numpy("a")
    assert arr.dtype == np.dtype(numpy_dtype)
    np.testing.assert_array_equal(arr, _EXPECTED_INTEGER.astype(numpy_dtype))
    np.testing.assert_array_equal(ds.to_numpy(f), arr)
    assert f.sum("a") == pytest.approx(21.0)


def test_read_numpy_f16() -> None:
    path = _FIXTURES / "multichunk_f16.tet"
    if not path.is_file():
        pytest.skip(f"missing fixture: {path}")
    f = tet.open(path)
    ds = f.dataset("a")
    assert ds.dtype == WIRE_DTYPE_TAG_V1["f16"]
    arr = f.read_numpy("a")
    assert arr.dtype == np.dtype("<f2")
    np.testing.assert_array_equal(arr.astype(np.float32), _EXPECTED_INTEGER.astype(np.float32))


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
