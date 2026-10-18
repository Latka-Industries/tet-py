"""tet.convert from s3:// : single-file download or Zarr chunk GETs (THI-507 / THI-516)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

import tet
import tet._native as _native


def test_s3_convert_rejects_unknown_and_empty_bucket(tmp_path: Path) -> None:
    dst = tmp_path / "out.tet"
    with pytest.raises(ValueError, match="unsupported s3 convert object"):
        tet.convert("s3://bucket/notes.txt", dst)
    with pytest.raises(ValueError, match="invalid s3 URI"):
        tet.convert("s3://bucket", dst)


def test_s3_convert_zarr_without_endpoint_reports_config(tmp_path: Path) -> None:
    """Zarr URIs probe S3; missing ``TETRATION_S3_ENDPOINT`` must not look like a bad store."""
    dst = tmp_path / "out.tet"
    if os.environ.get("TETRATION_S3_ENDPOINT", "").strip():
        pytest.skip("endpoint set; covered by minio zarr tests")
    with pytest.raises(_native.CatalogError, match="TETRATION_S3_ENDPOINT"):
        tet.convert("s3://bucket/store.zarr", dst)


def test_minio_s3_npy_convert_matches_local(tmp_path: Path) -> None:
    endpoint = os.environ.get("TETRATION_S3_ENDPOINT", "").strip()
    if not endpoint:
        if os.environ.get("TETRATION_S3_REQUIRED") == "1":
            pytest.fail("TETRATION_S3_ENDPOINT is required but unset")
        pytest.skip("set TETRATION_S3_ENDPOINT to run MinIO convert tests")

    bucket = os.environ.get("TETRATION_S3_BUCKET", "tetration-test")
    key = f"convert-s3/{time.time_ns()}/pair.npy"
    arr = np.array([1.5, 2.5], dtype=np.float32)
    local_npy = tmp_path / "pair.npy"
    np.save(local_npy, arr)
    _native.upload_s3_object(bucket, key, local_npy.read_bytes())

    local_tet = tmp_path / "local.tet"
    remote_tet = tmp_path / "remote.tet"
    tet.convert(local_npy, local_tet, dataset="data")
    tet.convert(f"s3://{bucket}/{key}", remote_tet, dataset="data")

    with tet.open(local_tet) as local, tet.open(remote_tet) as remote:
        np.testing.assert_array_equal(local.read_numpy("data"), remote.read_numpy("data"))
        np.testing.assert_array_equal(remote.read_numpy("data"), arr)


def _write_v3_pair(root: Path) -> None:
    zarr = pytest.importorskip("zarr")
    g = zarr.open_group(root, mode="w")
    g.create_array(
        "data",
        data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        chunks=(2,),
    )


def _write_v2_pair(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
    temp = root / "temp"
    temp.mkdir()
    (temp / ".zarray").write_text(
        '{"zarr_format":2,"shape":[4],"chunks":[2],"dtype":"<f4","order":"C",'
        '"compressor":null,"filters":null,"fill_value":0}',
        encoding="utf-8",
    )
    payload = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
    (temp / "0").write_bytes(payload[:8])
    (temp / "1").write_bytes(payload[8:])


def _upload_tree(bucket: str, prefix: str, root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        _native.upload_s3_object(bucket, f"{prefix}/{rel}", path.read_bytes())


def test_minio_s3_zarr_convert_matches_local_directory(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    endpoint = os.environ.get("TETRATION_S3_ENDPOINT", "").strip()
    if not endpoint:
        if os.environ.get("TETRATION_S3_REQUIRED") == "1":
            pytest.fail("TETRATION_S3_ENDPOINT is required but unset")
        pytest.skip("set TETRATION_S3_ENDPOINT to run MinIO convert tests")

    bucket = os.environ.get("TETRATION_S3_BUCKET", "tetration-test")
    nanos = time.time_ns()
    v3 = tmp_path / "v3.zarr"
    v2 = tmp_path / "v2store"
    _write_v3_pair(v3)
    _write_v2_pair(v2)
    v3_prefix = f"zarr-s3/{nanos}/v3.zarr"
    v2_prefix = f"zarr-s3/{nanos}/v2store"
    _upload_tree(bucket, v3_prefix, v3)
    _upload_tree(bucket, v2_prefix, v2)
    _native.upload_s3_object(bucket, f"{v3_prefix}/noise.bin", b"not-a-chunk")

    try:
        _assert_s3_matches_local(bucket, v3_prefix, v3, tmp_path / "v3")
        _assert_s3_matches_local(bucket, f"{v3_prefix}/", v3, tmp_path / "v3slash")
        _assert_s3_matches_local(bucket, v2_prefix, v2, tmp_path / "v2")
    finally:
        for key in _native.list_s3_prefix(bucket, f"zarr-s3/{nanos}"):
            # list returns relative keys; rebuild full key for delete via overwrite empty? 
            # no remove API — leave objects; unique nanos prefix is enough for CI
            pass


def _assert_s3_matches_local(
    bucket: str,
    prefix: str,
    local: Path,
    scratch: Path,
) -> None:
    from tet.convert import _S3ZarrMap, _commit, _load_zarr, _s3_zarr_memory_store

    scratch.mkdir(parents=True, exist_ok=True)
    local_tet = scratch / "local.tet"
    remote_tet = scratch / "remote.tet"
    tet.convert(local, local_tet, format="zarr")

    store = _s3_zarr_memory_store(bucket, prefix.strip("/"))
    backing = store._store_dict
    assert isinstance(backing, _S3ZarrMap)
    history = "zarr" if (local / "zarr.json").is_file() else "zarr2"
    imports, got_history = _load_zarr(store, history=history)
    assert got_history == history
    assert "noise.bin" not in backing.fetched
    assert "noise.bin" not in list(backing)
    _commit(remote_tet, imports, history)
    # Public URI path must match too.
    public_tet = scratch / "public.tet"
    tet.convert(f"s3://{bucket}/{prefix}", public_tet)
    with tet.open(remote_tet) as a, tet.open(public_tet) as b:
        for name in a.dataset_names:
            np.testing.assert_array_equal(a.read_numpy(name), b.read_numpy(name))

    with tet.open(local_tet) as left, tet.open(remote_tet) as right:
        assert left.dataset_names == right.dataset_names
        for name in left.dataset_names:
            np.testing.assert_array_equal(left.read_numpy(name), right.read_numpy(name))
