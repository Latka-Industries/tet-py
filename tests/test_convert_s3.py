"""tet.convert from s3:// : full object download, then the local importer (THI-507)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

import tet
import tet._native as _native


def test_s3_convert_rejects_zarr_and_unknown(tmp_path: Path) -> None:
    dst = tmp_path / "out.tet"
    with pytest.raises(ValueError, match="Zarr"):
        tet.convert("s3://bucket/store.zarr", dst)
    with pytest.raises(ValueError, match="Zarr"):
        tet.convert("s3://bucket/store/", dst)
    with pytest.raises(ValueError, match="unsupported s3 convert object"):
        tet.convert("s3://bucket/notes.txt", dst)
    with pytest.raises(ValueError, match="invalid s3 URI"):
        tet.convert("s3://bucket", dst)


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
