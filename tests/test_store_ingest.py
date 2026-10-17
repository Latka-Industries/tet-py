"""Store ingest / publish wrappers (THI-506). MinIO when TETRATION_S3_ENDPOINT is set."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import tet
from tet import MemoryStore, S3Store, write_dataset


def _write_part(path: Path, name: str, vals: list[float]) -> Path:
    write_dataset(path, name, np.array(vals, dtype=np.float32))
    return path


def _parallel_ingest_publish_pin(store: MemoryStore | S3Store, tmp_path: Path) -> None:
    parts = [
        ("d0", [1.0, 2.0]),
        ("d1", [3.0, 4.0]),
        ("d2", [5.0, 6.0]),
        ("d3", [7.0, 8.0]),
    ]

    def stage(i: int, name: str, vals: list[float]) -> str:
        src = _write_part(tmp_path / f"{name}.tet", name, vals)
        return store.ingest(src, session_id=f"s{i}", part_id=f"p{i}")

    with ThreadPoolExecutor(max_workers=len(parts)) as pool:
        futures = [
            pool.submit(stage, i, name, vals) for i, (name, vals) in enumerate(parts)
        ]
        ids = [f.result() for f in futures]
    assert ids == [f"p{i}" for i in range(len(parts))]

    generation = store.publish(ids)
    assert generation == 1

    if isinstance(store, MemoryStore):
        pinned = store.open("memory://current")
    else:
        pinned = store.open("current")
    assert pinned.generation == 1
    assert pinned.sum("d0") == pytest.approx(3.0)
    assert pinned.sum("d3") == pytest.approx(15.0)

    # Lost CAS: seal gen 2, then publish with expected=None (create-only) → keep gen 1.
    store.seal(ids, generation=2)
    with pytest.raises(tet.CatalogError):
        store.cas_publish(2, expected=None)

    if isinstance(store, MemoryStore):
        still = store.open("memory://current")
    else:
        still = store.open("current")
    assert still.generation == 1
    np.testing.assert_allclose(still.read_numpy("d1"), np.array([3.0, 4.0], dtype=np.float32))
    assert pinned.generation == 1
    assert pinned.sum("d0") == pytest.approx(3.0)


def test_memory_parallel_ingest_publish_and_lost_cas(tmp_path: Path) -> None:
    _parallel_ingest_publish_pin(MemoryStore(), tmp_path)


def _minio_store(prefix: str) -> S3Store | None:
    endpoint = os.environ.get("TETRATION_S3_ENDPOINT", "").strip()
    if not endpoint:
        if os.environ.get("TETRATION_S3_REQUIRED") == "1":
            pytest.fail("TETRATION_S3_ENDPOINT is required but unset")
        return None
    return S3Store.connect(
        endpoint=endpoint,
        bucket=os.environ.get("TETRATION_S3_BUCKET", "tetration-test"),
        access_key_id=os.environ.get("TETRATION_S3_ACCESS_KEY", "tetration"),
        secret_access_key=os.environ.get("TETRATION_S3_SECRET_KEY", "tetration-secret"),
        prefix=prefix,
        region=os.environ.get("TETRATION_S3_REGION", "us-east-1"),
    )


def test_minio_parallel_ingest_publish_and_lost_cas(tmp_path: Path) -> None:
    prefix = f"tests/tet-py-ingest-{time.time_ns()}"
    store = _minio_store(prefix)
    if store is None:
        pytest.skip("set TETRATION_S3_ENDPOINT to run MinIO store tests")
    _parallel_ingest_publish_pin(store, tmp_path)


def test_open_s3_package_range_reads_match_local(tmp_path: Path) -> None:
    prefix = f"tests/tet-py-open-{time.time_ns()}"
    store = _minio_store(prefix)
    if store is None:
        pytest.skip("set TETRATION_S3_ENDPOINT to run MinIO store tests")
    src = _write_part(tmp_path / "a.tet", "a", [1.0, 2.0, 3.0, 4.0])
    store.ingest(src, session_id="s", part_id="a")
    store.publish(["a"])
    bucket = os.environ.get("TETRATION_S3_BUCKET", "tetration-test")
    uri = f"s3://{bucket}/{prefix}"
    with tet.open(src) as local, tet.open(uri) as remote:
        assert remote.generation == 1
        assert remote.sum("a") == pytest.approx(local.sum("a"))
        np.testing.assert_allclose(remote.read_numpy("a"), local.read_numpy("a"))
        assert "a" in remote.datasets()
    with tet.open(f"{uri}?generation=1") as pinned:
        assert pinned.generation == 1
        assert pinned.sum("a") == pytest.approx(10.0)
