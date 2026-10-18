"""Open, ingest, and publish a local .tet.d package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tet
from tet import write_dataset


def test_open_ingest_publish_package(tmp_path: Path) -> None:
    src = tmp_path / "values.tet"
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    write_dataset(src, "values", arr)

    package = tmp_path / "demo.tet.d"
    part_id = tet.ingest(package, src, create=True)
    assert part_id == "values"
    generation = tet.publish(package, [part_id])
    assert generation == 1

    with tet.open(package) as f:
        np.testing.assert_array_equal(f.read_numpy("values"), arr)
        assert f.mean("values") == pytest.approx(float(arr.mean()))


def test_open_package_without_current(tmp_path: Path) -> None:
    package = tet.create(tmp_path / "empty.tet.d")
    with pytest.raises(tet.CatalogError, match="CURRENT"):
        tet.open(package)


def test_pack_keeps_package_by_default(tmp_path: Path) -> None:
    src = tmp_path / "values.tet"
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    write_dataset(src, "values", arr)

    package = tmp_path / "demo.tet.d"
    part_id = tet.ingest(package, src, create=True)
    generation = tet.publish(package, [part_id])
    dest = tmp_path / "demo.tet"
    packed = tet.pack(package, dest)
    assert packed == generation
    assert dest.is_file()
    assert package.is_dir()
    with tet.open(dest) as f:
        np.testing.assert_array_equal(f.read_numpy("values"), arr)


def test_pack_remove_package(tmp_path: Path) -> None:
    src = tmp_path / "values.tet"
    arr = np.arange(4, dtype=np.float32).reshape(2, 2)
    write_dataset(src, "values", arr)

    package = tmp_path / "rm.tet.d"
    part_id = tet.ingest(package, src, create=True)
    tet.publish(package, [part_id])
    dest = tmp_path / "rm.tet"
    tet.pack(package, dest, remove_package=True)
    assert dest.is_file()
    assert not package.exists()
    with tet.open(dest) as f:
        np.testing.assert_array_equal(f.read_numpy("values"), arr)


def test_publish_dry_run_leaves_current_unset(tmp_path: Path) -> None:
    src_a = tmp_path / "a.tet"
    src_b = tmp_path / "b.tet"
    write_dataset(src_a, "temp", np.array([1.0, 2.0], dtype=np.float64))
    write_dataset(src_b, "pressure", np.array([3.0, 4.0], dtype=np.float64))

    package = tmp_path / "dry.tet.d"
    tet.ingest(package, src_a, part_id="p0", session_id="s0", create=True)
    tet.ingest(package, src_b, part_id="p1", session_id="s1")
    assert tet.publish(package, ["p0", "p1"], dry_run=True) == 1
    with pytest.raises(tet.CatalogError, match="CURRENT"):
        tet.open(package)


def test_publish_rejects_differing_overlap(tmp_path: Path) -> None:
    src_a = tmp_path / "a.tet"
    src_b = tmp_path / "b.tet"
    write_dataset(src_a, "same", np.array([1.0, 2.0], dtype=np.float64))
    write_dataset(src_b, "same", np.array([9.0, 8.0], dtype=np.float64))

    package = tmp_path / "overlap.tet.d"
    tet.ingest(package, src_a, part_id="a", session_id="s0", create=True)
    tet.ingest(package, src_b, part_id="b", session_id="s1")
    with pytest.raises(tet.CatalogError, match="overlapping dataset claims"):
        tet.publish(package, ["a", "b"])
    with pytest.raises(tet.CatalogError, match="CURRENT"):
        tet.open(package)


def test_publish_identical_overlap_ok(tmp_path: Path) -> None:
    src = tmp_path / "same.tet"
    arr = np.array([1.0, 2.0], dtype=np.float64)
    write_dataset(src, "same", arr)

    package = tmp_path / "idem.tet.d"
    tet.ingest(package, src, part_id="a", session_id="s0", create=True)
    tet.ingest(package, src, part_id="b", session_id="s1")
    generation = tet.publish(package, ["a", "b"])
    assert generation == 1
    with tet.open(package) as f:
        np.testing.assert_array_equal(f.read_numpy("same"), arr)


def test_publish_schema_change_needs_replace(tmp_path: Path) -> None:
    src1 = tmp_path / "g1.tet"
    src2 = tmp_path / "g2.tet"
    write_dataset(src1, "temp", np.array([1.0, 2.0], dtype=np.float64))
    write_dataset(src2, "temp", np.arange(4, dtype=np.float64))

    package = tmp_path / "schema.tet.d"
    tet.ingest(package, src1, part_id="p0", session_id="s0", create=True)
    tet.publish(package, ["p0"])
    tet.ingest(package, src2, part_id="p1", session_id="s1")
    with pytest.raises(tet.CatalogError, match="schema conflict"):
        tet.publish(package, ["p1"])
    generation = tet.publish(package, ["p1"], replace=["temp"])
    assert generation == 2
    with tet.open(package) as f:
        np.testing.assert_array_equal(f.read_numpy("temp"), np.arange(4, dtype=np.float64))
