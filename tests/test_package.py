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
