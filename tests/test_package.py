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
