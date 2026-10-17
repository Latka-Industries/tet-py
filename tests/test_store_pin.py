"""Store URI open + generation pin (THI-504)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tet
from tet import MemoryStore, write_dataset


def _publish_values(
    package: Path,
    src: Path,
    arr: np.ndarray,
    *,
    create: bool,
    session_id: str = "s0",
    part_id: str | None = None,
) -> int:
    write_dataset(src, "values", arr)
    chosen = tet.ingest(package, src, create=create, session_id=session_id, part_id=part_id)
    return tet.publish(package, [chosen])


def test_package_pin_survives_current_advance(tmp_path: Path) -> None:
    package = tmp_path / "demo.tet.d"
    gen1 = _publish_values(
        package,
        tmp_path / "a.tet",
        np.arange(6, dtype=np.float64).reshape(2, 3),
        create=True,
    )
    assert gen1 == 1

    pinned = tet.open(package)
    assert pinned.generation == 1
    first = pinned.read_numpy("values")
    assert first.mean() == pytest.approx(2.5)

    gen2 = _publish_values(
        package,
        tmp_path / "b.tet",
        np.full((2, 3), 9.0, dtype=np.float64),
        create=False,
        session_id="s1",
        part_id="values-b",
    )
    assert gen2 == 2

    np.testing.assert_array_equal(pinned.read_numpy("values"), first)
    assert pinned.mean("values") == pytest.approx(2.5)

    latest = tet.open(package)
    assert latest.generation == 2
    np.testing.assert_array_equal(latest.read_numpy("values"), np.full((2, 3), 9.0))


def test_uri_generation_and_memory_store_match_local(tmp_path: Path) -> None:
    package = tmp_path / "pkg.tet.d"
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    _publish_values(package, tmp_path / "src.tet", arr, create=True)

    local = tet.open(package)
    via_uri = tet.open(f"{package}?generation=1")
    assert via_uri.generation == 1
    np.testing.assert_array_equal(via_uri.read_numpy("values"), local.read_numpy("values"))
    assert via_uri.mean("values") == pytest.approx(local.mean("values"))

    store = MemoryStore.from_package(package)
    mem = store.open("memory://current")
    assert mem.generation == 1
    np.testing.assert_array_equal(mem.read_numpy("values"), arr)
    assert mem.mean("values") == pytest.approx(float(arr.mean()))

    _publish_values(
        package,
        tmp_path / "next.tet",
        np.zeros((2, 3), dtype=np.float64),
        create=False,
        session_id="s1",
        part_id="values-next",
    )
    np.testing.assert_array_equal(mem.read_numpy("values"), arr)
