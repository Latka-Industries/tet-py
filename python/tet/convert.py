"""Optional-format → ``.tet`` orchestration (Phase 3 scaffolding).

NumPy ``.npy`` / ``.npz`` work with the core wheel. Other formats need optional
extras (``pip install tet-py[h5]``, …) and are stubbed until implemented.
Rust ``tet convert`` remains the fast path when native codecs are installed.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from tet._io.writer import write_dataset

_EXTRA_BY_SUFFIX: dict[str, tuple[str, str]] = {
    ".h5": ("h5", "h5py"),
    ".hdf5": ("h5", "h5py"),
    ".nc": ("netcdf", "netCDF4"),
    ".cdf": ("netcdf", "netCDF4"),
    ".zarr": ("zarr", "zarr"),
    ".csv": ("csv", "pandas"),
    ".parquet": ("parquet", "pyarrow"),
    ".pq": ("parquet", "pyarrow"),
}


def convert(
    src: str | PathLike[str],
    dst: str | PathLike[str],
    *,
    dataset: str = "data",
    format: str | None = None,
    **write_kw: Any,
) -> Path:
    """Convert ``src`` into a one-dataset ``.tet`` at ``dst``.

    Parameters
    ----------
    src : path-like
        Source file (``.npy`` / ``.npz`` supported in the core wheel).
    dst : path-like
        Destination ``.tet`` path.
    dataset : str, default ``"data"``
        Catalog name for the written dataset.
    format : str, optional
        Force format: ``npy``, ``npz``, ``h5``, ``netcdf``, ``zarr``, ``csv``,
        ``parquet``. Default: infer from ``src`` suffix.
    **write_kw
        Forwarded to :func:`~tet.write_dataset` (e.g. ``chunk_shape``, ``dim_names``).

    Returns
    -------
    pathlib.Path
        Path to the written ``.tet`` file.

    Raises
    ------
    FileNotFoundError
        If ``src`` does not exist.
    ImportError
        If an optional extra is required but not installed.
    NotImplementedError
        If the format is recognized but not yet wired in Python.
    ValueError
        If the format cannot be inferred or is unsupported.
    """
    src_path = Path(src).expanduser()
    dst_path = Path(dst).expanduser()
    kind = (format or _infer_format(src_path)).lower()

    if kind == "zarr":
        if not src_path.exists():
            raise FileNotFoundError(src_path)
    elif not src_path.is_file():
        raise FileNotFoundError(src_path)

    if kind in ("npy", "npz"):
        return _convert_numpy(src_path, dst_path, dataset=dataset, **write_kw)
    if kind in ("h5", "hdf5", "netcdf", "zarr", "csv", "parquet"):
        _require_or_stub(kind, src_path)
    raise ValueError(
        f"unsupported convert format {kind!r} for {src_path} "
        f"(supported now: npy, npz; extras stubbed: h5, netcdf, zarr, csv, parquet)"
    )


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix == ".npz":
        return "npz"
    if suffix in _EXTRA_BY_SUFFIX:
        return _EXTRA_BY_SUFFIX[suffix][0]
    raise ValueError(
        f"cannot infer convert format from {path.name!r}; pass format= "
        f"(npy, npz, h5, netcdf, zarr, csv, parquet)"
    )


def _convert_numpy(
    src: Path,
    dst: Path,
    *,
    dataset: str,
    **write_kw: Any,
) -> Path:
    loaded: Any = np.load(src, allow_pickle=False)
    if hasattr(loaded, "files"):
        keys = list(loaded.files)
        if len(keys) != 1:
            loaded.close()
            raise ValueError(
                f"npz {src} has {len(keys)} arrays {keys!r}; "
                "convert expects exactly one (or pass a .npy)"
            )
        array = np.asarray(loaded[keys[0]])
        loaded.close()
    else:
        array = np.asarray(loaded)
    return write_dataset(
        dst,
        dataset,
        array,
        history_op="convert",
        history_source=str(src),
        **write_kw,
    )


def _require_or_stub(kind: str, src: Path) -> None:
    """Raise ImportError with install hint, else NotImplementedError once deps exist."""
    extra_map = {
        "h5": ("h5", "h5py"),
        "hdf5": ("h5", "h5py"),
        "netcdf": ("netcdf", "netCDF4"),
        "zarr": ("zarr", "zarr"),
        "csv": ("csv", "pandas"),
        "parquet": ("parquet", "pyarrow"),
    }
    extra, pkg = extra_map[kind]
    try:
        __import__(pkg)
    except ImportError as exc:
        raise ImportError(
            f"convert from {src.suffix or kind!r} requires optional extra "
            f"`tet-py[{extra}]` (pip install 'tet-py[{extra}]' / "
            f"uv sync --extra {extra})"
        ) from exc
    raise NotImplementedError(
        f"tet.convert({kind!r}) is not implemented yet; use the Rust `tet convert` "
        "CLI or load the array in Python and call tet.write_dataset"
    )
