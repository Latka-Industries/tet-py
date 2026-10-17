"""``.tet`` → foreign formats (HDF5 / NetCDF / Zarr).

Uses optional extras (``h5py``, ``netCDF4``, ``zarr``), same shape as
:mod:`tet.convert` — not the native ``tetration-hdf5`` / ``tetration-netcdf``
features. Dense numeric datasets only; ``dim_names`` and basic string attrs are
carried when present in the footer.

Compared with Rust ``tet export``:

* values are densified via :meth:`~tet.TetFile.read_numpy` (full materialize)
* Zarr is written through the ``zarr`` library (not a byte-identical CLI store)
* typed CF packing attrs (``scale_factor``, ``_FillValue``, …) are not rewritten
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from tet.catalog import Dataset
from tet.file import TetFile

# Match tetration export `BASIC_EXPORT_ATTRS`.
_BASIC_ATTRS = frozenset(
    {"units", "long_name", "standard_name", "description", "title"}
)

_EXTRA_BY_KIND: dict[str, tuple[str, str]] = {
    "h5": ("h5", "h5py"),
    "hdf5": ("h5", "h5py"),
    "netcdf": ("netcdf", "netCDF4"),
    "zarr": ("zarr", "zarr"),
}

_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".h5": "h5",
    ".hdf5": "h5",
    ".hdf": "h5",
    ".nc": "netcdf",
    ".netcdf": "netcdf",
    ".nc4": "netcdf",
    ".cdf": "netcdf",
    ".zarr": "zarr",
}

_SUPPORTED_FORMATS = "h5, netcdf, zarr"


def export(
    src: str | PathLike[str],
    dst: str | PathLike[str],
    *,
    format: str | None = None,
) -> Path:
    """Export a ``.tet`` file to HDF5, NetCDF, or Zarr.

    Parameters
    ----------
    src : path-like
        Source ``.tet`` file.
    dst : path-like
        Destination path: ``.h5`` / ``.hdf5``, ``.nc`` / ``.netcdf``, or a
        ``.zarr`` directory (must not exist).
    format : str, optional
        Force format: ``h5``, ``netcdf``, ``zarr``. Default: infer from ``dst``.

    Returns
    -------
    pathlib.Path
        Path to the written file or Zarr store.

    Raises
    ------
    FileNotFoundError
        If ``src`` does not exist.
    FileExistsError
        If ``dst`` already exists.
    ImportError
        If an optional extra is required but not installed.
    ValueError
        If the format cannot be inferred or is unsupported, or the ``.tet`` has
        no datasets.
    """
    src_path = Path(src).expanduser()
    dst_path = Path(dst).expanduser()
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    if dst_path.exists():
        raise FileExistsError(dst_path)

    kind = (format or _infer_format(dst_path)).lower()
    if kind == "hdf5":
        kind = "h5"
    if kind not in ("h5", "netcdf", "zarr"):
        raise ValueError(
            f"unsupported export format {kind!r} for {dst_path} ({_SUPPORTED_FORMATS})"
        )
    _require(kind, dst_path)

    with TetFile.open(src_path) as handle:
        datasets = list(handle.iter_datasets())
        if not datasets:
            raise ValueError(f"no datasets in {src_path}")
        if kind == "h5":
            _write_h5(handle, dst_path, datasets)
        elif kind == "netcdf":
            _write_netcdf(handle, dst_path, datasets)
        else:
            _write_zarr(handle, dst_path, datasets)
    return dst_path


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _FORMAT_BY_SUFFIX:
        return _FORMAT_BY_SUFFIX[suffix]
    # Bare directory path → Zarr (CLI parity).
    if suffix == "" or path.name.endswith(".zarr"):
        return "zarr"
    raise ValueError(
        f"cannot infer export format from {path.name!r}; pass format= ({_SUPPORTED_FORMATS})"
    )


def _require(kind: str, dst: Path) -> None:
    extra, pkg = _EXTRA_BY_KIND[kind]
    try:
        __import__(pkg)
    except ImportError as exc:
        raise ImportError(
            f"export to {dst.suffix or kind!r} requires optional extra "
            f"`tet-py[{extra}]` (pip install 'tet-py[{extra}]' / "
            f"uv sync --extra {extra})"
        ) from exc


def _basic_attrs(handle: TetFile, name: str) -> dict[str, str]:
    meta = handle._metadata_for_dataset(name)
    raw = meta.get("attrs") or {}
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if key in _BASIC_ATTRS and isinstance(value, str):
            out[str(key)] = value
    return out


def _dim_names_for(ds: Dataset) -> list[str] | None:
    if ds.dim_names is None or len(ds.dim_names) != ds.ndim:
        return None
    if not all(isinstance(item, str) and item for item in ds.dim_names):
        return None
    return list(ds.dim_names)


def _write_h5(handle: TetFile, dst: Path, datasets: list[Dataset]) -> None:
    h5py = importlib.import_module("h5py")

    with h5py.File(dst, "w") as file:
        file.attrs["tetration_export"] = "tet-py"
        for ds in datasets:
            array = np.ascontiguousarray(handle.read_numpy(ds.name))
            created = file.create_dataset(ds.name, data=array)
            for key, value in _basic_attrs(handle, ds.name).items():
                created.attrs[key] = value
            dims = _dim_names_for(ds)
            if dims is not None:
                created.attrs["dim_names"] = ",".join(dims)
            created.attrs["tetration_dtype"] = str(array.dtype.name)


def _write_netcdf(handle: TetFile, dst: Path, datasets: list[Dataset]) -> None:
    netCDF4 = importlib.import_module("netCDF4")

    dim_lens: dict[tuple[str, str], int] = {}
    with netCDF4.Dataset(dst, "w", format="NETCDF4") as file:
        file.setncattr("tetration_export", "tet-py")
        for ds in datasets:
            array = np.ascontiguousarray(handle.read_numpy(ds.name))
            group_key, leaf = _split_path(ds.name)
            group = _ensure_nc_group(file, group_key)
            dims = _resolve_nc_dims(ds, leaf, array.shape)
            _ensure_nc_dims(group, group_key, dims, array.shape, dim_lens)
            var = group.createVariable(leaf, array.dtype, tuple(dims))
            var[:] = array
            for key, value in _basic_attrs(handle, ds.name).items():
                var.setncattr(key, value)
            var.setncattr("tetration_dtype", str(array.dtype.name))


def _split_path(name: str) -> tuple[str, str]:
    if "/" not in name:
        return "", name
    parent, _, leaf = name.rpartition("/")
    return parent, leaf


def _ensure_nc_group(root: Any, group_key: str) -> Any:
    if not group_key:
        return root
    group = root
    for part in group_key.split("/"):
        group = group.groups[part] if part in group.groups else group.createGroup(part)
    return group


def _resolve_nc_dims(
    ds: Dataset,
    leaf: str,
    shape: tuple[int, ...],
) -> list[str]:
    named = _dim_names_for(ds)
    if named is not None:
        return named
    return [f"{leaf}_d{index}" for index in range(len(shape))]


def _ensure_nc_dims(
    group: Any,
    group_key: str,
    dims: list[str],
    shape: tuple[int, ...],
    dim_lens: dict[tuple[str, str], int],
) -> None:
    for dim_name, length in zip(dims, shape, strict=True):
        key = (group_key, dim_name)
        existing = dim_lens.get(key)
        if existing is not None:
            if existing != length:
                raise ValueError(
                    f"dimension `{dim_name}` length conflict in group "
                    f"`{group_key or '/'}`: {existing} vs {length}"
                )
            continue
        if dim_name not in group.dimensions:
            group.createDimension(dim_name, length)
        dim_lens[key] = length


def _write_zarr(handle: TetFile, dst: Path, datasets: list[Dataset]) -> None:
    zarr = importlib.import_module("zarr")

    root = zarr.open_group(dst, mode="w")
    root.attrs["tetration_export"] = "tet-py"
    for ds in datasets:
        array = np.ascontiguousarray(handle.read_numpy(ds.name))
        group_key, leaf = _split_path(ds.name)
        parent = _ensure_zarr_group(root, group_key)
        created = parent.create_array(leaf, shape=array.shape, dtype=array.dtype)
        created[:] = array
        for key, value in _basic_attrs(handle, ds.name).items():
            created.attrs[key] = value
        dims = _dim_names_for(ds)
        if dims is not None:
            created.attrs["dim_names"] = dims
        created.attrs["tetration_dtype"] = str(array.dtype.name)


def _ensure_zarr_group(root: Any, group_key: str) -> Any:
    if not group_key:
        return root
    group = root
    for part in group_key.split("/"):
        if hasattr(group, "get"):
            child = group.get(part)
        else:
            child = group[part] if part in group else None
        if child is None:
            group = group.create_group(part)
        else:
            group = child
    return group
