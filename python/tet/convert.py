"""Foreign formats → ``.tet``.

``.npy``, multi-member ``.npz``, and ``.csv`` / ``.tsv`` ship in the core wheel
(stdlib ``csv``; no pandas). HDF5, NetCDF, Zarr v2/v3, and Parquet need optional
extras. Each numeric array or column becomes one dataset via :class:`~tet.TetWriter`.

Rust ``tet convert`` is still the fast path. Intentional Python differences:

* no CF ``scale_factor`` / ``add_offset`` decode
* the whole array is one chunk unless ``chunk_shape`` is passed (the CLI retile target is ~64 MiB)
* library codecs the Rust importer skips (HDF5 filters, Zarr blosc) are read when the extra can decode them
* unsupported columns are omitted; the call fails only when nothing remains
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike, fspath
from pathlib import Path
from typing import Any

import numpy as np

import tet._native as _native
from tet._io.writer import TetWriter

# Match tetration `MetadataLimitsV1::DEFAULT.attr_string_bytes` (footer attr cap).
_ATTR_STRING_BYTES = 1024
# Match tetration `convert::category::CATEGORY_ATTR` / `CATEGORY_MISSING`.
_CATEGORY_ATTR = "tetration_categories"
_CATEGORY_MISSING = -1
# Footer history `source` tags (same strings as Rust `tet convert`).
_HISTORY_H5 = "h5"
_HISTORY_NETCDF = "nc"
_HISTORY_ZARR = "zarr"
_HISTORY_ZARR2 = "zarr2"
_HISTORY_NPY = "npy"
_HISTORY_NPZ = "npz"
_HISTORY_CSV = "csv"
_HISTORY_PARQUET = "parquet"
_ZARR_V3_META = "zarr.json"

_EXTRA_BY_KIND: dict[str, tuple[str, str]] = {
    "h5": ("h5", "h5py"),
    "hdf5": ("h5", "h5py"),
    "netcdf": ("netcdf", "netCDF4"),
    "zarr": ("zarr", "zarr"),
    "parquet": ("parquet", "pyarrow"),
}

_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".h5": "h5",
    ".hdf5": "h5",
    ".nc": "netcdf",
    ".cdf": "netcdf",
    ".zarr": "zarr",
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
}

_SUPPORTED_FORMATS = "npy, npz, h5, netcdf, zarr, csv, tsv, parquet"
# Single-file suffixes `s3://` convert will download. Zarr stores are a later cut.
_S3_SINGLE_FILE_SUFFIXES = frozenset(set(_FORMAT_BY_SUFFIX) - {".zarr"}) | {".npy", ".npz"}
_S3_ZARR = (
    "s3:// convert does not read Zarr stores; pass a single file "
    "(.npy, .npz, .h5, .nc, .csv, .tsv, .parquet). Chunked Zarr-on-S3 reads are later"
)

# NumPy dtype name → layout-v1 wire dtype. ``bool`` and ``int8`` follow the HDF5
# importer and land as ``uint8``.
_WIRE_DTYPE: dict[str, np.dtype[Any]] = {
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "uint64": np.dtype("uint64"),
    "bool": np.dtype("uint8"),
    "int8": np.dtype("uint8"),
}


@dataclass(frozen=True)
class _Import:
    name: str
    array: np.ndarray
    attrs: dict[str, str] | None = None


def convert(
    src: str | PathLike[str],
    dst: str | PathLike[str],
    *,
    dataset: str = "data",
    format: str | None = None,
    **write_kw: Any,
) -> Path:
    """Convert ``src`` into a ``.tet`` at ``dst``.

    Parameters
    ----------
    src : path-like or ``s3://`` URI
        Source file, a Zarr directory store, or ``s3://bucket/object.ext``.
        An ``s3://`` object is downloaded in full (``TETRATION_S3_ENDPOINT`` required),
        then passed to the same importer as a local file. Zarr prefixes are rejected.
    dst : path-like
        Destination ``.tet`` path.
    dataset : str, default ``"data"``
        Catalog name for a ``.npy`` source. Other formats name datasets from the
        source (``.npz`` members, HDF5/NetCDF/Zarr paths, CSV/TSV and Parquet columns).
    format : str, optional
        Force format: ``npy``, ``npz``, ``h5``, ``netcdf``, ``zarr``, ``csv``,
        ``tsv``, ``parquet``. Default: infer from ``src`` suffix (``.tsv`` is tab,
        ``.csv`` is comma — same as Rust ``tet convert``).
    **write_kw
        Forwarded to :meth:`~tet.TetWriter.write_dataset` (for example ``chunk_shape``).

    Returns
    -------
    pathlib.Path
        Path to the written ``.tet`` file.

    Raises
    ------
    FileNotFoundError
        If a local ``src`` does not exist.
    ImportError
        If an optional extra is required but not installed.
    ValueError
        If the format cannot be inferred, is unsupported, the ``s3://`` URI is not a
        single file, or no layout-v1 arrays remain.
    CatalogError
        If an ``s3://`` download fails or ``TETRATION_S3_ENDPOINT`` is unset.
    """
    src_text = fspath(src).strip()
    dst_path = Path(dst).expanduser()
    if src_text.startswith("s3://"):
        return _convert_s3(src_text, dst_path, dataset=dataset, format=format, **write_kw)

    src_path = Path(src_text).expanduser()
    kind = (format or _infer_format(src_path)).lower()
    if kind == "hdf5":
        kind = "h5"

    if kind == "zarr":
        if not src_path.exists():
            raise FileNotFoundError(src_path)
    elif not src_path.is_file():
        raise FileNotFoundError(src_path)

    if kind == "npy":
        imports = [_Import(dataset, _load_npy(src_path))]
        history = _HISTORY_NPY
    elif kind == "npz":
        imports = _load_npz(src_path)
        history = _HISTORY_NPZ
    elif kind in ("csv", "tsv"):
        imports = _load_table(src_path, kind=kind)
        history = _HISTORY_CSV
    elif kind in ("h5", "netcdf", "zarr", "parquet"):
        _require(kind, src_path)
        imports, history = _load_extra(kind, src_path)
    else:
        raise ValueError(
            f"unsupported convert format {kind!r} for {src_path} ({_SUPPORTED_FORMATS})"
        )

    if not imports:
        raise ValueError(f"no supported arrays in {src_path}")
    return _commit(dst_path, imports, history, **write_kw)


def _convert_s3(
    uri: str,
    dst: Path,
    *,
    dataset: str,
    format: str | None,
    **write_kw: Any,
) -> Path:
    if format is not None and format.lower() == "zarr":
        raise ValueError(f"{_S3_ZARR} ({uri})")
    _bucket, key, file_name = _parse_s3_convert_object(uri)
    payload = _native.download_s3_object(_bucket, key)
    with tempfile.TemporaryDirectory(prefix="tet-convert-s3-") as scratch:
        staged = Path(scratch) / file_name
        staged.write_bytes(payload)
        return convert(staged, dst, dataset=dataset, format=format, **write_kw)


def _parse_s3_convert_object(uri: str) -> tuple[str, str, str]:
    """Return ``(bucket, key, file_name)`` for a single-file ``s3://`` convert source."""
    rest = uri[len("s3://") :]
    path = rest.split("?", 1)[0]
    bucket, sep, key = path.partition("/")
    if not sep or not bucket or not key:
        raise ValueError(f"invalid s3 URI (need s3://bucket/key): {uri}")
    if key.endswith("/"):
        raise ValueError(f"{_S3_ZARR} ({uri})")
    file_name = key.rsplit("/", 1)[-1]
    if file_name in ("", ".", ".."):
        raise ValueError(f"s3 object key has no file name: {key}")
    suffix = Path(file_name).suffix.lower()
    if suffix == ".zarr":
        raise ValueError(f"{_S3_ZARR} ({uri})")
    if suffix not in _S3_SINGLE_FILE_SUFFIXES:
        raise ValueError(
            f"unsupported s3 convert object {file_name!r} "
            f"(single-file extensions only; Zarr-on-S3 is later)"
        )
    return bucket, key, file_name


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix == ".npz":
        return "npz"
    if suffix in _FORMAT_BY_SUFFIX:
        return _FORMAT_BY_SUFFIX[suffix]
    raise ValueError(
        f"cannot infer convert format from {path.name!r}; pass format= ({_SUPPORTED_FORMATS})"
    )


def _require(kind: str, src: Path) -> None:
    extra, pkg = _EXTRA_BY_KIND[kind]
    try:
        __import__(pkg)
    except ImportError as exc:
        raise ImportError(
            f"convert from {src.suffix or kind!r} requires optional extra "
            f"`tet-py[{extra}]` (pip install 'tet-py[{extra}]' / "
            f"uv sync --extra {extra})"
        ) from exc


def _commit(dst: Path, imports: list[_Import], history: str, **write_kw: Any) -> Path:
    writer = TetWriter.create(dst)
    writer.push_history_event("convert", history)
    for item in imports:
        writer.write_dataset(item.name, item.array, attrs=item.attrs, **write_kw)
    return writer.commit()


def _as_wire(array: np.ndarray) -> np.ndarray | None:
    """Return a C-contiguous layout-v1 array, or ``None`` when the dtype is skipped."""
    arr = np.asarray(array)
    if arr.ndim == 0 or arr.size == 0:
        return None
    target = _WIRE_DTYPE.get(arr.dtype.name)
    if target is None:
        return None
    if arr.dtype != target:
        arr = arr.astype(target, copy=False)
    return np.ascontiguousarray(arr)


def _load_npy(src: Path) -> np.ndarray:
    loaded = np.load(src, allow_pickle=False)
    array = np.asarray(loaded)
    if array.dtype == object:
        raise ValueError(f"array in {src} is an object or pickled dtype; pickling is not supported")
    if array.ndim > 1 and array.flags.f_contiguous and not array.flags.c_contiguous:
        raise ValueError(f"array in {src} is Fortran-order; only C-order NumPy arrays are supported")
    wire = _as_wire(array)
    if wire is None:
        raise ValueError(f"no supported arrays in {src}")
    return wire


def _dataset_name_from_member(member: str) -> str:
    trimmed = member.rstrip("/")
    stem = trimmed[:-4] if trimmed.lower().endswith(".npy") else trimmed
    stem = stem.replace("\\", "/")
    return stem or "data"


def _load_npz(src: Path) -> list[_Import]:
    loaded: Any = np.load(src, allow_pickle=False)
    try:
        keys = sorted(loaded.files)
        imports: list[_Import] = []
        seen: set[str] = set()
        for key in keys:
            name = _dataset_name_from_member(str(key))
            if name in seen:
                raise ValueError(f"npz {src} has more than one array named `{name}`")
            seen.add(name)
            array = np.asarray(loaded[key])
            if array.dtype == object:
                raise ValueError(
                    f"array `{name}` is an object or pickled dtype; pickling is not supported"
                )
            if array.ndim > 1 and array.flags.f_contiguous and not array.flags.c_contiguous:
                raise ValueError(
                    f"array `{name}` is Fortran-order; only C-order NumPy arrays are supported"
                )
            wire = _as_wire(array)
            if wire is None:
                continue
            imports.append(_Import(name, wire))
        return imports
    finally:
        loaded.close()


def _load_extra(kind: str, src: Path) -> tuple[list[_Import], str]:
    if kind == "h5":
        return _load_h5(src), _HISTORY_H5
    if kind == "netcdf":
        return _load_netcdf(src), _HISTORY_NETCDF
    if kind == "zarr":
        return _load_zarr(src)
    if kind == "parquet":
        return _load_parquet(src), _HISTORY_PARQUET
    raise ValueError(f"unsupported convert format {kind!r}")


def _join(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _load_h5(src: Path) -> list[_Import]:
    h5py = importlib.import_module("h5py")

    imports: list[_Import] = []
    with h5py.File(src, "r") as handle:
        _walk_h5(handle, "", imports)
    imports.sort(key=lambda item: item.name)
    return imports


def _walk_h5(group: Any, prefix: str, imports: list[_Import]) -> None:
    h5py = importlib.import_module("h5py")

    for name in sorted(group.keys()):
        item = group[name]
        path = _join(prefix, str(name))
        if isinstance(item, h5py.Dataset):
            wire = _as_wire(item[()])
            if wire is not None:
                imports.append(_Import(path, wire))
        elif isinstance(item, h5py.Group):
            _walk_h5(item, path, imports)


def _load_netcdf(src: Path) -> list[_Import]:
    netCDF4 = importlib.import_module("netCDF4")

    imports: list[_Import] = []
    with netCDF4.Dataset(src, "r") as handle:
        _walk_nc(handle, "", imports)
    imports.sort(key=lambda item: item.name)
    return imports


def _walk_nc(group: Any, prefix: str, imports: list[_Import]) -> None:
    for name in sorted(group.variables):
        var = group.variables[name]
        wire = _as_wire(np.asarray(var[:]))
        if wire is not None:
            imports.append(_Import(_join(prefix, str(name)), wire))
    for name in sorted(group.groups):
        _walk_nc(group.groups[name], _join(prefix, str(name)), imports)


def _load_zarr(src: Path) -> tuple[list[_Import], str]:
    zarr = importlib.import_module("zarr")

    store = src
    history = _HISTORY_ZARR if (store / _ZARR_V3_META).is_file() else _HISTORY_ZARR2
    root = zarr.open_group(store, mode="r")
    imports: list[_Import] = []
    _walk_zarr(root, "", imports)
    imports.sort(key=lambda item: item.name)
    return imports, history


def _walk_zarr(node: Any, prefix: str, imports: list[_Import]) -> None:
    members = node.members() if hasattr(node, "members") else node.items()
    for name, child in sorted(members, key=lambda pair: str(pair[0])):
        path = _join(prefix, str(name))
        if hasattr(child, "members") or hasattr(child, "groups"):
            if not hasattr(child, "dtype"):
                _walk_zarr(child, path, imports)
                continue
        wire = _zarr_array(child)
        if wire is not None:
            imports.append(_Import(path, wire))


def _zarr_array(node: Any) -> np.ndarray | None:
    if not hasattr(node, "dtype") or not hasattr(node, "shape"):
        return None
    return _as_wire(np.asarray(node[:]))


def _load_table(src: Path, *, kind: str) -> list[_Import]:
    """Parse a comma (``csv``) or tab (``tsv``) table into 1-D datasets."""
    if kind == "tsv":
        delimiter = "\t"
    elif kind == "csv":
        delimiter = ","
    else:
        raise ValueError(f"expected csv or tsv, got {kind!r}")

    with src.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    rows = [row for row in rows if row]
    if not rows:
        return []
    width = len(rows[0])
    if width == 0:
        return []
    first_is_header = any(_classify(cell)[0] == "text" for cell in rows[0])
    if first_is_header:
        names = _header_names(rows[0])
        data_rows = rows[1:]
    else:
        names = [f"col{index}" for index in range(width)]
        data_rows = rows
    _ensure_unique(src, names)

    columns: list[_Column] = [_Column() for _ in names]
    for row_index, row in enumerate(data_rows, start=1):
        if len(row) != width:
            raise ValueError(f"csv {src} row {row_index} has {len(row)} fields, expected {width}")
        for column, field in zip(columns, row, strict=True):
            column.push(field)

    imports: list[_Import] = []
    for name, column in zip(names, columns, strict=True):
        finished = column.finish()
        if finished is not None:
            imports.append(_Import(name, finished[0], finished[1]))
    return imports


def _header_names(row: list[str]) -> list[str]:
    names: list[str] = []
    for index, field in enumerate(row):
        trimmed = field.strip().lstrip("\ufeff")
        names.append(trimmed or f"col{index}")
    return names


def _ensure_unique(src: Path, names: list[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"csv {src} has more than one column named `{name}`")
        seen.add(name)


def _classify(field: str) -> tuple[str, Any]:
    token = field.strip().lstrip("\ufeff")
    if not token:
        return ("blank", None)
    if _plain_integer(token):
        value = int(token)
        if -(2**63) <= value < 2**63:
            return ("int", value)
    try:
        numeric = float(token)
    except ValueError:
        return ("text", token)
    if math.isfinite(numeric) or _non_finite(token):
        return ("float", numeric)
    return ("text", token)


def _plain_integer(token: str) -> bool:
    digits = token[1:] if token[:1] in "+-" else token
    return bool(digits) and digits.isascii() and digits.isdigit()


def _non_finite(token: str) -> bool:
    return token.lower() in {
        "nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }


class _Category:
    def __init__(self) -> None:
        self.codes: dict[str, int] = {}
        self.values: list[int] = []
        self.overflow = False

    def push_missing(self) -> None:
        if not self.overflow:
            self.values.append(_CATEGORY_MISSING)

    def push_label(self, label: str) -> None:
        if self.overflow:
            return
        code = self.codes.get(label)
        if code is None:
            code = len(self.codes)
            self.codes[label] = code
            note = _category_json(self.codes)
            if len(note.encode()) > _ATTR_STRING_BYTES:
                self.overflow = True
                self.codes.clear()
                self.values.clear()
                return
        self.values.append(code)

    def finish(self) -> tuple[np.ndarray, dict[str, str]] | None:
        if self.overflow or not self.codes:
            return None
        array = np.asarray(self.values, dtype=np.int64)
        return array, {_CATEGORY_ATTR: _category_json(self.codes)}


def _category_json(codes: Mapping[str, int]) -> str:
    return json.dumps(
        {"codes": dict(sorted(codes.items())), "missing": _CATEGORY_MISSING},
        separators=(",", ":"),
        sort_keys=True,
    )


class _Column:
    def __init__(self) -> None:
        self.kind = "empty"
        self.blanks = 0
        self.ints: list[int] = []
        self.floats: list[float] = []
        self.category = _Category()

    def push(self, field: str) -> None:
        if self.kind == "skip":
            return
        kind, value = _classify(field)
        if kind == "text":
            self._text(str(value))
        elif kind == "blank":
            self._blank()
        elif kind == "int":
            self._int(int(value))
        else:
            self._float(float(value))

    def _text(self, label: str) -> None:
        if self.kind == "empty":
            self.kind = "category"
            self.category.push_label(label)
        elif self.kind == "blank":
            count = self.blanks
            self.kind = "category"
            for _ in range(count):
                self.category.push_missing()
            self.category.push_label(label)
        elif self.kind in ("int", "float"):
            self.kind = "skip"
        elif self.kind == "category":
            self.category.push_label(label)

    def _blank(self) -> None:
        if self.kind == "empty":
            self.kind = "blank"
            self.blanks = 1
        elif self.kind == "blank":
            self.blanks += 1
        elif self.kind == "int":
            self.floats = [float(value) for value in self.ints]
            self.floats.append(math.nan)
            self.kind = "float"
        elif self.kind == "float":
            self.floats.append(math.nan)
        elif self.kind == "category":
            self.category.push_missing()

    def _int(self, value: int) -> None:
        if self.kind == "empty":
            self.kind = "int"
            self.ints = [value]
        elif self.kind == "blank":
            self.floats = [math.nan] * self.blanks
            self.floats.append(float(value))
            self.kind = "float"
        elif self.kind == "int":
            self.ints.append(value)
        elif self.kind == "float":
            self.floats.append(float(value))
        elif self.kind == "category":
            self.kind = "skip"

    def _float(self, value: float) -> None:
        if self.kind == "empty":
            self.kind = "float"
            self.floats = [value]
        elif self.kind == "blank":
            self.floats = [math.nan] * self.blanks
            self.floats.append(value)
            self.kind = "float"
        elif self.kind == "int":
            self.floats = [float(item) for item in self.ints]
            self.floats.append(value)
            self.kind = "float"
        elif self.kind == "float":
            self.floats.append(value)
        elif self.kind == "category":
            self.kind = "skip"

    def finish(self) -> tuple[np.ndarray, dict[str, str] | None] | None:
        if self.kind in ("empty", "blank", "skip"):
            return None
        if self.kind == "int":
            if not self.ints:
                return None
            return np.asarray(self.ints, dtype=np.int64), None
        if self.kind == "float":
            if not self.floats or all(math.isnan(value) for value in self.floats):
                return None
            return np.asarray(self.floats, dtype=np.float64), None
        finished = self.category.finish()
        if finished is None:
            return None
        return finished


def _load_parquet(src: Path) -> list[_Import]:
    pq = importlib.import_module("pyarrow.parquet")

    table = pq.read_table(src)
    imports: list[_Import] = []
    for name in table.column_names:
        column = table.column(name).combine_chunks()
        finished = _parquet_column(column)
        if finished is not None:
            imports.append(_Import(str(name), finished[0], finished[1]))
    return imports


def _parquet_column(column: Any) -> tuple[np.ndarray, dict[str, str] | None] | None:
    pa = importlib.import_module("pyarrow")

    if pa.types.is_dictionary(column.type) and (
        pa.types.is_string(column.type.value_type) or pa.types.is_large_string(column.type.value_type)
    ):
        column = column.cast(column.type.value_type)
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        return _parquet_category(column.to_pylist())
    if pa.types.is_integer(column.type) or pa.types.is_floating(column.type):
        return _parquet_numeric(column)
    return None


def _parquet_numeric(column: Any) -> tuple[np.ndarray, None] | None:
    pa = importlib.import_module("pyarrow")

    values = column.to_numpy(zero_copy_only=False)
    if values.dtype == object or values.size == 0:
        return None
    mask = column.is_null().to_numpy(zero_copy_only=False)
    if pa.types.is_integer(column.type) and bool(mask.any()):
        floats = values.astype(np.float64, copy=True)
        floats[mask] = np.nan
        return floats, None
    if pa.types.is_floating(column.type) and bool(mask.any()):
        floats = values.astype(values.dtype, copy=True)
        floats[mask] = np.nan
        wire = _as_wire(floats)
        if wire is None:
            return None
        return wire, None
    wire = _as_wire(values)
    if wire is None:
        return None
    return wire, None


def _parquet_category(labels: list[Any]) -> tuple[np.ndarray, dict[str, str]] | None:
    column = _Category()
    for label in labels:
        if label is None:
            column.push_missing()
        else:
            column.push_label(str(label))
    return column.finish()
