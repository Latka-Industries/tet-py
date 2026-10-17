"""Local ``.tet.d`` packages: open ``CURRENT``, ingest a sealed ``.tet``, publish.

These wrap ``tetration`` ``ingest_tet_file`` / ``publish_parts``. They do not start
concurrent shard writers. :class:`MemoryStore` copies a published generation into
an in-memory store for URI pin tests (THI-504; no S3).
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import tet._native as _native
from tet.file import TetFile

# Match tetration package layout (`catalog::package`).
_PACKAGE_SUFFIX = ".tet.d"
_PROTOCOL_FILE = "PROTOCOL"
_DEFAULT_SESSION_ID = "s0"


class MemoryStore:
    """In-memory package store seeded from a local ``.tet.d`` (no credentials)."""

    def __init__(self, inner: _native.MemoryStore) -> None:
        self._inner = inner

    @classmethod
    def from_package(cls, path: str | PathLike[str]) -> MemoryStore:
        """Copy ``CURRENT`` / ``packed.tet`` from ``path`` into a new memory store."""
        return cls(_native.MemoryStore.from_package(Path(path).expanduser()))

    def open(self, uri: str = "memory://current") -> TetFile:
        """Pin ``uri`` (``memory://current`` or ``memory://g/<id>``) and open it."""
        return TetFile.from_native(self._inner.open(uri))


def is_package(path: str | PathLike[str]) -> bool:
    """True when ``path`` is a directory that looks like a ``.tet.d`` package."""
    resolved = Path(path).expanduser()
    if not resolved.is_dir():
        return False
    return resolved.name.endswith(_PACKAGE_SUFFIX) or (resolved / _PROTOCOL_FILE).is_file()


def create(path: str | PathLike[str]) -> Path:
    """Create an empty package directory and return its path."""
    resolved = Path(path).expanduser()
    _native.package_create(resolved)
    return resolved


def ingest(
    package: str | PathLike[str],
    source: str | PathLike[str],
    *,
    part_id: str | None = None,
    session_id: str = _DEFAULT_SESSION_ID,
    create: bool = False,
) -> str:
    """Stage ``source`` (a sealed ``.tet``) as one part. Returns the part id.

    Parameters
    ----------
    package
        ``.tet.d`` directory. Created when ``create`` is true and ``PROTOCOL`` is absent.
    source
        Existing ``.tet`` file.
    part_id
        Staging part id. Default: filesystem-safe stem of ``source``.
    session_id
        Staging session directory name. Default ``s0``.
    create
        Create the package when it does not exist yet.
    """
    root = Path(package).expanduser()
    src = Path(source).expanduser()
    if create and not (root / _PROTOCOL_FILE).is_file():
        _native.package_create(root)
    chosen = part_id if part_id is not None else _native.part_id_from_path(src)
    return str(_native.package_ingest(root, src, session_id, chosen))


def publish(
    package: str | PathLike[str],
    part_ids: Sequence[str],
    *,
    generation: int | None = None,
) -> int:
    """Seal ``part_ids`` into one generation and point ``CURRENT`` at it.

    ``generation`` defaults to one past the current pointer (``1`` on a new package).
    Returns the generation id that was sealed.
    """
    root = Path(package).expanduser()
    sealed = _native.package_publish(root, generation, list(part_ids))
    return int(sealed)
