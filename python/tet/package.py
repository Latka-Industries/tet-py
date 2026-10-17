"""Local ``.tet.d`` packages and store-backed ingest / publish.

Local helpers wrap ``tetration`` ``ingest_tet_file`` / ``publish_parts``. They do not
start concurrent shard writers. :class:`MemoryStore` and :class:`S3Store` speak the
same store ingest / publish path (``stage_part`` / ``publish_staged_parts``).
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


def _resolve_part_id(source: Path, part_id: str | None) -> str:
    if part_id is not None:
        return part_id
    return str(_native.part_id_from_path(source))


class MemoryStore:
    """In-memory package store (pin tests + store ingest without credentials)."""

    def __init__(self, inner: _native.MemoryStore | None = None) -> None:
        self._inner = inner if inner is not None else _native.MemoryStore()

    @classmethod
    def from_package(cls, path: str | PathLike[str]) -> MemoryStore:
        """Copy ``CURRENT`` / ``packed.tet`` from ``path`` into a new memory store."""
        return cls(_native.MemoryStore.from_package(Path(path).expanduser()))

    def ingest(
        self,
        source: str | PathLike[str],
        *,
        part_id: str | None = None,
        session_id: str = _DEFAULT_SESSION_ID,
    ) -> str:
        """Stage ``source`` (a sealed ``.tet``) as one part. Returns the part id."""
        src = Path(source).expanduser()
        chosen = _resolve_part_id(src, part_id)
        return str(self._inner.ingest(src, session_id=session_id, part_id=chosen))

    def publish(
        self,
        part_ids: Sequence[str],
        *,
        generation: int | None = None,
    ) -> int:
        """Seal ``part_ids`` and CAS-publish ``CURRENT``. Returns the generation id."""
        return int(self._inner.publish(list(part_ids), generation=generation))

    def seal(self, part_ids: Sequence[str], *, generation: int) -> int:
        """Seal ``part_ids`` into ``generation`` without flipping ``CURRENT``."""
        return int(self._inner.seal(list(part_ids), generation))

    def cas_publish(self, generation: int, *, expected: int | None) -> None:
        """CAS-write ``CURRENT``; raises when ``expected`` does not match."""
        self._inner.cas_publish(generation, expected=expected)

    def open(self, uri: str = "memory://current") -> TetFile:
        """Pin ``uri`` (``memory://current`` or ``memory://g/<id>``) and open it."""
        return TetFile.from_native(self._inner.open(uri))


class S3Store:
    """S3-compatible package store. Endpoint and credentials are constructor args."""

    def __init__(self, inner: _native.S3Store) -> None:
        self._inner = inner

    @classmethod
    def connect(
        cls,
        *,
        endpoint: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        prefix: str = "",
        region: str = "us-east-1",
    ) -> S3Store:
        """Open a bucket (path-style). Does not create the bucket."""
        return cls(
            _native.S3Store.connect(
                endpoint=endpoint,
                bucket=bucket,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                prefix=prefix,
                region=region,
            )
        )

    def ingest(
        self,
        source: str | PathLike[str],
        *,
        part_id: str | None = None,
        session_id: str = _DEFAULT_SESSION_ID,
    ) -> str:
        """Stage ``source`` (a sealed ``.tet``) as one part. Returns the part id."""
        src = Path(source).expanduser()
        chosen = _resolve_part_id(src, part_id)
        return str(self._inner.ingest(src, session_id=session_id, part_id=chosen))

    def publish(
        self,
        part_ids: Sequence[str],
        *,
        generation: int | None = None,
    ) -> int:
        """Seal ``part_ids`` and CAS-publish ``CURRENT``. Returns the generation id."""
        return int(self._inner.publish(list(part_ids), generation=generation))

    def seal(self, part_ids: Sequence[str], *, generation: int) -> int:
        """Seal ``part_ids`` into ``generation`` without flipping ``CURRENT``."""
        return int(self._inner.seal(list(part_ids), generation))

    def cas_publish(self, generation: int, *, expected: int | None) -> None:
        """CAS-write ``CURRENT``; a lost put leaves the last good generation."""
        self._inner.cas_publish(generation, expected=expected)

    def open(self, pin: str = "current") -> TetFile:
        """Pin ``current`` or ``g/<id>`` once and open that generation's ``packed.tet``."""
        return TetFile.from_native(self._inner.open(pin))


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
    chosen = _resolve_part_id(src, part_id)
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
