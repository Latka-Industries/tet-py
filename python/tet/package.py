"""Local ``.tet.d`` packages and store-backed ingest / publish.

Filesystem helpers wrap ``tetration`` ``ingest_tet_file`` / ``publish_parts``. They do
not start concurrent shard writers.

:class:`MemoryStore` and :class:`S3Store` use the store path
(``stage_part`` / ``publish_staged_parts``): N writers stage immutable parts, one
committer seals a generation and CAS-publishes ``CURRENT``. A rejected CAS leaves the
last good generation. :meth:`~S3Store.open` / :meth:`~MemoryStore.open` pin once —
advancing ``CURRENT`` later does not retarget an open handle (same rule as 0.3.0 local
packages).
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
    """In-memory package store for pin tests and store ingest without credentials.

    Construct empty with ``MemoryStore()``, or seed from a local ``.tet.d`` via
    :meth:`from_package`. Ingest / publish match :class:`S3Store` (same store trait).
    """

    def __init__(self, inner: _native.MemoryStore | None = None) -> None:
        self._inner = inner if inner is not None else _native.MemoryStore()

    @classmethod
    def from_package(cls, path: str | PathLike[str]) -> MemoryStore:
        """Copy ``CURRENT`` / ``packed.tet`` from ``path`` into a new memory store.

        Parameters
        ----------
        path
            Local ``.tet.d`` package directory.

        Raises
        ------
        CatalogError
            When the package has no published generation or I/O fails.
        """
        return cls(_native.MemoryStore.from_package(Path(path).expanduser()))

    def ingest(
        self,
        source: str | PathLike[str],
        *,
        part_id: str | None = None,
        session_id: str = _DEFAULT_SESSION_ID,
    ) -> str:
        """Stage ``source`` (a sealed ``.tet``) as one part. Returns the part id.

        Parameters
        ----------
        source
            Existing ``.tet`` file.
        part_id
            Staging part id. Default: filesystem-safe stem of ``source``.
        session_id
            Staging session directory name. Each parallel writer needs a unique id.
            Default ``s0``.

        Raises
        ------
        CatalogError
            When the source is missing or the session / part id collides.
        """
        src = Path(source).expanduser()
        chosen = _resolve_part_id(src, part_id)
        return str(self._inner.ingest(src, session_id=session_id, part_id=chosen))

    def publish(
        self,
        part_ids: Sequence[str],
        *,
        generation: int | None = None,
    ) -> int:
        """Seal ``part_ids`` and CAS-publish ``CURRENT``. Returns the generation id.

        ``generation`` defaults to one past the current pointer (``1`` when unset).
        """
        return int(self._inner.publish(list(part_ids), generation=generation))

    def seal(self, part_ids: Sequence[str], *, generation: int) -> int:
        """Seal ``part_ids`` into ``generation`` without flipping ``CURRENT``.

        Used with :meth:`cas_publish` to exercise a lost conditional put. Prefer
        :meth:`publish` for the normal seal-and-flip path.
        """
        return int(self._inner.seal(list(part_ids), generation))

    def cas_publish(self, generation: int, *, expected: int | None) -> None:
        """CAS-write ``CURRENT`` to ``generation``.

        Parameters
        ----------
        generation
            Generation id to name in ``CURRENT``.
        expected
            Prior ``CURRENT`` value that must still hold (``None`` = create-only).

        Raises
        ------
        CatalogError
            When ``expected`` does not match (last good generation is left as-is).
        """
        self._inner.cas_publish(generation, expected=expected)

    def open(self, uri: str = "memory://current") -> TetFile:
        """Pin ``uri`` once and open that generation's ``packed.tet``.

        Parameters
        ----------
        uri
            ``memory://current`` or ``memory://g/<id>``.

        Notes
        -----
        Materializes bytes to a temp file so mmap query / ``read_numpy`` work.
        Advancing ``CURRENT`` later does not retarget this handle.
        """
        return TetFile.from_native(self._inner.open(uri))


class S3Store:
    """S3-compatible package store (MinIO, AWS, and path-style endpoints).

    Endpoint and credentials are arguments to :meth:`connect` — they are not embedded
    in ``s3://`` URIs. Use this class for ingest / publish / pin.
    ``tet.open("s3://bucket/prefix")`` pins the same package from ``TETRATION_S3_*``
    and range-reads on query. :meth:`open` still downloads the whole ``packed.tet``.
    ``tet.convert("s3://bucket/object.ext", …)`` downloads one foreign object.
    ``tet.convert`` of a Zarr prefix fetches chunks on demand. Neither is a package open.
    """

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
        """Open a bucket (path-style client). Does not create the bucket.

        Parameters
        ----------
        endpoint
            HTTP(S) endpoint, e.g. ``http://127.0.0.1:9000`` for local MinIO.
        bucket
            Bucket that holds the package.
        access_key_id, secret_access_key
            SigV4 credentials.
        prefix
            Package root inside the bucket. Empty means the bucket root.
        region
            SigV4 region. MinIO accepts ``us-east-1``.

        Raises
        ------
        CatalogError
            When the endpoint or client configuration is invalid.
        """
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
        """Stage ``source`` (a sealed ``.tet``) as one part. Returns the part id.

        Parameters
        ----------
        source
            Existing ``.tet`` file (read from the local filesystem, then uploaded).
        part_id
            Staging part id. Default: filesystem-safe stem of ``source``.
        session_id
            Staging session directory name. Each parallel writer needs a unique id.
            Default ``s0``.

        Raises
        ------
        CatalogError
            When the source is missing or the session / part id collides.
        """
        src = Path(source).expanduser()
        chosen = _resolve_part_id(src, part_id)
        return str(self._inner.ingest(src, session_id=session_id, part_id=chosen))

    def publish(
        self,
        part_ids: Sequence[str],
        *,
        generation: int | None = None,
    ) -> int:
        """Seal ``part_ids`` and CAS-publish ``CURRENT``. Returns the generation id.

        ``generation`` defaults to one past the current pointer (``1`` when unset).
        A rejected conditional put leaves the previous generation's ``packed.tet``.
        """
        return int(self._inner.publish(list(part_ids), generation=generation))

    def seal(self, part_ids: Sequence[str], *, generation: int) -> int:
        """Seal ``part_ids`` into ``generation`` without flipping ``CURRENT``.

        Used with :meth:`cas_publish` to exercise a lost conditional put. Prefer
        :meth:`publish` for the normal seal-and-flip path.
        """
        return int(self._inner.seal(list(part_ids), generation))

    def cas_publish(self, generation: int, *, expected: int | None) -> None:
        """CAS-write ``CURRENT`` to ``generation``.

        Parameters
        ----------
        generation
            Generation id to name in ``CURRENT``.
        expected
            Prior ``CURRENT`` value that must still hold (``None`` = create-only).

        Raises
        ------
        CatalogError
            When the conditional put is rejected (last good generation kept).
        """
        self._inner.cas_publish(generation, expected=expected)

    def open(self, pin: str = "current") -> TetFile:
        """Pin ``current`` or ``g/<id>`` once and open that generation's ``packed.tet``.

        Parameters
        ----------
        pin
            ``current`` (read ``CURRENT`` once) or ``g/<id>`` for an explicit generation.

        Notes
        -----
        Same pin rule as local packages and :class:`MemoryStore`: a later publish does
        not retarget this handle. Materializes to a temp file for mmap query /
        ``read_numpy``.
        """
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
    dry_run: bool = False,
    replace: Sequence[str] | None = None,
) -> int:
    """Seal ``part_ids`` into one generation and point ``CURRENT`` at it.

    Parameters
    ----------
    package
        ``.tet.d`` directory.
    part_ids
        Staging part ids to include in the publish set.
    generation
        Generation id to seal. Default: one past the current pointer (``1`` on a
        new package).
    dry_run
        When true, run claim / schema checks only: no seal, no ``CURRENT`` flip.
        Returns the generation that a real publish would use.
    replace
        Dataset names allowed to change dtype/shape relative to sealed ``CURRENT``
        (opt-in G+1 replace). Default: reject schema changes.

    Returns
    -------
    int
        Generation id sealed (or that would be sealed when ``dry_run`` is true).
    """
    root = Path(package).expanduser()
    sealed = _native.package_publish(
        root,
        generation,
        list(part_ids),
        dry_run=dry_run,
        replace=list(replace) if replace is not None else None,
    )
    return int(sealed)


def pack(
    package: str | PathLike[str],
    dest: str | PathLike[str],
    *,
    generation: int | None = None,
    remove_package: bool = False,
) -> int:
    """Copy a sealed generation's ``packed.tet`` to a standalone ``.tet``.

    Parameters
    ----------
    package
        ``.tet.d`` directory.
    dest
        Destination ``.tet`` path (must not already exist).
    generation
        Generation to pack. Default: ``CURRENT``.
    remove_package
        When true, delete the package directory only after a verified copy
        (same as ``tet pack --remove-package``). Close any readers pinned on
        that package first.

    Returns
    -------
    int
        Generation id that was packed.
    """
    root = Path(package).expanduser()
    out = Path(dest).expanduser()
    return int(_native.package_pack(root, out, generation, remove_package))
