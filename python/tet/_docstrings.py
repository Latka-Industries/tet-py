"""Shared Parameters / Returns / Raises blocks for method docstrings."""

from __future__ import annotations

_RAISE_FILE_QUERY = """\
Raises
------
UnknownDatasetError
    If a named ``dataset`` is not in this file.
UnknownAxisError
    If an axis index or name is invalid.
TetError
    Query parse, validation, or execution failure.
TypeError
    If ``query`` is neither ``dict`` nor JSON string.\
"""

_REDUCE_IO = """\
Parameters
----------
dataset : str
    Dataset name in this file's catalog.
axes : sequence of int or str, optional
    Axis indices or ``dim_names`` to reduce. Omit with no ``axis`` to reduce all dimensions.
axis : int or str, optional
    Single axis to reduce (mutually exclusive with ``axes``).
device : str, optional
    Value for ``execution.device`` (e.g. ``"cpu"``, ``"cuda:0"``).
raw : bool, default False
    If False, return a scalar (full reduction) or :class:`~tet.QueryResult` (partial).
    If True, return the full wire response ``dict``.

Returns
-------
float, bool, int, QueryResult, or dict
    Full reduction and ``raw=False``: aggregate scalar. Partial reduction: :class:`~tet.QueryResult`
    (use ``.reduced``). ``raw=True``: wire ``dict``.

Raises
------
UnknownDatasetError, UnknownAxisError, TetError
    See :meth:`~tet.TetFile.query`.\
"""


def reduce_doc(summary: str) -> str:
    """One-line summary plus shared reduction I/O block."""
    return f"{summary.strip()}\n\n{_REDUCE_IO}"
