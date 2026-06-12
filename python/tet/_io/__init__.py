"""I/O helpers: NumPy interchange, spill files, and buffered writes."""

from tet._io.numpy import read_numpy_array, read_spill_array
from tet._io.spill import SpillReadResult, load_spill_array, normalize_path, resolve_spill_path
from tet._io.writer import TetWriter, write_dataset

__all__ = [
    "SpillReadResult",
    "TetWriter",
    "load_spill_array",
    "normalize_path",
    "read_numpy_array",
    "read_spill_array",
    "resolve_spill_path",
    "write_dataset",
]
