//! Convert tetration dense buffers into `numpy.ndarray` (copy, row-major).

use numpy::{PyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tetration::query::{DenseBuffer, DenseMaterializeOutcome};

/// Build a shaped `numpy.ndarray` from a dense materialize outcome.
pub(crate) fn outcome_to_py<'py>(
    py: Python<'py>,
    outcome: DenseMaterializeOutcome,
) -> PyResult<Bound<'py, PyAny>> {
    let shape: Vec<usize> = outcome
        .shape
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("shape dimension overflow")
            })
        })
        .collect::<PyResult<_>>()?;
    match outcome.buffer {
        DenseBuffer::F32(v) => PyArray::from_vec(py, v).reshape(shape).map(Bound::into_any),
        DenseBuffer::F64(v) => PyArray::from_vec(py, v).reshape(shape).map(Bound::into_any),
        DenseBuffer::I32(v) => PyArray::from_vec(py, v).reshape(shape).map(Bound::into_any),
        DenseBuffer::I64(v) => PyArray::from_vec(py, v).reshape(shape).map(Bound::into_any),
        DenseBuffer::U8(v) => PyArray::from_vec(py, v).reshape(shape).map(Bound::into_any),
        DenseBuffer::U16(v) => PyArray::from_vec(py, v).reshape(shape).map(Bound::into_any),
        DenseBuffer::I16(v) => PyArray::from_vec(py, v).reshape(shape).map(Bound::into_any),
    }
}
