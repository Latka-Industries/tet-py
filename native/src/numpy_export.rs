//! Convert tetration dense buffers into `numpy.ndarray` (copy, row-major).

use numpy::{PyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use tetration::query::{DenseBuffer, DenseMaterializeOutcome};

/// Build a shaped `numpy.ndarray` from a dense materialize outcome.
pub(crate) fn outcome_to_py(
    py: Python<'_>,
    outcome: DenseMaterializeOutcome,
) -> PyResult<Bound<'_, PyAny>> {
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
        DenseBuffer::U32(v) => PyArray::from_vec(py, v).reshape(shape).map(Bound::into_any),
        DenseBuffer::U64(v) => PyArray::from_vec(py, v).reshape(shape).map(Bound::into_any),
        DenseBuffer::F16(v) => {
            let mut bytes = Vec::with_capacity(v.len().saturating_mul(2));
            for h in v {
                bytes.extend_from_slice(&h.to_le_bytes());
            }
            let np = py.import("numpy")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "<f2")?;
            let arr = np.call_method("frombuffer", (PyBytes::new(py, &bytes),), Some(&kwargs))?;
            // `frombuffer` views our temporary bytes; copy before returning.
            let arr = arr.call_method0("copy")?;
            arr.call_method1("reshape", (shape,))
        }
    }
}
