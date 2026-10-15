//! Convert tetration dense buffers into `numpy.ndarray`.
//!
//! Prefer a zero-copy [`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
//! view when the read plan is one contiguous raw (uncompressed) payload span whose on-disk bytes
//! already match the logical row-major selection. Otherwise copy via [`DenseBuffer`].

use numpy::{PyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyTuple};
use tetration::catalog::{CHUNK_PAYLOAD_CODEC_V1, DATASET_DTYPE_TAG_V1};
use tetration::query::{
    DenseBuffer, DenseMaterializeOutcome, PlannedRead, ReadPlan, plan_read_for_document,
};

/// Contiguous raw byte range covering the full logical selection in on-disk order.
#[derive(Debug, Clone, Copy)]
struct ContiguousRawSpan {
    start: usize,
    len: usize,
}

/// Zero-copy eligibility: **one** raw chunk whose payload bytes are exactly the logical
/// row-major selection.
///
/// Multi-chunk adjacent payloads (tetration `detect_contiguous_raw_span`) are **not** safe to
/// reshape: tile order is chunk-raster, which differs from dense row-major whenever
/// `chunk_shape` does not cover full trailing axes (e.g. shape `(2,3)` / chunks `(2,2)`).
/// Scalar folds can still use that span; `NumPy` views cannot.
fn detect_single_chunk_raw_view(plan: &ReadPlan, elem_size: usize) -> Option<ContiguousRawSpan> {
    if plan.chunks.len() != 1 {
        return None;
    }
    if plan.selection_step.iter().any(|&s| s != 1) {
        return None;
    }
    let expected_raw = plan.logical_f32_element_count.checked_mul(elem_size)? as u64;
    if expected_raw == 0 {
        return None;
    }
    let c = &plan.chunks[0];
    if c.codec != CHUNK_PAYLOAD_CODEC_V1.raw || c.stored_byte_len != c.raw_byte_len {
        return None;
    }
    if c.raw_byte_len != expected_raw {
        return None;
    }
    let start = usize::try_from(c.payload_offset).ok()?;
    let len = usize::try_from(c.raw_byte_len).ok()?;
    Some(ContiguousRawSpan { start, len })
}

fn wire_dtype_elem_size(dtype: u32) -> Option<usize> {
    let tags = DATASET_DTYPE_TAG_V1;
    Some(if tags.is_f32(dtype) || tags.is_i32(dtype) || tags.is_u32(dtype) {
        4
    } else if tags.is_f64(dtype) || tags.is_i64(dtype) || tags.is_u64(dtype) {
        8
    } else if tags.is_u16(dtype) || tags.is_i16(dtype) || tags.is_f16(dtype) {
        2
    } else if tags.is_u8(dtype) {
        1
    } else {
        return None;
    })
}

fn numpy_le_dtype_str(dtype: u32) -> Option<&'static str> {
    let tags = DATASET_DTYPE_TAG_V1;
    if tags.is_f32(dtype) {
        Some("<f4")
    } else if tags.is_f64(dtype) {
        Some("<f8")
    } else if tags.is_f16(dtype) {
        Some("<f2")
    } else if tags.is_i32(dtype) {
        Some("<i4")
    } else if tags.is_i64(dtype) {
        Some("<i8")
    } else if tags.is_i16(dtype) {
        Some("<i2")
    } else if tags.is_u8(dtype) {
        Some("|u1")
    } else if tags.is_u16(dtype) {
        Some("<u2")
    } else if tags.is_u32(dtype) {
        Some("<u4")
    } else if tags.is_u64(dtype) {
        Some("<u8")
    } else {
        None
    }
}

fn logical_shape_usize(plan: &ReadPlan) -> PyResult<Vec<usize>> {
    plan.logical_selection_shape
        .iter()
        .map(|&d| {
            usize::try_from(d).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("shape dimension overflow")
            })
        })
        .collect()
}

/// Open a read-only `numpy.memmap` over a contiguous raw payload span.
fn memmap_zero_copy<'py>(
    py: Python<'py>,
    path: &str,
    planned: &PlannedRead,
    span: ContiguousRawSpan,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype_str = numpy_le_dtype_str(planned.dtype).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unsupported dtype {} for zero-copy memmap",
            planned.dtype
        ))
    })?;
    let shape = logical_shape_usize(&planned.read_plan)?;
    let np = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype_str)?;
    kwargs.set_item("mode", "r")?;
    kwargs.set_item("offset", span.start)?;
    kwargs.set_item("shape", PyTuple::new(py, &shape)?)?;
    let arr = np.call_method("memmap", (path,), Some(&kwargs))?;
    // Defensive: memmap length must match the planned span.
    let nbytes: usize = arr.getattr("nbytes")?.extract()?;
    if nbytes != span.len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "zero-copy memmap nbytes {nbytes} != planned span {}",
            span.len
        )));
    }
    Ok(arr)
}

/// Try zero-copy `numpy.memmap`; `None` means fall back to dense materialize (copy).
pub(crate) fn try_zero_copy_read_numpy<'py>(
    py: Python<'py>,
    path: &str,
    mmap: &[u8],
    doc: &tetration::query::QueryDocument,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let planned = plan_read_for_document(doc, mmap).map_err(crate::tet_err)?;
    let Some(elem_size) = wire_dtype_elem_size(planned.dtype) else {
        return Ok(None);
    };
    let Some(span) = detect_single_chunk_raw_view(&planned.read_plan, elem_size) else {
        return Ok(None);
    };
    // Bounds-check against the open mapping before opening a second memmap.
    let end = span.start.checked_add(span.len).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("payload span overflow")
    })?;
    if end > mmap.len() {
        return Ok(None);
    }
    Ok(Some(memmap_zero_copy(py, path, &planned, span)?))
}

/// Build a shaped `numpy.ndarray` from a dense materialize outcome (owned copy).
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
