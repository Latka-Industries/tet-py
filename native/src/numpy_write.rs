//! Copy a C-contiguous ``numpy.ndarray`` into row-major LE bytes for catalog write.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tetration::catalog::DATASET_DTYPE_TAG_V1;

/// Wire dtype tag, logical shape, and row-major little-endian payload bytes.
pub(crate) struct NumpyWriteBuffer {
    pub dtype: u32,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

const SUPPORTED_NUMPY_DTYPES: &str =
    "float16, float32, float64, int16, int32, int64, uint8, uint16, uint32, uint64";

/// Validate and copy one `NumPy` array into a write buffer (layout v1 wire dtypes).
pub(crate) fn array_to_write_buffer(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
) -> PyResult<NumpyWriteBuffer> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (array,))?;
    let dtype_name: String = arr.getattr("dtype")?.getattr("name")?.extract()?;
    let dtype = dtype_from_numpy_name(&dtype_name).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "unsupported numpy dtype {dtype_name:?} for write (supported: {SUPPORTED_NUMPY_DTYPES})"
        ))
    })?;
    let shape_py: Vec<usize> = arr.getattr("shape")?.extract()?;
    if shape_py.is_empty() {
        return Err(PyValueError::new_err(
            "write_dataset requires an array with at least one dimension",
        ));
    }
    let shape: Vec<u64> = shape_py
        .iter()
        .map(|&d| usize_to_u64(d))
        .collect::<PyResult<_>>()?;
    let data: Vec<u8> = arr.call_method0("tobytes")?.extract()?;
    validate_element_bytes(dtype, &shape, data.len())?;
    Ok(NumpyWriteBuffer { dtype, shape, data })
}

fn usize_to_u64(d: usize) -> PyResult<u64> {
    u64::try_from(d).map_err(|_| PyValueError::new_err("shape dimension overflow"))
}

fn dtype_from_numpy_name(name: &str) -> Option<u32> {
    let tags = DATASET_DTYPE_TAG_V1;
    match name {
        "float16" => Some(tags.f16),
        "float32" => Some(tags.f32),
        "float64" => Some(tags.f64),
        "int16" => Some(tags.i16),
        "int32" => Some(tags.i32),
        "int64" => Some(tags.i64),
        "uint8" => Some(tags.u8),
        "uint16" => Some(tags.u16),
        "uint32" => Some(tags.u32),
        "uint64" => Some(tags.u64),
        _ => None,
    }
}

fn validate_element_bytes(dtype: u32, shape: &[u64], nbytes: usize) -> PyResult<()> {
    let need = tetration::catalog::tensor_bytes_from_shape(shape, dtype)
        .ok_or_else(|| PyValueError::new_err("payload size overflow for shape and dtype"))?;
    if nbytes as u64 != need {
        return Err(PyValueError::new_err(format!(
            "array nbytes {nbytes} != expected {need} for shape and dtype"
        )));
    }
    Ok(())
}
