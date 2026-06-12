//! `PyO3` wrapper for [`tetration::catalog::TetWriterSession`].

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use tetration::catalog::{
    CatalogError, CoordAxisV1, TetDatasetWrite, TetWriterSession, DATASET_DTYPE_TAG_V1,
};

use crate::numpy_write;

fn catalog_err(err: &CatalogError) -> PyErr {
    crate::catalog_err(err)
}

fn extract_string_map(dict: Option<&Bound<'_, PyDict>>) -> PyResult<BTreeMap<String, String>> {
    let Some(dict) = dict else {
        return Ok(BTreeMap::new());
    };
    let mut out = BTreeMap::new();
    for (key, value) in dict.iter() {
        let k: String = key.extract()?;
        let v: String = value.extract()?;
        out.insert(k, v);
    }
    Ok(out)
}

fn extract_coords(
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<BTreeMap<String, CoordAxisV1>>> {
    let Some(dict) = dict else {
        return Ok(None);
    };
    let mut out = BTreeMap::new();
    for (key, value) in dict.iter() {
        let axis: String = key.extract()?;
        let labels: Vec<String> = value.extract()?;
        out.insert(axis, CoordAxisV1 { labels });
    }
    Ok(Some(out))
}

fn dataset_from_buffer(
    name: String,
    buf: numpy_write::NumpyWriteBuffer,
    chunk_shape: Vec<u64>,
) -> Result<TetDatasetWrite, CatalogError> {
    let tags = DATASET_DTYPE_TAG_V1;
    let dataset = if tags.is_f32(buf.dtype) {
        TetDatasetWrite::f32_row_major(name, &buf.shape, &chunk_shape, buf.data)?
    } else if tags.is_f64(buf.dtype) {
        TetDatasetWrite::f64_row_major(name, &buf.shape, &chunk_shape, buf.data)?
    } else {
        return Err(CatalogError::InvalidWriteSpec(
            "unsupported dataset dtype tag (tet-py write supports f32/f64 only)",
        ));
    };
    Ok(dataset)
}

/// Optional footer metadata and tiling for :meth:`PyTetWriterSession.write_dataset`.
#[pyclass(name = "WriteDatasetOptions")]
#[derive(Default)]
pub(crate) struct WriteDatasetOptions {
    #[pyo3(get, set)]
    chunk_shape: Option<Vec<u64>>,
    #[pyo3(get, set)]
    attrs: Option<Py<PyDict>>,
    #[pyo3(get, set)]
    dim_names: Option<Vec<String>>,
    #[pyo3(get, set)]
    coords: Option<Py<PyDict>>,
}

impl WriteDatasetOptions {
    fn attrs_bound<'py>(&self, py: Python<'py>) -> Option<&Bound<'py, PyDict>> {
        self.attrs.as_ref().map(|dict| dict.bind(py))
    }

    fn coords_bound<'py>(&self, py: Python<'py>) -> Option<&Bound<'py, PyDict>> {
        self.coords.as_ref().map(|dict| dict.bind(py))
    }
}

#[pymethods]
impl WriteDatasetOptions {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

/// Buffered ``.tet`` writer (create or append, then :meth:`commit`).
#[pyclass(name = "TetWriterSession")]
pub(crate) struct PyTetWriterSession {
    inner: Option<TetWriterSession>,
}

impl PyTetWriterSession {
    fn inner_mut(&mut self) -> PyResult<&mut TetWriterSession> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("TetWriterSession already committed"))
    }

    fn inner_ref(&self) -> PyResult<&TetWriterSession> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("TetWriterSession already committed"))
    }
}

#[pymethods]
impl PyTetWriterSession {
    /// Create a new file at ``path`` on :meth:`commit` (truncates any existing file).
    #[staticmethod]
    fn create(path: PathBuf) -> Self {
        Self {
            inner: Some(TetWriterSession::create(path)),
        }
    }

    /// Open an existing ``.tet`` and queue additional datasets on commit.
    #[staticmethod]
    fn open_append(path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: Some(TetWriterSession::open_append(path).map_err(|e| catalog_err(&e))?),
        })
    }

    #[getter]
    fn path(&self) -> PyResult<String> {
        Ok(self.inner_ref()?.path().display().to_string())
    }

    #[getter]
    fn dataset_count(&self) -> PyResult<usize> {
        Ok(self.inner_ref()?.dataset_count())
    }

    /// Queue one in-memory dataset from a row-major `NumPy` array.
    #[pyo3(signature = (name, array, options=None))]
    fn write_dataset(
        &mut self,
        py: Python<'_>,
        name: String,
        array: &Bound<'_, PyAny>,
        options: Option<Bound<'_, WriteDatasetOptions>>,
    ) -> PyResult<()> {
        let buf = numpy_write::array_to_write_buffer(py, array)?;
        let (chunk_shape, attrs, dim_names, coords) = if let Some(opts) = options {
            let opts = opts.borrow();
            let chunk_shape = opts
                .chunk_shape
                .clone()
                .unwrap_or_else(|| buf.shape.clone());
            let attrs = extract_string_map(opts.attrs_bound(py))?;
            let dim_names = opts.dim_names.clone();
            let coords = extract_coords(opts.coords_bound(py))?;
            (chunk_shape, attrs, dim_names, coords)
        } else {
            (buf.shape.clone(), BTreeMap::new(), None, None)
        };
        let mut dataset =
            dataset_from_buffer(name, buf, chunk_shape).map_err(|e| catalog_err(&e))?;
        dataset.attrs = attrs;
        dataset.coords = coords;
        dataset.dim_names = dim_names;
        self.inner_mut()?
            .push_dataset(dataset)
            .map_err(|e| catalog_err(&e))
    }

    /// Append a footer history row (``op``, ``source``, Unix timestamp added in Rust).
    fn push_history_event(&mut self, op: String, source: String) -> PyResult<()> {
        self.inner_mut()?.push_history_event(op, source);
        Ok(())
    }

    /// Flush queued datasets and footer metadata to disk.
    fn commit(&mut self) -> PyResult<String> {
        let mut inner = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("TetWriterSession already committed"))?;
        inner.metadata.tool = Some("tet-py".to_owned());
        inner.metadata.library_version = Some(env!("CARGO_PKG_VERSION").to_owned());
        let path = inner.commit().map_err(|e| catalog_err(&e))?;
        Ok(path.display().to_string())
    }
}
