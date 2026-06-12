//! `PyO3` extension: thin wrapper over the `tetration` crate.
//!
//! Python sees `tet._native`: `open`, `TetFile`, `TetError`, `CatalogError`, `core_version`.
//! Query documents are parsed/validated in Rust; responses are JSON strings for the pure-Python layer.

mod numpy_export;
mod numpy_write;
mod writer;

use std::path::PathBuf;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tetration::catalog::TetFile as CoreTetFile;
use tetration::query::{
    execute_query_document, materialize_query_selection, materialize_query_transform_ram,
    parse_query_json, validate_query, ExecuteQueryOptions, QueryDocument, TetError as QueryError,
};

create_exception!(_native, TetError, PyException, "Query parse, validation, or execution error.");
create_exception!(
    _native,
    CatalogError,
    PyException,
    "`.tet` catalog read error (layout, index, codec)."
);

/// Map a `tetration::query::TetError` to the appropriate Python exception.
pub(crate) fn tet_err(err: QueryError) -> PyErr {
    match err {
        QueryError::Catalog(catalog) => catalog_err(&catalog),
        other => TetError::new_err(other.to_string()),
    }
}

/// Map a catalog read failure to [`CatalogError`].
pub(crate) fn catalog_err(err: &tetration::catalog::CatalogError) -> PyErr {
    CatalogError::new_err(err.to_string())
}

/// Parse and validate a query from Python (`str`, `dict`, or other JSON-serializable value).
///
/// Accepts a JSON string, `str`, or any object passed through `json.dumps`.
/// Runs `validate_query` before returning.
fn parse_document(py: Python<'_>, query: &Bound<'_, PyAny>) -> PyResult<QueryDocument> {
    if let Ok(s) = query.extract::<String>() {
        let doc = parse_query_json(&s).map_err(tet_err)?;
        validate_query(&doc).map_err(tet_err)?;
        return Ok(doc);
    }
    if let Ok(s) = query.extract::<&str>() {
        let doc = parse_query_json(s).map_err(tet_err)?;
        validate_query(&doc).map_err(tet_err)?;
        return Ok(doc);
    }
    let json_mod = py.import("json")?;
    let dumped: String = json_mod.call_method1("dumps", (query,))?.extract()?;
    let doc = parse_query_json(&dumped).map_err(tet_err)?;
    validate_query(&doc).map_err(tet_err)?;
    Ok(doc)
}

/// Read-only mmap handle to one `.tet` file (catalog + query engine).
///
/// Construct via [`open`]. The underlying file mapping stays alive for the object's lifetime.
#[pyclass(name = "TetFile")]
struct PyTetFile {
    inner: CoreTetFile,
}

impl PyTetFile {
    /// Parse `query`, run or plan per `options`, serialize [`QueryResponse`] to JSON.
    ///
    /// # Errors
    ///
    /// * [`TetError`] — parse, validation, or execution failure
    /// * [`CatalogError`] — catalog errors surfaced through query path
    fn execute_query_json(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        options: ExecuteQueryOptions,
    ) -> PyResult<String> {
        let doc = parse_document(py, query)?;
        let response = execute_query_document(
            &doc,
            self.inner.path(),
            self.inner.mmap(),
            options,
            None,
        )
        .map_err(tet_err)?;
        serde_json::to_string(&response).map_err(|e| TetError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyTetFile {
    /// Resolved filesystem path to the open `.tet` file.
    #[getter]
    fn path(&self) -> String {
        self.inner.path().display().to_string()
    }

    /// Dataset names from the catalog (declaration order).
    ///
    /// # Errors
    ///
    /// [`CatalogError`] if the catalog cannot be read.
    fn datasets(&self) -> PyResult<Vec<String>> {
        let records = self.inner.datasets().map_err(|e| catalog_err(&e))?;
        Ok(records.into_iter().map(|r| r.name).collect())
    }

    /// Footer metadata and catalog summary as a JSON string.
    ///
    /// # Returns
    ///
    /// JSON object suitable for `json.loads` (datasets, dtypes, chunking, footer fields).
    ///
    /// # Errors
    ///
    /// [`CatalogError`] on catalog/summary read failure; [`CatalogError`] if JSON serialization fails.
    fn summary_json(&self) -> PyResult<String> {
        let summary = self.inner.summary().map_err(|e| catalog_err(&e))?;
        serde_json::to_string(&summary).map_err(|e| CatalogError::new_err(e.to_string()))
    }

    /// Execute a query document and return the full wire response as JSON.
    ///
    /// # Arguments
    ///
    /// * `query` — JSON string or a `dict` (serialized with `json.dumps` before parse)
    ///
    /// # Returns
    ///
    /// JSON string containing execution results (same shape as the `tet query -x` CLI).
    ///
    /// # Errors
    ///
    /// [`TetError`] on parse/validation/execution errors; [`CatalogError`] when the error is catalog-related.
    fn query(&self, py: Python<'_>, query: &Bound<'_, PyAny>) -> PyResult<String> {
        self.execute_query_json(py, query, ExecuteQueryOptions::execute_no_preview())
    }

    /// Plan a query without executing (parity with `tet query` without `-x`).
    ///
    /// # Arguments
    ///
    /// * `query` — same accepted types as [`Self::query`]
    ///
    /// # Returns
    ///
    /// JSON plan document (no materialized aggregates).
    ///
    /// # Errors
    ///
    /// Same as [`Self::query`].
    fn plan_only(&self, py: Python<'_>, query: &Bound<'_, PyAny>) -> PyResult<String> {
        self.execute_query_json(py, query, ExecuteQueryOptions::plan_only())
    }

    /// Materialize a selection-only query document into a `numpy.ndarray`.
    ///
    /// # Errors
    ///
    /// [`TetError`] on parse/validation/materialize failures; [`CatalogError`] when catalog-related.
    fn read_numpy(&self, py: Python<'_>, query: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let doc = parse_document(py, query)?;
        let outcome =
            materialize_query_selection(&doc, self.inner.mmap()).map_err(tet_err)?;
        Ok(numpy_export::outcome_to_py(py, outcome)?.unbind())
    }

    /// Materialize a transform query (`write: ram`) into a `numpy.ndarray`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::read_numpy`].
    fn transform_to_numpy(&self, py: Python<'_>, query: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let doc = parse_document(py, query)?;
        let outcome = materialize_query_transform_ram(
            &doc,
            self.inner.mmap(),
            self.inner.path(),
            None,
        )
        .map_err(tet_err)?;
        Ok(numpy_export::outcome_to_py(py, outcome)?.unbind())
    }
}

/// Version string of the linked `tetration` crate (from build-time `TETRATION_VERSION`).
#[pyfunction]
fn core_version() -> &'static str {
    env!("TETRATION_VERSION")
}

/// Open a `.tet` file read-only via mmap.
///
/// # Arguments
///
/// * `path` — filesystem path (not expanded here; Python layer expands `~`)
///
/// # Returns
///
/// [`PyTetFile`] wrapping the core [`CoreTetFile`].
///
/// # Errors
///
/// Propagates I/O and catalog errors from `tetration::catalog::TetFile::open` as Python exceptions.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts `PathBuf` by value from Python paths.
fn open(path: PathBuf) -> PyResult<PyTetFile> {
    let inner = CoreTetFile::open(&path).map_err(PyErr::from)?;
    Ok(PyTetFile { inner })
}

/// Register the `_native` module: exceptions, `open`, `TetFile`, `core_version`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TetError", m.py().get_type::<TetError>())?;
    m.add("CatalogError", m.py().get_type::<CatalogError>())?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_class::<PyTetFile>()?;
    m.add_class::<writer::PyTetWriterSession>()?;
    m.add_class::<writer::WriteDatasetOptions>()?;
    Ok(())
}
