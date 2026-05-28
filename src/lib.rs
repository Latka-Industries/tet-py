//! PyO3 extension: thin wrapper over the `tetration` crate.

use std::path::PathBuf;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use tetration::catalog::TetFile as CoreTetFile;
use tetration::query::{
    execute_query_document, parse_query_json, validate_query, ExecuteQueryOptions,
    QueryDocument, TetError as QueryError,
};

create_exception!(_native, TetError, PyException, "Query parse, validation, or execution error.");
create_exception!(
    _native,
    CatalogError,
    PyException,
    "`.tet` catalog read error (layout, index, codec)."
);

fn tet_err(err: QueryError) -> PyErr {
    match err {
        QueryError::Catalog(catalog) => catalog_err(catalog),
        other => TetError::new_err(other.to_string()),
    }
}

fn catalog_err(err: tetration::catalog::CatalogError) -> PyErr {
    CatalogError::new_err(err.to_string())
}

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

/// Open `.tet` read-only (mmap); keeps the file handle alive for the object lifetime.
#[pyclass(name = "TetFile")]
struct PyTetFile {
    inner: CoreTetFile,
}

#[pymethods]
impl PyTetFile {
    #[getter]
    fn path(&self) -> String {
        self.inner.path().display().to_string()
    }

    /// Catalog dataset names.
    fn datasets(&self) -> PyResult<Vec<String>> {
        let records = self.inner.datasets().map_err(catalog_err)?;
        Ok(records.into_iter().map(|r| r.name).collect())
    }

    /// Footer + catalog summary as JSON.
    fn summary_json(&self) -> PyResult<String> {
        let summary = self.inner.summary().map_err(catalog_err)?;
        serde_json::to_string(&summary).map_err(|e| CatalogError::new_err(e.to_string()))
    }

    /// Run a query document; returns JSON [`QueryResponse`].
    ///
    /// `query` may be a JSON string or a dict (serialized with `json.dumps`).
    fn query(&self, py: Python<'_>, query: &Bound<'_, PyAny>) -> PyResult<String> {
        let doc = parse_document(py, query)?;
        let response = execute_query_document(
            &doc,
            self.inner.path(),
            self.inner.mmap(),
            ExecuteQueryOptions::execute_no_preview(),
            None,
        )
        .map_err(tet_err)?;
        serde_json::to_string(&response).map_err(|e| TetError::new_err(e.to_string()))
    }
}

#[pyfunction]
fn core_version() -> &'static str {
    env!("TETRATION_VERSION")
}

#[pyfunction]
fn open(path: PathBuf) -> PyResult<PyTetFile> {
    let inner = CoreTetFile::open(&path).map_err(PyErr::from)?;
    Ok(PyTetFile { inner })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TetError", m.py().get_type::<TetError>())?;
    m.add("CatalogError", m.py().get_type::<CatalogError>())?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_class::<PyTetFile>()?;
    Ok(())
}
