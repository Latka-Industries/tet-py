//! `PyO3` extension: thin wrapper over the `tetration` crate.
//!
//! Python sees `tet._native`: `open`, `TetFile`, `TetError`, `CatalogError`, `core_version`,
//! `run_job`. Query documents are parsed/validated in Rust; responses are JSON strings for the
//! pure-Python layer.

mod numpy_export;
mod numpy_write;
mod writer;

use std::path::{Path, PathBuf};

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tempfile::TempDir;
use tetration::catalog::store::generation_manifest_key;
use tetration::catalog::{
    assemble_object_from_ranges, dry_run_publish_parts_with_options, ingest_tet_file,
    open_s3_package, packed_tet_key, parse_tet_uri, part_id_from_path as core_part_id_from_path,
    pin_packed_tet, plan_catalog_fetch, publish_parts_with_options, publish_staged_parts,
    read_superblock_at, read_tet_summary_at_store, seal_staged_generation, stage_part,
    GenerationPin, LocalStore, MemoryStore, OpenedS3Package, PackGenerationOptions, PackageStore,
    PublishGeneration, PublishPartsOptions, ReadAt, S3Config, S3Store, TetFile as CoreTetFile,
    TetFileSummaryV1, TetOpenTarget, TetPackage, WritePart,
};
use tetration::job::{run_job_path, JobError, RunJobOptions};
use tetration::query::{
    execute_query_document, materialize_query_selection, materialize_query_transform_ram,
    parse_query_json, parse_query_text, plan_query_fetch, plan_read_for_document,
    query_opened_s3_package, validate_query, ExecuteQueryOptions, QueryDocument, QueryInputFormat,
    TetError as QueryError,
};

create_exception!(
    _native,
    TetError,
    PyException,
    "Query parse, validation, or execution error."
);
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
/// String input uses [`parse_query_text`] with [`QueryInputFormat::Auto`] (leading `{` → JSON,
/// else TOML), matching the `tet query` CLI. Dicts and other objects go through `json.dumps`.
/// Runs `validate_query` before returning.
fn parse_document(py: Python<'_>, query: &Bound<'_, PyAny>) -> PyResult<QueryDocument> {
    let doc = if let Ok(s) = query.extract::<String>() {
        parse_query_text(&s, QueryInputFormat::Auto).map_err(tet_err)?
    } else if let Ok(s) = query.extract::<&str>() {
        parse_query_text(s, QueryInputFormat::Auto).map_err(tet_err)?
    } else {
        let json_mod = py.import("json")?;
        let dumped: String = json_mod.call_method1("dumps", (query,))?.extract()?;
        parse_query_json(&dumped).map_err(tet_err)?
    };
    validate_query(&doc).map_err(tet_err)?;
    Ok(doc)
}

/// Where the bytes for one [`PyTetFile`] live.
enum OpenedTet {
    /// Mapped `.tet` (a real file, or a temp copy of a fully downloaded store object).
    Local {
        file: CoreTetFile,
        /// Keeps a temp `packed.tet` alive for memory-store and `S3Store.open` downloads.
        _keepalive: Option<TempDir>,
    },
    /// Pinned `s3://` package. Queries range-read; they do not download `packed.tet`.
    S3(S3PackageHandle),
}

struct S3PackageHandle {
    opened: OpenedS3Package,
    uri: String,
}

/// Bytes a selection or transform materializes from.
enum SelectionImage<'a> {
    Mapped(&'a [u8]),
    Assembled(Vec<u8>),
}

impl SelectionImage<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mapped(bytes) => bytes,
            Self::Assembled(bytes) => bytes,
        }
    }
}

/// Read-only handle to one `.tet` file, or a pinned `s3://` package.
///
/// Construct via [`open`] / [`open_package`] / [`open_uri`]. A local file stays mapped for
/// the object's lifetime. An `s3://` package keeps [`OpenedS3Package`] and range-reads on
/// each query. Package / store opens pin one generation id.
#[pyclass(name = "TetFile")]
struct PyTetFile {
    source: OpenedTet,
    /// Sealed generation when opened from a package or store pin; `None` for plain `.tet`.
    generation: Option<u64>,
}

impl PyTetFile {
    fn local(file: CoreTetFile, generation: Option<u64>, keepalive: Option<TempDir>) -> Self {
        Self {
            source: OpenedTet::Local {
                file,
                _keepalive: keepalive,
            },
            generation,
        }
    }

    fn from_core(inner: CoreTetFile, generation: Option<u64>) -> Self {
        Self::local(inner, generation, None)
    }

    fn catalog_summary(&self) -> PyResult<TetFileSummaryV1> {
        match &self.source {
            OpenedTet::S3(s3) => s3_summary(&s3.opened),
            OpenedTet::Local { file, .. } => file.summary().map_err(|err| catalog_err(&err)),
        }
    }

    /// Mapped file bytes, or a sparse image of the ranges `doc` touches.
    fn selection_image<'a>(&'a self, doc: &QueryDocument) -> PyResult<SelectionImage<'a>> {
        match &self.source {
            OpenedTet::S3(s3) => Ok(SelectionImage::Assembled(s3_selection_image(
                &s3.opened, doc,
            )?)),
            OpenedTet::Local { file, .. } => Ok(SelectionImage::Mapped(file.mmap())),
        }
    }

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
        execute: bool,
        options: ExecuteQueryOptions,
    ) -> PyResult<String> {
        let doc = parse_document(py, query)?;
        let response = match &self.source {
            OpenedTet::S3(s3) => {
                query_opened_s3_package(&doc, &s3.opened, execute, options, None, Some(&s3.uri))
                    .map_err(tet_err)?
            }
            OpenedTet::Local { file, .. } => {
                execute_query_document(&doc, file.path(), file.mmap(), options, None)
                    .map_err(tet_err)?
            }
        };
        serde_json::to_string(&response).map_err(|e| TetError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyTetFile {
    /// Filesystem path of a local `.tet`, or the `s3://` URI of a remote package.
    #[getter]
    fn path(&self) -> String {
        match &self.source {
            OpenedTet::S3(s3) => s3.uri.clone(),
            OpenedTet::Local { file, .. } => file.path().display().to_string(),
        }
    }

    /// Pinned package generation id, or `None` for a standalone `.tet`.
    #[getter]
    fn generation(&self) -> Option<u64> {
        self.generation
    }

    /// Dataset names from the catalog (declaration order).
    ///
    /// # Errors
    ///
    /// [`CatalogError`] if the catalog cannot be read.
    fn datasets(&self) -> PyResult<Vec<String>> {
        let summary = self.catalog_summary()?;
        Ok(summary
            .datasets
            .into_iter()
            .map(|record| record.name)
            .collect())
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
        let summary = self.catalog_summary()?;
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
    #[pyo3(signature = (query, preview_max=None))]
    fn query(
        &self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        preview_max: Option<usize>,
    ) -> PyResult<String> {
        let options = match preview_max {
            Some(n) => ExecuteQueryOptions {
                preview_max: Some(n),
            },
            None => ExecuteQueryOptions::execute_no_preview(),
        };
        self.execute_query_json(py, query, true, options)
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
        self.execute_query_json(py, query, false, ExecuteQueryOptions::plan_only())
    }

    /// Materialize a selection-only query document into a `numpy.ndarray`.
    ///
    /// When raw payloads already form the logical row-major selection — one chunk, or
    /// several chunks that tile only the leading axis and sit in one contiguous span —
    /// returns a read-only
    /// [`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
    /// over that region (zero-copy). zstd, strided selections, and partial trailing tiles copy.
    ///
    /// # Errors
    ///
    /// [`TetError`] on parse/validation/materialize failures; [`CatalogError`] when catalog-related.
    fn read_numpy(&self, py: Python<'_>, query: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let doc = parse_document(py, query)?;
        if let OpenedTet::Local { file, .. } = &self.source {
            let path = file.path().display().to_string();
            if let Some(view) =
                numpy_export::try_zero_copy_read_numpy(py, &path, file.mmap(), &doc)?
            {
                return Ok(view.unbind());
            }
        }
        let image = self.selection_image(&doc)?;
        let outcome = materialize_query_selection(&doc, image.as_slice()).map_err(tet_err)?;
        Ok(numpy_export::outcome_to_py(py, outcome)?.unbind())
    }

    /// Materialize a transform query (`write: ram`) into a `numpy.ndarray`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::read_numpy`].
    fn transform_to_numpy(&self, py: Python<'_>, query: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let doc = parse_document(py, query)?;
        let image = self.selection_image(&doc)?;
        let path = match &self.source {
            OpenedTet::S3(s3) => PathBuf::from(&s3.uri),
            OpenedTet::Local { file, .. } => file.path().to_path_buf(),
        };
        let outcome = materialize_query_transform_ram(&doc, image.as_slice(), &path, None)
            .map_err(tet_err)?;
        Ok(numpy_export::outcome_to_py(py, outcome)?.unbind())
    }
}

/// Version string of the linked `tetration` crate (from build-time `TETRATION_VERSION`).
#[pyfunction]
fn core_version() -> &'static str {
    env!("TETRATION_VERSION")
}

fn job_err(err: &JobError) -> PyErr {
    TetError::new_err(err.to_string())
}

/// Run a `tet run` job file via [`tetration::job::run_job_path`].
///
/// Returns the [`JobReport`](tetration::job::JobReport) as a JSON string (same shape as
/// `tet run --json` on success). Does not re-implement steps in Python.
///
/// # Errors
///
/// [`TetError`] for parse / pre-flight / step failures (same messages as the CLI).
#[pyfunction]
#[pyo3(signature = (path, *, root=None, dry_run=false))]
#[allow(clippy::needless_pass_by_value)]
fn run_job(path: PathBuf, root: Option<PathBuf>, dry_run: bool) -> PyResult<String> {
    let opts = RunJobOptions {
        root,
        dry_run,
        allow_interactive: false,
        record_query_history: false,
    };
    let report = run_job_path(&path, &opts).map_err(|e| job_err(&e))?;
    serde_json::to_string(&report).map_err(|e| TetError::new_err(e.to_string()))
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
    Ok(PyTetFile::from_core(inner, None))
}

/// Open the `CURRENT` generation of a `.tet.d` package (`packed.tet`).
///
/// Pins once via [`TetPackage::pin_current`]; advancing `CURRENT` later does not retarget
/// this handle.
///
/// # Errors
///
/// [`CatalogError`] when the path is not a package or no generation is published.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn open_package(path: PathBuf) -> PyResult<PyTetFile> {
    open_local_package(&path, GenerationPin::Current)
}

fn open_local_package(path: &Path, pin: GenerationPin) -> PyResult<PyTetFile> {
    let package = TetPackage::open(path).map_err(|err| catalog_err(&err))?;
    let pinned = match pin {
        GenerationPin::Current => package.pin_current().map_err(|err| catalog_err(&err))?,
        GenerationPin::Explicit(g) => package.pin_generation(g).map_err(|err| catalog_err(&err))?,
    };
    let generation = pinned.generation();
    let inner = pinned.open().map_err(|err| catalog_err(&err))?;
    Ok(PyTetFile::from_core(inner, Some(generation)))
}

/// Open a path or tetration URI (`file://`, local `.tet` / `.tet.d`, `s3://bucket/prefix`, `?generation=`).
///
/// `memory://` requires [`PyMemoryStore::open`]. `s3://` pins the package with
/// [`open_s3_package`] (`TETRATION_S3_*`; credentials are not in the URI) and
/// range-reads on query. `S3Store.open` still materializes a full `packed.tet`.
///
/// # Errors
///
/// [`CatalogError`] / [`PyValueError`] for unsupported schemes or missing objects.
#[pyfunction]
fn open_uri(py: Python<'_>, uri: &str) -> PyResult<PyTetFile> {
    let parsed = parse_tet_uri(uri).map_err(|e| CatalogError::new_err(e.to_string()))?;
    match parsed.target {
        TetOpenTarget::LocalTet(path) => open(path),
        TetOpenTarget::LocalPackage { root, generation } => open_local_package(&root, generation),
        TetOpenTarget::StorePackage { .. } => Err(PyValueError::new_err(
            "memory:// URIs need MemoryStore.open(...); use MemoryStore.from_package(path)",
        )),
        TetOpenTarget::S3 { .. } => py.detach(|| open_s3_package_uri(uri)),
    }
}

fn open_s3_package_uri(uri: &str) -> PyResult<PyTetFile> {
    let opened = open_s3_package(uri).map_err(|err| CatalogError::new_err(err.to_string()))?;
    let generation = opened.pin().generation;
    Ok(PyTetFile {
        source: OpenedTet::S3(S3PackageHandle {
            opened,
            uri: uri.to_owned(),
        }),
        generation: Some(generation),
    })
}

fn s3_summary(opened: &OpenedS3Package) -> PyResult<TetFileSummaryV1> {
    let pin = opened.pin();
    read_tet_summary_at_store(opened.store(), &pin.object_key, pin.object_len)
        .map_err(|err| catalog_err(&err))
}

/// Catalog plus the chunk ranges `doc` touches, as a sparse `packed.tet` image.
fn s3_selection_image(opened: &OpenedS3Package, doc: &QueryDocument) -> PyResult<Vec<u8>> {
    let pin = opened.pin();
    let store = opened.store();
    let sb = read_superblock_at(store, &pin.object_key, pin.object_len)
        .map_err(|err| catalog_err(&err))?;
    let catalog_plan = plan_catalog_fetch(pin.object_len, &sb).map_err(|err| catalog_err(&err))?;
    let catalog =
        assemble_object_from_ranges(store, &pin.object_key, pin.object_len, &catalog_plan.ranges)
            .map_err(|err| catalog_err(&err))?;
    let planned = plan_read_for_document(doc, &catalog).map_err(tet_err)?;
    let fetch = plan_query_fetch(&pin.object_key, pin.object_len, &sb, &planned.read_plan)
        .map_err(|err| catalog_err(&err))?;
    assemble_object_from_ranges(store, &pin.object_key, pin.object_len, &fetch.ranges)
        .map_err(|err| catalog_err(&err))
}

/// Resolve `current` / `g/<id>` pin text into a [`GenerationPin`].
fn parse_store_pin(pin: &str) -> PyResult<GenerationPin> {
    let pin = pin.trim();
    if pin.is_empty() || pin == "current" {
        return Ok(GenerationPin::Current);
    }
    if let Some(id) = pin.strip_prefix("g/") {
        let generation = id
            .parse::<u64>()
            .map_err(|_| PyValueError::new_err(format!("invalid generation pin: {pin}")))?;
        return Ok(GenerationPin::Explicit(generation));
    }
    Err(PyValueError::new_err(
        "store pin must be 'current' or 'g/<id>'",
    ))
}

/// Next generation after `CURRENT`, or `1` when unset.
fn next_store_generation(store: &impl PackageStore) -> PyResult<u64> {
    Ok(store
        .current_generation()
        .map_err(|err| catalog_err(&err))?
        .map_or(1, |current| current + 1))
}

/// Materialize a pinned store object to a temp `.tet` for mmap query / `read_numpy`.
fn open_store_pin(store: &impl PackageStore, pin: GenerationPin) -> PyResult<PyTetFile> {
    let pinned = pin_packed_tet(store, pin).map_err(|err| catalog_err(&err))?;
    let bytes = store
        .read_all(&pinned.object_key)
        .map_err(|err| catalog_err(&err))?;
    let dir = TempDir::new().map_err(PyErr::from)?;
    let packed_path = dir.path().join("packed.tet");
    std::fs::write(&packed_path, &bytes).map_err(PyErr::from)?;
    let inner = CoreTetFile::open(&packed_path).map_err(PyErr::from)?;
    Ok(PyTetFile::local(inner, Some(pinned.generation), Some(dir)))
}

/// Stage a sealed `.tet` file into a [`PackageStore`].
fn store_ingest_path(
    store: &impl PackageStore,
    source: &Path,
    session_id: &str,
    part_id: &str,
) -> PyResult<String> {
    let bytes = std::fs::read(source).map_err(PyErr::from)?;
    let meta = stage_part(store, session_id, part_id, &bytes).map_err(|err| catalog_err(&err))?;
    Ok(meta.part_id)
}

/// Seal staged parts and CAS-publish `CURRENT`.
fn store_publish_parts(
    store: &impl PackageStore,
    generation: Option<u64>,
    part_ids: &[String],
) -> PyResult<u64> {
    let generation = match generation {
        Some(generation) => generation,
        None => next_store_generation(store)?,
    };
    publish_staged_parts(store, generation, part_ids).map_err(|err| catalog_err(&err))?;
    Ok(generation)
}

/// CAS-write `CURRENT` to `generation` (`expected` must match the live pointer).
fn store_cas_publish(
    store: &impl PackageStore,
    generation: u64,
    expected: Option<u64>,
) -> PyResult<()> {
    store
        .publish_generation(generation, expected)
        .map_err(|err| catalog_err(&err))
}

/// Seal staged parts without flipping `CURRENT` (lost-CAS tests).
fn store_seal_parts(
    store: &impl PackageStore,
    generation: u64,
    part_ids: &[String],
) -> PyResult<u64> {
    seal_staged_generation(store, generation, part_ids).map_err(|err| catalog_err(&err))?;
    Ok(generation)
}

/// In-memory package store seeded from a local `.tet.d` (THI-504).
#[pyclass(name = "MemoryStore")]
struct PyMemoryStore {
    inner: MemoryStore,
}

#[pymethods]
impl PyMemoryStore {
    /// Empty in-memory store (parallel ingest / publish tests).
    #[new]
    fn new() -> Self {
        Self {
            inner: MemoryStore::new(),
        }
    }

    /// Copy `CURRENT` (and its `packed.tet` / manifest) from a local package into memory.
    ///
    /// # Errors
    ///
    /// [`CatalogError`] when the package has no published generation or I/O fails.
    #[classmethod]
    #[allow(clippy::needless_pass_by_value)]
    fn from_package(_cls: &Bound<'_, pyo3::types::PyType>, path: PathBuf) -> PyResult<Self> {
        let local = LocalStore::new(&path);
        let Some(generation) = local
            .current_generation()
            .map_err(|err| catalog_err(&err))?
        else {
            return Err(CatalogError::new_err("package has no CURRENT generation"));
        };
        let mem = MemoryStore::new();
        let packed_key = packed_tet_key(generation);
        let packed = local
            .read_all(&packed_key)
            .map_err(|err| catalog_err(&err))?;
        mem.write_part(&packed_key, &packed)
            .map_err(|err| catalog_err(&err))?;
        let manifest_key = generation_manifest_key(generation);
        if let Ok(manifest) = local.read_all(&manifest_key) {
            mem.write_part(&manifest_key, &manifest)
                .map_err(|err| catalog_err(&err))?;
        }
        mem.publish_generation(generation, None)
            .map_err(|err| catalog_err(&err))?;
        Ok(Self { inner: mem })
    }

    /// Stage a sealed `.tet` under `staging/<session_id>/`.
    ///
    /// # Errors
    ///
    /// [`CatalogError`] when the source is missing or the session/part id collides.
    #[pyo3(signature = (source, *, session_id, part_id))]
    #[allow(clippy::needless_pass_by_value)]
    fn ingest(&self, source: PathBuf, session_id: String, part_id: String) -> PyResult<String> {
        store_ingest_path(&self.inner, &source, &session_id, &part_id)
    }

    /// Seal `part_ids` and CAS-publish `CURRENT` (`generation=None` → next id).
    ///
    /// # Errors
    ///
    /// [`CatalogError`] when parts are missing, claims overlap, or the CAS is rejected.
    #[pyo3(signature = (part_ids, *, generation=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn publish(&self, part_ids: Vec<String>, generation: Option<u64>) -> PyResult<u64> {
        store_publish_parts(&self.inner, generation, &part_ids)
    }

    /// Seal `part_ids` into `generation` without flipping `CURRENT`.
    ///
    /// # Errors
    ///
    /// [`CatalogError`] when parts are missing, claims overlap, or the publish lock is held.
    #[allow(clippy::needless_pass_by_value)]
    fn seal(&self, part_ids: Vec<String>, generation: u64) -> PyResult<u64> {
        store_seal_parts(&self.inner, generation, &part_ids)
    }

    /// CAS-write `CURRENT` to `generation` (`expected` must match the live pointer).
    ///
    /// # Errors
    ///
    /// [`CatalogError::InvalidCurrent`] when `expected` does not match.
    #[pyo3(signature = (generation, *, expected))]
    fn cas_publish(&self, generation: u64, expected: Option<u64>) -> PyResult<()> {
        store_cas_publish(&self.inner, generation, expected)
    }

    /// Pin a generation (`memory://current` or `memory://g/<id>`) and open `packed.tet`.
    ///
    /// Materializes the pinned object to a temp file so existing mmap query / `read_numpy`
    /// paths work. The temp directory lives for the life of the returned handle.
    ///
    /// # Errors
    ///
    /// [`CatalogError`] / [`PyValueError`] for bad URIs or missing objects.
    #[pyo3(signature = (uri="memory://current"))]
    fn open(&self, uri: &str) -> PyResult<PyTetFile> {
        let parsed = parse_tet_uri(uri).map_err(|e| CatalogError::new_err(e.to_string()))?;
        let TetOpenTarget::StorePackage { generation } = parsed.target else {
            return Err(PyValueError::new_err(
                "MemoryStore.open expects a memory:// URI (e.g. memory://current)",
            ));
        };
        open_store_pin(&self.inner, generation)
    }
}

/// S3-compatible package store (endpoint + credentials as arguments; THI-506).
#[pyclass(name = "S3Store")]
struct PyS3Store {
    inner: S3Store,
}

#[pymethods]
impl PyS3Store {
    /// Connect to an S3-compatible bucket (does not create the bucket).
    ///
    /// # Errors
    ///
    /// [`CatalogError`] for a bad endpoint / client configuration.
    #[classmethod]
    #[pyo3(signature = (*, endpoint, bucket, access_key_id, secret_access_key, prefix="", region="us-east-1"))]
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    fn connect(
        _cls: &Bound<'_, pyo3::types::PyType>,
        endpoint: String,
        bucket: String,
        access_key_id: String,
        secret_access_key: String,
        prefix: &str,
        region: &str,
    ) -> PyResult<Self> {
        let inner = S3Store::connect(S3Config {
            endpoint,
            bucket,
            prefix: prefix.to_owned(),
            region: region.to_owned(),
            access_key_id,
            secret_access_key,
        })
        .map_err(|err| catalog_err(&err))?;
        Ok(Self { inner })
    }

    /// Stage a sealed `.tet` under `staging/<session_id>/`.
    ///
    /// # Errors
    ///
    /// [`CatalogError`] when the source is missing or the session/part id collides.
    #[pyo3(signature = (source, *, session_id, part_id))]
    #[allow(clippy::needless_pass_by_value)]
    fn ingest(
        &self,
        py: Python<'_>,
        source: PathBuf,
        session_id: String,
        part_id: String,
    ) -> PyResult<String> {
        py.detach(|| store_ingest_path(&self.inner, &source, &session_id, &part_id))
    }

    /// Seal `part_ids` and CAS-publish `CURRENT` (`generation=None` → next id).
    ///
    /// # Errors
    ///
    /// [`CatalogError`] when parts are missing, claims overlap, or the CAS is rejected.
    #[pyo3(signature = (part_ids, *, generation=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn publish(
        &self,
        py: Python<'_>,
        part_ids: Vec<String>,
        generation: Option<u64>,
    ) -> PyResult<u64> {
        py.detach(|| store_publish_parts(&self.inner, generation, &part_ids))
    }

    /// Seal `part_ids` into `generation` without flipping `CURRENT`.
    ///
    /// # Errors
    ///
    /// [`CatalogError`] when parts are missing, claims overlap, or the publish lock is held.
    #[allow(clippy::needless_pass_by_value)]
    fn seal(&self, py: Python<'_>, part_ids: Vec<String>, generation: u64) -> PyResult<u64> {
        py.detach(|| store_seal_parts(&self.inner, generation, &part_ids))
    }

    /// CAS-write `CURRENT` to `generation` (`expected` must match the live pointer).
    ///
    /// # Errors
    ///
    /// [`CatalogError`] when the conditional put is rejected (last good generation kept).
    #[pyo3(signature = (generation, *, expected))]
    fn cas_publish(&self, py: Python<'_>, generation: u64, expected: Option<u64>) -> PyResult<()> {
        py.detach(|| store_cas_publish(&self.inner, generation, expected))
    }

    /// Pin `current` or `g/<id>` and open that generation's `packed.tet`.
    ///
    /// Same pin rule as 0.3.0: advancing `CURRENT` later does not retarget this handle.
    ///
    /// # Errors
    ///
    /// [`CatalogError`] / [`PyValueError`] for a bad pin or missing objects.
    #[pyo3(signature = (pin="current"))]
    fn open(&self, py: Python<'_>, pin: &str) -> PyResult<PyTetFile> {
        let generation = parse_store_pin(pin)?;
        py.detach(|| open_store_pin(&self.inner, generation))
    }
}

/// Bucket-root client using `TETRATION_S3_*` (`S3Config::from_env`). `key` is the full object key.
fn connect_s3_from_env(bucket: String) -> PyResult<S3Store> {
    let config = S3Config::from_env(bucket, String::new()).map_err(|err| catalog_err(&err))?;
    S3Store::connect(config).map_err(|err| catalog_err(&err))
}

/// Download one object for `tet.convert` (`s3://` single-file sources).
///
/// Credentials come from `TETRATION_S3_*` via [`S3Config::from_env`]. The bucket is the
/// one in the URI; the prefix is empty so `key` is the full object key.
///
/// # Errors
///
/// [`CatalogError`] when the endpoint is unset, the client fails, or the object is missing.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn download_s3_object(py: Python<'_>, bucket: String, key: String) -> PyResult<Vec<u8>> {
    py.detach(|| {
        let store = connect_s3_from_env(bucket)?;
        store.read_all(&key).map_err(|err| catalog_err(&err))
    })
}

/// Upload one object so convert tests can stage a fixture. Not a package ingest.
///
/// # Errors
///
/// [`CatalogError`] when the endpoint is unset or the put fails.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn upload_s3_object(py: Python<'_>, bucket: String, key: String, body: Vec<u8>) -> PyResult<()> {
    py.detach(|| {
        let store = connect_s3_from_env(bucket)?;
        store
            .write_part(&key, &body)
            .map_err(|err| catalog_err(&err))
    })
}

/// List object keys under `prefix` (inclusive). Keys are relative to `prefix`.
///
/// Used by `tet.convert` for an `s3://` Zarr store so the zarr library can walk
/// the hierarchy without downloading chunk bodies first.
///
/// # Errors
///
/// [`CatalogError`] when the endpoint is unset or the list fails.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn list_s3_prefix(py: Python<'_>, bucket: String, prefix: String) -> PyResult<Vec<String>> {
    py.detach(|| {
        let store = connect_s3_from_env(bucket)?;
        let trimmed = prefix.trim_matches('/').to_string();
        let keys = store
            .list_prefix(&trimmed)
            .map_err(|err| catalog_err(&err))?;
        if trimmed.is_empty() {
            return Ok(keys);
        }
        let head = format!("{trimmed}/");
        Ok(keys
            .into_iter()
            .filter_map(|key| {
                if key == trimmed {
                    None
                } else {
                    key.strip_prefix(&head).map(str::to_owned)
                }
            })
            .collect())
    })
}

/// Create an empty `.tet.d` package directory.
///
/// # Errors
///
/// [`CatalogError`] when the path already exists or cannot be created.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn package_create(path: PathBuf) -> PyResult<()> {
    TetPackage::create(&path).map_err(|err| catalog_err(&err))?;
    Ok(())
}

/// Suggest a part id from a source path stem.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn part_id_from_path(path: PathBuf) -> String {
    core_part_id_from_path(&path)
}

/// Copy a sealed `.tet` into `staging/<session_id>/` and write `PART_META`.
///
/// # Errors
///
/// [`CatalogError`] when the package or source is missing, or the part id collides.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn package_ingest(
    package: PathBuf,
    source: PathBuf,
    session_id: String,
    part_id: String,
) -> PyResult<String> {
    let pkg = TetPackage::open(&package).map_err(|err| catalog_err(&err))?;
    let meta =
        ingest_tet_file(&pkg, &source, session_id, part_id).map_err(|err| catalog_err(&err))?;
    Ok(meta.part_id)
}

/// Seal `part_ids` into generation `generation` (`None` = next id after `CURRENT`).
///
/// When `dry_run` is true, run claim / schema checks only (no seal, no `CURRENT` flip).
/// `replace` lists dataset names allowed to change dtype/shape vs sealed `CURRENT`.
///
/// # Errors
///
/// [`CatalogError`] when parts are missing, claims overlap, schema conflicts, or the
/// publish lock is held.
#[pyfunction]
#[pyo3(signature = (package, generation, part_ids, *, dry_run=false, replace=None))]
#[allow(clippy::needless_pass_by_value)]
fn package_publish(
    package: PathBuf,
    generation: Option<u64>,
    part_ids: Vec<String>,
    dry_run: bool,
    replace: Option<Vec<String>>,
) -> PyResult<u64> {
    let pkg = TetPackage::open(&package).map_err(|err| catalog_err(&err))?;
    let generation = match generation {
        Some(generation) => generation,
        None => pkg.next_generation().map_err(|err| catalog_err(&err))?,
    };
    let options = PublishPartsOptions {
        replace: replace.unwrap_or_default(),
    };
    if dry_run {
        let validation = dry_run_publish_parts_with_options(&pkg, generation, &part_ids, &options)
            .map_err(|err| catalog_err(&err))?;
        return Ok(validation.generation);
    }
    publish_parts_with_options(&pkg, generation, &part_ids, &options)
        .map_err(|err| catalog_err(&err))?;
    Ok(generation)
}

/// Copy a sealed generation's `packed.tet` to a standalone `.tet`.
///
/// When `remove_package` is true, delete the `.tet.d` only after a verified copy
/// (same as `tet pack --remove-package`).
///
/// # Errors
///
/// [`CatalogError`] when the package or generation is missing, the destination exists,
/// verify fails, or remove is refused.
#[pyfunction]
#[pyo3(signature = (package, dest, generation=None, remove_package=false))]
#[allow(clippy::needless_pass_by_value)]
fn package_pack(
    package: PathBuf,
    dest: PathBuf,
    generation: Option<u64>,
    remove_package: bool,
) -> PyResult<u64> {
    let pkg = TetPackage::open(&package).map_err(|err| catalog_err(&err))?;
    let generation = match generation {
        Some(generation) => generation,
        None => pkg
            .current()
            .map_err(|err| catalog_err(&err))?
            .ok_or_else(|| CatalogError::new_err("package has no CURRENT generation"))?,
    };
    pkg.pack_generation_with(generation, &dest, &PackGenerationOptions { remove_package })
        .map_err(|err| catalog_err(&err))?;
    Ok(generation)
}

/// Register the `_native` module: exceptions, `open`, `TetFile`, `core_version`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TetError", m.py().get_type::<TetError>())?;
    m.add("CatalogError", m.py().get_type::<CatalogError>())?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(run_job, m)?)?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(open_package, m)?)?;
    m.add_function(wrap_pyfunction!(open_uri, m)?)?;
    m.add_function(wrap_pyfunction!(download_s3_object, m)?)?;
    m.add_function(wrap_pyfunction!(upload_s3_object, m)?)?;
    m.add_function(wrap_pyfunction!(list_s3_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(package_create, m)?)?;
    m.add_function(wrap_pyfunction!(part_id_from_path, m)?)?;
    m.add_function(wrap_pyfunction!(package_ingest, m)?)?;
    m.add_function(wrap_pyfunction!(package_publish, m)?)?;
    m.add_function(wrap_pyfunction!(package_pack, m)?)?;
    m.add_class::<PyTetFile>()?;
    m.add_class::<PyMemoryStore>()?;
    m.add_class::<PyS3Store>()?;
    m.add_class::<writer::PyTetWriterSession>()?;
    m.add_class::<writer::WriteDatasetOptions>()?;
    Ok(())
}
