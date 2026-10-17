//! Emit `TETRATION_VERSION` for `core_version()`.
//!
//! A sibling `tetration` checkout wins (local path builds). Otherwise the version is the
//! `tetration` pin in this crate's `Cargo.toml`, so the wheel fallback cannot drift from the pin.

/// Read the sibling checkout when present, else the crates.io pin in `Cargo.toml`.
fn main() {
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let sibling = manifest_dir.join("../../tetration/Cargo.toml");
    println!("cargo:rerun-if-changed={}", sibling.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("Cargo.toml").display()
    );

    let version = if sibling.exists() {
        let text = std::fs::read_to_string(&sibling).expect("read sibling tetration Cargo.toml");
        parse_package_version(&text).expect("version = not found in sibling tetration Cargo.toml")
    } else {
        let text = std::fs::read_to_string(manifest_dir.join("Cargo.toml"))
            .expect("read native Cargo.toml");
        parse_dep_version(&text, "tetration").expect("tetration dependency version pin")
    };
    println!("cargo:rustc-env=TETRATION_VERSION={version}");
}

/// Extract `[package].version` from a `Cargo.toml` string (first `[package]` section only).
fn parse_package_version(cargo_toml: &str) -> Option<String> {
    let mut in_package = false;
    for line in cargo_toml.lines() {
        if line.trim() == "[package]" {
            in_package = true;
            continue;
        }
        if in_package && line.starts_with('[') {
            break;
        }
        if in_package {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("version = ") {
                return Some(rest.trim().trim_matches('"').to_owned());
            }
        }
    }
    None
}

/// Extract `name = { version = "…" }` from a `Cargo.toml` dependency line.
fn parse_dep_version(cargo_toml: &str, name: &str) -> Option<String> {
    let needle = format!("{name} = {{ version = \"");
    for line in cargo_toml.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(&needle) {
            let version = rest.split('"').next()?.trim();
            if !version.is_empty() {
                return Some(version.to_owned());
            }
        }
    }
    None
}
