//! Emit `TETRATION_VERSION` from the sibling `tetration` checkout (path dependency).

fn main() {
    let manifest = std::path::Path::new("../tetration/Cargo.toml");
    let version = if manifest.exists() {
        let text = std::fs::read_to_string(manifest).expect("read ../tetration/Cargo.toml");
        parse_package_version(&text).unwrap_or_else(|| {
            panic!("version = not found in ../tetration/Cargo.toml");
        })
    } else {
        // crates.io-only build: keep in sync with Cargo.toml `[dependencies.tetration].version`
        "0.1.5".to_owned()
    };
    println!("cargo:rustc-env=TETRATION_VERSION={version}");
}

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
