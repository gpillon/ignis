//! Write an artifact's six frontend resources into a directory, one file per
//! resource under its base name (GitHub #176). The vision fixture recorder
//! (`tools/vision-fixtures`) feeds these files to the reference frontend, so
//! both engines prepare prompts from the same container bytes.
//!
//! Usage: `cargo run -p ignis-artifact --example dump_frontend -- <artifact.ninfer> <out-dir>`

use std::path::Path;

use ignis_artifact::{Reader, FRONTEND_RESOURCES};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [artifact, out] = args.as_slice() else {
        eprintln!("usage: dump_frontend <artifact.ninfer> <out-dir>");
        std::process::exit(2);
    };
    let reader = Reader::open(Path::new(artifact)).unwrap_or_else(|e| panic!("open {artifact}: {e}"));
    let out = Path::new(out);
    std::fs::create_dir_all(out).expect("create out dir");
    for name in FRONTEND_RESOURCES {
        let object = reader
            .find(&format!("frontend/{name}"))
            .unwrap_or_else(|| panic!("frontend/{name} missing"));
        let span = reader.payload_at(object).unwrap_or_else(|e| panic!("read {name}: {e}"));
        std::fs::write(out.join(name), span.data).unwrap_or_else(|e| panic!("write {name}: {e}"));
        println!("{name}: {} bytes", span.data.len());
    }
}
