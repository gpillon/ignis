//! ignis-artifact-inspect: dump the identity + object inventory of a
//! `.ninfer` v2 artifact.
//!
//! Usage:
//!   ignis-artifact-inspect <artifact.ninfer>
//!   ignis-artifact-inspect <artifact.ninfer> --find NAME
//!
//! Prints the identity, file geometry, object counts, the tensor format
//! histogram, and every resource entry. With `--find NAME`, also prints the
//! full descriptor of one object (for binder planning, ticket 03).

use std::collections::BTreeMap;
use std::path::Path;

use ignis_artifact::{Object, Reader};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: ignis-artifact-inspect <artifact.ninfer> [--find NAME]");
        std::process::exit(2);
    }
    let path = args[0].clone();
    let find = if args.len() == 3 && args[1] == "--find" {
        Some(args[2].clone())
    } else if args.len() > 1 {
        eprintln!("unknown argument: {}", args[1]);
        std::process::exit(2);
    } else {
        None
    };

    let reader = match Reader::open(Path::new(&path)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let identity = reader.identity();
    let objects = reader.objects();
    let tensors = objects.iter().filter(|o| matches!(o, Object::Tensor(_))).count();
    let resources = objects.len() - tensors;

    let mut formats: BTreeMap<&str, usize> = BTreeMap::new();
    for o in objects {
        if let Object::Tensor(t) = o {
            *formats.entry(t.format.name()).or_insert(0) += 1;
        }
    }

    println!("identity: {}/{}", identity.model_id, identity.weights_id);
    println!(
        "file: {} bytes, payload_offset {}",
        reader.file_bytes(),
        reader.payload_offset()
    );
    println!(
        "objects: {} ({} tensors, {} resources)",
        objects.len(),
        tensors,
        resources
    );
    for (name, count) in &formats {
        println!("  format {name}: {count} tensors");
    }
    println!("resources:");
    for o in objects.iter().filter_map(|o| match o {
        Object::Resource(r) => Some(r),
        _ => None,
    }) {
        println!(
            "  {}  ({} bytes @ offset {})",
            o.name, o.bytes, o.offset
        );
    }

    if let Some(name) = &find {
        match reader.find(name) {
            Some(Object::Tensor(t)) => {
                let shape = t
                    .shape
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "tensor {name}: [{}] {} {} @ offset {} ({} bytes)",
                    shape,
                    t.format.name(),
                    t.layout.name(),
                    t.offset,
                    t.bytes
                );
            }
            Some(Object::Resource(r)) => {
                println!(
                    "resource {name}: {} @ offset {} ({} bytes)",
                    r.encoding.name(),
                    r.offset,
                    r.bytes
                );
            }
            None => {
                eprintln!("object not found: {name}");
                std::process::exit(1);
            }
        }
    }
}