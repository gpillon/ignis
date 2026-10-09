//! Write one Flash-Next layer's expert records into a pool file, so the MoE
//! microbenchmark times its decode routes on real weights at that layer's K
//! mix (`kernel/tests/bench_moe.cu --pool`, GitHub #306) without a model load.
//!
//! Usage: `cargo run -p ignis-artifact --example dump_expert_pool -- <artifact.ninfer> <layer> <out.pool>`
//!
//! Format, little-endian: the 8 bytes `IGNMOEP1`, u32 layer, u32 expert
//! count; then per expert in id order: u32 id, and for gate/up then down a
//! u32 k2 (2 K), a u64 byte count and the record's bytes as stored.

use std::io::Write;
use std::path::Path;

use ignis_artifact::flash_next::{self, FlashNextGeometry, Projection};
use ignis_artifact::Reader;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [artifact, layer, out] = args.as_slice() else {
        eprintln!("usage: dump_expert_pool <artifact.ninfer> <layer> <out.pool>");
        std::process::exit(2);
    };
    let layer: usize = layer.parse().expect("layer is a number");
    let reader = Reader::open(Path::new(artifact)).unwrap_or_else(|e| panic!("open {artifact}: {e}"));
    let plan = flash_next::bind(&reader, &FlashNextGeometry::qwen38_flash_next())
        .unwrap_or_else(|e| panic!("bind the Flash-Next artifact: {e:?}"));
    let experts = plan.experts.experts_per_layer();
    let mut file = std::io::BufWriter::new(std::fs::File::create(out).unwrap_or_else(|e| panic!("create {out}: {e}")));
    file.write_all(b"IGNMOEP1").unwrap();
    file.write_all(&(layer as u32).to_le_bytes()).unwrap();
    file.write_all(&(experts as u32).to_le_bytes()).unwrap();
    let mut counts = [[0usize; 4]; 2];
    for expert in 0..experts {
        file.write_all(&(expert as u32).to_le_bytes()).unwrap();
        for (p, projection) in Projection::ALL.into_iter().enumerate() {
            let k2 = plan.experts.get(layer, expert, projection).expect("expert index entry").k.k2();
            let span = reader
                .payload(&flash_next::expert_name(layer, expert as u64, projection))
                .unwrap_or_else(|e| panic!("L{layer} expert {expert} {projection:?}: {e:?}"));
            file.write_all(&u32::from(k2).to_le_bytes()).unwrap();
            file.write_all(&(span.data.len() as u64).to_le_bytes()).unwrap();
            file.write_all(span.data).unwrap();
            counts[p][match k2 {
                4 => 0,
                5 => 1,
                6 => 2,
                _ => 3,
            }] += 1;
        }
    }
    file.flush().unwrap();
    println!(
        "layer {layer}: {experts} experts; K = 2/2.5/3/4 gate/up {:?}, down {:?}",
        counts[0], counts[1]
    );
}
