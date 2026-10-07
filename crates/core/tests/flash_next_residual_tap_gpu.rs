//! The Flash-Next residual-stack tap on the real artifact (spec
//! flash-next/07 phase A; `kernel/include/ignis_fn_residual_tap.h`).
//!
//! - Every position gets its row, chunk after chunk: a span over three
//!   prefill chunks fills all of them, and the rows of its first chunk equal,
//!   bit for bit, the rows of a span that is only that chunk (the same
//!   computation, so a row landing at the wrong offset shows).
//! - The tap only reads: the picks of a tapped span equal the picks of the
//!   same span untapped, and a second tapped span repeats the first's rows.
//!
//! That a row is the stack the head reads -- the trunk's mixer and lm_head
//! over it give the engine's pick -- is checked where the head's weights are,
//! by the phase A prototype (`tools/flash-next-mtp/phase_a.py check`).
//!
//! Machine-local, explicit GPU profile (ADR 0006) and the non-default
//! `residual-tap` feature:
//! `cargo test -p ignis-core --features cuda,residual-tap --test flash_next_residual_tap_gpu -- --ignored`.
//! Needs ~38 GB of free RAM for the pinned expert pool.

#![cfg(all(feature = "cuda", feature = "residual-tap"))]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::flash_next::{EngineOptions, FlashNextEngine};
use ignis_core::gpu_profile;

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const CHUNK: usize = 128;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

/// A deterministic prompt of ordinary token ids (the low vocab is text).
fn prompt(len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| 1000 + (i * 7919) % 60_000).collect()
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn the_tap_fills_every_position_and_only_reads() {
    let dir = model_dir();
    if !dir.join(ARTIFACT_FILE_NAME).exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact in {}", dir.display()));
        return;
    }
    let options = EngineOptions {
        prefill_chunk_tokens: CHUNK as u32,
        max_context_tokens: 1024,
        decode_lanes: 1,
        capture_graphs: false,
        ..EngineOptions::default()
    };
    let mut engine = match FlashNextEngine::load(&dir, options) {
        Ok(engine) => engine,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine: {e}"));
            return;
        }
    };
    let tokens = prompt(3 * CHUNK - 40);
    let span = engine.tapped_span(&tokens).expect("tapped span");
    let width = span.width;
    assert_eq!(width, 4 * 2560);
    assert_eq!(span.stacks.len(), tokens.len() * width);
    for (p, row) in span.stacks.chunks_exact(width).enumerate() {
        let finite = row.iter().all(|&b| f32::from_bits(u32::from(b) << 16).is_finite());
        assert!(finite && row.iter().any(|&b| b & 0x7fff != 0), "row {p} is not a written stack");
    }

    let first = engine.tapped_span(&tokens[..CHUNK]).expect("one-chunk span");
    assert!(first.stacks == span.stacks[..CHUNK * width], "the first chunk's rows moved");

    let again = engine.tapped_span(&tokens).expect("second tapped span");
    assert!(again.stacks == span.stacks, "a second tapped span differs");
    let mut untapped = Vec::new();
    let vocab = engine.vocab();
    engine
        .span_logits(&tokens, &mut |_, rows| {
            for row in rows.chunks_exact(vocab) {
                let best = (0..vocab).max_by(|&a, &b| {
                    let (x, y) = (f32::from_bits(u32::from(row[a]) << 16), f32::from_bits(u32::from(row[b]) << 16));
                    x.total_cmp(&y).then(b.cmp(&a))
                });
                untapped.push(best.unwrap() as u32);
            }
            Ok(())
        })
        .expect("untapped span");
    assert_eq!(untapped, span.argmax, "the tap changed a pick");
}
