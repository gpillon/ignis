//! Flash-Next's forward takes an image (GitHub #557, spec #555, ADR 0087), on
//! the real artifact, with columns the test places in the load's embedding
//! pool where the vision companion's encode will write the tower's
//! (`FlashNextEngine::place_media`): the forward's half of the image path,
//! checked before any tower is bound.
//!
//! - **Three equal axes are the text path, with vision off and on.** A
//!   300-token span of real text, in 128-token chunks: its logits on a vision
//!   load, as a text span and as a span of a multimodal prompt whose three
//!   axes are its positions, equal the text load's bit for bit, every row.
//! - **Media columns replace their placeholders in every stream.** 150
//!   placeholders at positions 100..250 (two chunks, two pool pages) take
//!   columns that are the embedding rows of the text they stand for (read
//!   back from the pool as placed): every row equals, bit for bit, the text
//!   span whose embedding is that text and
//!   whose n-gram rows hash the placeholders, as the multimodal one's do --
//!   under the host residency and the device one alike, whose rows are the
//!   same. Without the columns, the placeholders' own embeddings, every row
//!   from the first placeholder on is another; every row before it the same.
//!   The pool is the plan's: the leaf's line equals `Vision::pool_bytes` at
//!   Flash-Next's 2,560-wide merger in VRAM, none under the host residency,
//!   and what the load holds is what it planned.
//! - **The axes and the rope delta reach the forward.** An 8x8 image at
//!   positions 40..104 of a 200-token span, at its three axes (the image's
//!   start plus its row and column; the text after it resumes past the larger
//!   of the two, rope delta -56): rows up to the image's first token equal
//!   the same span at equal axes, bit for bit, and every later row is
//!   another. The first decode round after it (its graph) rotates at its
//!   position plus the delta: its logits pick what the prefill of the same
//!   token there picks, but for a near-tie (one BF16 ulp, the routes' sums
//!   differing in order) -- the prefill rotates a text span on a sequence with
//!   a rope delta at its position plus it, the 27B's rule -- and sit nearer
//!   them than the prefill at the position alone (the same span with a delta
//!   of 0).
//!
//! The rows of a 2,560-wide pool and of a placeholder's embedding are what
//! the companion's encode and the server's prompt will feed this forward;
//! neither is here. The three-axis rotation's arithmetic, pair by pair, is
//! `kernel/tests/test_flash_next_rope.cu`'s.
//!
//! ```text
//! cargo test -p ignis-core --features cuda --test flash_next_vision_forward_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`), its `references/long8192/tokens.u32` the prompt.
//! Explicit GPU profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing
//! artifact or GPU is a skip, under it a failure. Needs ~38 GB of free RAM
//! for the pinned expert pool; each test loads the model once or twice, one
//! load at a time.

#![cfg(feature = "cuda")]

use std::fs;
use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::vision::IMAGE_PAD_ID;
use ignis_artifact::Reader;
use ignis_core::compute::ModelConfig;
use ignis_core::flash_next::{EngineOptions, FlashNextEngine};
use ignis_core::step::{MultimodalPrefill, SamplingParams, SpanMediaColumns};
use ignis_core::{gpu_profile, EmbeddingWidth, Vision, VisionResidency};

#[path = "support/embedding_rows.rs"]
mod embedding_rows;

use embedding_rows::embedding_rows;

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MAX_CONTEXT: u32 = 8192;
/// Small chunks, so a span of a few hundred tokens crosses chunk boundaries.
const CHUNK: u32 = 128;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

/// The vision every vision load here takes: the context's envelope, the
/// pool at its floor, the tower where `residency` puts it.
fn vision(residency: VisionResidency) -> Vision {
    Vision::new(MAX_CONTEXT).expect("an envelope").with_residency(residency)
}

/// A load with `vision` (`None`: a text load), or `None` through
/// `skip_or_fail` when the artifact or the GPU is not there.
fn engine(vision: Option<Vision>) -> Option<FlashNextEngine> {
    let dir = model_dir();
    if !dir.join(ARTIFACT_FILE_NAME).exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact in {}", dir.display()));
        return None;
    }
    let options =
        EngineOptions { max_context_tokens: MAX_CONTEXT, prefill_chunk_tokens: CHUNK, vision, ..EngineOptions::default() };
    match FlashNextEngine::load(&dir, options) {
        Ok(engine) => Some(engine),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next engine: {e}"));
            None
        }
    }
}

/// The first `count` tokens of the converter's `long8192` window 0: real text.
fn prompt(count: usize) -> Option<Vec<u32>> {
    let path = model_dir().join("references").join("long8192").join("tokens.u32");
    let Ok(bytes) = fs::read(&path) else {
        gpu_profile::skip_or_fail(&format!("{} does not exist", path.display()));
        return None;
    };
    let tokens: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().expect("four bytes"))).collect();
    Some(tokens[..count].to_vec())
}

/// The artifact, for its embedding table.
fn artifact() -> Reader {
    let path = model_dir().join(ARTIFACT_FILE_NAME);
    Reader::open(&path).unwrap_or_else(|e| panic!("open {}: {e:?}", path.display()))
}

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// One BF16 ulp at `x`'s magnitude: 2^(exponent - 7).
fn bf16_ulp(x: f32) -> f32 {
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent - 7)
}

/// Three axes equal to positions 0.., axis-major `[3][count]`: a text span
/// as a span of a multimodal prompt.
fn equal_axes(count: usize) -> Vec<i32> {
    (0..3).flat_map(|_| 0..count as i32).collect()
}

/// `tokens` with `count` placeholders from `first` on.
fn with_placeholders(tokens: &[u32], first: usize, count: usize) -> Vec<u32> {
    let mut out = tokens.to_vec();
    out[first..first + count].fill(IMAGE_PAD_ID);
    out
}

/// Which rows of two `[rows][vocab]` logit blocks differ.
fn differing_rows(a: &[u16], b: &[u16], vocab: usize) -> Vec<usize> {
    assert_eq!(a.len(), b.len());
    a.chunks_exact(vocab).zip(b.chunks_exact(vocab)).enumerate().filter(|(_, (x, y))| x != y).map(|(i, _)| i).collect()
}

fn assert_bit_identical(a: &[u16], b: &[u16], vocab: usize, what: &str) {
    let rows = differing_rows(a, b, vocab);
    assert!(rows.is_empty(), "{what}: {} rows differ, the first {:?}", rows.len(), &rows[..rows.len().min(8)]);
}

#[test]
#[ignore = "GPU profile only: Flash-Next on the RTX 5090 (scripts/gpu-profile.ps1)"]
fn three_equal_axes_rotate_as_the_text_path_with_vision_off_and_on() {
    let Some(tokens) = prompt(300) else { return };
    let text = {
        let Some(engine) = engine(None) else { return };
        let (_, _, logits) = engine.span(&tokens, None, None).expect("the text load's span");
        logits
    };
    let Some(engine) = engine(Some(vision(VisionResidency::Host))) else { return };
    let vocab = engine.vocab();
    let (_, _, on) = engine.span(&tokens, None, None).expect("the vision load's text span");
    assert_bit_identical(&text, &on, vocab, "a text span with vision on");
    let axes = equal_axes(tokens.len());
    let span = MultimodalPrefill { positions: &axes, rope_delta: 0, media: None };
    let (_, _, multimodal) = engine.span(&tokens, None, Some(span)).expect("the span at three equal axes");
    assert_bit_identical(&text, &multimodal, vocab, "the span at three equal axes");
    println!("{} rows of 300 bit for bit: vision off, on, and at three equal axes", text.len() / vocab);
}

#[test]
#[ignore = "GPU profile only: Flash-Next on the RTX 5090 (scripts/gpu-profile.ps1)"]
fn media_columns_replace_their_placeholders_in_every_stream() {
    let Some(tokens) = prompt(300) else { return };
    let (first, count) = (100usize, 150usize);
    let placeholders = with_placeholders(&tokens, first, count);
    let columns = embedding_rows(&artifact(), &tokens[first..first + count]);
    let scatter: Vec<i32> = (first as i32..(first + count) as i32).collect();
    let axes = equal_axes(tokens.len());
    let config = ModelConfig::qwen38_flash_next();
    let width = EmbeddingWidth::from_inventory(&ignis_artifact::vision_scope(config.hidden), config.hidden)
        .expect("Flash-Next's merger writes its hidden width");
    let mut placed_by_residency = Vec::new();
    for residency in [VisionResidency::Host, VisionResidency::Device] {
        let asked = vision(residency);
        let Some(engine) = engine(Some(asked)) else { return };
        let vocab = engine.vocab();
        let pool = match residency {
            VisionResidency::Host => 0,
            VisionResidency::Device => asked.pool_bytes(MAX_CONTEXT, width),
        };
        assert_eq!(engine.planned().media_embedding_bytes, pool, "{residency:?}: the plan's pool line");
        assert_eq!(engine.reserved().media_embedding_bytes, pool, "{residency:?}: the pool the load holds");
        assert_eq!(engine.reserved().activation_bytes, engine.planned().activation_bytes, "{residency:?}: activations");
        // The reference embeds the text and hashes the placeholders.
        let (_, _, reference) = engine.span(&tokens, Some(&placeholders), None).expect("the reference span");
        let media = engine.place_media(&columns).expect("place the columns");
        assert_eq!(media.read_bits(config.hidden).expect("read them back"), columns, "{residency:?}: the placed columns");
        let span = MultimodalPrefill {
            positions: &axes,
            rope_delta: 0,
            media: Some(SpanMediaColumns { embedding: &media, first_column: 0, scatter_indices: &scatter }),
        };
        let (_, _, placed) = engine.span(&placeholders, None, Some(span)).expect("the span with the columns");
        assert_bit_identical(&reference, &placed, vocab, &format!("{residency:?}: the placed columns"));
        let bare = MultimodalPrefill { positions: &axes, rope_delta: 0, media: None };
        let (_, _, unplaced) = engine.span(&placeholders, None, Some(bare)).expect("the span without the columns");
        let moved = differing_rows(&placed, &unplaced, vocab);
        assert_eq!(moved, (first..tokens.len()).collect::<Vec<_>>(), "{residency:?}: the rows the columns move");
        println!("{residency:?}: {} rows bit for bit with the placed columns; {} rows moved without them", tokens.len(), moved.len());
        drop(media);
        placed_by_residency.push(placed);
    }
    let vocab = ModelConfig::qwen38_flash_next().vocab as usize;
    assert_bit_identical(&placed_by_residency[0], &placed_by_residency[1], vocab, "the host and the device pool");
}

#[test]
#[ignore = "GPU profile only: Flash-Next on the RTX 5090 (scripts/gpu-profile.ps1)"]
fn the_axes_and_the_rope_delta_reach_the_forward() {
    const N: usize = 200;
    const FIRST: usize = 40;
    const SIDE: usize = 8;
    let Some(tokens) = prompt(N) else { return };
    let count = SIDE * SIDE;
    let placeholders = with_placeholders(&tokens, FIRST, count);
    let columns = embedding_rows(&artifact(), &tokens[FIRST..FIRST + count]);
    let scatter: Vec<i32> = (FIRST as i32..(FIRST + count) as i32).collect();
    // The reference's rule (Qwen3.5's get_rope_index): text on all three
    // axes; the image's tokens at its start plus their row and column; the
    // text after it from the start plus the larger side. Axis-major [3][N].
    let mut grid = vec![[0i32; 3]; N];
    for (t, at) in grid.iter_mut().enumerate() {
        let start = FIRST as i32;
        *at = if t < FIRST {
            [t as i32; 3]
        } else if t < FIRST + count {
            let (y, x) = (((t - FIRST) / SIDE) as i32, ((t - FIRST) % SIDE) as i32);
            [start, start + y, start + x]
        } else {
            [start + SIDE as i32 + (t - FIRST - count) as i32; 3]
        };
    }
    let grid: Vec<i32> = (0..3).flat_map(|axis| grid.iter().map(move |at| at[axis])).collect();
    let delta = grid.iter().max().copied().expect("positions") + 1 - N as i32;
    assert_eq!(delta, SIDE as i32 - count as i32, "the rope delta of one 8x8 image");
    let flat = equal_axes(N);

    let Some(engine) = engine(Some(vision(VisionResidency::Host))) else { return };
    let vocab = engine.vocab();
    let media = engine.place_media(&columns).expect("place the columns");
    let columns_of = || Some(SpanMediaColumns { embedding: &media, first_column: 0, scatter_indices: &scatter });
    let at_grid = |rope_delta| MultimodalPrefill { positions: &grid, rope_delta, media: columns_of() };

    let (_, _, flat_rows) = engine
        .span(&placeholders, None, Some(MultimodalPrefill { positions: &flat, rope_delta: 0, media: columns_of() }))
        .expect("the image at equal axes");
    let (seq, context, grid_rows) = engine.span(&placeholders, None, Some(at_grid(delta))).expect("the image at its axes");
    let moved = differing_rows(&flat_rows, &grid_rows, vocab);
    assert_eq!(moved, (FIRST + 1..N).collect::<Vec<_>>(), "the rows the image's axes move: every one past its first token");
    assert!(grid_rows.iter().all(|&v| bf16(v).is_finite()), "every logit finite");

    // The first round after the prompt, from its graph, against the prefill
    // of its token at the same frontier with the delta and without it.
    let mut sequences = vec![seq];
    let mut contexts = vec![context];
    let mut decoded = vec![0u16; vocab];
    engine
        .decode_standing(&mut sequences, &mut contexts, &[(SamplingParams::greedy(), &[][..])], Some(&mut decoded))
        .expect("the round");
    let decoded: Vec<f32> = decoded.iter().map(|&v| bf16(v)).collect();
    let probe = |rope_delta| {
        let (mut seq, mut context, _) = engine.span(&placeholders, None, Some(at_grid(rope_delta))).expect("the span");
        engine.probe_logits(&mut seq, &mut context).expect("the prefill of the round's token")
    };
    let shifted = probe(delta);
    let plain = probe(0);
    let argmax = |row: &[f32]| (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a))).expect("a row");
    let mean_gap = |row: &[f32]| row.iter().zip(&decoded).map(|(a, b)| f64::from((a - b).abs())).sum::<f64>() / row.len() as f64;
    let (to_shifted, to_plain) = (mean_gap(&shifted), mean_gap(&plain));
    println!(
        "the round after an 8x8 image (rope delta {delta}): mean |logit gap| {to_shifted:.4} to the prefill at position + delta, \
         {to_plain:.4} to the prefill at the position; argmax {} / {} / {}",
        argmax(&decoded),
        argmax(&shifted),
        argmax(&plain)
    );
    // The two routes' sums differ in order (flash_next_forward_gpu.rs): the
    // round's pick is the prefill's, or within one BF16 ulp of its top logit.
    let (pick, top) = (argmax(&decoded), argmax(&shifted));
    assert!(
        pick == top || shifted[top] - shifted[pick] <= bf16_ulp(shifted[top]),
        "the round picks {pick}, the prefill at position + delta {top} ({} against {})",
        shifted[pick],
        shifted[top]
    );
    assert!(to_shifted < to_plain, "the round sits nearer the prefill at position + delta than at the position");
}
