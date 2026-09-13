//! GPU test for the DFlash2 feature taps and the drafter's window (P5-03,
//! GitHub #152): after a prefill on a drafter-bearing load, the drafter's
//! window holds the projected context of the prompt's positions, checked
//! against a host recomputation.
//!
//! What the recomputation takes from the device, and what it redoes:
//!
//! - the target's hidden states at layers 5, 19, 33, 47 and 61 come from the
//!   device, by running the same prompt layer by layer (`run_gdn_layer` /
//!   `run_gqa_layer`) on a second sequence from a host-decoded embedding. The
//!   f64 layer oracle cannot run 62 layers inside a test, and the taps are the
//!   target's output, not the drafter's work;
//! - everything the drafter does with them is recomputed on the host in f64
//!   from the stored weights: the feature projection, the context norm, each
//!   drafter layer's fused query/key/value parent, the key norm and the
//!   drafter's own base-1e7 RoPE at absolute positions.
//!
//! The window is read out of a snapshot blob of the prefilled sequence, so
//! the test also reads what snapshot carries. That blob layout is leaf
//! internal (ADR 0024) and is spelled out below for version 2 only: a format
//! change fails the version assertion rather than misreading bytes.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{
    bind_model_scope_27b, f64_reference, materialize, CudaDevice, Device, DraftModule, FrontendSet,
    Reader,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gdn_layer::run_gdn_layer;
use ignis_core::gpu_profile;
use ignis_core::gqa_layer::run_gqa_layer;
use ignis_core::model_load::load_qwen38_27b_with_speculation;
use ignis_core::seq::{snapshot_format_version, SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program;
use ignis_core::{KvFormat, Speculation, SpeculativeBackend};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 512;
/// The load's prefill chunk: the prompt below spans several, so the taps of
/// one chunk and the next both land in the window.
const PREFILL_CHUNK: u32 = 128;
/// The prompt is its sentence repeated until it is at least this long.
const PROMPT_TOKENS: usize = 200;
const HIDDEN: usize = 5120;
/// The target layers the drafter taps, in feature order.
const TAP_LAYERS: [u32; 5] = [5, 19, 33, 47, 61];
const DRAFTER_LAYERS: usize = 5;
const WINDOW: usize = 2048;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const QUERY_SIZE: usize = 4096;
const ROPE_THETA: f64 = 1.0e7;

/// The version-2 blob layout (`kernel/include/ignis_seq_sections.h`).
const HEADER_BYTES: usize = 128;
const RECORD_BYTES: usize = 24;
const SECTION_PROGRESS: i32 = 4;
const SECTION_DFLASH_WINDOW: i32 = 5;
const SECTION_DFLASH_CHECKPOINT: i32 = 6;

/// Relative L2 between the window's BF16 rows and the f64 recomputation,
/// per role over every drafter layer, KV head and prompt position. Bounds the
/// drafter's A16 projections, its norms, its RoPE and every BF16 rounding in
/// between. Measured 2026-09-14: 0.002623 (keys) / 0.002705 (values) on a
/// single 13-token chunk, 0.002800 / 0.002809 on the 210-token, two-chunk
/// prompt below (deterministic); this is that maximum with ~25% margin. A wrong tap layer, row offset, position or missing RoPE lands
/// orders of magnitude above it.
const WINDOW_TOLERANCE: f64 = 0.0035;

fn bf16_to_f64(bits: u16) -> f64 {
    f32::from_bits(u32::from(bits) << 16) as f64
}

/// f32 -> BF16, round-to-nearest-even on the low 16 bits.
fn f32_to_bf16(x: f32) -> u16 {
    let bits = u64::from(x.to_bits());
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// `(offset, bytes)` of the blob's section of `kind`.
fn section(blob: &[u8], kind: i32) -> (usize, usize) {
    assert_eq!(read_u32(blob, 12) as usize, HEADER_BYTES, "snapshot header bytes");
    assert_eq!(read_u32(blob, 16) as usize, RECORD_BYTES, "snapshot record bytes");
    let count = read_u32(blob, 20) as usize;
    for i in 0..count {
        let at = HEADER_BYTES + i * RECORD_BYTES;
        if read_u32(blob, at) as i32 == kind {
            return (read_u64(blob, at + 8) as usize, read_u64(blob, at + 16) as usize);
        }
    }
    panic!("the snapshot lists no section of kind {kind}");
}

/// Split-half NeoX RoPE over the whole 128-wide head, key side (no factor).
fn rope_key(x: &[f64], position: usize) -> Vec<f64> {
    let half = HEAD_DIM / 2;
    let mut out = x.to_vec();
    for i in 0..half {
        let angle = position as f64 * ROPE_THETA.powf(-2.0 * i as f64 / HEAD_DIM as f64);
        let (sin, cos) = angle.sin_cos();
        out[i] = x[i] * cos - x[i + half] * sin;
        out[i + half] = x[i + half] * cos + x[i] * sin;
    }
    out
}

#[derive(Default)]
struct Drift {
    squared_error: f64,
    squared_reference: f64,
    max_abs_error: f64,
}

impl Drift {
    fn add(&mut self, device: f64, host: f64) {
        let error = device - host;
        self.squared_error += error * error;
        self.squared_reference += host * host;
        self.max_abs_error = self.max_abs_error.max(error.abs());
    }

    fn relative_l2(&self) -> f64 {
        self.squared_error.sqrt() / self.squared_reference.sqrt().max(1e-30)
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_drafter_window_holds_the_projected_context_of_the_prompt() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let sentence: Vec<i32> = frontend
        .tokenizer()
        .encode("In one sentence, explain what a paged KV cache is. ")
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    assert!(!sentence.is_empty());
    let mut prompt = Vec::new();
    while prompt.len() < PROMPT_TOKENS {
        prompt.extend_from_slice(&sentence);
    }
    let n = prompt.len();
    assert!(n > PREFILL_CHUNK as usize && n < WINDOW, "a multi-chunk prompt under the window: {n}");
    // The host recomputation projects 25,600 columns per position, so it
    // checks a sample: every 16th position, both sides of every chunk
    // boundary, and the last.
    let checked: Vec<usize> = (0..n)
        .filter(|&p| {
            let chunk = PREFILL_CHUNK as usize;
            p % 16 == 0 || p + 1 == n || p % chunk == 0 || (p + 1) % chunk == 0
        })
        .collect();

    let (plan, handles) = bind_model_scope_27b(&reader, Some(DraftModule::Dflash2))
        .unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + dflash2: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
    let model = load_qwen38_27b_with_speculation(
        &reader,
        &artifact,
        &handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::Bf16,
        Some(speculation),
    )
    .unwrap_or_else(|e| panic!("load with dflash2: {e}"));
    let cfg = ModelConfig::qwen38_27b();
    let budget = SeqPoolBudget {
        kv_format: KvFormat::Bf16,
        kv_page_group_count: 16,
        max_context_tokens: MAX_CONTEXT,
        slot_count: 2,
    };

    // The drafter's weights and its per-sequence window come as a pair: a
    // pool without the window is refused before any device work.
    {
        let plain = SeqPool::create(&cfg, &budget).unwrap_or_else(|e| panic!("plain pool: {e}"));
        let mut sequence = plain.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("plain alloc: {e}"));
        let err = prefill_program(&model, &plain, &mut sequence, &prompt, 0, None)
            .expect_err("a drafter load refuses a pool without the drafter");
        assert!(err.contains("speculative backend"), "{err}");
    }

    let pool = SeqPool::create_with_speculation(&cfg, &budget, Some(SpeculativeBackend::Dflash2))
        .unwrap_or_else(|e| panic!("drafter pool: {e}"));
    let mut prefilled = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    if let Err(e) = prefill_program(&model, &pool, &mut prefilled, &prompt, 0, None) {
        if gpu_profile::skip_or_fail(&format!("prefill with the drafter: {e}")) {
            return;
        }
        unreachable!("skip_or_fail panics under the profile");
    }

    assert_eq!(snapshot_format_version(), 2, "this test reads the version-2 blob layout");
    let blob = prefilled.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}"));
    let plane = HEAD_DIM * WINDOW * KV_HEADS * 2;
    let (window_at, window_bytes) = section(&blob, SECTION_DFLASH_WINDOW);
    assert_eq!(window_bytes, DRAFTER_LAYERS * 2 * plane, "the window section is one lane");
    let (checkpoint_at, checkpoint_bytes) = section(&blob, SECTION_DFLASH_CHECKPOINT);
    assert!(
        blob[checkpoint_at..checkpoint_at + checkpoint_bytes].iter().all(|&b| b == 0),
        "a prefill does not write the rewrite checkpoint"
    );
    let (progress_at, _) = section(&blob, SECTION_PROGRESS);
    assert_eq!(read_u64(&blob, progress_at) as usize, n, "the program frontier");
    assert_eq!(
        read_u64(&blob, progress_at + 16) as usize,
        n,
        "the drafter frontier stands at the prompt's end"
    );

    let window = &blob[window_at..window_at + window_bytes];
    // Lane layout: every layer's K then V, each `[head_dim, 2048, kv_heads]`
    // with position p in ring slot p mod 2048.
    let word = |layer: usize, role: usize, position: usize, head: usize, dim: usize| -> u16 {
        let at = (layer * 2 + role) * plane + (dim + HEAD_DIM * (position % WINDOW + WINDOW * head)) * 2;
        u16::from_le_bytes([window[at], window[at + 1]])
    };
    for layer in 0..DRAFTER_LAYERS {
        for role in 0..2 {
            for head in 0..KV_HEADS {
                for position in n..WINDOW {
                    assert!(
                        (0..HEAD_DIM).all(|dim| word(layer, role, position, head, dim) == 0),
                        "ring slot {position} past the prompt was written (layer {layer}, role {role})"
                    );
                }
            }
        }
    }

    // The target's features, layer by layer on a second sequence.
    let layered = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc layered: {e}"));
    let mut residual = Vec::with_capacity(n * HIDDEN * 2);
    for &id in &prompt {
        let row = f64_reference::matrix_row(&reader, "text/token_embedding", cfg.vocab as usize, HIDDEN, id as usize)
            .unwrap_or_else(|e| panic!("embedding row {id}: {e}"));
        for value in row {
            residual.extend_from_slice(&f32_to_bf16(value as f32).to_le_bytes());
        }
    }
    // In the load's own chunks: the layer entry points share the prefill
    // scratch, which is sized for one chunk, and the same cut keeps these
    // taps on the kernels the chunked prefill ran.
    let chunk_bytes = PREFILL_CHUNK as usize * HIDDEN * 2;
    let mut buffers = [
        device.allocate(chunk_bytes as u64).unwrap_or_else(|e| panic!("allocate residual: {e}")),
        device.allocate(chunk_bytes as u64).unwrap_or_else(|e| panic!("allocate residual: {e}")),
    ];
    let mut taps: Vec<Vec<u16>> = vec![Vec::with_capacity(n * HIDDEN); TAP_LAYERS.len()];
    for chunk in residual.chunks(chunk_bytes) {
        let tokens = (chunk.len() / (HIDDEN * 2)) as u64;
        device.copy_h2d(&buffers[0], 0, chunk).unwrap_or_else(|e| panic!("H2D embedding: {e}"));
        device.synchronize().unwrap_or_else(|e| panic!("synchronize: {e}"));
        for layer in 0..=TAP_LAYERS[TAP_LAYERS.len() - 1] {
            let stepped = if layer % 4 == 3 {
                run_gqa_layer(&model, &pool, &layered, layer, &buffers[0], &buffers[1], tokens)
            } else {
                run_gdn_layer(&model, &pool, &layered, layer, &buffers[0], &buffers[1], tokens)
            };
            if let Err(e) = stepped {
                if gpu_profile::skip_or_fail(&format!("layer {layer}: {e}")) {
                    return;
                }
                unreachable!("skip_or_fail panics under the profile");
            }
            buffers.swap(0, 1);
            if let Some(j) = TAP_LAYERS.iter().position(|&tap| tap == layer) {
                let mut out = vec![0u8; chunk.len()];
                device.copy_d2h(&buffers[0], 0, &mut out).unwrap_or_else(|e| panic!("D2H tap: {e}"));
                device.synchronize().unwrap_or_else(|e| panic!("synchronize after D2H: {e}"));
                taps[j].extend(out.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
            }
        }
    }

    // The drafter's context append, in f64.
    let context_norm = f64_reference::vector_weight(&reader, "dflash2/context_norm", HIDDEN)
        .unwrap_or_else(|e| panic!("context norm: {e}"));
    let key_norms: Vec<Vec<f64>> = (0..DRAFTER_LAYERS)
        .map(|l| {
            f64_reference::vector_weight(&reader, &format!("dflash2/layers/{l}/attention/key_norm"), HEAD_DIM)
                .unwrap_or_else(|e| panic!("key norm {l}: {e}"))
        })
        .collect();
    let mut keys = Drift::default();
    let mut values = Drift::default();
    for &position in &checked {
        let features: Vec<f64> = taps
            .iter()
            .flat_map(|tap| tap[position * HIDDEN..(position + 1) * HIDDEN].iter().map(|&b| bf16_to_f64(b)))
            .collect();
        let projected = f64_reference::matrix_product(
            &reader,
            "dflash2/feature_projection",
            HIDDEN,
            TAP_LAYERS.len() * HIDDEN,
            &features,
        )
        .unwrap_or_else(|e| panic!("feature projection: {e}"));
        let context = f64_reference::rms_norm(&projected, &context_norm, false)
            .unwrap_or_else(|e| panic!("context norm: {e}"));
        for layer in 0..DRAFTER_LAYERS {
            let qkv = f64_reference::matrix_product(
                &reader,
                &format!("dflash2/layers/{layer}/attention/query_key_value"),
                QUERY_SIZE + 2 * KV_HEADS * HEAD_DIM,
                HIDDEN,
                &context,
            )
            .unwrap_or_else(|e| panic!("query_key_value {layer}: {e}"));
            for head in 0..KV_HEADS {
                let key_at = QUERY_SIZE + head * HEAD_DIM;
                let value_at = QUERY_SIZE + KV_HEADS * HEAD_DIM + head * HEAD_DIM;
                let normed = f64_reference::rms_norm(&qkv[key_at..key_at + HEAD_DIM], &key_norms[layer], false)
                    .unwrap_or_else(|e| panic!("key norm: {e}"));
                let key = rope_key(&normed, position);
                for dim in 0..HEAD_DIM {
                    keys.add(bf16_to_f64(word(layer, 0, position, head, dim)), key[dim]);
                    values.add(bf16_to_f64(word(layer, 1, position, head, dim)), qkv[value_at + dim]);
                }
            }
        }
    }

    for (label, drift) in [("key", &keys), ("value", &values)] {
        let relative_l2 = drift.relative_l2();
        println!(
            "drafter window {label} at {} of {n} positions: relative L2 {relative_l2:.6} of {WINDOW_TOLERANCE} \
             ({:.1}% of budget), max abs error {:.6}",
            checked.len(),
            100.0 * relative_l2 / WINDOW_TOLERANCE,
            drift.max_abs_error
        );
        assert!(
            relative_l2 <= WINDOW_TOLERANCE,
            "drafter window {label}: relative L2 {relative_l2} exceeds {WINDOW_TOLERANCE}"
        );
    }

    drop(layered);
    drop(prefilled);
    drop(pool);
    drop(model);
    for buffer in buffers {
        let _ = device.deallocate(buffer);
    }
    let _ = artifact.release_arena(&mut device);
}
