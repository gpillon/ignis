//! GPU integration coverage for sequence state transfer against the real
//! model (P4-06, GitHub #124, ADR 0024).
//!
//! The leaf's own test (`kernel/tests/test_seq_snapshot.cpp`) proves that a
//! sequence's bytes round-trip: a dirtied sequence, released and
//! re-allocated as a zeroed one, restores to a byte-identical image. What it
//! cannot say is whether those bytes are *the state a sequence generates
//! from* — for that the sequence has to keep decoding, and the tokens have
//! to come out the same.
//!
//! So this test snapshots mid-generation, releases the sequence, restores it
//! into a fresh handle, and continues: the tail a restored sequence produces
//! must be the tail the unevicted one produced from the same point. That is
//! the claim the KV-RAM host tier is built on (GitHub #125) — restore
//! instead of re-prefill — and it is false unless every section of the
//! sequence actually crossed.
//!
//! It also drives the two refusals at this level, where "leaves the target
//! untouched" and "is refused mid-chunk" mean something about a running
//! request rather than about a struct: a refused restore is attempted on the
//! live sequence *before* it produces the control tail, so a restore that
//! had touched it would show up as a different tail below.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU or artifact is a **skip**; under the profile it is a **hard
//! failure**. Run via `scripts/gpu-profile.ps1` (stops the reference
//! `ninfer-serve` first — the RTX 5090 is exclusive).

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, Device, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::gqa_layer::run_gqa_layer;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{decode_program_batch, prefill_program};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 512;
const HIDDEN: usize = 5120;
/// The first GQA layer of the Qwen 3.8 topology — the one whose frontier a
/// single-layer call advances, leaving the sequence mid-chunk.
const FIRST_GQA_LAYER: u32 = 3;
/// Decoded before the snapshot, and again after it: enough tokens that a
/// sequence restored with a stale GDN slot or a truncated KV history would
/// diverge, short enough to stay inside `MAX_CONTEXT`.
const GENERATED: usize = 6;

fn decode_n(
    model: &ignis_core::model_load::Model,
    pool: &SeqPool,
    sequence: &mut ignis_core::seq::Seq<'_>,
    count: usize,
    label: &str,
) -> Vec<i32> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.extend(
            decode_program_batch(model, pool, &mut [&mut *sequence])
                .unwrap_or_else(|e| panic!("{label}: decode: {e}")),
        );
    }
    out
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_restored_sequence_continues_to_the_same_tokens() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let prompt: Vec<i32> = frontend
        .tokenizer()
        .encode("In one sentence, explain what a paged KV cache is.")
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    assert!(!prompt.is_empty());

    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b(
        &reader,
        &artifact,
        &handles,
        MAX_CONTEXT,
        MAX_CONTEXT,
        ignis_core::KvFormat::Bf16,
    )
    .unwrap_or_else(|e| panic!("load model: {e}"));
    // Two slots and room for two full-context sequences plus a short one:
    // the restore target is allocated after the source is released, but the
    // mid-chunk sequence below lives alongside it.
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: 24,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 2,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));

    // ---- the source sequence, mid-generation -------------------------------
    let mut source = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc source: {e}"));
    let fresh_bytes = source
        .snapshot_bytes()
        .unwrap_or_else(|e| panic!("fresh snapshot size: {e}"));

    prefill_program(&model, &pool, &mut source, &prompt, 0, None)
        .unwrap_or_else(|e| panic!("prefill: {e}"));
    let head = decode_n(&model, &pool, &mut source, GENERATED, "source head");
    assert_eq!(head.len(), GENERATED);

    let blob = source.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}"));
    assert!(
        blob.len() as u64 > fresh_bytes,
        "a snapshot grows with the history the sequence has written: \
         {fresh} bytes fresh, {warm} after a prompt",
        fresh = fresh_bytes,
        warm = blob.len()
    );

    // ---- a refused restore must not disturb a running sequence -------------
    //
    // Attempted here, before the control tail below is generated: if this
    // wrote anything into `source`, the control tail would not be the tail
    // the restored sequence produces, and the comparison at the end would
    // fail. The refusal is therefore checked by its return code *and* by the
    // tokens that follow it.
    let mut foreign = blob.clone();
    foreign[0] ^= 0xFF; // the blob's magic — not an ignis snapshot any more
    let refusal = source
        .restore(&foreign)
        .expect_err("a foreign blob must be refused");
    assert!(
        refusal.is_bad_snapshot(),
        "a foreign blob is a bad snapshot, not a bad call: {refusal}"
    );

    let mut truncated = blob.clone();
    truncated.pop();
    let refusal = source
        .restore(&truncated)
        .expect_err("a size that disagrees with the blob's own header must be refused");
    assert!(refusal.is_bad_snapshot(), "{refusal}");

    // ---- the control tail, from the sequence that was never evicted --------
    let control = decode_n(&model, &pool, &mut source, GENERATED, "control tail");
    drop(source);

    // ---- restore into a fresh handle and continue --------------------------
    let mut restored = pool
        .alloc(MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("alloc restore target: {e}"));
    assert_eq!(
        restored
            .snapshot_bytes()
            .unwrap_or_else(|e| panic!("target snapshot size: {e}")),
        fresh_bytes,
        "a fresh handle is back at the state floor"
    );
    restored
        .restore(&blob)
        .unwrap_or_else(|e| panic!("restore: {e}"));
    // Byte-identity at the real geometry, before a single token is decoded.
    // The token comparison below is the claim that matters, but it runs
    // greedy, so a section that only sampling reads — the penalty-count row
    // — could go missing without changing a token. This catches that, and
    // catches it at the model's real 248,320-entry vocabulary.
    assert_eq!(
        restored
            .snapshot()
            .unwrap_or_else(|e| panic!("re-snapshot: {e}")),
        blob,
        "a restored sequence snapshots to the same bytes as its source"
    );

    let tail = decode_n(&model, &pool, &mut restored, GENERATED, "restored tail");
    assert_eq!(
        tail, control,
        "a sequence restored from a snapshot continues to the same tokens it \
         would have produced unevicted"
    );
    assert!(tail.iter().all(|&id| id >= 0), "generated ids are valid token ids");
    drop(restored);

    // ---- mid-chunk is refused, not captured --------------------------------
    //
    // One GQA layer's frontier advanced and the program's not: exactly what
    // the per-layer entry point leaves between layers of a chunk, and the
    // state in which the sections of a sequence disagree with one another.
    let midchunk = pool
        .alloc(128)
        .unwrap_or_else(|e| panic!("alloc mid-chunk sequence: {e}"));
    let token_bytes = HIDDEN * std::mem::size_of::<u16>();
    let in_buf = device
        .allocate(token_bytes as u64)
        .unwrap_or_else(|e| panic!("allocate residual in: {e}"));
    let out_buf = device
        .allocate(token_bytes as u64)
        .unwrap_or_else(|e| panic!("allocate residual out: {e}"));
    device
        .copy_h2d(&in_buf, 0, &vec![0u8; token_bytes])
        .unwrap_or_else(|e| panic!("H2D residual: {e}"));
    device
        .synchronize()
        .unwrap_or_else(|e| panic!("sync after H2D: {e}"));

    assert!(
        midchunk.snapshot_bytes().is_ok(),
        "a fresh sequence is at a chunk boundary"
    );
    if let Err(e) = run_gqa_layer(
        &model,
        &pool,
        &midchunk,
        FIRST_GQA_LAYER,
        &in_buf,
        &out_buf,
        1,
    ) {
        if gpu_profile::skip_or_fail(&format!("ignis_gqa_layer_step: {e}")) {
            return;
        }
        unreachable!("skip_or_fail panics under the profile");
    }

    let refusal = midchunk
        .snapshot_bytes()
        .expect_err("a mid-chunk sequence has no coherent snapshot to price");
    assert!(
        refusal.is_not_at_boundary(),
        "mid-chunk is its own refusal, distinct from a bad argument: {refusal}"
    );
    let refusal = midchunk
        .snapshot_into(&mut vec![0u8; blob.len()])
        .expect_err("a mid-chunk sequence is not snapshotted");
    assert!(refusal.is_not_at_boundary(), "{refusal}");
}
