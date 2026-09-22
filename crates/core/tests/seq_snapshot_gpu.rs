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
//! GitHub #194 adds a multimodal sequence to the same claim: one whose rope
//! delta is not 0 restores to the tokens it would have produced unevicted,
//! which it cannot unless the blob carried the delta. The model is loaded
//! with vision for it — a multimodal span is refused on a load without.
//!
//! GitHub #257 adds hq-e8-2b's residual window: indexed by slot, not by page,
//! so a sequence restored into a slot another sequence used decodes its own
//! history only if the blob carried the window. The same claim, under hq, into
//! the other slot — plus the ring words a prefill and a run of decode rounds
//! leave, read out of the blob, against the host rule (`ignis_core::hq_ring`).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing GPU or artifact is a **skip**; under the profile it is a **hard
//! failure**. Run via `scripts/gpu-profile.ps1` (stops the reference
//! `ninfer-serve` first — the RTX 5090 is exclusive).

#![cfg(feature = "cuda")]

#[path = "support/snapshot_blob.rs"]
mod snapshot_blob;

use std::path::Path;

use ignis_artifact::{
    CudaDevice, Device, FrontendSet, ModelScope, Reader, bind_model_scope_27b_with, bind_text_scope_27b,
    materialize,
};
use ignis_core::Vision;
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::gqa_layer::run_gqa_layer;
use ignis_core::hq_ring::HqRing;
use ignis_core::model_load::{Model, load_qwen38_27b, load_qwen38_27b_with_options};
use ignis_core::RopeScaling;
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget, snapshot_format_version};
use ignis_core::step::{
    MultimodalPrefill, SamplingParams, decode_program_batch, prefill_program,
    prefill_program_multimodal,
};

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
    model: &Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
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

/// A fresh sequence prefilled with `prompt` as a multimodal span rotated at
/// `positions` and left holding `rope_delta`, and the first tokens it decodes.
fn prefill_multimodal_and_decode<'p>(
    model: &Model,
    pool: &'p SeqPool,
    prompt: &[i32],
    positions: &[i32],
    rope_delta: i32,
    label: &str,
) -> (Seq<'p>, Vec<i32>) {
    let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("{label}: alloc: {e}"));
    prefill_program_multimodal(
        model,
        pool,
        &mut sequence,
        prompt,
        0,
        SamplingParams::greedy(),
        // No permitted set (GitHub #242): this prefills a prompt.
        &[],
        MultimodalPrefill { positions, rope_delta, media: None },
        None,
    )
    .unwrap_or_else(|e| panic!("{label}: multimodal prefill: {e}"));
    let head = decode_n(model, pool, &mut sequence, GENERATED, label);
    (sequence, head)
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

    let (plan, handles) = bind_model_scope_27b_with(&reader, ModelScope { draft: None, vision: true })
        .unwrap_or_else(|e| panic!("bind: {e}"));
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
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b_with_options(
        &reader,
        &artifact,
        &handles,
        MAX_CONTEXT,
        MAX_CONTEXT,
        ignis_core::KvFormat::Bf16,
        None,
        Some(Vision::default()),
        RopeScaling::NONE,
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
            retained_slot_count: 0,
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
    drop(midchunk);

    // ---- GitHub #194: a multimodal sequence restores with its rope delta ---
    //
    // A multimodal span over a real question, left holding a rope delta
    // every later decode round rotates at. The delta is far larger than an
    // image leaves (tens to hundreds of positions): this model rotates 64 of
    // 256 head dims at theta 1e7 in 16 of its 64 layers, and at an image's
    // delta its greedy tokens do not move, so a restore that dropped one
    // would pass unnoticed. The leaf treats the delta as an opaque scalar
    // either way. No media columns: a span rotates at its positions with or
    // without an embedding.
    const FAR_DELTA: i32 = 200_000;
    // Even at that delta the first dozen greedy tokens agree with delta 0's;
    // the tail after the restore is long enough to part company.
    const MULTIMODAL_TAIL: usize = 34;
    assert_eq!(snapshot_format_version(), 4, "this leg reads the version-4 blob layout");
    let long_prompt: Vec<i32> = frontend
        .tokenizer()
        .encode(
            "<|im_start|>user\nWrite a short, original poem about a lighthouse keeper who \
             collects lost letters from the sea, using vivid and unusual imagery.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        .unwrap_or_else(|e| panic!("tokenize multimodal prompt: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    let positions: Vec<i32> = (0..3).flat_map(|_| 0..long_prompt.len() as i32).collect();

    let (mut multimodal, _) = prefill_multimodal_and_decode(
        &model,
        &pool,
        &long_prompt,
        &positions,
        FAR_DELTA,
        "multimodal source",
    );
    let multimodal_blob =
        multimodal.snapshot().unwrap_or_else(|e| panic!("multimodal snapshot: {e}"));
    let (progress, _) = snapshot_blob::section(&multimodal_blob, snapshot_blob::SECTION_PROGRESS);
    assert_eq!(
        snapshot_blob::read_i32(&multimodal_blob, progress + snapshot_blob::PROGRESS_ROPE_DELTA),
        FAR_DELTA,
        "the blob's progress section carries the sequence's rope delta"
    );
    let multimodal_control = decode_n(&model, &pool, &mut multimodal, MULTIMODAL_TAIL, "multimodal control");
    drop(multimodal);

    let mut multimodal_restored =
        pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc multimodal target: {e}"));
    multimodal_restored
        .restore(&multimodal_blob)
        .unwrap_or_else(|e| panic!("multimodal restore: {e}"));
    assert_eq!(
        multimodal_restored.snapshot().unwrap_or_else(|e| panic!("multimodal re-snapshot: {e}")),
        multimodal_blob,
        "a restored multimodal sequence snapshots to the same bytes as its source"
    );
    let multimodal_tail =
        decode_n(&model, &pool, &mut multimodal_restored, MULTIMODAL_TAIL, "multimodal restored tail");
    drop(multimodal_restored);
    assert_eq!(
        multimodal_tail, multimodal_control,
        "a multimodal sequence restored from a snapshot continues to the same tokens"
    );

    // The comparison above proves the delta crossed only if the delta decides
    // what the sequence generates: the same prompt at delta 0 — what a
    // restore that lost it would decode — must not produce the same text.
    let (mut unrotated, _) =
        prefill_multimodal_and_decode(&model, &pool, &long_prompt, &positions, 0, "delta-0 twin");
    let unrotated_tail = decode_n(&model, &pool, &mut unrotated, MULTIMODAL_TAIL, "delta-0 twin tail");
    drop(unrotated);
    assert_ne!(
        multimodal_control, unrotated_tail,
        "the rope delta must change the decoded tokens, or this leg proves nothing about it"
    );

    // The arena outlives a drop: without the release, the next test in this
    // process finds the card still full.
    drop(pool);
    drop(model);
    let _ = artifact.release_arena(&mut device);
}

/// GitHub #257 (spec runtime/06): a sequence restored under hq-e8-2b into a
/// slot another sequence used continues to the tokens it would have produced
/// unevicted -- which it cannot unless the blob carried its residual window,
/// since the target slot's own rows were zeroed at alloc and its ring bits
/// with them. The ring words the source holds after its prefill and a run of
/// one-token decode rounds are read out of the blob and held to the host rule
/// (the decode route has no tap: this is its observation).
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn an_hq_sequence_restored_into_another_slot_continues_to_the_same_tokens() {
    const CONTEXT: u32 = 1024;
    const CHUNK: u32 = 256;
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
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
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let encode = |text: &str| -> Vec<i32> {
        frontend
            .tokenizer()
            .encode(text)
            .unwrap_or_else(|e| panic!("tokenize: {e}"))
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect()
    };
    // Long enough for the ring to hold keys past the sinks, short enough that
    // it does not wrap: every bit then names one position, and a wrong one
    // shows.
    let paragraph = "A paged KV cache stores each sequence's keys and values in fixed-size \
         pages, and a block table maps every logical page to a physical one. ";
    let prompt = encode(&format!(
        "<|im_start|>user\n{}Summarise that in one line.<|im_end|>\n<|im_start|>assistant\n",
        paragraph.repeat(4)
    ));
    let squatter_prompt = encode("<|im_start|>user\nName three rivers in Europe.<|im_end|>\n<|im_start|>assistant\n");
    assert!(
        prompt.len() > 64 && prompt.len() + 2 * GENERATED < 512,
        "the prompt must fill the ring past the sinks without wrapping it: {} tokens",
        prompt.len()
    );

    let model = load_qwen38_27b(&reader, &artifact, &handles, CHUNK, CONTEXT, ignis_core::KvFormat::HqE8_2b)
        .unwrap_or_else(|e| panic!("load hq model: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::HqE8_2b,
            kv_page_group_count: 2 * CONTEXT / 64,
            max_context_tokens: CONTEXT,
            slot_count: 2,
            retained_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));
    assert_eq!(snapshot_format_version(), 4, "this test reads the version-4 blob layout");

    let mut source = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc source: {e}"));
    let source_slot = source.stats().slot;
    prefill_program(&model, &pool, &mut source, &prompt, 0, None).unwrap_or_else(|e| panic!("prefill: {e}"));
    let _head = decode_n(&model, &pool, &mut source, GENERATED, "source head");
    let blob = source.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}"));

    // The ring the prefill's chunks and the decode rounds left: each chunk
    // appended its keys, each round the one column it committed.
    let mut ring = HqRing::new();
    let mut start = 0u64;
    while start < prompt.len() as u64 {
        let len = u64::from(CHUNK).min(prompt.len() as u64 - start);
        ring.append_prefill(start, len);
        start += len;
    }
    ring.append_decode(prompt.len() as u64, GENERATED as u64);
    assert_eq!(
        snapshot_blob::hq_ring_words(&blob),
        ring.words(),
        "the ring words after the prefill and {GENERATED} decode rounds are the host rule's"
    );

    let control = decode_n(&model, &pool, &mut source, GENERATED, "control tail");

    // Another sequence takes the other slot and writes a window of its own.
    let mut squatter = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc squatter: {e}"));
    let squatter_slot = squatter.stats().slot;
    assert_ne!(squatter_slot, source_slot, "two live sequences, two slots");
    prefill_program(&model, &pool, &mut squatter, &squatter_prompt, 0, None)
        .unwrap_or_else(|e| panic!("squatter prefill: {e}"));
    let _ = decode_n(&model, &pool, &mut squatter, GENERATED, "squatter");
    // Released in this order, the next alloc takes the squatter's slot.
    drop(source);
    drop(squatter);

    let mut restored = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("alloc restore target: {e}"));
    assert_eq!(restored.stats().slot, squatter_slot, "the restore lands in the slot the squatter used");
    let fresh = restored.snapshot().unwrap_or_else(|e| panic!("fresh snapshot: {e}"));
    assert!(
        snapshot_blob::hq_window(&fresh).iter().all(|&b| b == 0),
        "a re-allocated slot holds nothing of its previous occupant's window"
    );
    restored.restore(&blob).unwrap_or_else(|e| panic!("restore: {e}"));
    assert!(
        restored.snapshot().unwrap_or_else(|e| panic!("re-snapshot: {e}")) == blob,
        "a restored hq sequence snapshots to the same bytes as its source, window included"
    );
    let tail = decode_n(&model, &pool, &mut restored, GENERATED, "restored tail");
    assert_eq!(
        tail, control,
        "an hq sequence restored into another slot continues to the same tokens it would have \
         produced unevicted"
    );
    drop(restored);
    drop(pool);
    drop(model);
    let _ = artifact.release_arena(&mut device);
}
