//! GPU coverage for the sequence pool's **retained slots** (GitHub #211, ADR
//! 0030): the places past the lanes that hold a lane's mutable state each.
//!
//! What is pinned here, on a pool with the DFlash2 drafter's sections at the
//! real 27B state geometry (no weights needed — the pool is the subject):
//!
//! - the pool reserves exactly one lane's state per retained slot more, and
//!   the lanes' own line does not move;
//! - no sequence ever stands on a retained slot;
//! - a lane's state copied into a retained slot and back is bit-exact, for
//!   every section, and two retained slots never alias each other or a lane.
//!
//! The copy is checked twice. Once on a pattern written through
//! `ignis_seq_restore`: every byte of every section is distinct from its
//! neighbours and from the other pattern, which a copy that dropped, shifted
//! or aliased any region cannot reproduce. And once on a real DFlash2 load,
//! whose sequence has prefilled and drafted, so the drafter's window and
//! checkpoint hold what the drafter wrote.
//!
//! Explicit GPU profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing
//! GPU is a skip; under the profile it is a hard failure.

#![cfg(feature = "cuda")]

#[path = "support/snapshot_blob.rs"]
mod snapshot_blob;

use std::path::Path;

use ignis_artifact::{CudaDevice, DraftModule, FrontendSet, Reader, bind_model_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_speculation;
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget, snapshot_format_version};
use ignis_core::step::{SamplingParams, VerifyLane, decode_program_verify_runs, prefill_program_sampled};
use ignis_core::{KvFormat, RetainedSlots, Speculation, SpeculativeBackend};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// The mutable-state sections a retained slot carries: GDN conv, GDN
/// recurrent, penalty counts, the drafter's window and its checkpoint.
const STATE_SECTIONS: [i32; 5] = [1, 2, 3, 5, 6];
const LANES: u32 = 2;
const RETAINED: u32 = 2;

fn cuda_device_or_skip() -> Option<CudaDevice> {
    match CudaDevice::create(0) {
        Ok(d) => Some(d),
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                None
            } else {
                unreachable!("skip_or_fail panics under the profile");
            }
        }
    }
}

fn budget(retained_slot_count: u32) -> SeqPoolBudget {
    SeqPoolBudget {
        kv_format: KvFormat::Bf16,
        kv_page_group_count: 8,
        max_context_tokens: 128,
        slot_count: LANES,
        retained_slot_count,
    }
}

/// Overwrite every state section of `seq` with a pattern keyed by `seed`,
/// through a snapshot of it and a restore.
fn write_pattern(seq: &mut Seq<'_>, seed: u32) -> Vec<u8> {
    let mut blob = seq.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}"));
    for kind in STATE_SECTIONS {
        let (offset, bytes) = snapshot_blob::section(&blob, kind);
        for (i, byte) in blob[offset..offset + bytes].iter_mut().enumerate() {
            // A cheap hash of (seed, kind, index): no two neighbouring
            // regions repeat, so a shifted or truncated copy shows.
            let x = (i as u32).wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503) ^ (kind as u32) << 24;
            *byte = (x ^ (x >> 13)) as u8;
        }
    }
    seq.restore(&blob).unwrap_or_else(|e| panic!("restore the pattern: {e}"));
    blob
}

/// The bytes of every state section of `seq`, in section order.
fn state_of(seq: &Seq<'_>) -> Vec<Vec<u8>> {
    let blob = seq.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}"));
    sections_of(&blob)
}

fn sections_of(blob: &[u8]) -> Vec<Vec<u8>> {
    STATE_SECTIONS
        .iter()
        .map(|&kind| {
            let (offset, bytes) = snapshot_blob::section(blob, kind);
            assert!(bytes > 0, "section {kind} is empty on a DFlash2 pool");
            blob[offset..offset + bytes].to_vec()
        })
        .collect()
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn retained_slots_reserve_one_lane_state_each_and_are_never_a_lane() {
    let Some(_device) = cuda_device_or_skip() else { return };
    let cfg = ModelConfig::qwen38_27b();
    let backend = Some(SpeculativeBackend::Dflash2);

    let without = SeqPool::plan(&cfg, &budget(0), backend).unwrap_or_else(|e| panic!("plan: {e}"));
    let with = SeqPool::plan(&cfg, &budget(RETAINED), backend).unwrap_or_else(|e| panic!("plan: {e}"));
    assert!(with.slot_state_bytes > 80 * 1024 * 1024, "a slot carries the drafter's 80 MiB");
    assert_eq!(without.slot_state_bytes, with.slot_state_bytes);
    assert_eq!(without.retained_state_bytes, 0);
    assert_eq!(with.retained_state_bytes, u64::from(RETAINED) * with.slot_state_bytes);
    assert_eq!(
        with.lane_state_bytes, without.lane_state_bytes,
        "the lanes' line does not move: the retained slots are exactly one lane's state each more"
    );
    assert_eq!(
        without.lane_state_bytes,
        u64::from(LANES) * without.slot_state_bytes,
        "a lane's state is one slot's"
    );
    assert_eq!(with.kv_bytes, without.kv_bytes, "a retained slot has no KV block-table row");

    let pool = SeqPool::create_with_speculation(&cfg, &budget(RETAINED), backend)
        .unwrap_or_else(|e| panic!("create: {e}"));
    let stats = pool.stats();
    assert_eq!(stats.slot_count, LANES);
    assert_eq!(stats.free_slot_count, LANES, "only lanes are free to sequences");
    assert_eq!(stats.retained_slot_count, RETAINED);
    assert_eq!(stats.slot_state_bytes, with.slot_state_bytes);
    assert_eq!(stats.retained_state_bytes, with.retained_state_bytes);
    assert_eq!(stats.lane_state_bytes, with.lane_state_bytes, "the built pool holds its plan");
    assert_eq!(stats.kv_arena_bytes, with.kv_bytes);

    let first = pool.alloc(64).unwrap_or_else(|e| panic!("alloc: {e}"));
    let second = pool.alloc(64).unwrap_or_else(|e| panic!("alloc: {e}"));
    let mut slots = [first.stats().slot, second.stats().slot];
    slots.sort_unstable();
    assert_eq!(slots, [0, 1], "sequences stand on lane slots");
    let Err(err) = pool.alloc(64) else {
        panic!("every lane is taken; the retained slots are not lanes");
    };
    assert!(err.contains("no free slot"), "{err}");
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_lane_state_copied_into_a_retained_slot_and_back_is_bit_exact() {
    let Some(_device) = cuda_device_or_skip() else { return };
    assert_eq!(snapshot_format_version(), 3, "support/snapshot_blob.rs reads format 3");
    let cfg = ModelConfig::qwen38_27b();
    let pool = SeqPool::create_with_speculation(&cfg, &budget(RETAINED), Some(SpeculativeBackend::Dflash2))
        .unwrap_or_else(|e| panic!("create: {e}"));
    let mut retained = RetainedSlots::new(RETAINED);
    let (r0, r1) = (retained.take().unwrap(), retained.take().unwrap());

    let mut a = pool.alloc(64).unwrap_or_else(|e| panic!("alloc: {e}"));
    let mut b = pool.alloc(64).unwrap_or_else(|e| panic!("alloc: {e}"));

    let first = sections_of(&write_pattern(&mut a, 1));
    assert_eq!(state_of(&a), first, "the pattern landed");
    pool.retained_store(&a, &r0).unwrap_or_else(|e| panic!("store into r0: {e}"));
    let second = sections_of(&write_pattern(&mut a, 2));
    assert_ne!(first, second);
    pool.retained_store(&a, &r1).unwrap_or_else(|e| panic!("store into r1: {e}"));
    assert_eq!(state_of(&a), second, "a store reads the lane and changes nothing");

    // Dirty both lanes, then bring each retained slot back into the other
    // lane and into its own.
    write_pattern(&mut a, 3);
    write_pattern(&mut b, 4);
    pool.retained_load(&r0, &mut b).unwrap_or_else(|e| panic!("load r0 into b: {e}"));
    assert_eq!(state_of(&b), first, "r0 into another lane is bit-exact");
    pool.retained_load(&r1, &mut a).unwrap_or_else(|e| panic!("load r1 into a: {e}"));
    assert_eq!(state_of(&a), second, "r1 back into its own lane is bit-exact");
    pool.retained_load(&r0, &mut a).unwrap_or_else(|e| panic!("load r0 into a: {e}"));
    assert_eq!(state_of(&a), first, "a load leaves the retained slot as it was");
    assert_eq!(state_of(&b), first, "a load into one lane leaves the other alone");

    // A retained index past the pool's is refused.
    let mut wide = RetainedSlots::new(RETAINED + 1);
    let _ = (wide.take(), wide.take());
    let past = wide.take().unwrap();
    let err = pool.retained_store(&a, &past).expect_err("retained slot 2 of 2");
    assert!(err.message.contains("out of range"), "{err}");
    let err = pool.retained_load(&past, &mut a).expect_err("retained slot 2 of 2");
    assert!(err.message.contains("out of range"), "{err}");
    assert_eq!(state_of(&a), first, "a refused load changes nothing");

    // A pool with no retained slots has none to copy into.
    let plain = SeqPool::create_with_speculation(&cfg, &budget(0), Some(SpeculativeBackend::Dflash2))
        .unwrap_or_else(|e| panic!("create: {e}"));
    let lone = plain.alloc(64).unwrap_or_else(|e| panic!("alloc: {e}"));
    let err = plain.retained_store(&lone, &r0).expect_err("no retained slots");
    assert!(err.message.contains("out of range"), "{err}");
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_drafting_sequence_state_comes_back_from_a_retained_slot_bit_exact_on_a_dflash2_load() {
    const MAX_CONTEXT: u32 = 256;
    const WINDOW: u32 = 7;
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    assert_eq!(snapshot_format_version(), 3, "support/snapshot_blob.rs reads format 3");
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let prompt: Vec<i32> = frontend
        .tokenizer()
        .encode(
            "<|im_start|>user\nList five Rust crates for parsing command-line arguments, one per \
             line.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        .unwrap_or_else(|e| panic!("tokenize: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    let (plan, handles) = bind_model_scope_27b(&reader, Some(DraftModule::Dflash2))
        .unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
    let Some(mut device) = cuda_device_or_skip() else { return };
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + dflash2: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    {
        let speculation = Speculation::new(SpeculativeBackend::Dflash2, WINDOW).unwrap();
        let model = load_qwen38_27b_with_speculation(
            &reader,
            &artifact,
            &handles,
            MAX_CONTEXT,
            MAX_CONTEXT,
            KvFormat::Bf16,
            Some(speculation),
        )
        .unwrap_or_else(|e| panic!("model load (dflash2): {e}"));
        let pool = SeqPool::create_with_speculation(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: KvFormat::Bf16,
                kv_page_group_count: MAX_CONTEXT.div_ceil(64),
                max_context_tokens: MAX_CONTEXT,
                slot_count: 1,
                retained_slot_count: 1,
            },
            Some(SpeculativeBackend::Dflash2),
        )
        .unwrap_or_else(|e| panic!("create: {e}"));
        let mut retained = RetainedSlots::new(1);
        let slot = retained.take().unwrap();

        let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill_program_sampled(&model, &pool, &mut seq, &prompt, 0, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
        let rounds = |seq: &mut Seq<'_>, count: usize| {
            for _ in 0..count {
                let lanes = [VerifyLane {
                    remaining_tokens: 64,
                    ..VerifyLane::greedy(&[])
                }];
                decode_program_verify_runs(&model, &pool, &mut [&mut *seq], &lanes, WINDOW)
                    .unwrap_or_else(|e| panic!("verify round: {e}"));
            }
        };
        rounds(&mut seq, 2);

        let drafted = state_of(&seq);
        // The GDN state and the drafter's window and checkpoint hold what the
        // load wrote. The penalty counts are left out: greedy rounds with no
        // penalty keep them at zero, and the pattern test above carries them.
        for (&kind, section) in STATE_SECTIONS.iter().zip(&drafted) {
            if kind != 3 {
                assert!(section.iter().any(|&b| b != 0), "section {kind} holds no state");
            }
        }
        pool.retained_store(&seq, &slot).unwrap_or_else(|e| panic!("store: {e}"));
        rounds(&mut seq, 2);
        assert_ne!(state_of(&seq), drafted, "two more rounds moved the lane's state");
        pool.retained_load(&slot, &mut seq).unwrap_or_else(|e| panic!("load: {e}"));
        assert_eq!(state_of(&seq), drafted, "the retained slot gives back the drafted state bit for bit");
    }
    let _ = artifact.release_arena(&mut device);
}
