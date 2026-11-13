//! Does a **repair** draw the waiting position again from the scores its call kept,
//! and does the next round consume the result as if it had been drawn?
//! (GitHub #586, spec #287.)
//!
//! Check-first decoding samples freely, checks each draw against a grammar on
//! the host before the model consumes it, and on a **miss** asks the leaf to
//! draw that position again, restricted to what the grammar allows there
//! (`ignis_program_repair`). Everything that makes that exact lives on the
//! device -- the kept scores, the mask, the sampler, the penalty counts -- so
//! nothing CPU-side can check it.
//!
//! What each part pins, with the ticket's acceptance beside it:
//!
//! 1. **A replaced draw is consumed exactly as if it had been drawn.** The
//!    control draws token X under a short permitted set {X} (the forced
//!    literal's route, which feeds X through the same rounds); the subject
//!    draws freely and has its draw repaired to X. Their following tokens are
//!    the same, bit for bit, for the prefill's draw and for a plain round's.
//! 2. **A greedy repair is the masked argmax of the kept scores.** For the
//!    prefill's draw the readout of the same call is the oracle: the repair
//!    equals the masked argmax computed on the host from it, under a sparse
//!    mask and under one at the top of the distribution, and a second repair
//!    of the same position reads the same scores. For a plain round's draw the
//!    oracle is the leaf's own short-set draw over the same round: repairing a
//!    free draw to a set equals drawing under that set.
//! 3. **Under a temperature a repair never leaves its mask, and the position
//!    counts once**: the draw's penalty count is taken back and the repaired
//!    token's added, nothing else in the row moves (read from a snapshot's
//!    penalty-count section), and eight seeds draw more than one token -- a
//!    sampler, not an argmax.
//! 4. **The prefill's own draw** is repaired in 1, 2 and 3 alike.
//! 5. **The refusals**: scores a later round wrote over, a draw the
//!    test-only per-token route made (it keeps none), a prefill that was not
//!    asked to keep its draw's scores (#287: a request with no grammar, whose
//!    prefill makes no copy), a sequence with no pending token, and a mask
//!    built for another vocabulary.
//!
//! The prefill draws repaired here are a held request's: the prefill keeps
//! their scores (`keeps_scores`, [`held_prefill`]).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile it is a failure.

#![cfg(feature = "cuda")]

use std::collections::BTreeSet;
use std::path::Path;

use ignis_artifact::{CudaDevice, Reader, bind_text_scope_27b, materialize};
use ignis_core::KvFormat;
use ignis_core::compute::ModelConfig;
use ignis_core::constrained::PermittedMask;
use ignis_core::gpu_profile;
use ignis_core::model_load::{Model, load_qwen38_27b};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    LaneDraw, PermittedLane, PrefillRoute, SamplingParams, decode_program_batch, decode_program_batch_permitted,
    decode_program_batch_sampled, prefill_program_draw, prefill_program_permitted, prefill_program_sampled,
    prefill_program_with_route, repair_program,
};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 128;
/// Tokens compared after a replaced draw: enough that a state the replaced
/// token left subtly different would have shown by the end.
const FOLLOWING: usize = 6;
/// `IGNIS_SEQ_SECTION_PENALTY_COUNTS` (`kernel/include/ignis_seq_sections.h`).
const PENALTY_COUNTS_SECTION: i32 = 3;

fn new_pool(slot_count: u32) -> Result<SeqPool, String> {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
    )
}

fn prompt() -> Vec<i32> {
    (0..6).map(|i| 11 + i).collect()
}

/// The prompt's prefill for a request held to a grammar (#287): its draw keeps
/// the scores a repair reads, as the serving path asks for a checked job.
fn held_prefill(
    model: &Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
    sampling: SamplingParams,
    logits: Option<&mut [f32]>,
) -> Result<(), String> {
    prefill_program_draw(model, pool, sequence, &prompt(), 0, 0, None, false, true, LaneDraw::free(sampling), logits)
        .map(|_| ())
}

fn vocab() -> u32 {
    ModelConfig::qwen38_27b().vocab as u32
}

fn mask_of(ids: &[i32]) -> PermittedMask {
    let ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
    PermittedMask::from_ids(vocab(), &ids).unwrap_or_else(|e| panic!("mask: {e}"))
}

/// The host's masked argmax over a readout: the highest permitted logit, the
/// lower id on a tie -- the vendored sampler's own greedy rule.
fn masked_argmax(logits: &[f32], mask: &PermittedMask) -> i32 {
    let mut best: Option<u32> = None;
    for id in mask.ids() {
        if best.is_none_or(|b| logits[id as usize] > logits[b as usize]) {
            best = Some(id);
        }
    }
    best.expect("a mask permits a token") as i32
}

/// The readout's ids, best first, the lower id on a tie.
fn ranked(logits: &[f32]) -> Vec<i32> {
    let mut ids: Vec<i32> = (0..logits.len() as i32).collect();
    ids.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]).then(a.cmp(&b)));
    ids
}

fn hot(seed: u64) -> SamplingParams {
    SamplingParams {
        temperature: 2.0,
        top_k: 0,
        top_p: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed,
    }
}

/// `sequence`'s penalty-count row, read from a snapshot's own section.
fn penalty_counts(sequence: &Seq<'_>) -> Vec<i32> {
    let blob = sequence.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}"));
    let u32_at = |at: usize| u32::from_le_bytes(blob[at..at + 4].try_into().unwrap());
    let u64_at = |at: usize| u64::from_le_bytes(blob[at..at + 8].try_into().unwrap()) as usize;
    let (header_bytes, record_bytes, count) = (u32_at(12) as usize, u32_at(16) as usize, u32_at(20) as usize);
    for record in 0..count {
        let at = header_bytes + record * record_bytes;
        if u32_at(at) as i32 == PENALTY_COUNTS_SECTION {
            let (offset, bytes) = (u64_at(at + 8), u64_at(at + 16));
            return blob[offset..offset + bytes]
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
        }
    }
    panic!("the snapshot carries no penalty-count section");
}

/// The rows that differ between two penalty-count snapshots, as
/// (id, before, after).
fn moved(before: &[i32], after: &[i32]) -> Vec<(usize, i32, i32)> {
    before
        .iter()
        .zip(after)
        .enumerate()
        .filter(|(_, (b, a))| b != a)
        .map(|(id, (&b, &a))| (id, b, a))
        .collect()
}

fn decode_n(model: &Model, pool: &SeqPool, sequence: &mut Seq<'_>, count: usize, label: &str) -> Vec<i32> {
    (0..count)
        .flat_map(|_| {
            decode_program_batch(model, pool, &mut [&mut *sequence]).unwrap_or_else(|e| panic!("{label}: decode: {e}"))
        })
        .collect()
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_repair_draws_the_waiting_position_again_from_the_scores_its_call_kept() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
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
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT, KvFormat::Bf16)
        .unwrap_or_else(|e| panic!("model load: {e}"));
    let pool = new_pool(4).unwrap_or_else(|e| panic!("seq pool: {e}"));
    let alloc = || pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    let vocab = vocab() as usize;

    // The prompt's readout, and its free (greedy) successor.
    let mut readout = vec![0f32; vocab];
    let free = {
        let mut sequence = alloc();
        prefill_program_sampled(&model, &pool, &mut sequence, &prompt(), 0, SamplingParams::greedy(), Some(&mut readout))
            .unwrap_or_else(|e| panic!("readout prefill: {e}"));
        sequence.pending_token().expect("the prefill drew")
    };
    let ranking = ranked(&readout);
    assert_eq!(free, ranking[0], "the greedy draw is the readout's argmax");
    // A plausible token that is not the free one: the runner-up.
    let forced = ranking[1];

    // ── 1. a replaced draw is consumed as if it had been drawn ───────────
    // The prefill's draw. Control: X drawn under the short set {X}.
    let control = {
        let mut sequence = alloc();
        prefill_program_permitted(&model, &pool, &mut sequence, &prompt(), 0, SamplingParams::greedy(), &[forced], None)
            .unwrap_or_else(|e| panic!("control prefill: {e}"));
        decode_n(&model, &pool, &mut sequence, FOLLOWING, "control")
    };
    let subject = {
        let mut sequence = alloc();
        held_prefill(&model, &pool, &mut sequence, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("subject prefill: {e}"));
        assert_eq!(sequence.pending_token(), Some(free));
        let repaired = repair_program(&model, &pool, &mut sequence, SamplingParams::greedy(), &mask_of(&[forced]))
            .unwrap_or_else(|e| panic!("repair of the prefill's draw: {e}"));
        assert_eq!(repaired, forced, "a mask of one is that one token");
        assert_eq!(sequence.pending_token(), Some(forced), "the repaired token is the pending token now");
        decode_n(&model, &pool, &mut sequence, FOLLOWING, "subject")
    };
    eprintln!("ignis repair gpu: prefill draw {free} -> {forced}, then {subject:?}");
    assert_eq!(control[0], forced, "the control's first round emits its forced draw");
    assert_eq!(
        subject, control,
        "the next round consumed the repaired prefill draw exactly as the forced one"
    );

    // A plain round's draw: the subject's round draws freely and is repaired
    // to Y; the control's round draws Y under {Y}. Both rounds consumed the
    // same anchor, so only the draw differs, and then it does not.
    let (subject, round_forced) = {
        let mut sequence = alloc();
        prefill_program_sampled(&model, &pool, &mut sequence, &prompt(), 0, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("subject prefill: {e}"));
        decode_program_batch(&model, &pool, &mut [&mut sequence]).unwrap_or_else(|e| panic!("subject round: {e}"));
        let drawn = sequence.pending_token().expect("the round drew");
        let round_forced = if forced != drawn { forced } else { ranking[2] };
        let repaired = repair_program(&model, &pool, &mut sequence, SamplingParams::greedy(), &mask_of(&[round_forced]))
            .unwrap_or_else(|e| panic!("repair of a round's draw: {e}"));
        assert_eq!(repaired, round_forced);
        (decode_n(&model, &pool, &mut sequence, FOLLOWING, "subject"), round_forced)
    };
    let control = {
        let mut sequence = alloc();
        prefill_program_sampled(&model, &pool, &mut sequence, &prompt(), 0, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("control prefill: {e}"));
        decode_program_batch_permitted(
            &model,
            &pool,
            &mut [&mut sequence],
            &[PermittedLane { sampling: SamplingParams::greedy(), permitted: &[round_forced] }],
        )
        .unwrap_or_else(|e| panic!("control round: {e}"));
        decode_n(&model, &pool, &mut sequence, FOLLOWING, "control")
    };
    assert_eq!(control[0], round_forced);
    assert_eq!(subject, control, "the next round consumed the repaired round draw exactly as the forced one");

    // ── 2. a greedy repair is the masked argmax of the kept scores ───────
    {
        let mut sequence = alloc();
        let mut logits = vec![0f32; vocab];
        held_prefill(&model, &pool, &mut sequence, SamplingParams::greedy(), Some(&mut logits))
            .unwrap_or_else(|e| panic!("prefill: {e}"));
        // Sparse, across the whole vocabulary, without the free draw: a miss.
        let sparse: Vec<i32> = (0..vocab as i32).filter(|&id| id % 7 == 3 && id != free).collect();
        let sparse = mask_of(&sparse);
        let repaired = repair_program(&model, &pool, &mut sequence, SamplingParams::greedy(), &sparse)
            .unwrap_or_else(|e| panic!("sparse repair: {e}"));
        assert_eq!(repaired, masked_argmax(&logits, &sparse), "the masked argmax of the readout of the same position");
        // At the top of the distribution: the next twenty without the argmax.
        let top = mask_of(&ranking[1..21]);
        let again = repair_program(&model, &pool, &mut sequence, SamplingParams::greedy(), &top)
            .unwrap_or_else(|e| panic!("second repair: {e}"));
        assert_eq!(
            again,
            masked_argmax(&logits, &top),
            "a second repair of the position reads the scores the draw read, not the first repair's mask"
        );
        assert_eq!(again, ranking[1]);
        eprintln!("ignis repair gpu: greedy repairs {repaired} (sparse) and {again} (top)");
    }
    // A plain round's draw against the leaf's own short-set draw.
    {
        let mut subject = alloc();
        let mut control = alloc();
        for sequence in [&mut subject, &mut control] {
            prefill_program_sampled(&model, &pool, sequence, &prompt(), 0, SamplingParams::greedy(), None)
                .unwrap_or_else(|e| panic!("prefill: {e}"));
        }
        decode_program_batch(&model, &pool, &mut [&mut subject]).unwrap_or_else(|e| panic!("free round: {e}"));
        let drawn = subject.pending_token().expect("the round drew");
        let set: Vec<i32> = (0..32).map(|k| (drawn + 1 + k * 7_757) % vocab as i32).collect();
        assert!(!set.contains(&drawn));
        let repaired = repair_program(&model, &pool, &mut subject, SamplingParams::greedy(), &mask_of(&set))
            .unwrap_or_else(|e| panic!("round repair: {e}"));
        decode_program_batch_permitted(
            &model,
            &pool,
            &mut [&mut control],
            &[PermittedLane { sampling: SamplingParams::greedy(), permitted: &set }],
        )
        .unwrap_or_else(|e| panic!("short-set round: {e}"));
        assert_eq!(
            Some(repaired),
            control.pending_token(),
            "repairing a round's free draw to a set is drawing under that set, from the same scores"
        );
    }

    // ── 3. under a temperature: inside the mask, counted once ────────────
    let mut repaired_tokens = BTreeSet::new();
    for seed in 0..8u64 {
        let sampling = hot(0x586_0000 + seed);
        // The prefill's draw.
        let mut sequence = alloc();
        held_prefill(&model, &pool, &mut sequence, sampling, None)
            .unwrap_or_else(|e| panic!("hot prefill: {e}"));
        let drawn = sequence.pending_token().expect("the prefill drew");
        let before = penalty_counts(&sequence);
        assert_eq!(before[drawn as usize], 1, "a temperature draw counts its token");
        let near: Vec<i32> = ranking[..64].iter().copied().filter(|&id| id != drawn).collect();
        let mask = mask_of(&near);
        let repaired = repair_program(&model, &pool, &mut sequence, sampling, &mask)
            .unwrap_or_else(|e| panic!("hot repair: {e}"));
        assert!(mask.permits(repaired as u32), "a temperature repair drew {repaired}, outside its mask");
        let after = penalty_counts(&sequence);
        assert_eq!(
            moved(&before, &after),
            vec![
                (drawn.min(repaired) as usize, i32::from(drawn < repaired), i32::from(drawn > repaired)),
                (drawn.max(repaired) as usize, i32::from(drawn > repaired), i32::from(drawn < repaired)),
            ],
            "the draw's count taken back, the repaired token's added, nothing else"
        );
        repaired_tokens.insert(repaired);
        // The round that follows consumes it.
        let emitted = decode_program_batch_sampled(&model, &pool, &mut [&mut sequence], &[sampling])
            .unwrap_or_else(|e| panic!("hot round: {e}"));
        assert_eq!(emitted, vec![repaired]);

        // That round's own draw, repaired the same way.
        let drawn = sequence.pending_token().expect("the round drew");
        let before = penalty_counts(&sequence);
        let mask = mask_of(&ranking[..64].iter().copied().filter(|&id| id != drawn).collect::<Vec<_>>());
        let repaired = repair_program(&model, &pool, &mut sequence, sampling, &mask)
            .unwrap_or_else(|e| panic!("hot round repair: {e}"));
        assert!(mask.permits(repaired as u32), "a round's temperature repair drew {repaired}, outside its mask");
        let after = penalty_counts(&sequence);
        assert_eq!(after[drawn as usize], before[drawn as usize] - 1, "the round's draw's count taken back");
        assert_eq!(after[repaired as usize], before[repaired as usize] + 1, "the repaired token counted once");
        assert_eq!(moved(&before, &after).len(), 2, "and nothing else moved");
    }
    eprintln!("ignis repair gpu: eight temperature repairs drew {repaired_tokens:?}");
    assert!(
        repaired_tokens.len() > 1,
        "a temperature of 2.0 over 63 ids repaired to the same token eight times, which is an argmax, \
         not a draw: {repaired_tokens:?}"
    );

    // ── 5. the refusals ──────────────────────────────────────────────────
    {
        let mut left = alloc();
        let mut right = alloc();
        for sequence in [&mut left, &mut right] {
            prefill_program_sampled(&model, &pool, sequence, &prompt(), 0, SamplingParams::greedy(), None)
                .unwrap_or_else(|e| panic!("prefill: {e}"));
        }
        decode_program_batch(&model, &pool, &mut [&mut left, &mut right]).unwrap_or_else(|e| panic!("round: {e}"));
        // `right` alone rewrites the column `left`'s draw was kept in.
        decode_program_batch(&model, &pool, &mut [&mut right]).unwrap_or_else(|e| panic!("round: {e}"));
        let error = repair_program(&model, &pool, &mut left, SamplingParams::greedy(), &mask_of(&[forced]))
            .expect_err("scores written over since the draw");
        assert!(error.contains("written over"), "{error}");
        assert!(
            repair_program(&model, &pool, &mut right, SamplingParams::greedy(), &mask_of(&[forced])).is_ok(),
            "while the round that wrote them can repair its own draw"
        );

        // Each sequence below goes back to the pool before the next is
        // taken: the pool has four slots, and `left` and `right` hold two.
        {
            let mut per_token = alloc();
            prefill_program_with_route(&model, &pool, &mut per_token, &prompt(), 0, PrefillRoute::PerToken, None)
                .unwrap_or_else(|e| panic!("per-token prefill: {e}"));
            let error = repair_program(&model, &pool, &mut per_token, SamplingParams::greedy(), &mask_of(&[forced]))
                .expect_err("the per-token route keeps no scores");
            assert!(error.contains("kept no scores"), "{error}");
        }

        // #287: a prefill not asked to keep its draw's scores -- a request
        // with no grammar -- keeps none.
        {
            let mut unheld = alloc();
            prefill_program_sampled(&model, &pool, &mut unheld, &prompt(), 0, SamplingParams::greedy(), None)
                .unwrap_or_else(|e| panic!("unheld prefill: {e}"));
            let error = repair_program(&model, &pool, &mut unheld, SamplingParams::greedy(), &mask_of(&[forced]))
                .expect_err("an unheld prefill keeps no scores");
            assert!(error.contains("kept no scores"), "{error}");
        }

        let mut fresh = alloc();
        let error = repair_program(&model, &pool, &mut fresh, SamplingParams::greedy(), &mask_of(&[forced]))
            .expect_err("nothing drawn yet");
        assert!(error.contains("no pending token"), "{error}");

        let narrow = PermittedMask::from_ids(1000, &[1]).expect("a mask over 1000 columns");
        let error = repair_program(&model, &pool, &mut right, SamplingParams::greedy(), &narrow)
            .expect_err("a mask for another vocabulary");
        assert!(error.contains("for a vocabulary of"), "{error}");
    }
}
