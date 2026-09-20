//! Does a **permitted token set** actually constrain the draw? (GitHub #242,
//! ADR 0034.)
//!
//! Spec 06's acceptance 1: "a constrained decode returns only tokens from its
//! permitted set, under greedy and under temperature". That is a property of
//! the device — the mask runs between the model's traversal and the vendored
//! sampler — so nothing CPU-side can check it.
//!
//! **Where a constrained run starts.** A decode round returns the successor
//! the *previous* call made ready, so the first token of a run is the one the
//! **prefill** draws — a run that constrained only its rounds would commit one
//! free token in the middle of its own forced text. This test found that by
//! failing: the first version constrained a round and got the prompt's free
//! successor back.
//!
//! What each part of this pins:
//!
//! 1. **Greedy** takes the set's argmax. A mask that did nothing would still
//!    pass a test that only asked for a token id back, so the set here
//!    deliberately excludes what the unconstrained round commits — the same
//!    prompt is run twice, once free and once constrained away from its own
//!    free answer.
//! 2. **Temperature** draws from the set alone. A mask applied after the
//!    sampler, or one that left a few columns alive, would show up here and
//!    not under greedy: a spread distribution samples the tail.
//! 3. **The reported probability** is the committed token's share of its own
//!    set. Checked against the same softmax computed on the host from the
//!    round's own logits, which is the only independent statement of what it
//!    should be.
//! 4. **An unconstrained lane in the same round is untouched**, because the
//!    mask is per lane and a round mixes them.
//! 5. **The refusals**: a set larger than the leaf's cap, and an id outside
//!    the vocabulary, are errors rather than truncations or reads past the
//!    end of a row.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile it is a failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, Reader, bind_text_scope_27b, materialize};
use ignis_core::KvFormat;
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{Model, load_qwen38_27b};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    PermittedLane, SamplingParams, decode_program_batch_permitted, prefill_program_permitted,
    prefill_program_sampled,
};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 128;
/// The leaf's own cap (`IGNIS_MAX_PERMITTED_TOKENS`), restated here so the
/// refusal test asks for one past it rather than for a number.
const CAP: usize = 32;

fn new_pool(slot_count: u32) -> Result<SeqPool, String> {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
            retained_slot_count: 0,
        },
    )
}

fn prompt() -> Vec<i32> {
    (0..6).map(|i| 11 + i).collect()
}

/// Prefill `sequence` with the shared prompt, keeping the round's logits.
fn prefill(model: &Model, pool: &SeqPool, sequence: &mut Seq<'_>, logits: &mut [f32]) {
    prefill_program_sampled(
        model,
        pool,
        sequence,
        &prompt(),
        0,
        SamplingParams::greedy(),
        Some(logits),
    )
    .unwrap_or_else(|e| panic!("prefill: {e}"));
}

/// The softmax of `ids`' logits over `ids` alone — what the device reports
/// for the token it committed, computed here from the host's own copy.
fn restricted(logits: &[f32], ids: &[i32], chosen: i32) -> f64 {
    let highest = ids
        .iter()
        .map(|&id| logits[id as usize] as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = ids.iter().map(|&id| (logits[id as usize] as f64 - highest).exp()).sum();
    let own = (logits[chosen as usize] as f64 - highest).exp();
    own / total
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_permitted_set_is_the_only_thing_a_lane_can_commit() {
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
    let model = load_qwen38_27b(
        &reader,
        &artifact,
        &handles,
        MAX_CONTEXT,
        MAX_CONTEXT,
        KvFormat::Bf16,
    )
    .unwrap_or_else(|e| panic!("model load: {e}"));
    let pool = new_pool(2).unwrap_or_else(|e| panic!("seq pool: {e}"));
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut logits = vec![0f32; vocab];

    // ── what the prompt's own successor is with nothing in its way ───────
    let free_token = {
        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill(&model, &pool, &mut sequence, &mut logits);
        let (tokens, probabilities) = decode_program_batch_permitted(
            &model,
            &pool,
            &mut [&mut sequence],
            &[PermittedLane { sampling: SamplingParams::greedy(), permitted: &[] }],
        )
        .unwrap_or_else(|e| panic!("unconstrained round: {e}"));
        assert_eq!(
            probabilities[0], 0.0,
            "a lane that declared no set reports no probability, rather than a \
             number computed over a set it does not have"
        );
        tokens[0]
    };
    assert!(free_token >= 0, "the free round committed a token");

    /// Prefill the shared prompt with `set` in force, then surface the
    /// successor it drew — which the following round emits.
    macro_rules! constrained_draw {
        ($set:expr, $sampling:expr) => {{
            let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
            let probability = prefill_program_permitted(
                &model,
                &pool,
                &mut sequence,
                &prompt(),
                0,
                $sampling,
                $set,
                None,
            )
            .unwrap_or_else(|e| panic!("constrained prefill: {e}"));
            let (tokens, _) = decode_program_batch_permitted(
                &model,
                &pool,
                &mut [&mut sequence],
                &[PermittedLane { sampling: SamplingParams::greedy(), permitted: &[] }],
            )
            .unwrap_or_else(|e| panic!("round after a constrained prefill: {e}"));
            (tokens[0], probability)
        }};
    }

    // ── 1. greedy takes the set's argmax, not the model's own ────────────
    // A set of ten ids that deliberately excludes the free answer: if the
    // mask did nothing, greedy would commit `free_token` again.
    let set: Vec<i32> = (0..10).map(|k| (free_token + 1 + k) % vocab as i32).collect();
    assert!(!set.contains(&free_token), "the set excludes the free answer by construction");

    let (greedy, greedy_probability) = constrained_draw!(&set, SamplingParams::greedy());
    assert!(
        set.contains(&greedy),
        "greedy committed {greedy}, which is not in its permitted set {set:?}"
    );
    assert_ne!(
        greedy, free_token,
        "and the constraint really moved it off the free answer"
    );

    // 3. The probability is the committed token's share of the set, and the
    //    host can say what that is from the same logits.
    let expected = restricted(&logits, &set, greedy);
    eprintln!(
        "ignis permitted gpu: greedy {greedy} of {set:?}, p {greedy_probability} \
         (host {expected:.6})"
    );
    assert!(
        (greedy_probability as f64 - expected).abs() < 5e-3,
        "the reported probability {greedy_probability} is the restricted softmax {expected:.6}"
    );
    // Greedy over a set takes its argmax, so the winner holds the most of it.
    for &id in &set {
        assert!(
            logits[greedy as usize] >= logits[id as usize],
            "the committed token is the set's argmax: {greedy} vs {id}"
        );
    }
    // ── 2. temperature draws from the set, and only from it ──────────────
    // Spread wide on purpose: a peaked distribution would commit the argmax
    // every round and a mask applied in the wrong place could hide behind it.
    let hot = SamplingParams {
        temperature: 2.0,
        top_k: 0,
        top_p: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0x2420_0001,
    };
    let mut drawn = std::collections::BTreeSet::new();
    for round in 0..8 {
        let mut hot = hot;
        hot.seed = 0x2420_0001 + round;
        let (token, probability) = constrained_draw!(&set, hot);
        assert!(
            set.contains(&token),
            "a temperature draw committed {token}, which is not in its permitted set {set:?}"
        );
        assert!(
            probability > 0.0 && probability <= 1.0,
            "and reports a probability: {probability}"
        );
        drawn.insert(token);
    }
    eprintln!("ignis permitted gpu: eight hot draws committed {drawn:?}");
    assert!(
        drawn.len() > 1,
        "a temperature of 2.0 over ten ids drew the same one eight times, which \
         means this is not sampling at all: {drawn:?}"
    );

    // ── 4. a constrained lane does not constrain the one beside it ───────
    let mut left = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    let mut right = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    prefill(&model, &pool, &mut left, &mut logits);
    prefill(&model, &pool, &mut right, &mut logits);
    // Both lanes carry the free successor now; the sets here constrain the
    // token each will emit next, and only one lane declares one.
    let (_, probabilities) = decode_program_batch_permitted(
        &model,
        &pool,
        &mut [&mut left, &mut right],
        &[
            PermittedLane { sampling: SamplingParams::greedy(), permitted: &set },
            PermittedLane { sampling: SamplingParams::greedy(), permitted: &[] },
        ],
    )
    .unwrap_or_else(|e| panic!("mixed round: {e}"));
    assert!(
        probabilities[0] > 0.0,
        "the constrained lane reported its draw's share of the set: {}",
        probabilities[0]
    );
    assert_eq!(probabilities[1], 0.0, "an unconstrained lane reports no probability");
    let (tokens, _) = decode_program_batch_permitted(
        &model,
        &pool,
        &mut [&mut left, &mut right],
        &[
            PermittedLane { sampling: SamplingParams::greedy(), permitted: &[] },
            PermittedLane { sampling: SamplingParams::greedy(), permitted: &[] },
        ],
    )
    .unwrap_or_else(|e| panic!("round after the mixed one: {e}"));
    assert!(
        set.contains(&tokens[0]),
        "the constrained lane emitted from its set: {} of {set:?}",
        tokens[0]
    );
    assert!(
        !set.contains(&tokens[1]) || tokens[1] == tokens[0],
        "and the lane beside it drew freely — the mask is per lane, and a round \
         mixes them: {}",
        tokens[1]
    );
    drop(left);
    drop(right);

    // ── 5. the refusals ──────────────────────────────────────────────────
    let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    prefill(&model, &pool, &mut sequence, &mut logits);
    let oversized: Vec<i32> = (0..CAP as i32 + 1).collect();
    let error = decode_program_batch_permitted(
        &model,
        &pool,
        &mut [&mut sequence],
        &[PermittedLane { sampling: SamplingParams::greedy(), permitted: &oversized }],
    )
    .expect_err("a set past the cap is refused");
    assert!(error.contains("refused, never truncated"), "{error}");

    let outside = [vocab as i32];
    let error = decode_program_batch_permitted(
        &model,
        &pool,
        &mut [&mut sequence],
        &[PermittedLane { sampling: SamplingParams::greedy(), permitted: &outside }],
    )
    .expect_err("an id past the vocabulary is refused");
    assert!(error.contains("outside the vocabulary"), "{error}");
}
