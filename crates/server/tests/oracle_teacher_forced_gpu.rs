//! The G1 cross-engine canary gate (ADR 0014, GitHub #76): **teacher-forced
//! next-token agreement** against the recorded canary oracle, with a >= 95%
//! floor.
//!
//! For each oracle position the engine is fed the *oracle's own* token
//! prefix, and its greedy next-token argmax is compared with the oracle's
//! recorded token. Every position is scored independently, so one divergence
//! cannot cascade into the positions after it.
//!
//! That is the whole reason this test exists. The gate used to score a
//! *free-running* continuation (`ignis-bench oracle compare`): ignis fed its
//! own emitted token back in, so the two engines stopped sharing a prefix at
//! the first divergence. GitHub #72 showed the first divergence on two
//! canaries is an exact BF16 logit tie between ignis's pick and the oracle's
//! — a coin flip, not a compute error — and the cascade dragged the suite to
//! 52%. Free-running comparison is now a diagnostic (ADR 0014); this is the
//! correctness floor.
//!
//! What this gate is for: catching **gross implementation errors** — wrong
//! tensor layouts, missing ops, broken state wiring, wrong RoPE positions,
//! corrupted activations, an output head wired backwards. It is deliberately
//! a sanity floor. It does **not** assert numerical or token-level
//! equivalence with the reference (ADR 0007: correctness is self-checked,
//! not reference-matched), and it does not replace the numeric checks — the
//! vendored op tests, the f64 layer/program references and the determinism
//! checks remain separate hard checks.
//!
//! No mismatch class is waived. The two known BF16 exact-tie positions count
//! as mismatches and the suite still clears the floor (99/102 = 97.1%,
//! measured 2026-09-07; see ADR 0014 for the per-canary numbers).
//!
//! The scoring arithmetic lives in `ignis_bench::oracle` so it is
//! CPU-unit-tested without a GPU; this file is the GPU driver that produces
//! the predictions.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program;

use ignis_bench::oracle::{
    Fixture, G1_AGREEMENT_FLOOR, TeacherForcedResult, meets_g1_floor,
    overall_teacher_forced_agreement, score_teacher_forced,
};

use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 512;
/// The G1 window: the same `--first-n` the comparer defaults to.
const FIRST_N: usize = 32;

/// The recorded oracle (`crates/bench/tests/fixtures/oracle_canary.json`) —
/// the same fixture the gate has always scored against, read through
/// `ignis_bench`'s own type so the prompt text and token ids cannot drift
/// from what the bench tooling produces.
fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("tests")
        .join("fixtures")
        .join("oracle_canary.json");
    Fixture::read(&path).unwrap_or_else(|e| panic!("read the oracle fixture: {e}"))
}

/// Argmax with the lowest-token-id tie-break — the convention the vendored
/// `ninfer::ops::argmax` kernel uses, so this host-side pick is the token the
/// leaf itself would have emitted. Not a policy choice: the kernel is
/// vendored verbatim from the reference (ADR 0010) and this mirrors it.
fn argmax_lowest_id(logits: &[f32]) -> u32 {
    let mut best_id = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (id, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_id = id;
        }
    }
    best_id as u32
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn teacher_forced_canary_agreement_meets_the_g1_floor() {
    let path = Path::new(ARTIFACT);
    if !path.exists()
        && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}"))
    {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let provider = ArtifactTemplateProvider::new(frontend);
    // GitHub #68: the oracle was recorded with thinking disabled -- the
    // candidate prompt must match, or this scores a template difference
    // instead of engine agreement.
    let thinking = ThinkingOptions {
        enable_thinking: false,
        ..ThinkingOptions::default()
    };

    let (plan, handles) =
        bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind text scope: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize the text scope on the device: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));

    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let fixture = load_fixture();
    let mut results: Vec<TeacherForcedResult> = Vec::with_capacity(fixture.prompts.len());

    for canary in &fixture.prompts {
        let prompt_tokens: Vec<i32> = provider
            .apply_chat_template(&[ChatMessage::text("user", canary.prompt.clone())], &thinking, &[])
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect();
        assert!(
            !prompt_tokens.is_empty(),
            "{}: the prompt must template",
            canary.id
        );

        let compared = FIRST_N.min(canary.token_ids.len());
        let pool = SeqPool::create(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_page_group_count: 8,
                max_context_tokens: MAX_CONTEXT,
                slot_count: 1,
            },
        )
        .unwrap_or_else(|e| panic!("{}: seq pool create: {e}", canary.id));
        let mut sequence = pool
            .alloc(MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("{}: seq alloc: {e}", canary.id));

        // Teacher forcing rides the step ABI's existing span + start-position
        // contract (`ignis_program_prefill` requires `start_position` to be
        // the sequence frontier and advances it per token), so appending the
        // oracle's next token reuses the KV/GDN state already built. The
        // next-token logits are the ones `prefill_program` already copies
        // back for the span's last position (GitHub #72) -- no production
        // change was needed to expose them.
        let mut logits = vec![0f32; vocab];
        let mut position = 0u64;
        if let Err(e) = prefill_program(
            &model,
            &pool,
            &mut sequence,
            &prompt_tokens,
            position,
            Some(&mut logits),
        ) {
            if gpu_profile::skip_or_fail(&format!("{}: prefill_program(prompt): {e}", canary.id)) {
                continue;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
        position += prompt_tokens.len() as u64;

        let mut predictions: Vec<u32> = Vec::with_capacity(compared);
        for i in 0..compared {
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "{} position {i}: the logits must be finite",
                canary.id
            );
            predictions.push(argmax_lowest_id(&logits));

            // Feed the ORACLE's token, not ours: that is what keeps position
            // i + 1 judged on a prefix both engines share.
            let forced = [i32::try_from(canary.token_ids[i]).expect("token id fits i32")];
            if let Err(e) = prefill_program(
                &model,
                &pool,
                &mut sequence,
                &forced,
                position,
                Some(&mut logits),
            ) {
                if gpu_profile::skip_or_fail(&format!(
                    "{}: prefill_program(forced position {i}): {e}",
                    canary.id
                )) {
                    break;
                }
                unreachable!("skip_or_fail panics under the profile");
            }
            position += 1;
        }

        let result = score_teacher_forced(&canary.id, &canary.token_ids, &predictions, FIRST_N);
        for m in &result.mismatches {
            eprintln!(
                "G1 {} position {}: mismatch -- ours={:?} oracle={}",
                result.id, m.position, m.predicted, m.expected
            );
        }
        eprintln!(
            "G1 {}: teacher-forced agreement {}/{} = {:.1}%",
            result.id,
            result.agree,
            result.compared,
            result.agreement * 100.0
        );
        results.push(result);
    }

    let overall = overall_teacher_forced_agreement(&results);
    let agree: usize = results.iter().map(|r| r.agree).sum();
    let compared: usize = results.iter().map(|r| r.compared).sum();
    eprintln!(
        "G1 OVERALL teacher-forced agreement {agree}/{compared} = {:.1}% (floor {:.0}%)",
        overall * 100.0,
        G1_AGREEMENT_FLOOR * 100.0
    );

    assert!(
        compared > 0,
        "the fixture must contribute scored positions -- an empty suite is not a pass"
    );
    assert!(
        meets_g1_floor(overall),
        "G1 canary floor: teacher-forced agreement {agree}/{compared} = {:.1}% < {:.0}%. \
         This gate catches gross implementation errors (layouts, missing ops, state wiring, \
         positions, corrupted activations); it is not a parity check, so a drop here means \
         something in the forward pass is structurally wrong, not merely numerically different.",
        overall * 100.0,
        G1_AGREEMENT_FLOOR * 100.0
    );
}
