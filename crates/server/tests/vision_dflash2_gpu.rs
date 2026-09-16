//! DFlash2 on a multimodal prefill span (GitHub #195, spec
//! `.scratch/vision/specs/01-image-input.md` §Compute seam, ADR 0014).
//!
//! GitHub #178 fenced `--vision` and `--spec dflash2` apart at load. What
//! stood in the way was one thing: the verify round staged each lane's raw KV
//! frontier as the rotation position, so a sequence with a nonzero
//! `rope_delta` -- every prompt with an image -- verified its columns at
//! different positions from the ones its own decode rounds use. The drafter
//! needed nothing: its context append consumes the span's KV positions, which
//! is exactly what the reference's own prefill sink captures on a multimodal
//! span (`tap.capture_positions(positions, ...)` over the cache positions,
//! `impl/runtime/text_context_impl.h`), and its proposal block rotates at the
//! lane frontier the same way. No vision-specific drafter state exists.
//!
//! One load carries all four checks, because the card fits one artifact at a
//! time and each is a question about the *same* load:
//!
//! - AC 1: `--vision` with `--spec dflash2` loads -- the tower and the
//!   drafter bind together and the vision reservation is still reported.
//! - AC 2: the vision canary's teacher-forced floor (ADR 0014, >= 95%) holds
//!   on this load, so the drafter riding the multimodal prefill does not
//!   disturb the target's own next-token agreement with the reference.
//! - AC 3: greedy spec-on equals spec-off on the canary images. Spec-off here
//!   is a plain decode round *on this same load*, which is the sharpest form
//!   of the question: the two rounds differ only in shape, so a rotation the
//!   verify round got wrong shows up immediately. A divergence passes only
//!   where the engine's own two multimodal prefill routes (the whole span and
//!   the placeholder-crossing 48-token spans) call the two candidates a
//!   near-tie -- the same rule `crates/core/tests/support/near_tie.rs` states
//!   for text, with the per-token route (which has no multimodal form)
//!   replaced by the spanning one.
//! - AC 4: acceptance on the text after an image, printed per round and as
//!   the mean committed tokens per full-window round against the reference's
//!   3.4-5.75 band. Below the band is a finding, not a failure; never landing
//!   a draft at all is a failure, because that is a drafter reading the wrong
//!   window. AC 3 and AC 4 run over two questions per image -- the fixture's
//!   own, whose answer is one sentence, and a request for a long description
//!   that runs to the token budget, because a one-sentence answer is two or
//!   three rounds and that is not a sample.
//!
//! BF16 KV throughout: it is the correctness oracle (ADR 0022).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

#[path = "support/mod.rs"]
mod support;

use std::path::Path;

use ignis_artifact::vision::VisionProcessor;
use ignis_artifact::{
    bind_model_scope_27b_with, materialize, ChatMessage, ContentPart, CudaDevice, DraftModule,
    FrontendSet, MessageContent, ModelScope, Reader, Role,
};
use ignis_bench::oracle::{
    meets_g1_floor, overall_teacher_forced_agreement, score_teacher_forced, TeacherForcedResult,
    G1_AGREEMENT_FLOOR,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_options, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    self, decode_program_batch_sampled, decode_program_verify_runs, MediaEmbedding, SamplingParams,
    VerifyLane,
};
use ignis_core::vision::{vision_item_control, Multimodal};
use ignis_core::{KvFormat, Speculation, SpeculativeBackend, Vision};

use support::vision_canary::{argmax_lowest_id, load_canaries, prefill_prompt, Canary};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 2048;
const PREFILL_CHUNK: u32 = 1024;
/// Narrower than every canary image's placeholder run (64..196 tokens), so
/// the second prefill route cuts the run (`vision_canary_gpu.rs`'s "spanning").
const SPANNING_CHUNK: u32 = 48;
/// DFlash2-7, the reference lane G5 measures.
const WINDOW: u32 = 7;
/// Scored teacher-forced positions per canary (ADR 0014's first 32).
const FIRST_N: usize = 32;
/// Tokens each canary emits in the spec-on/spec-off comparison.
const TOTAL: usize = 48;
/// The reference's committed tokens per round at draft 7 (the issue's band).
const REFERENCE_BAND: (f64, f64) = (3.4, 5.75);
/// A second question per canary image, whose answer runs to the token budget
/// rather than stopping after one sentence, so the acceptance figure has a
/// sample. Never scored against the fixture -- the reference recorded only
/// the canary's own question.
const DETAIL_QUESTION: &str = "Describe everything you can see in this image, in detail.";

/// The prompt a canary turns into, once: its token ids, and the three-axis
/// positions, rope delta and media item the frontend prepared with them.
struct Prompt {
    tokens: Vec<i32>,
    multimodal: Multimodal,
}

fn drafter_pool(slot_count: u32) -> SeqPool {
    SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
        },
        Some(SpeculativeBackend::Dflash2),
    )
    .unwrap_or_else(|e| panic!("seq pool create (dflash2): {e}"))
}

fn prompt_for(
    frontend: &FrontendSet,
    processor: &VisionProcessor,
    canary: &Canary,
    question: &str,
) -> Prompt {
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Parts(vec![
            ContentPart::Image { url: None },
            ContentPart::Text(question.to_owned()),
        ]),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let prepared = frontend
        .prepare_prompt(processor, &messages, &[&canary.image], false, None, None)
        .unwrap_or_else(|e| panic!("{}: prepare prompt: {e}", canary.id));
    let (token_ids, multimodal) = Multimodal::from_prepared(prepared);
    assert_eq!(multimodal.media.len(), 1, "{}: one image per canary", canary.id);
    assert_ne!(
        multimodal.rope_delta, 0,
        "{}: an image prompt whose rope delta is 0 cannot tell the two rotations apart",
        canary.id
    );
    Prompt { tokens: token_ids.iter().map(|&t| t as i32).collect(), multimodal }
}

/// A fresh sequence carrying `prompt`, prefilled in `chunk`-wide spans.
fn prefilled<'p>(
    model: &Model,
    pool: &'p SeqPool,
    prompt: &Prompt,
    embedding: &MediaEmbedding<'_>,
    chunk: u32,
    logits: &mut [f32],
) -> Seq<'p> {
    let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
    prefill_prompt(model, pool, &mut sequence, &prompt.tokens, &prompt.multimodal, embedding, chunk, logits)
        .unwrap_or_else(|e| panic!("multimodal prefill at chunk {chunk}: {e}"));
    sequence
}

/// `total` greedy tokens from a plain decode round -- one token per round on
/// the load's own decode path -- cut at the turn's first stop id.
fn spec_off(
    model: &Model,
    pool: &SeqPool,
    prompt: &Prompt,
    embedding: &MediaEmbedding<'_>,
    stop_ids: &[i32],
    logits: &mut [f32],
) -> Vec<i32> {
    let mut sequence = prefilled(model, pool, prompt, embedding, PREFILL_CHUNK, logits);
    let params = [SamplingParams::greedy()];
    let mut stream = Vec::with_capacity(TOTAL);
    while stream.len() < TOTAL {
        let round = decode_program_batch_sampled(model, pool, &mut [&mut sequence], &params)
            .unwrap_or_else(|e| panic!("spec-off decode: {e}"));
        stream.push(round[0]);
        if stop_ids.contains(&round[0]) {
            break;
        }
    }
    stream
}

/// What one spec-on run recorded: the text, and per round its
/// `(extent, committed)`.
struct SpecRun {
    emitted: Vec<i32>,
    rounds: Vec<(u32, usize)>,
}

/// The same `total` tokens through verify rounds, the drafter proposing from
/// its window. One lane: the question is the rotation, not the batching.
fn spec_on(
    model: &Model,
    pool: &SeqPool,
    prompt: &Prompt,
    embedding: &MediaEmbedding<'_>,
    stop_ids: &[i32],
    logits: &mut [f32],
) -> SpecRun {
    let mut sequence = prefilled(model, pool, prompt, embedding, PREFILL_CHUNK, logits);
    let mut emitted: Vec<i32> = Vec::with_capacity(TOTAL);
    let mut rounds = Vec::new();
    while emitted.len() < TOTAL {
        let remaining_tokens = (TOTAL - emitted.len()) as u32;
        let lanes = [VerifyLane { remaining_tokens, stop_ids, ..VerifyLane::greedy(&[]) }];
        let results = decode_program_verify_runs(model, pool, &mut [&mut sequence], &lanes, WINDOW)
            .unwrap_or_else(|e| panic!("verify round: {e}"));
        let run = &results[0];
        assert!(
            !run.tokens.is_empty() && run.tokens.len() <= run.extent as usize + 1,
            "a run of {} tokens at extent {}",
            run.tokens.len(),
            run.extent
        );
        rounds.push((run.extent, run.tokens.len()));
        let stopped = run.tokens.iter().any(|t| stop_ids.contains(t));
        emitted.extend(run.tokens.iter().copied());
        if stopped {
            break;
        }
        assert!(rounds.len() <= TOTAL, "the run does not converge");
    }
    SpecRun { emitted, rounds }
}

/// The final logits after `prompt` prefilled at `chunk` plus `agreed` fed as
/// text, which is how teacher forcing already continues a multimodal prompt.
fn gap_after(
    model: &Model,
    pool: &SeqPool,
    prompt: &Prompt,
    embedding: &MediaEmbedding<'_>,
    agreed: &[i32],
    chunk: u32,
    a: i32,
    b: i32,
) -> f32 {
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut logits = vec![0f32; vocab];
    let mut sequence = prefilled(model, pool, prompt, embedding, chunk, &mut logits);
    let mut position = prompt.tokens.len() as u64;
    for &token in agreed {
        step::prefill_program(model, pool, &mut sequence, &[token], position, Some(&mut logits))
            .unwrap_or_else(|e| panic!("near-tie probe at position {position}: {e}"));
        position += 1;
    }
    logits[a as usize] - logits[b as usize]
}

/// Spec-off's pick `a` against spec-on's pick `b` after `agreed`, by the
/// engine's own two multimodal prefill routes (the whole span, and 48-token
/// spans that cut the placeholder run). A near-tie when the routes disagree
/// about the winner or rate the gap no wider than the spread between them;
/// spec-off's own drift when both prefer `b`. Panics naming both gaps
/// otherwise. The tolerance is never a constant
/// (`crates/core/tests/support/near_tie.rs`).
#[allow(clippy::too_many_arguments)]
fn assert_near_tie(
    model: &Model,
    pool: &SeqPool,
    prompt: &Prompt,
    embedding: &MediaEmbedding<'_>,
    agreed: &[i32],
    a: i32,
    b: i32,
    what: &str,
) {
    let whole = gap_after(model, pool, prompt, embedding, agreed, PREFILL_CHUNK, a, b);
    let spanning = gap_after(model, pool, prompt, embedding, agreed, SPANNING_CHUNK, a, b);
    let routes_disagree = (whole > 0.0) != (spanning > 0.0) || whole == 0.0 || spanning == 0.0;
    let within_spread = whole.abs().min(spanning.abs()) <= (whole - spanning).abs();
    let routes_side_with_spec_on = whole < 0.0 && spanning < 0.0;
    assert!(
        routes_disagree || within_spread || routes_side_with_spec_on,
        "{what}: the verify round picked {b} where the decode round picked {a} after {} tokens, it is not a \
         near-tie, and both multimodal prefill routes side with the decode round: logit[{a}] - logit[{b}] = \
         {whole} (whole span), {spanning} (spanning spans) -- the verify round is rotating at the wrong \
         positions",
        agreed.len()
    );
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_drafter_follows_an_image_prompt_and_its_text_stays_the_decode_rounds_text() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let processor = frontend.vision_processor().unwrap_or_else(|e| panic!("vision processor: {e}"));
    let eos = frontend.eos_token_id().expect("eos");
    let stop_ids: Vec<i32> = [eos]
        .into_iter()
        .chain(frontend.tokenizer().encode("<|im_end|>").expect("tokenize the turn marker"))
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    let canaries = load_canaries(&frontend);
    assert!(!canaries.is_empty());

    // --- AC 1: the two load options together -----------------------------
    let (plan, handles) =
        bind_model_scope_27b_with(&reader, ModelScope { draft: Some(DraftModule::Dflash2), vision: true })
            .unwrap_or_else(|e| panic!("bind with the drafter and vision: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(a) => a,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + dflash2 + vision: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, WINDOW).expect("dflash2-7");
    let model = load_qwen38_27b_with_options(
        &reader,
        &artifact,
        &handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::Bf16,
        Some(speculation),
        Some(Vision::default()),
    )
    .unwrap_or_else(|e| panic!("ignis_model_load with vision and dflash2: {e}"));
    let stats = model.stats();
    assert!(
        stats.vision_reserved_bytes > 0,
        "a vision load must still report its reservation with the drafter loaded beside it"
    );
    eprintln!(
        "vision + dflash2 load: {} bound tensors, {:.2} GiB, vision reservation {:.2} MiB",
        stats.bound_tensor_count,
        stats.vram_bytes as f64 / (1u64 << 30) as f64,
        stats.vision_reserved_bytes as f64 / (1u64 << 20) as f64
    );

    let pool = drafter_pool(1);
    // Both rounds run through their captured graphs, which bake the address
    // the rotation positions are staged at (ADR 0019/0020).
    step::capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    let program = step::program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
    assert!(program.decode_graph_ready_mask & 1 != 0, "no width-1 decode graph");
    assert!(
        program.verify_graph_ready_mask & 1 != 0,
        "no width-1 verify graph with the drafter in it (mask {:#010b})",
        program.verify_graph_ready_mask
    );
    let vocab = ModelConfig::qwen38_27b().vocab as usize;

    let mut results: Vec<TeacherForcedResult> = Vec::new();
    let mut committed_full = 0usize;
    let mut full_rounds = 0usize;
    let mut accepted = 0usize;
    for canary in &canaries {
        let prompt = prompt_for(&frontend, &processor, canary, &canary.question);
        let item = &prompt.multimodal.media[0];
        eprintln!(
            "vision+dflash2 canary {}: {} prompt tokens, image {}x{} patches ({} merged), rope_delta {}",
            canary.id,
            prompt.tokens.len(),
            item.grid.h,
            item.grid.w,
            item.token_span.count,
            prompt.multimodal.rope_delta
        );
        let control = vision_item_control(item.grid);
        let embedding = step::encode_media(&model, item.grid, &item.patches, &control)
            .unwrap_or_else(|e| panic!("{}: encode: {e}", canary.id));
        assert_eq!(embedding.columns() as usize, item.token_span.count);
        let mut logits = vec![0f32; vocab];

        // --- AC 2: ADR 0014's floor, on this load ------------------------
        {
            let mut sequence = prefilled(&model, &pool, &prompt, &embedding, PREFILL_CHUNK, &mut logits);
            let compared = FIRST_N.min(canary.expected.len());
            let mut predictions = Vec::with_capacity(compared);
            let mut position = prompt.tokens.len() as u64;
            for i in 0..compared {
                predictions.push(argmax_lowest_id(&logits));
                let forced = [canary.expected[i] as i32];
                step::prefill_program(&model, &pool, &mut sequence, &forced, position, Some(&mut logits))
                    .unwrap_or_else(|e| panic!("{}: forced position {i}: {e}", canary.id));
                position += 1;
            }
            let result = score_teacher_forced(&canary.id, &canary.expected, &predictions, FIRST_N);
            for m in &result.mismatches {
                eprintln!(
                    "vision+dflash2 canary {} position {}: ours={:?} reference={}",
                    result.id, m.position, m.predicted, m.expected
                );
            }
            eprintln!(
                "vision+dflash2 canary {}: teacher-forced agreement {}/{} = {:.1}%",
                result.id,
                result.agree,
                result.compared,
                result.agreement * 100.0
            );
            results.push(result);
        }

        // --- AC 3 + AC 4: the drafter's text, and its acceptance ---------
        // Two prompts over the same image and the same encoded columns: the
        // fixture's own question, whose answer is one sentence, and a request
        // for a long description, so the acceptance figure rests on more than
        // the two or three rounds a one-sentence answer affords.
        let detail = prompt_for(&frontend, &processor, canary, DETAIL_QUESTION);
        assert_eq!(
            detail.multimodal.media[0].grid, item.grid,
            "{}: the second question must resize the image the same way, or its scatter would index \
             columns this embedding does not hold",
            canary.id
        );
        for (label, prompt) in [(canary.id.clone(), &prompt), (format!("{}/detail", canary.id), &detail)] {
            assert!(
                prompt.tokens.len() + TOTAL <= MAX_CONTEXT as usize,
                "{label}: the prompt and its generation outgrow the context"
            );
            let off = spec_off(&model, &pool, prompt, &embedding, &stop_ids, &mut logits);
            let on = spec_on(&model, &pool, prompt, &embedding, &stop_ids, &mut logits);
            let divergence = off
                .iter()
                .zip(&on.emitted)
                .position(|(x, y)| x != y)
                .or_else(|| (off.len() != on.emitted.len()).then(|| off.len().min(on.emitted.len())));
            match divergence {
                None => eprintln!(
                    "vision+dflash2 {label}: the verify rounds' text equals the decode rounds' over {} tokens",
                    off.len()
                ),
                Some(d) => {
                    // A length-only divergence is not a near-tie and the probe
                    // cannot adjudicate one: there is no pair of candidates to
                    // weigh. It is also not reachable through the model -- both
                    // streams share the stop set and the same TOTAL budget, so
                    // equal prefixes end at equal lengths. Reaching it means the
                    // round bookkeeping is wrong, which is a failure outright.
                    assert!(
                        d < off.len() && d < on.emitted.len(),
                        "{label}: the two streams agree on every token but end at different lengths \
                         (decode rounds {} tokens, verify rounds {}) -- with one stop set and one budget \
                         that cannot happen, so the round accounting is wrong",
                        off.len(),
                        on.emitted.len()
                    );
                    eprintln!(
                        "vision+dflash2 {label}: the two rounds part at token {d} ({} vs {}) -- probing",
                        off[d], on.emitted[d]
                    );
                    assert_near_tie(&model, &pool, prompt, &embedding, &off[..d], off[d], on.emitted[d], &label);
                }
            }
            for (index, &(extent, committed)) in on.rounds.iter().enumerate() {
                eprintln!("vision+dflash2 {label} round {index}: accepted/drafted {}/{extent}", committed - 1);
                accepted += committed - 1;
                if extent == WINDOW {
                    committed_full += committed;
                    full_rounds += 1;
                }
            }
        }
        drop(embedding);
    }

    let overall = overall_teacher_forced_agreement(&results);
    let agree: usize = results.iter().map(|r| r.agree).sum();
    let compared: usize = results.iter().map(|r| r.compared).sum();
    eprintln!(
        "vision+dflash2 canary OVERALL teacher-forced agreement {agree}/{compared} = {:.1}% (floor {:.0}%)",
        overall * 100.0,
        G1_AGREEMENT_FLOOR * 100.0
    );
    // AC 4 is a report, so it must never read as a measurement it did not
    // make: no full-window round is no data, not an acceptance of zero.
    if full_rounds == 0 {
        eprintln!(
            "vision+dflash2: NOT MEASURED -- no round reached the full {WINDOW}-token window, so \
             there is no acceptance to compare with the reference's band"
        );
    } else {
        let mean = committed_full as f64 / full_rounds as f64;
        eprintln!(
            "vision+dflash2: {mean:.2} committed tokens per full-window round over {full_rounds} rounds \
             (reference band {:.2}-{:.2} at draft {WINDOW})",
            REFERENCE_BAND.0, REFERENCE_BAND.1
        );
        if mean < REFERENCE_BAND.0 {
            eprintln!(
                "vision+dflash2: FINDING -- acceptance on the text after an image is below the reference's band"
            );
        }
    }

    drop(pool);
    drop(model);
    let _ = artifact.release_arena(&mut device);

    assert!(compared > 0, "the fixture must contribute scored positions");
    // A generation budgeted to TOTAL tokens opens at extent `WINDOW`, so no
    // full-window round at all means the rounds never ran as this test
    // intends and AC 4 went unmeasured -- which must fail, not pass quietly.
    assert!(
        full_rounds > 0,
        "no round reached the full {WINDOW}-token window: the acceptance on the text after an \
         image was never measured"
    );
    assert!(
        meets_g1_floor(overall),
        "vision canary on a DFlash2 load: teacher-forced agreement {agree}/{compared} = {:.1}% < {:.0}%",
        overall * 100.0,
        G1_AGREEMENT_FLOOR * 100.0
    );
    assert!(
        accepted > 0,
        "the drafter never landed a draft after an image: it is not proposing from its window"
    );
}
