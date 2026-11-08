//! DFlash2 on a multimodal prefill span (GitHub #195, spec
//! `docs/specs/vision/01-image-input.md` §Compute seam, ADR 0014).
//!
//! GitHub #178 fenced `--vision-enabled` and `--spec-backend dflash2` apart at load. What
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
//! - AC 1: `--vision-enabled` with `--spec-backend dflash2` loads -- the tower and the
//!   drafter bind together and the vision reservation is still reported.
//! - AC 2: the vision canary's teacher-forced floor (ADR 0014, >= 95%) holds
//!   on this load, so the drafter riding the multimodal prefill does not
//!   disturb the target's own next-token agreement with the reference.
//! - AC 3: greedy spec-on equals spec-off on the canary images. Spec-off here
//!   is a plain decode round *on this same load*, which is the sharpest form
//!   of the question: the two rounds differ only in shape, so a rotation the
//!   verify round got wrong shows up immediately. A split passes only as a
//!   near-tie judged by the two **round shapes** (GitHub #534, glossary
//!   Near-tie and Round shape, `support/round_shapes.rs`), each computed
//!   outside the rounds through the prefill at the position whose logits drew
//!   the split token:
//!   - the *decode shape*: the prompt prefilled as both runs prefill it, then
//!     the agreed text one token at a time, a decode round's one column;
//!   - the *verify shape*: the speculative run's own state at the start of
//!     the round that drew the split token, its snapshot restored into a
//!     fresh sequence, then that round computed again, independently, in its
//!     own shape: one span `WINDOW + 1` columns wide from its base over the
//!     decode text, read at the split's row. The verify round's width, so its
//!     activation routes (ADR 0016), and none of its graph, staging or
//!     drafter.
//!
//!   The split is numerics when the two shapes disagree about the winner, when
//!   the verify shape ranks the verify's pick at least as high, or when either
//!   shape is within one BF16 ulp of a tie at its larger logit's magnitude
//!   (the computation's own rounding, never a tuned constant). It is a fault of
//!   the verify round only when the verify shape sides with the decode round
//!   by more than that. Each split prints its round, row, position, both picks,
//!   both shapes' gaps and ulps, and the verdict. What the verify shape gives
//!   up: the rounds before the split's are the run's own computation, restored,
//!   not rebuilt independently; only the split's round is (`verify_shape`). The
//!   text rule of `crates/core/tests/support/near_tie.rs` is not this one.
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
#[path = "support/round_shapes.rs"]
mod round_shapes;

use std::path::Path;

use ignis_artifact::vision::VisionProcessor;
use ignis_artifact::{
    bind_model_scope_27b_with, materialize, ChatMessage, ChatRenderOptions, ContentPart, CudaDevice,
    FrontendSet, MessageContent, VisionResidency, Reader, Role,
};
use ignis_bench::oracle::{
    meets_g1_floor, overall_teacher_forced_agreement, score_teacher_forced, TeacherForcedResult,
    G1_AGREEMENT_FLOOR,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_options, model_scope, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    self, decode_program_batch_sampled, decode_program_verify_runs, MediaEmbedding, SamplingParams,
    VerifyLane,
};
use ignis_core::vision::{vision_item_control, Multimodal};
use ignis_core::{ExceptionWeights, KvFormat, RopeScaling, Speculation, SpeculativeBackend, Vision};

use round_shapes::{bf16, judge, span_tokens, Shape, Verdict, VerifyRound};
use support::vision_canary::{argmax_lowest_id, load_canaries, prefill_prompt, Canary};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 2048;
const PREFILL_CHUNK: u32 = 1024;
/// DFlash2-7, the reference lane G5 measures.
const WINDOW: u32 = 7;
/// A verify round's columns per lane, its round shape: `WINDOW + 1`
/// whatever the round's extent, the columns past it padded with the anchor
/// (`lane_columns` in `run_verify_round`, `kernel/src/step.cu`).
const VERIFY_COLUMNS: u32 = WINDOW + 1;
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
            retained_slot_count: 0,
            retained_host_slot_count: 0,
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
    // GitHub #185 folded the render flags into `ChatRenderOptions`. These must
    // stay the options `vision_canary_gpu.rs` renders with, because the
    // fixture this test scores against was recorded with thinking off and a
    // different prompt would move the agreement without failing to compile:
    // `enable_thinking: false` (the fixture's own condition), and the two
    // defaults `reasoning_effort: None` and `preserve_thinking: false`, which
    // are what the pre-#185 `false, None, None` rendered.
    let prepared = frontend
        .prepare_prompt(
            processor,
            &messages,
            &[&canary.image],
            ChatRenderOptions { enable_thinking: false, ..Default::default() },
            None,
        )
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

/// What one spec-on run recorded: the text, per round its base (the
/// sequence's own frontier before the round, where its anchor sits), its
/// extent and the tokens it committed (its `LaneVerifyRun`), and the state the
/// verify-shape probe starts from.
struct SpecRun {
    emitted: Vec<i32>,
    rounds: Vec<VerifyRound>,
    departed: Option<Departed>,
}

/// The snapshot blob of a spec-on sequence at the base of `rounds[round]`,
/// the round whose columns drew the first token departing from the decode
/// rounds' text ([`VerifyRound::drew_the_departure`]): the KV, GDN state and
/// drafter window the verify rounds before it left. A restore is checked
/// against the pool it writes into, so it restores into a sequence of the
/// pool the run drew from, the one pool this test has.
struct Departed {
    round: usize,
    snapshot: Vec<u8>,
}

/// The same `total` tokens through verify rounds, the drafter proposing from
/// its window. One lane: the question is the rotation, not the batching.
///
/// Until a round has drawn a token departing from `off`, the decode rounds'
/// text, each round's base is snapshotted, once the previous round's commit
/// has completed, and the snapshot kept only if the round drew the departure
/// (`Departed`): one blob at a time. A round's last column draws its
/// correction, the sequence's pending token, which the run commits only as
/// the next round's anchor, so the round's draws are compared with the text it
/// committed and its pending token after it.
fn spec_on(
    model: &Model,
    pool: &SeqPool,
    prompt: &Prompt,
    embedding: &MediaEmbedding<'_>,
    off: &[i32],
    stop_ids: &[i32],
    logits: &mut [f32],
) -> SpecRun {
    let mut sequence = prefilled(model, pool, prompt, embedding, PREFILL_CHUNK, logits);
    let first = prompt.tokens.len() as u64;
    let mut emitted: Vec<i32> = Vec::with_capacity(TOTAL);
    let mut rounds = Vec::new();
    let mut departed = None;
    while emitted.len() < TOTAL {
        let base = sequence.stats().position;
        assert_eq!(
            base,
            (prompt.tokens.len() + emitted.len()) as u64,
            "a round's anchor must sit right after the prompt and the text committed before it"
        );
        let snapshot = departed.is_none().then(|| {
            sequence
                .snapshot()
                .unwrap_or_else(|e| panic!("snapshot at round {}'s base {base}: {e}", rounds.len()))
        });
        let anchor = sequence.pending_token();
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
        assert_eq!(
            Some(run.tokens[0]),
            anchor,
            "a round commits first its anchor, the pending token the round before it drew"
        );
        let round = VerifyRound { base, extent: run.extent, committed: run.tokens.len() };
        rounds.push(round);
        let stopped = run.tokens.iter().any(|t| stop_ids.contains(t));
        emitted.extend(run.tokens.iter().copied());
        if let Some(snapshot) = snapshot {
            let known: Vec<i32> = emitted.iter().copied().chain(sequence.pending_token()).collect();
            if round.drew_the_departure(first, off, &known) {
                departed = Some(Departed { round: rounds.len() - 1, snapshot });
            }
        }
        if stopped {
            break;
        }
        assert!(rounds.len() <= TOTAL, "the run does not converge");
    }
    SpecRun { emitted, rounds, departed }
}

/// The decode shape: `prompt` prefilled as both runs prefill it, then
/// `agreed` fed one token at a time, each a one-column span -- a decode
/// round's shape -- which is how teacher forcing already continues a
/// multimodal prompt; read at the last one, `a` (the decode round's pick)
/// against `b` (the verify round's).
fn decode_shape(
    model: &Model,
    pool: &SeqPool,
    prompt: &Prompt,
    embedding: &MediaEmbedding<'_>,
    agreed: &[i32],
    a: i32,
    b: i32,
) -> Shape {
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut logits = vec![0f32; vocab];
    let mut sequence = prefilled(model, pool, prompt, embedding, PREFILL_CHUNK, &mut logits);
    let mut position = prompt.tokens.len() as u64;
    for &token in agreed {
        step::prefill_program(model, pool, &mut sequence, &[token], position, Some(&mut logits))
            .unwrap_or_else(|e| panic!("decode-shape probe at position {position}: {e}"));
        position += 1;
    }
    Shape { decode_pick: logits[a as usize], verify_pick: logits[b as usize] }
}

/// The verify shape: the speculative run's own state at the start of the
/// round that drew the split token -- its snapshot (`departed`), restored into
/// a fresh sequence of `pool`, the pool the run drew from -- then that round
/// computed again, independently, in its own shape: one span
/// [`VERIFY_COLUMNS`] wide from its `base` over the decode `text` (`text[0]`
/// at `first`, the prompt's end), read at `row`; `a` (the decode round's
/// pick) against `b` (the verify round's). The span takes the verify round's
/// activation routes (ADR 0016) through the prefill, and the probe runs none
/// of the round's graph, staging or drafter; the diagnosis of #531 found such
/// a span's row equal to the verify column's logits at bf16 on both GDN
/// prefill routes.
///
/// The span's columns up to the split's row hold the round's anchor and the
/// drafts it accepted, the text both runs agree on. Past the row they hold
/// the decode text, and past its end its last token, where the verify round
/// held its own drafts and, past its extent, its anchor again: no row reads a
/// later column (the attention and the recurrent GDN pass are causal, and the
/// W4A4 activations quantize at each weight's static input scale, not a
/// per-call maximum), so the row read is the same whatever fills them.
///
/// The history is the run's own: the KV and GDN state the earlier verify
/// rounds left, each computed `VERIFY_COLUMNS` wide with only its committed
/// columns kept. What this gives up: that computation is restored, not
/// rebuilt independently; only the split's round is recomputed.
#[allow(clippy::too_many_arguments)]
fn verify_shape(
    model: &Model,
    pool: &SeqPool,
    departed: &Departed,
    base: u64,
    text: &[i32],
    first: u64,
    row: usize,
    a: i32,
    b: i32,
) -> Shape {
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
    sequence.restore(&departed.snapshot).unwrap_or_else(|e| {
        panic!("verify-shape probe, restoring round {}'s snapshot: {e}", departed.round)
    });
    assert_eq!(
        sequence.stats().position,
        base,
        "the snapshot of round {} must restore to the round's base",
        departed.round
    );
    let span = base..base + u64::from(VERIFY_COLUMNS);
    let columns = span_tokens(text, first, span.clone());
    let mut rows = vec![0u16; columns.len() * vocab];
    step::prefill_program_span_logits(model, pool, &mut sequence, &columns, base, &mut rows)
        .unwrap_or_else(|e| panic!("verify-shape probe, the split's round span {span:?}: {e}"));
    let row = &rows[row * vocab..(row + 1) * vocab];
    Shape { decode_pick: bf16(row[a as usize]), verify_pick: bf16(row[b as usize]) }
}

/// The split at token `split` -- the decode rounds' `off[split]` against the
/// verify rounds' `on.emitted[split]`, after the text both agree on -- judged
/// by the two round shapes ([`judge`]), the verify shape starting from `on`'s
/// own state at the base of the round that drew the split. Prints the
/// judgement, then panics on a fault of the verify round, naming both shapes'
/// gaps.
#[allow(clippy::too_many_arguments)]
fn assert_near_tie(
    model: &Model,
    pool: &SeqPool,
    prompt: &Prompt,
    embedding: &MediaEmbedding<'_>,
    off: &[i32],
    on: &SpecRun,
    split: usize,
    label: &str,
) {
    let (a, b) = (off[split], on.emitted[split]);
    let first = prompt.tokens.len() as u64;
    // The position whose logits drew the split token: the prompt's last for
    // the first token, which the prefill draws in both runs alike.
    let drawn_at = first + split as u64 - 1;
    let (departed, row) = on
        .departed
        .as_ref()
        .and_then(|departed| Some((departed, on.rounds[departed.round].row(first, split)?)))
        .unwrap_or_else(|| {
            panic!(
                "{label}: the two runs part at token {split} ({a} vs {b}), drawn at position {drawn_at}, and the \
                 run kept no snapshot of a round that drew it -- token 0 is the prompt's own draw, one \
                 computation in both runs, and a later token no round drew means the round accounting is wrong"
            )
        });
    let base = on.rounds[departed.round].base;
    let decode = decode_shape(model, pool, prompt, embedding, &off[..split], a, b);
    let verify = verify_shape(model, pool, departed, base, off, first, row, a, b);
    let verdict = judge(decode, verify);
    let judgement = format!(
        "{label}: round {} row {row} (position {drawn_at}, token {split}), decode picked {a}, verify picked {b}: \
         logit[{a}] - logit[{b}] = {:+} in the decode shape ({} / {}, ulp {}), {:+} in the verify shape \
         ({} / {}, ulp {}) -- {verdict}",
        departed.round,
        decode.gap(),
        decode.decode_pick,
        decode.verify_pick,
        decode.ulp(),
        verify.gap(),
        verify.decode_pick,
        verify.verify_pick,
        verify.ulp(),
    );
    eprintln!("vision+dflash2 near-tie {judgement}");
    assert!(
        verdict != Verdict::VerifyFault,
        "{judgement}: the verify shape, computed outside the round, sides with the decode round beyond the \
         computation's rounding -- the verify round computed this position wrong"
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
    // Today's load, the tower on the device (ADR 0075's `device`): the scope
    // is bound from the vision the model loads with, so the two agree.
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, WINDOW).expect("dflash2-7");
    let vision = Vision::default().with_residency(VisionResidency::Device);
    let (plan, handles) =
        bind_model_scope_27b_with(&reader, model_scope(Some(speculation), Some(vision), ExceptionWeights::Bf16))
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
    let model = load_qwen38_27b_with_options(
        &reader,
        &artifact,
        &handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::Bf16,
        Some(speculation),
        Some(vision),
        RopeScaling::NONE,
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
            .unwrap_or_else(|(_, e)| panic!("{}: encode: {e}", canary.id));
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
                prompt.tokens.len() + TOTAL + VERIFY_COLUMNS as usize <= MAX_CONTEXT as usize,
                "{label}: the prompt, its generation and the verify-shape probe's span outgrow the context"
            );
            let off = spec_off(&model, &pool, prompt, &embedding, &stop_ids, &mut logits);
            let on = spec_on(&model, &pool, prompt, &embedding, &off, &stop_ids, &mut logits);
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
                    assert_near_tie(&model, &pool, prompt, &embedding, &off, &on, d, &label);
                }
            }
            for (index, round) in on.rounds.iter().enumerate() {
                eprintln!(
                    "vision+dflash2 {label} round {index}: accepted/drafted {}/{}",
                    round.committed - 1,
                    round.extent
                );
                accepted += round.committed - 1;
                if round.extent == WINDOW {
                    committed_full += round.committed;
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
