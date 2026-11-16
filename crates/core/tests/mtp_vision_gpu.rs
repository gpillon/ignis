//! GPU tests for the 27B's MTP head beside vision (GitHub #624, spec #618):
//! on a load with both, the head's state after an image is the head's own
//! weights over what the target saw, in double precision.
//!
//! - A prefill over an image prompt writes the head's row at every position
//!   it licenses, the image's included. A row whose next token is an image
//!   placeholder embeds the image's column the target's embedding took there
//!   (Flash-Next's rule, GitHub #564), and every row rotates where the
//!   target's hidden at its position did: at that position's three MRoPE
//!   axes. The rows are held to `ignis_artifact::f64_reference::
//!   evaluate_mtp_head_at_axes` from the exact BF16 inputs the device read
//!   (the target's hiddens through the span readout, the image's columns read
//!   back, each text token's embedding rounded as the device rounds it); the
//!   same rows built from the placeholder ids' embeddings, or rotated at their
//!   KV positions, land far outside the bound, so the check tells the rule
//!   apart from its two likely mistakes.
//! - The prompt prefilled in one span per chunk, as the scheduler sends it,
//!   leaves the head's rows of the prompt prefilled as one span, bit for bit:
//!   each span's first row, at the position before it, rotates at the axes the
//!   caller passes with the span (`mrope_previous`), and those are the axes the
//!   one span's own positions hold there. The prompt's chunk boundary falls
//!   inside the image.
//! - The head's first draft at the frontier is held to the f64 head over every
//!   row, its query rotated at the frontier plus the sequence's rope delta;
//!   the round's chain, which rotates at the verify round's own rotation
//!   positions, scores its first step as the f64 hidden scores. The first
//!   verify round after the image drafts the whole window, and the rows its
//!   commit writes, and the row a plain round writes after it, rotate at their
//!   positions plus the rope delta, as the target's decode and verify columns
//!   do: held to f64 again, with a rotation at the bare positions as the
//!   control.
//! - Under the host residency (the tower staged, the media pool host-mapped)
//!   the same prefill leaves the same rows, bit for bit, as under the device
//!   residency: the head reads the image's columns wherever the pool lives.
//!
//! BF16 KV throughout: it is the correctness oracle (ADR 0022).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

#[path = "support/bf16.rs"]
mod bf16;
#[path = "support/mtp_bounds.rs"]
mod mtp_bounds;

use std::path::{Path, PathBuf};

use ignis_artifact::{
    bind_model_scope_27b_with, f64_reference, materialize, ChatMessage, ChatRenderOptions, ContentPart, CudaDevice,
    FrontendSet, MaterializedArtifact, MessageContent, Reader, Role, VisionResidency,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_options, model_scope, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    self, decode_program_batch, decode_program_verify_runs, prefill_program_multimodal_span_hidden, MediaEmbedding,
    MultimodalPrefill, SpanMediaColumns, VerifyLane,
};
use ignis_core::vision::{vision_item_control, Multimodal};
use ignis_core::{mtp_readout, ExceptionWeights, KvFormat, RopeScaling, Speculation, SpeculativeBackend, Vision};

use bf16::{bf16_to_f64, f32_to_bf16};
use mtp_bounds::{tolerance, HIDDEN_ROUNDINGS, KEY_ROUNDINGS, SCORE_ROUNDINGS, VALUE_ROUNDINGS};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 1024;
/// The load's chunk: number.png's 196 placeholders start a few tokens into
/// the prompt, so the chunk boundary at 128 falls inside the image, and a
/// chunk is one head call.
const CHUNK: u32 = 128;
const WINDOW: u32 = 7;
const HIDDEN: usize = 5120;
const VOCAB: usize = 248_320;

/// A control lands at least this many bounds out: a wrong embedding or
/// rotation moves a row by its own size, not by a rounding.
const CONTROL_FACTOR: f64 = 10.0;

fn relative_l2(device: impl IntoIterator<Item = f64>, host: impl IntoIterator<Item = f64>) -> f64 {
    let (mut error, mut reference) = (0.0, 0.0);
    for (d, h) in device.into_iter().zip(host) {
        error += (d - h) * (d - h);
        reference += h * h;
    }
    error.sqrt() / reference.sqrt().max(1e-30)
}

/// The device's embedding of `token`, rounded to BF16 as the op writes it.
fn embedding(reader: &Reader, token: i32) -> Vec<f64> {
    f64_reference::matrix_row(reader, "text/token_embedding", VOCAB, HIDDEN, token as usize)
        .unwrap_or_else(|e| panic!("embedding row {token}: {e}"))
        .into_iter()
        .map(|value| bf16_to_f64(f32_to_bf16(value as f32)))
        .collect()
}

fn column(bits: &[u16], at: usize) -> Vec<f64> {
    bits[at * HIDDEN..(at + 1) * HIDDEN].iter().copied().map(bf16_to_f64).collect()
}

fn image(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../server/tests/fixtures/vision_canary").join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

struct Loaded {
    reader: Reader,
    device: CudaDevice,
    artifact: MaterializedArtifact,
    model: Model,
}

fn spec() -> Speculation {
    Speculation::new(SpeculativeBackend::Mtp, WINDOW).expect("mtp-7")
}

/// The 27B with its MTP head and vision under `residency`, or `None` where
/// the profile allows a skip.
fn load(residency: VisionResidency) -> Option<Loaded> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the artifact is absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let vision = Vision::default().with_residency(residency);
    let (plan, handles) =
        bind_model_scope_27b_with(&reader, model_scope(Some(spec()), Some(vision), ExceptionWeights::Bf16))
            .unwrap_or_else(|e| panic!("bind the head and vision: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + mtp + vision: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b_with_options(
        &reader,
        &artifact,
        &handles,
        CHUNK,
        MAX_CONTEXT,
        KvFormat::Bf16,
        Some(spec()),
        Some(vision),
        RopeScaling::NONE,
    )
    .unwrap_or_else(|e| panic!("load the head beside vision ({residency:?}): {e}"));
    assert!(model.stats().vision_reserved_bytes > 0, "{residency:?}: the load reserves vision beside the head");
    Some(Loaded { reader, device, artifact, model })
}

fn pool() -> SeqPool {
    SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: 2 * MAX_CONTEXT.div_ceil(64),
            max_context_tokens: MAX_CONTEXT,
            slot_count: 2,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
        Some(SpeculativeBackend::Mtp),
    )
    .unwrap_or_else(|e| panic!("an MTP pool: {e}"))
}

/// number.png with its question, thinking off: tokens and the multimodal part.
fn prompt(reader: &Reader) -> (Vec<i32>, Multimodal) {
    let frontend = FrontendSet::from_reader(reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let processor = frontend.vision_processor().unwrap_or_else(|e| panic!("vision processor: {e}"));
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Parts(vec![
            ContentPart::Image { url: None },
            ContentPart::Text("What number is shown in the image?".to_owned()),
        ]),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let number = image("number.png");
    let prepared = frontend
        .prepare_prompt(&processor, &messages, &[&number], ChatRenderOptions { enable_thinking: false, ..Default::default() }, None)
        .unwrap_or_else(|e| panic!("prepare the prompt: {e}"));
    let (tokens, multimodal) = Multimodal::from_prepared(prepared);
    assert_eq!(multimodal.media.len(), 1);
    assert_ne!(multimodal.rope_delta, 0, "an image whose delta is 0 cannot tell the rotations apart");
    (tokens.iter().map(|&t| t as i32).collect(), multimodal)
}

/// Prefill `tokens` into `sequence` in spans of at most `span` tokens (each
/// capped where a media item would begin, as the scheduler cuts them), each
/// with the axes before it, reading the target's hidden of every position.
#[allow(clippy::too_many_arguments)]
fn prefill(
    model: &Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
    tokens: &[i32],
    multimodal: &Multimodal,
    embedding: &MediaEmbedding<'_>,
    span: u32,
    hidden_bits: &mut [u16],
) {
    let total = tokens.len() as u32;
    let mut start = 0u32;
    while start < total {
        let len = multimodal.cap_chunk(start, span.min(total - start));
        let end = (start + len) as usize;
        let positions = multimodal.span_positions(start as usize, len as usize);
        let previous = multimodal.axes_before(start as usize);
        let media = multimodal.chunk_media(start, len);
        let next = (end < tokens.len()).then(|| tokens[end] as u32);
        prefill_program_multimodal_span_hidden(
            model,
            pool,
            sequence,
            &tokens[start as usize..end],
            u64::from(start),
            next,
            MultimodalPrefill {
                positions: &positions,
                rope_delta: multimodal.rope_delta,
                media: media.as_ref().map(|media| SpanMediaColumns {
                    embedding,
                    first_column: media.first_column,
                    scatter_indices: &media.scatter_indices,
                }),
                previous: previous.as_ref(),
            },
            &mut hidden_bits[start as usize * HIDDEN..end * HIDDEN],
            None,
        )
        .unwrap_or_else(|e| panic!("prefill the span at {start} ({len} tokens): {e}"));
        start += len;
    }
}

/// What a row at prompt position `p` is built from.
struct RowInputs {
    hidden: Vec<f64>,
    embedding: Vec<f64>,
    axes: [i32; 3],
}

/// The f64 head's keys and values of `rows` against the device's rows at
/// `positions` of `sequence`: (key drift, value drift).
fn drift(
    reader: &Reader,
    pool: &SeqPool,
    sequence: &Seq<'_>,
    positions: &[u32],
    rows: &[RowInputs],
) -> (f64, f64) {
    let reference = f64_reference::evaluate_mtp_head_at_axes(
        reader,
        &rows.iter().map(|r| r.hidden.clone()).collect::<Vec<_>>(),
        &rows.iter().map(|r| r.embedding.clone()).collect::<Vec<_>>(),
        &rows.iter().map(|r| r.axes).collect::<Vec<_>>(),
    )
    .unwrap_or_else(|e| panic!("f64 head: {e}"));
    let (mut keys, mut values, mut want_keys, mut want_values) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, &p) in positions.iter().enumerate() {
        let (k, v) = mtp_readout::head_state(pool, sequence, p, 1).unwrap_or_else(|e| panic!("head state at {p}: {e}"));
        keys.extend(k.into_iter().map(bf16_to_f64));
        values.extend(v.into_iter().map(bf16_to_f64));
        want_keys.extend_from_slice(&reference.keys[i]);
        want_values.extend_from_slice(&reference.values[i]);
    }
    (relative_l2(keys, want_keys), relative_l2(values, want_values))
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn after_an_image_the_heads_rows_and_first_draft_are_its_weights_in_double_precision_and_the_first_round_drafts() {
    let Some(loaded) = load(VisionResidency::Device) else {
        return;
    };
    let reader = &loaded.reader;
    let model = &loaded.model;
    let pool = pool();
    let (tokens, multimodal) = prompt(reader);
    let n = tokens.len();
    let item = &multimodal.media[0];
    let (image_begin, image_count) = (item.token_span.begin, item.token_span.count);
    let image_end = image_begin + image_count;
    assert!(
        (image_begin..image_end).contains(&(CHUNK as usize)) && n < MAX_CONTEXT as usize - 2 * WINDOW as usize,
        "the chunk boundary falls inside the image: image {image_begin}..{image_end}, prompt {n}"
    );
    let item_control = vision_item_control(item.grid);
    let media =
        step::encode_media(model, item.grid, &item.patches, &item_control).unwrap_or_else(|(_, e)| panic!("encode: {e}"));
    let media_bits = media.read_bits(HIDDEN as u64).unwrap_or_else(|e| panic!("read the image's columns: {e}"));
    let delta = multimodal.rope_delta;
    eprintln!("prompt {n} tokens, image {image_begin}..{image_end}, rope delta {delta}");

    // The prompt in one span a chunk, as the scheduler sends it, and as one
    // span the leaf cuts into the same chunks.
    let mut spans = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    let mut whole = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    let mut hidden_bits = vec![0u16; n * HIDDEN];
    let mut whole_bits = vec![0u16; n * HIDDEN];
    prefill(model, &pool, &mut spans, &tokens, &multimodal, &media, CHUNK, &mut hidden_bits);
    prefill(model, &pool, &mut whole, &tokens, &multimodal, &media, n as u32, &mut whole_bits);
    assert!(hidden_bits == whole_bits, "the target's hiddens are the same either way");
    let (span_keys, span_values) = mtp_readout::head_state(&pool, &spans, 0, n as u32 - 1).expect("head state");
    let (whole_keys, whole_values) = mtp_readout::head_state(&pool, &whole, 0, n as u32 - 1).expect("head state");
    assert!(
        span_keys == whole_keys && span_values == whole_values,
        "the rows of the prompt in one span a chunk are the one span's, bit for bit: the axes before each span are \
         the ones its positions hold"
    );
    drop(whole);
    // A span that continues the sequence without the axes before it is
    // refused by name, before anything runs: the head would rotate its first
    // row at a position nobody named.
    {
        let mut bare = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        let mut bits = vec![0u16; n * HIDDEN];
        prefill(model, &pool, &mut bare, &tokens[..CHUNK as usize], &multimodal, &media, CHUNK, &mut bits[..CHUNK as usize * HIDDEN]);
        let len = multimodal.cap_chunk(CHUNK, CHUNK.min(n as u32 - CHUNK));
        let positions = multimodal.span_positions(CHUNK as usize, len as usize);
        let chunk_media = multimodal.chunk_media(CHUNK, len);
        let end = (CHUNK + len) as usize;
        let refused = prefill_program_multimodal_span_hidden(
            model,
            &pool,
            &mut bare,
            &tokens[CHUNK as usize..end],
            u64::from(CHUNK),
            (end < n).then(|| tokens[end] as u32),
            MultimodalPrefill {
                positions: &positions,
                rope_delta: multimodal.rope_delta,
                media: chunk_media.as_ref().map(|m| SpanMediaColumns {
                    embedding: &media,
                    first_column: m.first_column,
                    scatter_indices: &m.scatter_indices,
                }),
                previous: None,
            },
            &mut bits[CHUNK as usize * HIDDEN..end * HIDDEN],
            None,
        )
        .expect_err("a continuing multimodal span without the axes before it");
        assert!(refused.contains("mrope_previous"), "{refused}");
        assert_eq!(bare.stats().position, u64::from(CHUNK), "the refused span moved nothing");
    }

    // The rows at the prompt's positions: before the image (its next token
    // the image's first column), inside it, at the chunk boundary inside it,
    // its last (its next token text), after it, and the prompt's last but one.
    let pending = spans.pending_token().expect("the prompt draws");
    let next_token = |p: usize| if p + 1 < n { tokens[p + 1] } else { pending };
    let next_embedding = |p: usize| {
        if (image_begin..image_end).contains(&(p + 1)) {
            column(&media_bits, p + 1 - image_begin)
        } else {
            embedding(reader, next_token(p))
        }
    };
    let axes = |p: usize| -> [i32; 3] {
        if p < n {
            std::array::from_fn(|axis| multimodal.positions[axis * n + p])
        } else {
            [p as i32 + delta; 3]
        }
    };
    let chosen: Vec<usize> = [
        image_begin - 1,
        image_begin,
        image_begin + 1,
        CHUNK as usize - 1,
        CHUNK as usize,
        image_end - 2,
        image_end - 1,
        image_end,
        n - 2,
    ]
    .into_iter()
    .collect();
    let rows = |embed: &dyn Fn(usize) -> Vec<f64>, turn: &dyn Fn(usize) -> [i32; 3]| -> Vec<RowInputs> {
        chosen.iter().map(|&p| RowInputs { hidden: column(&hidden_bits, p), embedding: embed(p), axes: turn(p) }).collect()
    };
    let positions: Vec<u32> = chosen.iter().map(|&p| p as u32).collect();
    let (key, value) = drift(reader, &pool, &spans, &positions, &rows(&next_embedding, &axes));
    eprintln!(
        "prompt rows {chosen:?}: keys {key:.6} (bound {:.6}), values {value:.6} (bound {:.6})",
        tolerance(KEY_ROUNDINGS),
        tolerance(VALUE_ROUNDINGS)
    );
    assert!(key <= tolerance(KEY_ROUNDINGS) && value <= tolerance(VALUE_ROUNDINGS), "keys {key}, values {value}");
    // The controls: the placeholders' own ids embedded, and the rows rotated
    // at their KV positions.
    let placeholder = |p: usize| embedding(reader, next_token(p));
    let media_rows: Vec<u32> = chosen.iter().filter(|&&p| (image_begin..image_end).contains(&(p + 1))).map(|&p| p as u32).collect();
    let (_, by_id) = {
        let picked: Vec<RowInputs> = rows(&placeholder, &axes)
            .into_iter()
            .zip(&chosen)
            .filter(|(_, p)| media_rows.contains(&(**p as u32)))
            .map(|(row, _)| row)
            .collect();
        drift(reader, &pool, &spans, &media_rows, &picked)
    };
    let rotated_rows: Vec<u32> = chosen.iter().filter(|&&p| p >= image_begin).map(|&p| p as u32).collect();
    let (at_positions, _) = {
        let flat = |p: usize| [p as i32; 3];
        let picked: Vec<RowInputs> = rows(&next_embedding, &flat)
            .into_iter()
            .zip(&chosen)
            .filter(|(_, p)| **p >= image_begin)
            .map(|(row, _)| row)
            .collect();
        drift(reader, &pool, &spans, &rotated_rows, &picked)
    };
    eprintln!("controls: placeholder ids' embeddings -> values {by_id:.4}; KV positions' rotation -> keys {at_positions:.4}");
    assert!(by_id > CONTROL_FACTOR * tolerance(VALUE_ROUNDINGS), "the placeholder-id control {by_id}");
    assert!(at_positions > CONTROL_FACTOR * tolerance(KEY_ROUNDINGS), "the KV-position control {at_positions}");

    // The first draft at the frontier: the f64 head over every row, the
    // frontier's row from the continuation hidden and the pending token.
    let every: Vec<RowInputs> =
        (0..n).map(|p| RowInputs { hidden: column(&hidden_bits, p), embedding: next_embedding(p), axes: axes(p) }).collect();
    let reference = f64_reference::evaluate_mtp_head_at_axes(
        reader,
        &every.iter().map(|r| r.hidden.clone()).collect::<Vec<_>>(),
        &every.iter().map(|r| r.embedding.clone()).collect::<Vec<_>>(),
        &every.iter().map(|r| r.axes).collect::<Vec<_>>(),
    )
    .unwrap_or_else(|e| panic!("f64 head over the prompt: {e}"));
    let proposal = mtp_readout::propose(model, &pool, &mut spans).unwrap_or_else(|e| panic!("frontier readout: {e}"));
    let hidden_drift = relative_l2(proposal.head_hidden.iter().copied().map(bf16_to_f64), reference.hidden.iter().copied());
    // The chain, at the verify round's own rotation positions: its first
    // step's scores against the device's output head over the f64 hidden.
    let f64_hidden: Vec<u16> = reference.hidden.iter().map(|&x| f32_to_bf16(x as f32)).collect();
    let f64_scores = mtp_readout::score(model, &f64_hidden).unwrap_or_else(|e| panic!("score the f64 hidden: {e}"));
    let chain = mtp_readout::chain(model, &pool, &mut spans, WINDOW, WINDOW, true).unwrap_or_else(|e| panic!("chain: {e}"));
    let chain_scores = chain.scores.as_ref().expect("asked for the scores");
    let score_drift = relative_l2(chain_scores[..VOCAB].iter().copied().map(bf16_to_f64), f64_scores.iter().map(|&s| s as f64));
    eprintln!(
        "frontier: head hidden {hidden_drift:.6} (bound {:.6}), chain's first scores {score_drift:.6} (bound {:.6}); \
         drafts {:?}, probabilities {:?}",
        tolerance(HIDDEN_ROUNDINGS),
        tolerance(SCORE_ROUNDINGS),
        chain.drafts(),
        chain.probabilities()
    );
    assert!(hidden_drift <= tolerance(HIDDEN_ROUNDINGS), "frontier head hidden {hidden_drift}");
    assert!(score_drift <= tolerance(SCORE_ROUNDINGS), "the chain's first step {score_drift}");

    // The first verify round drafts the window; its commit's rows, and the row
    // the plain round after it writes, rotate at position + delta.
    let anchor = pending;
    let lanes = [VerifyLane::greedy(&[])];
    let run = decode_program_verify_runs(model, &pool, &mut [&mut spans], &lanes, WINDOW)
        .unwrap_or_else(|e| panic!("the first verify round: {e}"))
        .remove(0);
    let committed = run.tokens.len();
    eprintln!("first round: extent {}, committed {committed} ({:?})", run.extent, run.tokens);
    assert_eq!(run.extent, WINDOW, "the first round after the image drafts the whole window");
    assert_eq!(run.tokens[0], anchor);
    let round = mtp_readout::round_hidden(model, 1, WINDOW).unwrap_or_else(|e| panic!("round hidden: {e}"));
    // Row n - 1 + j: the hidden at n - 1 + j (the prompt's last for j = 0,
    // else the round's column j - 1) and the run's token j.
    let committed_rows = |turn: &dyn Fn(usize) -> [i32; 3]| -> Vec<RowInputs> {
        (0..committed)
            .map(|j| RowInputs {
                hidden: if j == 0 { column(&hidden_bits, n - 1) } else { column(&round, j - 1) },
                embedding: embedding(reader, run.tokens[j]),
                axes: turn(n - 1 + j),
            })
            .collect()
    };
    let commit_positions: Vec<u32> = (0..committed).map(|j| (n - 1 + j) as u32).collect();
    let decoded = |p: usize| [p as i32 + delta; 3];
    let flat = |p: usize| [p as i32; 3];
    let (key, value) = drift(reader, &pool, &spans, &commit_positions, &committed_rows(&decoded));
    let (control, _) = drift(reader, &pool, &spans, &commit_positions, &committed_rows(&flat));
    eprintln!("commit rows: keys {key:.6}, values {value:.6}; rotated at the bare positions: keys {control:.4}");
    assert!(key <= tolerance(KEY_ROUNDINGS) && value <= tolerance(VALUE_ROUNDINGS), "commit keys {key}, values {value}");
    assert!(control > CONTROL_FACTOR * tolerance(KEY_ROUNDINGS), "the commit's bare-position control {control}");

    let frontier = spans.stats().position as usize;
    assert_eq!(frontier, n + committed);
    let consumed = spans.pending_token().expect("the round leaves a pending token");
    decode_program_batch(model, &pool, &mut [&mut spans]).unwrap_or_else(|e| panic!("a plain round: {e}"));
    let plain = |turn: &dyn Fn(usize) -> [i32; 3]| {
        vec![RowInputs {
            hidden: column(&round, committed - 1),
            embedding: embedding(reader, consumed),
            axes: turn(frontier - 1),
        }]
    };
    let (key, value) = drift(reader, &pool, &spans, &[frontier as u32 - 1], &plain(&decoded));
    let (control, _) = drift(reader, &pool, &spans, &[frontier as u32 - 1], &plain(&flat));
    eprintln!("plain round's row: keys {key:.6}, values {value:.6}; at the bare position: keys {control:.4}");
    assert!(key <= tolerance(KEY_ROUNDINGS) && value <= tolerance(VALUE_ROUNDINGS), "plain keys {key}, values {value}");
    assert!(control > CONTROL_FACTOR * tolerance(KEY_ROUNDINGS), "the plain round's bare-position control {control}");

    // The same from the captured graphs, which serving replays: a verify round
    // and a plain round at two lanes, the image's beside a text sequence's
    // (rope delta 0), each lane's rows at its own rotation.
    let text_tokens: Vec<i32> = FrontendSet::from_reader(reader)
        .expect("frontend")
        .tokenizer()
        .encode("Explain in two sentences how a bicycle gear changes the effort a rider needs.")
        .expect("tokenize")
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let mut text = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc the text lane: {e}"));
    let mut text_bits = vec![0u16; text_tokens.len() * HIDDEN];
    step::prefill_program_span_hidden(model, &pool, &mut text, &text_tokens, 0, None, &mut text_bits, None)
        .unwrap_or_else(|e| panic!("prefill the text lane: {e}"));
    step::capture_decode_graphs(model, &pool).unwrap_or_else(|e| panic!("capture the graphs: {e}"));
    let program = step::program_stats(model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
    assert!(
        program.decode_graph_ready_mask & 2 != 0 && program.verify_graph_ready_mask & 2 != 0,
        "the width-2 graphs are captured (decode {:#b}, verify {:#b})",
        program.decode_graph_ready_mask,
        program.verify_graph_ready_mask
    );
    // What each lane's next rows are built from: its continuation hidden and
    // frontier, and (the image lane's) the chain's drafts, read eagerly.
    let image_continuation = mtp_readout::propose(model, &pool, &mut spans).expect("readout").continuation;
    let text_continuation = mtp_readout::propose(model, &pool, &mut text).expect("readout").continuation;
    let eager_drafts = mtp_readout::chain(model, &pool, &mut spans, WINDOW, WINDOW, false).expect("chain").drafts();
    let bases = [spans.stats().position as usize, text.stats().position as usize];
    let deltas = [delta, 0];
    let lanes = [VerifyLane::greedy(&[]), VerifyLane::greedy(&[])];
    let runs = decode_program_verify_runs(model, &pool, &mut [&mut spans, &mut text], &lanes, WINDOW)
        .unwrap_or_else(|e| panic!("a verify round from its graph: {e}"));
    let round = mtp_readout::round_hidden(model, 2, WINDOW).unwrap_or_else(|e| panic!("round hidden: {e}"));
    let lane_column = |lane: usize, j: usize| column(&round, lane * (WINDOW as usize + 1) + j);
    let image_run = &runs[0].tokens;
    eprintln!("graph round: image lane {:?}, text lane {:?}; eager chain {eager_drafts:?}", image_run, runs[1].tokens);
    assert_eq!(
        image_run[1..],
        eager_drafts[..image_run.len() - 1],
        "the graph's chain drafted what the eager chain drafts: the accepted run is its drafts"
    );
    for (lane, continuation) in [(0usize, &image_continuation), (1, &text_continuation)] {
        let run = &runs[lane].tokens;
        let rows = |turn: &dyn Fn(usize) -> [i32; 3]| -> Vec<RowInputs> {
            (0..run.len())
                .map(|j| RowInputs {
                    hidden: if j == 0 {
                        continuation.iter().copied().map(bf16_to_f64).collect()
                    } else {
                        lane_column(lane, j - 1)
                    },
                    embedding: embedding(reader, run[j]),
                    axes: turn(bases[lane] - 1 + j),
                })
                .collect()
        };
        let positions: Vec<u32> = (0..run.len()).map(|j| (bases[lane] - 1 + j) as u32).collect();
        let at_delta = |p: usize| [p as i32 + deltas[lane]; 3];
        let sequence = if lane == 0 { &spans } else { &text };
        let (key, value) = drift(reader, &pool, sequence, &positions, &rows(&at_delta));
        eprintln!("graph round, lane {lane} (delta {}): commit keys {key:.6}, values {value:.6}", deltas[lane]);
        assert!(key <= tolerance(KEY_ROUNDINGS) && value <= tolerance(VALUE_ROUNDINGS), "lane {lane}: {key}, {value}");
        if lane == 0 {
            let (control, _) = drift(reader, &pool, sequence, &positions, &rows(&flat));
            assert!(control > CONTROL_FACTOR * tolerance(KEY_ROUNDINGS), "the graph commit's control {control}");
        }
    }
    let consumed = [spans.pending_token().expect("pending"), text.pending_token().expect("pending")];
    let frontiers = [spans.stats().position as usize, text.stats().position as usize];
    decode_program_batch(model, &pool, &mut [&mut spans, &mut text]).unwrap_or_else(|e| panic!("a plain round from its graph: {e}"));
    for lane in 0..2 {
        let last = runs[lane].tokens.len() - 1;
        let row = |turn: &dyn Fn(usize) -> [i32; 3]| {
            vec![RowInputs {
                hidden: lane_column(lane, last),
                embedding: embedding(reader, consumed[lane]),
                axes: turn(frontiers[lane] - 1),
            }]
        };
        let at_delta = |p: usize| [p as i32 + deltas[lane]; 3];
        let sequence = if lane == 0 { &spans } else { &text };
        let (key, value) = drift(reader, &pool, sequence, &[frontiers[lane] as u32 - 1], &row(&at_delta));
        eprintln!("graph plain round, lane {lane}: keys {key:.6}, values {value:.6}");
        assert!(key <= tolerance(KEY_ROUNDINGS) && value <= tolerance(VALUE_ROUNDINGS), "lane {lane}: {key}, {value}");
        if lane == 0 {
            let (control, _) = drift(reader, &pool, sequence, &[frontiers[lane] as u32 - 1], &row(&flat));
            assert!(control > CONTROL_FACTOR * tolerance(KEY_ROUNDINGS), "the graph plain round's control {control}");
        }
    }
    drop(text);
    drop(spans);
    drop(media);

    // Under the host residency the same prefill leaves the same rows.
    drop(pool);
    let Loaded { mut device, mut artifact, model, .. } = loaded;
    drop(model);
    let _ = artifact.release_arena(&mut device);
    drop(device);
    let Some(host) = load(VisionResidency::Host) else {
        return;
    };
    let pool = self::pool();
    let media = step::encode_media(&host.model, item.grid, &item.patches, &item_control)
        .unwrap_or_else(|(_, e)| panic!("encode under host: {e}"));
    assert!(media.read_bits(HIDDEN as u64).expect("read") == media_bits, "the image's columns are the same under host");
    let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    let mut host_bits = vec![0u16; n * HIDDEN];
    prefill(&host.model, &pool, &mut sequence, &tokens, &multimodal, &media, CHUNK, &mut host_bits);
    let (keys, values) = mtp_readout::head_state(&pool, &sequence, 0, n as u32 - 1).expect("head state");
    assert!(
        host_bits == hidden_bits && keys == span_keys && values == span_values,
        "under the host residency the prefill leaves the device residency's rows, bit for bit"
    );
    drop(sequence);
    drop(media);
    drop(pool);
    let Loaded { mut device, mut artifact, model, .. } = host;
    drop(model);
    let _ = artifact.release_arena(&mut device);
}
