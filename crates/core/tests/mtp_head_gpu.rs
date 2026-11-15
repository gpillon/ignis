//! GPU tests for the 27B's MTP head (GitHub #620, spec #618): a 27B load
//! built for MTP binds the checkpoint's head and the head computes.
//!
//! - The load binds the head's 12 objects and nothing of DFlash2, and a load
//!   without the option binds nothing of the head; its VRAM plan states the
//!   head's lines -- its weights, what its prefill and its rounds add to the
//!   scratch, the continuation hidden in the lane state, the head's layer in
//!   the KV pool's bytes per token and in the hq window -- and the CPU's
//!   arithmetic (`ignis_core::mtp_head`, `lane_lines`) is the leaf's plan to
//!   the byte, as the load holds it.
//! - A prefill writes the head's state at every prompt position it licenses
//!   and leaves the continuation hidden with the pending token; a plain round
//!   (eager, and replayed from its captured graph) writes the head's row at the
//!   position before the one it consumes and moves the continuation hidden on.
//! - That state and the head's first draft at the frontier are held to a
//!   double-precision recomputation from the stored `mtp/` weights
//!   (`ignis_artifact::f64_reference::evaluate_mtp_head`), BF16 KV, on the
//!   NVFP4 artifact and on the 16 GB artifact. The recomputation is fed the
//!   exact BF16 inputs the device's head read (the target's hiddens through
//!   the span readout, each next token's embedding rounded as the device
//!   rounds it), so what it measures is the head's own arithmetic.
//!
//! The 16 GB artifact's output head is trellis, which no host code decodes;
//! its head is W8 like the NVFP4 artifact's (the same bytes). There the head's
//! state and hidden are held to f64 as on NVFP4, and the first draft's scores
//! to the device's own output head applied to the f64 hidden: the head's every
//! weight is in the first two checks, and the third adds only the target's
//! head, a fixed map of the hidden.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{
    bind_model_scope_27b, f64_reference, materialize, text_scope_27b, CudaDevice, DraftModule, FrontendSet,
    MaterializedArtifact, Reader,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::lane_lines::lane_lines;
use ignis_core::model_load::{load_qwen38_27b_with_speculation, plan_qwen38_27b_reservations};
use ignis_core::seq::{kv_geometry, Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{capture_decode_graphs, decode_program_batch, prefill_program_span_hidden};
use ignis_core::{mtp_head, mtp_readout, KvFormat, PrefillAttention, PrefillLinear, RopeScaling, Speculation, SpeculativeBackend};

const NVFP4: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const TRELLIS: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_trellis_sc30-v2.ninfer";
const MAX_CONTEXT: u32 = 1024;
/// The load's chunk (a multiple of 128): wider than the head's call
/// (`mtp_head::PREFILL_ROWS`), so a chunk's rows run in two calls, and the
/// prompt below spans two, so a chunk's first row is built from the
/// continuation hidden the one before it left.
const PREFILL_CHUNK: u32 = 256;
const WINDOW: u32 = 7;
const HIDDEN: usize = 5120;
const VOCAB: usize = 248_320;
const KV_WIDTH: usize = mtp_readout::KV_WIDTH;
/// The prompt is its sentence repeated to at least this many tokens, cut into
/// two spans at `SPLIT`: the first with its next prompt token, the second
/// drawing the pending token.
const PROMPT_TOKENS: usize = 300;
const SPLIT: usize = 140;
/// Plain rounds after the prefill: the first two eager, the rest replayed
/// from the decode graphs captured between them.
const EAGER_ROUNDS: usize = 2;
const GRAPH_ROUNDS: usize = 2;

/// BF16's unit roundoff: an 8-bit significand rounds to the nearest value,
/// within half an ulp, at most 2^-8 of the value.
const BF16_UNIT: f64 = 1.0 / 256.0;

/// The BF16 roundings on the device's path from the head's inputs to each
/// quantity read (kernel/src/mtp_head.cu): the stem's normed embedding and
/// hidden, its projection, the layer's input norm and fused projection, then
/// the key's norm and rotation; the value is the projection's. The head's
/// hidden adds the query's norm and rotation, the attention, the output
/// projection and its residual add, the MLP's input norm, gate and up, the
/// SwiGLU, down and its residual add, and the final norm; the scores add the
/// output head's.
const KEY_ROUNDINGS: f64 = 6.0;
const VALUE_ROUNDINGS: f64 = 5.0;
const HIDDEN_ROUNDINGS: f64 = 16.0;
const SCORE_ROUNDINGS: f64 = 17.0;

/// The relative L2 a quantity after `roundings` BF16 roundings stays within:
/// independent roundings, each at most the unit roundoff, compound in
/// quadrature, and the bound takes each at its largest (the RMS of one is a
/// third of that squared), which leaves the norms and the softmax, whose
/// gains are near one, room. A wrong row, position, token, concatenation
/// order or gain lands far above it: those move the result by its own size.
fn tolerance(roundings: f64) -> f64 {
    BF16_UNIT * roundings.sqrt()
}

fn bf16_to_f64(bits: u16) -> f64 {
    f32::from_bits(u32::from(bits) << 16) as f64
}

/// f32 -> BF16, round to nearest even on the low 16 bits.
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

#[derive(Default)]
struct Drift {
    squared_error: f64,
    squared_reference: f64,
}

impl Drift {
    fn add(&mut self, device: f64, host: f64) {
        self.squared_error += (device - host) * (device - host);
        self.squared_reference += host * host;
    }

    fn relative_l2(&self) -> f64 {
        self.squared_error.sqrt() / self.squared_reference.sqrt().max(1e-30)
    }
}

fn relative_l2(device: impl IntoIterator<Item = f64>, host: &[f64]) -> f64 {
    let mut drift = Drift::default();
    for (d, h) in device.into_iter().zip(host) {
        drift.add(d, *h);
    }
    drift.relative_l2()
}

/// The device's embedding of `token`: the W8 row decoded (an int8 code times
/// its FP16 scale, exact in f32) and rounded to BF16, as the embedding op
/// writes it.
fn embedding(reader: &Reader, token: i32) -> Vec<f64> {
    f64_reference::matrix_row(reader, "text/token_embedding", VOCAB, HIDDEN, token as usize)
        .unwrap_or_else(|e| panic!("embedding row {token}: {e}"))
        .into_iter()
        .map(|value| bf16_to_f64(f32_to_bf16(value as f32)))
        .collect()
}

fn hidden_of(bits: &[u16]) -> Vec<f64> {
    bits.iter().copied().map(bf16_to_f64).collect()
}

fn argmax(scores: impl Iterator<Item = f64>) -> (usize, f64) {
    scores.enumerate().fold((0, f64::NEG_INFINITY), |best, (i, s)| if s > best.1 { (i, s) } else { best })
}

/// A device's or the skip's: `Ok(None)` outside the profile when the card or
/// the artifact is missing.
struct Loaded {
    reader: Reader,
    device: CudaDevice,
    artifact: MaterializedArtifact,
    handles: Vec<ignis_artifact::ObjectHandle>,
    plan: ignis_artifact::MaterializationPlan,
}

fn materialized(path: &str) -> Option<Loaded> {
    if !Path::new(path).exists() && gpu_profile::skip_or_fail(&format!("the artifact is absent: {path}")) {
        return None;
    }
    let reader = Reader::open(Path::new(path)).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let (plan, handles) =
        bind_model_scope_27b(&reader, Some(DraftModule::Mtp)).unwrap_or_else(|e| panic!("bind with the MTP head: {e}"));
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
            if gpu_profile::skip_or_fail(&format!("materialize text + mtp: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    Some(Loaded { reader, device, artifact, handles, plan })
}

fn mtp() -> Speculation {
    Speculation::new(SpeculativeBackend::Mtp, WINDOW).unwrap()
}

fn budget(kv_format: KvFormat, slot_count: u32, retained_slot_count: u32) -> SeqPoolBudget {
    SeqPoolBudget {
        kv_format,
        kv_page_group_count: 64,
        max_context_tokens: MAX_CONTEXT,
        slot_count,
        retained_slot_count,
        retained_host_slot_count: 0,
    }
}

fn prompt(reader: &Reader) -> Vec<i32> {
    let frontend = FrontendSet::from_reader(reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let sentence: Vec<i32> = frontend
        .tokenizer()
        .encode("In one sentence, explain what a paged KV cache is and why a drafter needs one. ")
        .unwrap_or_else(|e| panic!("tokenize: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("a token id fits i32"))
        .collect();
    let mut prompt = Vec::new();
    while prompt.len() < PROMPT_TOKENS {
        prompt.extend_from_slice(&sentence);
    }
    prompt
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn an_mtp_load_binds_the_head_and_plans_its_lines_as_the_cpu_does_to_the_byte() {
    let Some(mut loaded) = materialized(NVFP4) else {
        return;
    };
    let text_len = text_scope_27b().len();
    let cfg = ModelConfig::qwen38_27b();
    // The head's 12 objects and nothing of DFlash2 cross the ABI: the scope is
    // the text's and then the head's, whose bytes are the head's weight line.
    assert_eq!(loaded.handles.len(), text_len + 12);
    let text_device_objects = text_len - loaded.plan.host_mapped_objects.len();
    assert_eq!(loaded.plan.device_objects.len(), text_device_objects + 12);
    let head_bytes: u64 = loaded.plan.device_objects[text_device_objects..].iter().map(|p| p.bytes).sum();
    assert_eq!(head_bytes, mtp_head::WEIGHT_BYTES, "the head's weights line");

    let spec = mtp();
    let verify_only = Speculation::new(SpeculativeBackend::VerifyOnly, WINDOW).unwrap();
    let lanes = ignis_core::N_DECODE_LANES as u32;
    let plan = |handles: &[ignis_artifact::ObjectHandle], speculation: Option<Speculation>, kv_format: KvFormat| {
        plan_qwen38_27b_reservations(
            &loaded.reader,
            &loaded.plan,
            handles,
            PREFILL_CHUNK,
            MAX_CONTEXT,
            kv_format,
            speculation,
            None,
            RopeScaling::default(),
            PrefillAttention::default(),
            PrefillLinear::default(),
            lanes,
        )
        .unwrap_or_else(|e| panic!("plan ({speculation:?}): {e}"))
    };
    for kv_format in [KvFormat::Bf16, KvFormat::HqE8_2b] {
        let head = plan(&loaded.handles, Some(spec), kv_format);
        let substrate = plan(&loaded.handles[..text_len], Some(verify_only), kv_format);
        // Beside the verify substrate at the same window, the head adds its
        // prefill's buffers to the workspace and its rounds' to the decode
        // round's scratch, and no drafter round.
        assert_eq!(head.workspace_bytes - substrate.workspace_bytes, spec.prefill_scratch_bytes(PREFILL_CHUNK, false), "{kv_format:?}");
        assert_eq!(head.decode_graph_bytes - substrate.decode_graph_bytes, spec.decode_round_bytes(lanes, false), "{kv_format:?}");
        assert_eq!((head.drafter_round_bytes, spec.round_scratch_bytes(lanes)), (0, 0), "{kv_format:?}");
        assert_eq!(head.verify_round_bytes, substrate.verify_round_bytes, "{kv_format:?}");
        assert_eq!(head.sampling_bytes, substrate.sampling_bytes, "{kv_format:?}");

        // The pool: the continuation hidden in every slot, the head's layer
        // in every page and in the hq window, as the lane lines state them.
        let lines = lane_lines(&cfg, kv_format, Some(spec), false, lanes, 2);
        let pool = SeqPool::plan(&cfg, &budget(kv_format, lanes, 2), Some(SpeculativeBackend::Mtp))
            .unwrap_or_else(|e| panic!("pool plan: {e}"));
        let substrate_pool = SeqPool::plan(&cfg, &budget(kv_format, lanes, 2), Some(SpeculativeBackend::VerifyOnly))
            .unwrap_or_else(|e| panic!("pool plan: {e}"));
        assert_eq!(pool.lane_state_bytes, lines.lane_state, "{kv_format:?}: the lane state line");
        assert_eq!(pool.hq_residual_bytes, lines.hq_residual_window, "{kv_format:?}: the hq window line");
        assert_eq!(pool.lane_state_bytes - substrate_pool.lane_state_bytes, spec.drafter_state_bytes(lanes));
        assert_eq!(pool.retained_state_bytes - substrate_pool.retained_state_bytes, spec.drafter_state_bytes(2));
        let built = SeqPool::create_with_speculation(&cfg, &budget(kv_format, 1, 0), Some(SpeculativeBackend::Mtp))
            .unwrap_or_else(|e| panic!("pool create: {e}"));
        let geometry = kv_geometry(&cfg, Some(SpeculativeBackend::Mtp));
        assert_eq!(geometry.gqa_layers, 17);
        assert_eq!(built.stats().kv_page_bytes, kv_format.page_bytes(geometry), "{kv_format:?}: the head's layer per page");
        assert_eq!(
            kv_format.bytes_per_token(geometry) * 16,
            kv_format.bytes_per_token(ignis_core::KvGeometry::qwen38_27b()) * 17,
            "one layer in sixteen"
        );
    }

    // The load holds what its plan said, and binds the head's 12 objects
    // past the text scope's.
    let model = load_qwen38_27b_with_speculation(
        &loaded.reader,
        &loaded.artifact,
        &loaded.handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::HqE8_2b,
        Some(spec),
    )
    .unwrap_or_else(|e| panic!("load with the MTP head: {e}"));
    assert_eq!(model.stats().reserved, plan(&loaded.handles, Some(spec), KvFormat::HqE8_2b), "planned == held");
    let plain = load_qwen38_27b_with_speculation(
        &loaded.reader,
        &loaded.artifact,
        &loaded.handles[..text_len],
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::HqE8_2b,
        None,
    )
    .unwrap_or_else(|e| panic!("plain load: {e}"));
    assert_eq!(model.stats().bound_tensor_count, plain.stats().bound_tensor_count + 12);
    drop(plain);
    // A load without the option binds nothing of the head: handed its
    // objects, the load refuses them as outside its scope.
    let err = load_qwen38_27b_with_speculation(
        &loaded.reader,
        &loaded.artifact,
        &loaded.handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::HqE8_2b,
        None,
    )
    .map(|_| ())
    .expect_err("a load without MTP refuses the head's objects");
    assert!(err.contains("does not match the inventory"), "{err}");

    // Under hq-e8-2b the head runs too: its prompt route over the chunks, its
    // decode route in plain rounds, its own ring words. The prompt is shorter
    // than the residual window (32 sink rows and a 512-row ring), so every row
    // the readout's attention reads is the exact row in the head's own planes,
    // under the head's own ring bits: its hidden is the f64 head's over the
    // hiddens this prefill read, within the BF16 bound.
    let pool = SeqPool::create_with_speculation(&cfg, &budget(KvFormat::HqE8_2b, 2, 0), Some(SpeculativeBackend::Mtp))
        .unwrap_or_else(|e| panic!("hq pool: {e}"));
    let tokens = prompt(&loaded.reader);
    let n = tokens.len();
    assert!(n < 544, "the prompt fits the residual window");
    let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    let mut hidden_bits = vec![0u16; n * HIDDEN];
    prefill_program_span_hidden(&model, &pool, &mut sequence, &tokens, 0, None, &mut hidden_bits, None)
        .unwrap_or_else(|e| panic!("hq prefill: {e}"));
    let pending = sequence.pending_token().expect("the prompt draws");
    let proposal = mtp_readout::propose(&model, &pool, &mut sequence).unwrap_or_else(|e| panic!("hq readout: {e}"));
    let hiddens: Vec<Vec<f64>> = (0..n).map(|p| hidden_of(&hidden_bits[p * HIDDEN..(p + 1) * HIDDEN])).collect();
    let next_tokens: Vec<i32> = (1..n).map(|p| tokens[p]).chain(std::iter::once(pending)).collect();
    let reference_head = reference(&loaded.reader, &hiddens, &next_tokens, &(0..n as i32).collect::<Vec<_>>());
    let hidden_drift = relative_l2(proposal.head_hidden.iter().copied().map(bf16_to_f64), &reference_head.hidden);
    eprintln!("hq-e8-2b: head hidden {hidden_drift:.6} (bound {:.6})", tolerance(HIDDEN_ROUNDINGS));
    assert!(hidden_drift <= tolerance(HIDDEN_ROUNDINGS), "hq head hidden {hidden_drift}");
    for _ in 0..2 {
        decode_program_batch(&model, &pool, &mut [&mut sequence]).unwrap_or_else(|e| panic!("hq round: {e}"));
    }
    let proposal = mtp_readout::propose(&model, &pool, &mut sequence).unwrap_or_else(|e| panic!("hq readout: {e}"));
    assert!((0..VOCAB as i32).contains(&proposal.draft), "{}", proposal.draft);
    // GitHub #622: the head's state is in the section table, so its sequences
    // move: a restored sequence holds the head's ring words, and its own
    // snapshot is the source's bytes.
    let blob = sequence.snapshot().unwrap_or_else(|e| panic!("an MTP pool's snapshot: {e}"));
    let mut restored = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc the restore's target: {e}"));
    restored.restore(&blob).unwrap_or_else(|e| panic!("restore: {e}"));
    assert_eq!(
        mtp_readout::ring_words(&pool, &restored).unwrap_or_else(|e| panic!("ring words: {e}")),
        mtp_readout::ring_words(&pool, &sequence).unwrap_or_else(|e| panic!("ring words: {e}")),
        "the restored sequence holds the target's and the head's ring words"
    );
    assert!(restored.snapshot().unwrap_or_else(|e| panic!("snapshot the restored: {e}")) == blob);
    drop(restored);

    drop(sequence);
    drop(pool);
    drop(model);
    let _ = loaded.artifact.release_arena(&mut loaded.device);
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_heads_state_and_first_draft_are_its_weights_in_double_precision_on_the_nvfp4_artifact() {
    check_against_f64(NVFP4, true);
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_heads_state_and_first_draft_are_its_weights_in_double_precision_on_the_16gb_artifact() {
    check_against_f64(TRELLIS, false);
}

/// The head's rows at `positions` from `hiddens` and the embeddings of
/// `next_tokens`, in f64.
fn reference(reader: &Reader, hiddens: &[Vec<f64>], next_tokens: &[i32], positions: &[i32]) -> f64_reference::MtpHeadReference {
    let embeddings: Vec<Vec<f64>> = next_tokens.iter().map(|&t| embedding(reader, t)).collect();
    f64_reference::evaluate_mtp_head(reader, hiddens, &embeddings, positions).unwrap_or_else(|e| panic!("f64 head: {e}"))
}

/// The device's keys and values of rows `first..first + count` against the
/// reference's rows `0..count`.
fn kv_drift(keys: &[u16], values: &[u16], reference: &f64_reference::MtpHeadReference, count: usize) -> (f64, f64) {
    let mut key = Drift::default();
    let mut value = Drift::default();
    for row in 0..count {
        for i in 0..KV_WIDTH {
            key.add(bf16_to_f64(keys[row * KV_WIDTH + i]), reference.keys[row][i]);
            value.add(bf16_to_f64(values[row * KV_WIDTH + i]), reference.values[row][i]);
        }
    }
    (key.relative_l2(), value.relative_l2())
}

fn check_against_f64(path: &str, host_head: bool) {
    let Some(mut loaded) = materialized(path) else {
        return;
    };
    let cfg = ModelConfig::qwen38_27b();
    let model = load_qwen38_27b_with_speculation(
        &loaded.reader,
        &loaded.artifact,
        &loaded.handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::Bf16,
        Some(mtp()),
    )
    .unwrap_or_else(|e| panic!("load with the MTP head: {e}"));
    let pool = SeqPool::create_with_speculation(&cfg, &budget(KvFormat::Bf16, 2, 0), Some(SpeculativeBackend::Mtp))
        .unwrap_or_else(|e| panic!("pool: {e}"));
    let reader = &loaded.reader;
    let tokens = prompt(reader);
    let n = tokens.len();
    // The sequence the readouts read, and a witness that runs the same prefill
    // and the same rounds with no readout at all -- the readout writes the
    // head's row at the frontier, the very row a round then writes, so only
    // the witness's rows show what the rounds wrote -- in the pool's other
    // slot, so a slot read where a lane was meant shows too.
    let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    let mut witness = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc the witness: {e}"));
    assert_ne!(sequence.stats().slot, witness.stats().slot);

    // The prefill, in two spans of several chunks, reading the target's
    // post-final-norm hidden of every position as the head read it.
    let mut hidden_bits = vec![0u16; n * HIDDEN];
    let mut witness_bits = vec![0u16; n * HIDDEN];
    let mut logits = vec![0f32; VOCAB];
    for (seq, bits, logits) in
        [(&mut sequence, &mut hidden_bits, Some(logits.as_mut_slice())), (&mut witness, &mut witness_bits, None)]
    {
        let (first, second) = bits.split_at_mut(SPLIT * HIDDEN);
        prefill_program_span_hidden(&model, &pool, seq, &tokens[..SPLIT], 0, Some(tokens[SPLIT] as u32), first, None)
            .unwrap_or_else(|e| panic!("prefill, first span: {e}"));
        prefill_program_span_hidden(&model, &pool, seq, &tokens[SPLIT..], SPLIT as u64, None, second, logits)
            .unwrap_or_else(|e| panic!("prefill, second span: {e}"));
    }
    assert!(witness_bits == hidden_bits, "the witness's prefill read the sequence's hiddens");
    let pending = sequence.pending_token().expect("the prompt's last span draws");
    assert_eq!(sequence.stats().position, n as u64);

    // The continuation hidden is the target's hidden at the prompt's last
    // position: the one its draw's scores came from, bit for bit.
    let proposal = mtp_readout::propose(&model, &pool, &mut sequence).unwrap_or_else(|e| panic!("readout: {e}"));
    assert_eq!(proposal.continuation, hidden_bits[(n - 1) * HIDDEN..]);
    assert_eq!(mtp_readout::score(&model, &proposal.continuation).unwrap(), logits, "the draw's own scores");

    // The head holds every position but the last; the last's entry is the
    // next step's to build.
    let (keys, values) = mtp_readout::head_state(&pool, &sequence, 0, (n - 1) as u32)
        .unwrap_or_else(|e| panic!("the head's state: {e}"));
    let err = mtp_readout::head_state(&pool, &sequence, 0, n as u32).map(|_| ()).expect_err("past the head's frontier");
    assert!(err.contains("written"), "{err}");

    // The f64 head over the prompt's rows, the last one's token the pending
    // one: its state, and its hidden and scores at the frontier.
    let hiddens: Vec<Vec<f64>> = (0..n).map(|p| hidden_of(&hidden_bits[p * HIDDEN..(p + 1) * HIDDEN])).collect();
    let next_tokens: Vec<i32> = (1..n).map(|p| tokens[p]).chain(std::iter::once(pending)).collect();
    let positions: Vec<i32> = (0..n as i32).collect();
    let reference_head = reference(reader, &hiddens, &next_tokens, &positions);
    let (key_drift, value_drift) = kv_drift(&keys, &values, &reference_head, n - 1);
    for row in 0..n - 1 {
        let one = f64_reference::MtpHeadReference {
            keys: vec![reference_head.keys[row].clone()],
            values: vec![reference_head.values[row].clone()],
            hidden: Vec::new(),
        };
        let (k, v) = kv_drift(&keys[row * KV_WIDTH..], &values[row * KV_WIDTH..], &one, 1);
        if k > tolerance(KEY_ROUNDINGS) || v > tolerance(VALUE_ROUNDINGS) {
            eprintln!("{path}: row {row}: keys {k:.6}, values {v:.6}");
        }
    }
    let hidden_drift = relative_l2(proposal.head_hidden.iter().copied().map(bf16_to_f64), &reference_head.hidden);
    let device_scores_of_reference = mtp_readout::score(
        &model,
        &reference_head.hidden.iter().map(|&x| f32_to_bf16(x as f32)).collect::<Vec<_>>(),
    )
    .unwrap();
    let head_drift =
        relative_l2(proposal.scores.iter().map(|&s| f64::from(s)), &device_scores_of_reference.iter().map(|&s| f64::from(s)).collect::<Vec<_>>());
    eprintln!(
        "{path}: {} rows; keys {key_drift:.6} (bound {:.6}), values {value_drift:.6} (bound {:.6}), head hidden \
         {hidden_drift:.6} (bound {:.6}), scores against the device's head over the f64 hidden {head_drift:.6} (bound \
         {:.6})",
        n - 1,
        tolerance(KEY_ROUNDINGS),
        tolerance(VALUE_ROUNDINGS),
        tolerance(HIDDEN_ROUNDINGS),
        tolerance(SCORE_ROUNDINGS),
    );
    assert!(key_drift <= tolerance(KEY_ROUNDINGS), "keys {key_drift}");
    assert!(value_drift <= tolerance(VALUE_ROUNDINGS), "values {value_drift}");
    assert!(hidden_drift <= tolerance(HIDDEN_ROUNDINGS), "head hidden {hidden_drift}");
    assert!(head_drift <= tolerance(SCORE_ROUNDINGS), "scores {head_drift}");
    // The first draft is the scores' best token, the f64's unless the two
    // best are within the scores' bound of each other.
    let reference_scores: Vec<f64> = if host_head {
        f64_reference::matrix_product(reader, "text/output_head", VOCAB, HIDDEN, &reference_head.hidden)
            .unwrap_or_else(|e| panic!("f64 output head: {e}"))
    } else {
        device_scores_of_reference.iter().map(|&s| f64::from(s)).collect()
    };
    if host_head {
        let score_drift = relative_l2(proposal.scores.iter().map(|&s| f64::from(s)), &reference_scores);
        eprintln!("{path}: scores against the f64 output head {score_drift:.6} (bound {:.6})", tolerance(SCORE_ROUNDINGS));
        assert!(score_drift <= tolerance(SCORE_ROUNDINGS), "scores {score_drift}");
    }
    let (best, best_score) = argmax(reference_scores.iter().copied());
    assert_eq!(proposal.draft as usize, argmax(proposal.scores.iter().map(|&s| f64::from(s))).0);
    if proposal.draft as usize != best {
        let scale = reference_scores.iter().fold(0f64, |m, s| m.max(s.abs()));
        let gap = best_score - reference_scores[proposal.draft as usize];
        assert!(gap <= tolerance(SCORE_ROUNDINGS) * scale, "draft {} against the f64's {best}, {gap} apart", proposal.draft);
    }

    // The witness holds what the sequence does: its prefill's rows, bit for
    // bit, from its own slot.
    let (witness_keys, witness_values) = mtp_readout::head_state(&pool, &witness, 0, (n - 1) as u32)
        .unwrap_or_else(|e| panic!("the witness's state: {e}"));
    assert!(witness_keys == keys && witness_values == values, "the head's rows follow the sequence, not its slot");

    // Plain rounds: each writes the head's row at the position before the one
    // it consumes, from the continuation hidden and that token, and leaves
    // the hidden of the position it consumed, whose scores drew its token.
    // The inputs come from the sequence's readouts; the rows the rounds wrote
    // are read off the witness, which ran the same rounds and no readout.
    let mut rows = Vec::new();
    let round = |sequence: &mut Seq<'_>, witness: &mut Seq<'_>, rows: &mut Vec<(u32, Vec<u16>, i32)>, what: &str| {
        let before = mtp_readout::propose(&model, &pool, sequence).unwrap_or_else(|e| panic!("{what}: readout: {e}"));
        let consumed = sequence.pending_token().expect("a pending token");
        let position = sequence.stats().position;
        let emitted = decode_program_batch(&model, &pool, &mut [&mut *sequence]).unwrap_or_else(|e| panic!("{what}: round: {e}"));
        let witnessed = decode_program_batch(&model, &pool, &mut [&mut *witness]).unwrap_or_else(|e| panic!("{what}: witness: {e}"));
        assert_eq!(emitted, witnessed, "{what}: the witness emits what the sequence does");
        assert_eq!(witness.pending_token(), sequence.pending_token(), "{what}: and draws what it does");
        rows.push(((position - 1) as u32, before.continuation, consumed));
        let after = mtp_readout::propose(&model, &pool, sequence).unwrap_or_else(|e| panic!("{what}: readout: {e}"));
        let scores = mtp_readout::score(&model, &after.continuation).unwrap();
        let drawn = sequence.pending_token().expect("the round drew");
        assert_eq!(argmax(scores.iter().map(|&s| f64::from(s))).0, drawn as usize, "{what}: the round's hidden drew its token");
        assert_ne!(after.continuation, rows.last().unwrap().1, "{what}: the continuation hidden moved on");
    };
    for i in 0..EAGER_ROUNDS {
        round(&mut sequence, &mut witness, &mut rows, &format!("eager round {i}"));
    }
    let capture = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    assert!(capture.is_ready(1), "the one-lane round, the head's step in it, is captured and replayed");
    for i in 0..GRAPH_ROUNDS {
        round(&mut sequence, &mut witness, &mut rows, &format!("replayed round {i}"));
    }
    for (i, (row, continuation, consumed)) in rows.iter().enumerate() {
        let (keys, values) =
            mtp_readout::head_state(&pool, &witness, *row, 1).unwrap_or_else(|e| panic!("round {i}: state: {e}"));
        let reference_row = reference(reader, &[hidden_of(continuation)], &[*consumed], &[*row as i32]);
        let (key_drift, value_drift) = kv_drift(&keys, &values, &reference_row, 1);
        eprintln!("{path}: round {i} row {row}: keys {key_drift:.6}, values {value_drift:.6}");
        assert!(key_drift <= tolerance(KEY_ROUNDINGS) && value_drift <= tolerance(VALUE_ROUNDINGS), "round {i}: row {row}");
    }
    let last = mtp_readout::propose(&model, &pool, &mut witness).unwrap_or_else(|e| panic!("the witness's readout: {e}"));
    let first = mtp_readout::propose(&model, &pool, &mut sequence).unwrap_or_else(|e| panic!("the readout: {e}"));
    assert_eq!(last.continuation, first.continuation, "the witness's continuation hidden is the sequence's");
    assert_eq!(model.stats().reserved.drafter_round_bytes, 0);
    drop(witness);

    drop(sequence);
    drop(pool);
    drop(model);
    let _ = loaded.artifact.release_arena(&mut loaded.device);
}
