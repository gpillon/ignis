//! GitHub #572 (spec #570): the 27B's decode and verify rounds over hq-e8-2b
//! on ignis's warp-specialized attention route, chosen once per load by card
//! (`kernel/src/hq_decode_attention.h`), against today's route, which the
//! measurement switch `IGNIS_HQ_DECODE_ATTENTION=vendored` selects. On the card
//! at hand, with the NVFP4 artifact and its DFlash2 drafter:
//!
//! - each load reports the route it took (`Model::hq_decode_attention`): the
//!   card's own with the switch unset (ours on a Blackwell card whose image the
//!   build holds, today's on any other), today's under `vendored` and on a BF16
//!   load whatever the card, and a value the switch does not know refuses the
//!   load, naming the variable;
//! - every decode graph and every verify graph captures at widths 1..8 with the
//!   node count today's route captures (the route takes the vendored partial
//!   kernel's place, the vendored reducer follows it);
//! - every lane's text is the same on both routes, graph replays all: greedy
//!   decode rounds at widths 1..8, greedy verify rounds at widths 1..8, and
//!   sampled verify rounds (temperature 1, top-p 0.95, top-k 20, a fixed seed
//!   per lane) under the sampled draft at widths 1 and 8, with the same drafts
//!   committed per round. The route writes the reference route's bytes, so no
//!   lane may move at all.
//!
//! The prompts run past the residual window's 512 keys, and the load's context
//! is 65,536 tokens, so the rounds launch the split capacity a long-context load
//! launches (85 splits a KV head). On a card where the default is today's route
//! the two loads compare it with itself, which says only that the switch is
//! read. This binary holds this one test: it sets the process environment
//! before each load. Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a
//! failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{
    bind_model_scope_27b, built_cuda_architectures, materialize, text_scope_27b, CudaDevice, DraftModule, FrontendSet,
    MaterializedArtifact, ObjectHandle, Reader,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b, load_qwen38_27b_with_speculation, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_batch_sampled, decode_program_verify_runs, prefill_program_sampled,
    program_stats, SamplingParams, VerifyLane,
};
use ignis_core::{DraftRule, DrafterSampling, HqDecodeAttention, KvFormat, Speculation, SpeculativeBackend};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The measurement switch the leaf reads at each load (`kernel/src/model.cu`).
const SWITCH: &str = "IGNIS_HQ_DECODE_ATTENTION";
/// The load's context: past 16,390 keys the split policy reaches its cap.
const MAX_CONTEXT: u32 = 65_536;
/// The BF16 load's, which only reports its route.
const BF16_CONTEXT: u32 = 2_048;
const CHUNK: u32 = 512;
/// What one lane's sequence reserves: its prompt and the tokens it emits.
const SEQ_TOKENS: u32 = 2_048;
const LANES: u32 = 8;
/// DFlash2-7, the 27B's served window.
const WINDOW: u32 = 7;
/// Tokens each lane emits in a run.
const TOTAL: usize = 24;

/// What one route's loads recorded.
#[derive(Debug)]
struct RouteRun {
    /// The text load's decode graphs, the drafter load's decode graphs, and
    /// its verify graphs: each width's node count.
    decode_nodes: [u32; 8],
    drafter_decode_nodes: [u32; 8],
    verify_nodes: [u32; 8],
    /// Per width 1..8: each lane's greedy decode text.
    decode: Vec<Vec<Vec<i32>>>,
    /// Per run: each lane's text, and per round the `(lane, extent, committed)`
    /// of every lane that rode it.
    verify: Vec<(Vec<Vec<i32>>, Vec<Vec<(usize, u32, usize)>>)>,
}

/// The route a load takes with the switch unset: ours on a Blackwell card
/// whose arch-specific target the build names, today's on any other.
fn card_default() -> HqDecodeAttention {
    let (name, (major, minor)) = CudaDevice::identity(0).unwrap_or_else(|e| panic!("device identity: {e}"));
    let built = built_cuda_architectures();
    let image = built.split(';').any(|arch| arch == format!("{major}{minor}a"));
    println!("card: {name} (sm_{major}{minor}), build {built}");
    if major == 12 && image { HqDecodeAttention::Ours } else { HqDecodeAttention::Vendored }
}

fn set_switch(value: Option<&str>) {
    // SAFETY: this binary holds this one test, which runs on one thread; nothing else reads or
    // writes the environment while it is set, and the leaf reads it inside the next load.
    match value {
        Some(value) => unsafe { std::env::set_var(SWITCH, value) },
        None => unsafe { std::env::remove_var(SWITCH) },
    }
}

fn pool_for(backend: Option<SpeculativeBackend>) -> SeqPool {
    SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::HqE8_2b,
            // One whole context of pages, as a served pool holds at least; the lanes take 32 each.
            kv_page_group_count: MAX_CONTEXT.div_ceil(64),
            max_context_tokens: MAX_CONTEXT,
            slot_count: LANES,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
        backend,
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"))
}

/// Captures the load's graphs, every width of both kinds the load serves, and
/// returns their node counts.
fn capture(model: &Model, pool: &SeqPool, verify: bool) -> ([u32; 8], [u32; 8]) {
    let capture = capture_decode_graphs(model, pool).unwrap_or_else(|e| panic!("capture: {e}"));
    let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("stats: {e}"));
    assert_eq!(capture.ready_mask, 0xff, "every decode width captured");
    assert_eq!(stats.verify_graph_ready_mask, if verify { 0xff } else { 0 }, "every verify width captured");
    assert!(stats.decode_graph_nodes.iter().all(|&n| n > 0), "{:?}", stats.decode_graph_nodes);
    assert!(stats.verify_graph_nodes.iter().all(|&n| (n > 0) == verify), "{:?}", stats.verify_graph_nodes);
    (stats.decode_graph_nodes, stats.verify_graph_nodes)
}

fn prefill_all<'p>(model: &Model, pool: &'p SeqPool, prompts: &[Vec<i32>]) -> Vec<Seq<'p>> {
    prompts
        .iter()
        .map(|prompt| {
            let mut seq = pool.alloc(SEQ_TOKENS).unwrap_or_else(|e| panic!("alloc: {e}"));
            prefill_program_sampled(model, pool, &mut seq, prompt, 0, SamplingParams::greedy(), None)
                .unwrap_or_else(|e| panic!("prefill: {e}"));
            seq
        })
        .collect()
}

/// `TOTAL` greedy decode rounds over `prompts`, one batch, each a graph replay.
fn decode_rounds(model: &Model, pool: &SeqPool, prompts: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let mut sequences = prefill_all(model, pool, prompts);
    let params = vec![SamplingParams::greedy(); prompts.len()];
    let mut streams = vec![Vec::with_capacity(TOTAL); prompts.len()];
    for round in 0..TOTAL {
        let mut refs: Vec<&mut Seq<'_>> = sequences.iter_mut().collect();
        let tokens = decode_program_batch_sampled(model, pool, &mut refs, &params)
            .unwrap_or_else(|e| panic!("decode round {round}: {e}"));
        let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.graph_launches, 1, "decode round {round} at width {}: a graph replay", prompts.len());
        for (stream, token) in streams.iter_mut().zip(tokens) {
            stream.push(token);
        }
    }
    streams
}

/// Verify rounds over `prompts` under `samplings` until every lane has emitted
/// `TOTAL` tokens, each a graph replay; lanes leave as they finish.
fn verify_rounds(
    model: &Model,
    pool: &SeqPool,
    prompts: &[Vec<i32>],
    samplings: &[SamplingParams],
) -> (Vec<Vec<i32>>, Vec<Vec<(usize, u32, usize)>>) {
    let mut sequences = prefill_all(model, pool, prompts);
    let mut emitted: Vec<Vec<i32>> = vec![Vec::new(); prompts.len()];
    let mut rounds = Vec::new();
    while emitted.iter().any(|e| e.len() < TOTAL) {
        let active: Vec<usize> = (0..prompts.len()).filter(|&i| emitted[i].len() < TOTAL).collect();
        let lanes: Vec<VerifyLane<'_>> = active
            .iter()
            .map(|&i| VerifyLane {
                sampling: samplings[i],
                remaining_tokens: (TOTAL - emitted[i].len()) as u32,
                ..VerifyLane::greedy(&[])
            })
            .collect();
        let mut remaining: Vec<Option<&mut Seq<'_>>> = sequences.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut Seq<'_>> =
            active.iter().map(|&i| remaining[i].take().expect("each active lane once")).collect();
        let results = decode_program_verify_runs(model, pool, &mut refs, &lanes, WINDOW)
            .unwrap_or_else(|e| panic!("verify round {}: {e}", rounds.len()));
        let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.graph_launches, 1, "verify round {} at width {}: a graph replay", rounds.len(), active.len());
        let mut this_round = Vec::new();
        for (&i, result) in active.iter().zip(results) {
            assert!(!result.tokens.is_empty(), "lane {i}: a round commits at least its anchor");
            this_round.push((i, result.extent, result.tokens.len()));
            emitted[i].extend(result.tokens);
        }
        rounds.push(this_round);
        assert!(rounds.len() <= TOTAL, "the run does not converge");
    }
    (emitted, rounds)
}

/// The sampling a coding agent sends: temperature 1, top-k 20, top-p 0.95.
fn agent(seed: u64) -> SamplingParams {
    SamplingParams { temperature: 1.0, top_k: 20, top_p: 0.95, presence_penalty: 0.0, frequency_penalty: 0.0, seed }
}

/// One route's two loads, the switch set to `switch`: the text load's decode
/// rounds, then the drafter load's verify rounds, under the sampled draft as
/// main serves it.
fn run_route(
    reader: &Reader,
    artifact: &MaterializedArtifact,
    handles: &[ObjectHandle],
    text_len: usize,
    prompts: &[Vec<i32>],
    switch: Option<&str>,
    expected: HqDecodeAttention,
) -> RouteRun {
    set_switch(switch);
    let label = switch.unwrap_or("unset");
    let (decode_nodes, decode) = {
        let model = load_qwen38_27b(reader, artifact, &handles[..text_len], CHUNK, MAX_CONTEXT, KvFormat::HqE8_2b)
            .unwrap_or_else(|e| panic!("{label}: model load (hq): {e}"));
        assert_eq!(model.hq_decode_attention(), expected, "the route the hq load reports, the switch {label}");
        let pool = pool_for(None);
        let (decode_nodes, _) = capture(&model, &pool, false);
        let decode = (1..=LANES as usize).map(|width| decode_rounds(&model, &pool, &prompts[..width])).collect();
        (decode_nodes, decode)
    };
    let spec = Speculation::new(SpeculativeBackend::Dflash2, WINDOW)
        .unwrap()
        .with_draft_rule(DraftRule::Sampled, DrafterSampling::default());
    let model = load_qwen38_27b_with_speculation(reader, artifact, handles, CHUNK, MAX_CONTEXT, KvFormat::HqE8_2b, Some(spec))
        .unwrap_or_else(|e| panic!("{label}: model load (hq, dflash2, the sampled draft): {e}"));
    assert_eq!(model.hq_decode_attention(), expected, "the route the drafter load reports, the switch {label}");
    let pool = pool_for(Some(SpeculativeBackend::Dflash2));
    let (drafter_decode_nodes, verify_nodes) = capture(&model, &pool, true);
    let mut verify: Vec<_> = (1..=LANES as usize)
        .map(|width| verify_rounds(&model, &pool, &prompts[..width], &vec![SamplingParams::greedy(); width]))
        .collect();
    for width in [1usize, LANES as usize] {
        let samplings: Vec<SamplingParams> = (0..width).map(|lane| agent(0x572 + lane as u64)).collect();
        verify.push(verify_rounds(&model, &pool, &prompts[..width], &samplings));
    }
    println!(
        "{label}: {expected:?}, decode nodes {decode_nodes:?} (the drafter load's {drafter_decode_nodes:?}), \
         verify nodes {verify_nodes:?}"
    );
    RouteRun { decode_nodes, drafter_decode_nodes, verify_nodes, decode, verify }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_blackwell_load_reads_hq_on_its_own_route_and_every_lane_keeps_todays_text() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let encode = |text: &str| -> Vec<i32> {
        frontend
            .tokenizer()
            .encode(text)
            .unwrap_or_else(|e| panic!("tokenize: {e}"))
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect()
    };
    // Eight user turns, each past the residual window's 512 keys and each its own length, so a
    // lane that read another's rows, or a split cut at the wrong key, moves a token.
    let topics = [
        "what does the block table map?",
        "why do pages let many sequences share one pool?",
        "what happens when a sequence outgrows its last page?",
        "how does prefix sharing avoid a copy?",
        "when does a shared page get copied?",
        "what does freeing a sequence give back?",
        "why is the pool sized in bytes and not in tokens?",
        "what limits how many sequences decode at once?",
    ];
    let prompts: Vec<Vec<i32>> = topics
        .iter()
        .enumerate()
        .map(|(i, topic)| {
            let context = "A paged KV cache keeps each sequence's keys and values in fixed-size pages, and a block \
                           table maps every logical page to a physical one, so many sequences share one pool. "
                .repeat(16 + 2 * i);
            encode(&format!(
                "<|im_start|>user\n{context}Now answer in one sentence: {topic}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ))
        })
        .collect();
    assert!(prompts.iter().all(|p| p.len() > 544), "every prompt runs past the residual window");
    assert!(prompts.iter().all(|p| p.len() + TOTAL + WINDOW as usize <= SEQ_TOKENS as usize), "a prompt outgrows its sequence");

    let (plan, handles) = bind_model_scope_27b(&reader, Some(DraftModule::Dflash2))
        .unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
    let text_len = text_scope_27b().len();
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
            if gpu_profile::skip_or_fail(&format!("materialize text + dflash2: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let default = card_default();

    // A value the switch does not know refuses the load by name.
    set_switch(Some("throughput"));
    match load_qwen38_27b(&reader, &artifact, &handles[..text_len], CHUNK, MAX_CONTEXT, KvFormat::HqE8_2b) {
        Ok(_) => panic!("IGNIS_HQ_DECODE_ATTENTION=throughput loaded"),
        Err(e) => assert!(e.contains(SWITCH), "the refusal names the switch: {e}"),
    }
    // A BF16 load runs today's route whatever the card.
    set_switch(None);
    let bf16 = load_qwen38_27b(&reader, &artifact, &handles[..text_len], CHUNK, BF16_CONTEXT, KvFormat::Bf16)
        .unwrap_or_else(|e| panic!("model load (bf16): {e}"));
    assert_eq!(bf16.hq_decode_attention(), HqDecodeAttention::Vendored, "a BF16 load's route");
    drop(bf16);

    let todays = run_route(&reader, &artifact, &handles, text_len, &prompts, Some("vendored"), HqDecodeAttention::Vendored);
    let cards = run_route(&reader, &artifact, &handles, text_len, &prompts, None, default);
    set_switch(None);
    if default == HqDecodeAttention::Vendored {
        println!("this card keeps today's route: the two runs compare it with itself");
    }
    assert_eq!(cards.decode_nodes, todays.decode_nodes, "the decode graphs' node counts at widths 1..8");
    assert_eq!(cards.drafter_decode_nodes, todays.drafter_decode_nodes, "a drafter load's decode graphs' node counts");
    assert_eq!(cards.verify_nodes, todays.verify_nodes, "the verify graphs' node counts at widths 1..8");
    for (width, (ours, theirs)) in cards.decode.iter().zip(&todays.decode).enumerate() {
        assert_eq!(ours, theirs, "decode rounds at width {}: every lane's text", width + 1);
    }
    for (run, (ours, theirs)) in cards.verify.iter().zip(&todays.verify).enumerate() {
        assert_eq!(ours.0, theirs.0, "verify run {run}: every lane's text");
        assert_eq!(ours.1, theirs.1, "verify run {run}: every round's drafts committed");
    }
    let committed: usize = cards.verify.iter().flat_map(|(_, rounds)| rounds.iter().flatten()).map(|&(_, _, n)| n - 1).sum();
    println!("drafts committed over the verify runs: {committed}");
    assert!(committed > 0, "no draft landed: the verify rounds checked no draft");
}
