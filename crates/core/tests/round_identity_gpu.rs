//! The round identity test (GitHub #674, spec #672): two arms of the 27B's
//! decode round, chosen by measurement switches, run the same prompts and the
//! same draws and must commit the same tokens, at every width a load serves
//! and at its draft window, on every 27B load the spec names, under each
//! exception setting the load has, with greedy lanes and with sampled ones.
//!
//! Every arm is paired with the reference: the decode and verify graphs
//! replayed with every measurement switch unset, today's path. The first two
//! arms are pairs main already keeps byte for byte: `eager`, the same
//! traversal enqueued directly with no graph captured, and `round-timing`,
//! `IGNIS_ROUND_TIMING=1`, an instrument that moves no token, whose arm also
//! captures the graphs a second time on the same load. Each lever of spec
//! #672 appends its switch as an arm.
//!
//! # How a lever adds its arm
//!
//! One entry in [`ARMS`], under its ticket's line: a name for
//! `IGNIS_ROUND_IDENTITY_ARMS`, the switches the arm sets (main's path, as
//! `("IGNIS_DECODE_GRAPH_COPIES", "1")`) and `graphs: true`. Before each arm
//! the test unsets every switch any arm names and sets the arm's own, captures
//! the graphs again on the same load and pool (a switch read at capture takes
//! effect there, a switch read per round at its next round), runs the session
//! once with greedy lanes and once with sampled ones, and holds both to the
//! reference's tokens. An arm with `graphs: false` runs eagerly, before the
//! load's first capture: a load has no way back to eager rounds once a width
//! is captured, so a switch read only at capture never reaches such an arm.
//! A lever that needs a load the table lacks adds a [`LOADS`] entry; a wider
//! draft window (#666) is a load at that window: `window` is a field of the
//! load, and the session's context grows with it.
//!
//! # The session
//!
//! On one sequence pool of the load's lanes, lanes join one at a time: lane
//! `w - 1` is prefilled between the rounds of the lanes already there, then
//! the `w` lanes ride [`PLAIN_ROUNDS`] plain rounds and [`VERIFY_ROUNDS`]
//! verify rounds together, each verify round drafting the load's whole window
//! (no budget, no stop ids). At the widest, every odd lane is released and
//! the rest ride verify rounds, then plain ones. So every width from 1 to the
//! load's lanes runs both kinds of round, from the graphs or eagerly as the
//! arm says (every round asserts which), and the transitions a lever that
//! keeps work in flight has to survive all happen: a prefill and a release
//! between rounds, a width that grows and one that shrinks, a plain round
//! after a verify round and the reverse. Every round's tokens are compared,
//! each verify run ended by -1 in its lane's stream.
//!
//! On the vision load, lane 0's prompt is a multimodal span whose first four
//! tokens rotate as a 2x2 image at the prompt's start would, which leaves the
//! sequence a rope delta of [`IMAGE_ROPE_DELTA`]: every round after rotates
//! the lane at its positions plus that delta, the input a round stages for a
//! lane with an image.
//!
//! # Loads and weights
//!
//! [`LOADS`]: the NVFP4 body with DFlash2 at 7, with its MTP head at the
//! window the 12 GB class drafts at, and with vision bound beside DFlash2 (the
//! RTX 5090 profile's load, the tower in host memory); the trellis bodies of
//! the 16 GB class (the `sc30` artifact, DFlash2 at 7) and of the 12 GB class
//! (`ig28`, MTP at 7). Each drafts by the server's default rule for its
//! drafter, over hq-e8-2b KV, at 8 lanes: past the 16 and 12 GB profiles' 2
//! and 1, which an operator can raise, so that widths 1-8 run on both bodies.
//! Every load names its exception weights. The NVFP4 body runs under `bf16`
//! and under `fp8`, which needs compute capability 12.0 (on another card, set
//! `IGNIS_ROUND_IDENTITY_WEIGHTS=bf16`). The trellis body holds no BF16
//! exception tensor, so `fp8` would load it byte for byte as `bf16` does: it
//! runs once, as `bf16`. The test binds and loads the model itself and reads
//! no config file, the user's included. One materialize per load and weights,
//! released before the next; one model load and one pool serve every arm.
//!
//! # The decode graph's nodes
//!
//! On the NVFP4 body with DFlash2 the reference's capture and the
//! `decode-graph-copies` arms' are also held to their copies (#677, #680,
//! [`copy_free_graphs`]): on a card of compute capability 12 the reference's
//! decode graphs copy nothing at any width, its width-1 graph is
//! [`COPY_FREE_WIDTH_1_NODES`] nodes, and its verify graphs keep only the
//! drafter's [`DRAFTER_COPIES`]; the `decode-graph-copies` arm's are main's,
//! with their copies, and the `decode-graph-copies-wide` arm's main's but the
//! width-1 decode graph, which is the reference's.
//!
//! # Narrow runs
//!
//! `IGNIS_ROUND_IDENTITY_LOADS` (a comma list of load names),
//! `IGNIS_ROUND_IDENTITY_ARMS` (a comma list of arm names) and
//! `IGNIS_ROUND_IDENTITY_WEIGHTS` (`bf16`, `fp8`, or both) narrow the run;
//! each, unset, is everything. A name the test does not know fails it, naming
//! the ones it knows, and so does a run that compares nothing.
//!
//! ```text
//! IGNIS_ROUND_IDENTITY_LOADS=nvfp4-dflash2 cargo test -p ignis-core --features cuda --test round_identity_gpu -- --ignored --nocapture
//! ```
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a **skip**; under the profile the same
//! condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b_with, materialize, CudaDevice, FrontendSet, Reader, VisionResidency};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, model_scope, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::speculation::QWEN38_MTP_DEFAULT_DRAFT_TOKENS;
use ignis_core::step::{
    capture_decode_graphs, decode_program_batch_sampled, decode_program_verify_runs, prefill_program_multimodal,
    prefill_program_sampled, program_stats, MultimodalPrefill, Redirect, SamplingParams, VerifyLane,
};
use ignis_core::{
    DraftRule, DrafterSampling, ExceptionWeights, KvFormat, PrefillAttention, PrefillLinear, RopeScaling,
    Speculation, SpeculativeBackend, Vision,
};

/// One arm of the round: the measurement switches it sets (every other switch
/// an arm names is unset) and whether its rounds replay the captured graphs.
#[derive(Debug)]
struct Arm {
    name: &'static str,
    switches: &'static [(&'static str, &'static str)],
    graphs: bool,
}

/// What every arm is held to: the graphs replayed, every switch unset.
const REFERENCE: Arm = Arm { name: "reference", switches: &[], graphs: true };

/// The arms, each held to [`REFERENCE`]. A lever appends its own under its
/// ticket's line, so that levers built side by side merge without a conflict.
const ARMS: &[Arm] = &[
    // #674: the eager round, which main keeps byte for byte with the graph's replay.
    Arm { name: "eager", switches: &[], graphs: false },
    // #674: the round timing, an instrument; its arm captures the graphs a second time.
    Arm { name: "round-timing", switches: &[("IGNIS_ROUND_TIMING", "1")], graphs: true },
    // #676's arm, below this line.

    // #677's arm, below this line.
    Arm { name: "decode-graph-copies", switches: &[("IGNIS_DECODE_GRAPH_COPIES", "1")], graphs: true },

    // #678's arm, below this line.
    // The FP8 q|k|gate|v projection's one-row GEMV: an `fp8` load's lever (`bf16` runs no FP8 GEMV).
    Arm { name: "fp8-gemv-one-row", switches: &[("IGNIS_FP8_GEMV_PAIRED", "0")], graphs: true },

    // #679's arm, below this line.

    // #680's arm, below this line.
    Arm { name: "decode-graph-copies-wide", switches: &[("IGNIS_DECODE_GRAPH_COPIES", "wide")], graphs: true },

    // #681's arm, below this line.

];

/// One 27B load every arm runs on.
struct Load {
    /// Its name in `IGNIS_ROUND_IDENTITY_LOADS`.
    name: &'static str,
    /// The artifact, in the models folder.
    artifact: &'static str,
    /// The drafter, the draft window every verify round drafts, and the draft
    /// rule the server's default gives that drafter.
    backend: SpeculativeBackend,
    window: u32,
    draft_rule: DraftRule,
    /// The widest round: the session runs every width from 1 to it.
    lanes: u32,
    /// Vision bound, the tower in host memory as the RTX 5090 profile has it.
    vision: bool,
    /// Whether the body holds BF16 exception tensors, which `fp8` converts.
    exceptions: bool,
}

const NVFP4: &str = "F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer";
/// The 16 GB class's trellis artifact, and the 12 GB class's.
const TRELLIS_16GB: &str = "F:/ai/q38/ninfer-models/qwen3_8_27b_trellis_sc30-v2.ninfer";
const TRELLIS_12GB: &str = "F:/ai/q38/ninfer-models/qwen3_8_27b_trellis_ig28-v2.ninfer";

const LOADS: &[Load] = &[
    Load {
        name: "nvfp4-dflash2",
        artifact: NVFP4,
        backend: SpeculativeBackend::Dflash2,
        window: 7,
        draft_rule: DraftRule::Sampled,
        lanes: 8,
        vision: false,
        exceptions: true,
    },
    Load {
        name: "nvfp4-mtp",
        artifact: NVFP4,
        backend: SpeculativeBackend::Mtp,
        window: QWEN38_MTP_DEFAULT_DRAFT_TOKENS,
        draft_rule: DraftRule::Fixed,
        lanes: 8,
        vision: false,
        exceptions: true,
    },
    Load {
        name: "nvfp4-vision",
        artifact: NVFP4,
        backend: SpeculativeBackend::Dflash2,
        window: 7,
        draft_rule: DraftRule::Sampled,
        lanes: 8,
        vision: true,
        exceptions: true,
    },
    Load {
        name: "trellis-16gb",
        artifact: TRELLIS_16GB,
        backend: SpeculativeBackend::Dflash2,
        window: 7,
        draft_rule: DraftRule::Sampled,
        lanes: 8,
        vision: false,
        exceptions: false,
    },
    Load {
        name: "trellis-12gb",
        artifact: TRELLIS_12GB,
        backend: SpeculativeBackend::Mtp,
        window: QWEN38_MTP_DEFAULT_DRAFT_TOKENS,
        draft_rule: DraftRule::Fixed,
        lanes: 8,
        vision: false,
        exceptions: false,
    },
];

/// #677, #680: the load whose decode and verify graphs [`copy_free_graphs`]
/// pins, and its copy-free width-1 decode graph's nodes on a card of compute
/// capability 12, every one a kernel; main's decode round copies each of the
/// 64 layers' residual and each of the 48 GDN layers' q, k and v out of the
/// conv plane, at every width, and main's verify round each layer's residual
/// beside the drafter's copies: its five feature taps, the final hidden it
/// keeps and its own forward's sixteen, which no copy-free round removes.
const COPY_FREE_LOAD: &str = "nvfp4-dflash2";
const COPY_FREE_WIDTH_1_NODES: u32 = 677;
const MAIN_COPIES: u32 = 64 + 3 * 48;
const RESIDUAL_COPIES: u32 = 64;
const DRAFTER_COPIES: u32 = 5 + 1 + 16;

/// Plain rounds, then verify rounds, at each width of the session.
const PLAIN_ROUNDS: u32 = 4;
const VERIFY_ROUNDS: u32 = 4;
/// The most tokens a prompt below takes.
const PROMPT_TOKENS: u32 = 64;
/// The load's prefill chunk: every prompt fits one.
const CHUNK: u32 = 128;
/// The vision lane's rope delta: a 2x2 image's four tokens take two positions.
const IMAGE_ROPE_DELTA: i32 = -2;

/// One prompt per lane, in a user turn with thinking closed.
const PROMPTS: [&str; 8] = [
    "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
    "List the planets of the solar system in order, one per line.",
    "Explain in two sentences what a hash map is.",
    "Write a JSON object describing a book with its title, author, year and three tags.",
    "Compute, step by step, 17 * 23 and give the final number on the last line.",
    "Write a Rust function that reverses a string, with a doc comment.",
    "Name five prime numbers greater than 100 and say why each one is prime.",
    "Describe how a binary search works, as a numbered list of steps.",
];

impl Load {
    fn speculation(&self) -> Speculation {
        Speculation::new(self.backend, self.window)
            .unwrap_or_else(|e| panic!("{}: speculation: {e}", self.name))
            .with_draft_rule(self.draft_rule, DrafterSampling::default())
    }

    fn vision(&self) -> Option<Vision> {
        self.vision.then(|| Vision::default().with_residency(VisionResidency::Host))
    }

    /// The context the session's longest-lived lane needs: its prompt, then
    /// at each width and once more after the release its plain tokens and
    /// whole verify runs.
    fn context(&self) -> u32 {
        let tokens = PROMPT_TOKENS + (self.lanes + 1) * (PLAIN_ROUNDS + VERIFY_ROUNDS * (self.window + 1));
        tokens.div_ceil(128) * 128
    }
}

/// What a session commits, by lane: every plain round's token and every
/// verify round's run, each run followed by -1, in the order the lane rode
/// them.
type Committed = Vec<Vec<i32>>;

fn greedy(_lane: usize) -> SamplingParams {
    SamplingParams::greedy()
}

/// Each lane its own seed, temperature, truncation and penalties, so that the
/// sampler's per-lane settings, the penalty counts and a verify round's
/// take-back of a cut run's counts are all under the comparison.
fn sampled(lane: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.6 + 0.1 * lane as f32,
        top_k: if lane % 2 == 0 { 10 } else { 0 },
        top_p: if lane % 3 == 0 { 0.95 } else { 1.0 },
        presence_penalty: 0.1 * (lane % 3) as f32,
        frequency_penalty: 0.2 * (lane % 2) as f32,
        seed: 0x0674_0000 + lane as u64,
    }
}

const KINDS: [(&str, fn(usize) -> SamplingParams); 2] = [("greedy", greedy), ("sampled", sampled)];

/// The names `variable` selects among `known`: all of them when it is unset
/// or empty. Panics on a name it does not know, naming the ones it knows.
fn selected(variable: &str, known: &[&'static str]) -> Vec<&'static str> {
    let Ok(value) = std::env::var(variable) else {
        return known.to_vec();
    };
    if value.trim().is_empty() {
        return known.to_vec();
    }
    value
        .split(',')
        .map(str::trim)
        .map(|name| {
            *known
                .iter()
                .find(|&&k| k == name)
                .unwrap_or_else(|| panic!("{variable}: `{name}` is none of {}", known.join(", ")))
        })
        .collect()
}

/// Unsets every switch an arm names, then sets `arm`'s own.
fn apply(arm: &Arm) {
    for &(name, _) in ARMS.iter().flat_map(|a| a.switches) {
        // SAFETY (test-only): this binary's one test reads and writes the
        // environment on its own thread, where the leaf reads it too.
        unsafe { std::env::remove_var(name) };
    }
    for &(name, value) in arm.switches {
        // SAFETY: as above.
        unsafe { std::env::set_var(name, value) };
    }
}

fn chat_prompts(reader: &Reader) -> Vec<Vec<i32>> {
    let frontend = FrontendSet::from_reader(reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    PROMPTS
        .iter()
        .map(|prompt| {
            let tokens: Vec<i32> = frontend
                .tokenizer()
                .encode(&format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
                .unwrap_or_else(|e| panic!("tokenize: {e}"))
                .into_iter()
                .map(|id| i32::try_from(id).expect("token id fits i32"))
                .collect();
            assert!(tokens.len() as u32 <= PROMPT_TOKENS, "a prompt of {} tokens", tokens.len());
            tokens
        })
        .collect()
}

/// Axis-major `[3, n]` positions of a prompt whose first four tokens rotate
/// as a 2x2 image at its start would (temporal 0, height and width over the
/// grid) and whose later tokens rotate at their index plus
/// [`IMAGE_ROPE_DELTA`].
fn image_positions(n: usize) -> Vec<i32> {
    let mut axes = vec![0i32; 3 * n];
    for i in 0..n {
        let (t, h, w) = if i < 4 {
            (0, (i / 2) as i32, (i % 2) as i32)
        } else {
            let p = i as i32 + IMAGE_ROPE_DELTA;
            (p, p, p)
        };
        axes[i] = t;
        axes[n + i] = h;
        axes[2 * n + i] = w;
    }
    axes
}

fn prefill(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, prompt: &[i32], sampling: SamplingParams, image: bool, what: &str) {
    if image {
        let positions = image_positions(prompt.len());
        let span = MultimodalPrefill { positions: &positions, rope_delta: IMAGE_ROPE_DELTA, media: None, previous: None };
        prefill_program_multimodal(model, pool, seq, prompt, 0, 0, None, false, false, sampling, &[], Redirect::NONE, span, None, None)
            .unwrap_or_else(|e| panic!("{what}: multimodal prefill: {e}"));
    } else {
        prefill_program_sampled(model, pool, seq, prompt, 0, sampling, None).unwrap_or_else(|e| panic!("{what}: prefill: {e}"));
    }
}

/// The riding lanes' sequences, in `riding`'s order.
fn riders<'a, 'p>(seqs: &'a mut [Option<Seq<'p>>], riding: &[usize]) -> Vec<&'a mut Seq<'p>> {
    let mut remaining: Vec<Option<&'a mut Seq<'p>>> = seqs.iter_mut().map(Option::as_mut).collect();
    riding.iter().map(|&lane| remaining[lane].take().expect("a live lane, once")).collect()
}

/// #677, #680: on [`COPY_FREE_LOAD`], the decode and verify graphs the
/// arm's capture left, for the reference and the arms that name main's
/// copying rounds. On a card of compute capability 12 with every switch
/// unset, no decode graph copies, the width-1 one is
/// [`COPY_FREE_WIDTH_1_NODES`] nodes, and every verify graph copies only the
/// drafter's [`DRAFTER_COPIES`]; under `IGNIS_DECODE_GRAPH_COPIES=1`, and on
/// any other card, every decode graph copies main's [`MAIN_COPIES`] and every
/// verify graph the drafter's and the [`RESIDUAL_COPIES`]; under `wide`, the
/// same but the width-1 decode graph, the reference's.
fn copy_free_graphs(model: &Model, pool: &SeqPool, load: &Load, arm: &Arm, label: &str) {
    let named = arm.switches.iter().find(|&&(name, _)| name == "IGNIS_DECODE_GRAPH_COPIES").map(|&(_, value)| value);
    if load.name != COPY_FREE_LOAD || !(arm.switches.is_empty() || named.is_some()) {
        return;
    }
    let (_, (major, _)) = CudaDevice::identity(0).unwrap_or_else(|e| panic!("{label}: device identity: {e}"));
    let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("{label}: stats: {e}"));
    let lanes = load.lanes as usize;
    let (nodes, copies) = (&stats.decode_graph_nodes[..lanes], &stats.decode_graph_copy_nodes[..lanes]);
    let (verify_nodes, verify_copies) = (&stats.verify_graph_nodes[..lanes], &stats.verify_graph_copy_nodes[..lanes]);
    println!(
        "round_identity_gpu {label}: arm `{}`: decode graph nodes {nodes:?}, copies {copies:?}; verify graph nodes \
         {verify_nodes:?}, copies {verify_copies:?}",
        arm.name
    );
    let blackwell = major == 12;
    let width_1_free = blackwell && named != Some("1");
    let wide_free = blackwell && named.is_none();
    assert_eq!(copies[0], if width_1_free { 0 } else { MAIN_COPIES }, "{label}: arm `{}`: width 1's copy nodes", arm.name);
    if blackwell {
        assert_eq!(nodes[0], COPY_FREE_WIDTH_1_NODES + copies[0], "{label}: arm `{}`: width 1's nodes", arm.name);
    }
    let wide = if wide_free { 0 } else { MAIN_COPIES };
    assert!(copies[1..].iter().all(|&c| c == wide), "{label}: arm `{}`: widths 2-{lanes} copy {wide} nodes", arm.name);
    let verify = DRAFTER_COPIES + if wide_free { 0 } else { RESIDUAL_COPIES };
    assert!(verify_copies.iter().all(|&c| c == verify), "{label}: arm `{}`: the verify graphs copy {verify} nodes", arm.name);
}

fn expect_replay(model: &Model, pool: &SeqPool, graphs: bool, what: &str) {
    let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("{what}: stats: {e}"));
    assert_eq!(stats.graph_launches, u64::from(graphs), "{what}: graph replay");
}

/// The session's rounds of one kind over the riding lanes.
#[allow(clippy::too_many_arguments)]
fn ride(
    model: &Model,
    pool: &SeqPool,
    load: &Load,
    seqs: &mut [Option<Seq<'_>>],
    riding: &[usize],
    sampling: fn(usize) -> SamplingParams,
    graphs: bool,
    verify: bool,
    committed: &mut Committed,
    label: &str,
) {
    let width = riding.len();
    for round in 0..if verify { VERIFY_ROUNDS } else { PLAIN_ROUNDS } {
        let what = format!("{label}: width {width} {} round {round}", if verify { "verify" } else { "plain" });
        let mut refs = riders(seqs, riding);
        if verify {
            let lanes: Vec<VerifyLane<'_>> =
                riding.iter().map(|&lane| VerifyLane { sampling: sampling(lane), ..VerifyLane::greedy(&[]) }).collect();
            let runs = decode_program_verify_runs(model, pool, &mut refs, &lanes, load.window)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            expect_replay(model, pool, graphs, &what);
            for (&lane, run) in riding.iter().zip(runs) {
                assert_eq!(run.extent, load.window, "{what}: lane {lane} drafts the whole window");
                committed[lane].extend(run.tokens);
                committed[lane].push(-1);
            }
        } else {
            let params: Vec<SamplingParams> = riding.iter().map(|&lane| sampling(lane)).collect();
            let drawn = decode_program_batch_sampled(model, pool, &mut refs, &params).unwrap_or_else(|e| panic!("{what}: {e}"));
            expect_replay(model, pool, graphs, &what);
            for (&lane, token) in riding.iter().zip(drawn) {
                committed[lane].push(token);
            }
        }
    }
}

/// The session (see the module's note) on `pool`, under the arm's graphs or
/// eagerly.
fn session(model: &Model, pool: &SeqPool, load: &Load, prompts: &[Vec<i32>], sampling: fn(usize) -> SamplingParams, graphs: bool, label: &str) -> Committed {
    let lanes = load.lanes as usize;
    let mut seqs: Vec<Option<Seq<'_>>> = Vec::with_capacity(lanes);
    let mut committed: Committed = vec![Vec::new(); lanes];
    for width in 1..=lanes {
        let lane = width - 1;
        let mut seq = pool.alloc(load.context()).unwrap_or_else(|e| panic!("{label}: alloc lane {lane}: {e}"));
        prefill(model, pool, &mut seq, &prompts[lane], sampling(lane), load.vision && lane == 0, &format!("{label}: lane {lane}"));
        seqs.push(Some(seq));
        let riding: Vec<usize> = (0..width).collect();
        ride(model, pool, load, &mut seqs, &riding, sampling, graphs, false, &mut committed, label);
        ride(model, pool, load, &mut seqs, &riding, sampling, graphs, true, &mut committed, label);
    }
    if lanes > 1 {
        for seq in seqs.iter_mut().skip(1).step_by(2) {
            *seq = None;
        }
        let riding: Vec<usize> = (0..lanes).step_by(2).collect();
        ride(model, pool, load, &mut seqs, &riding, sampling, graphs, true, &mut committed, label);
        ride(model, pool, load, &mut seqs, &riding, sampling, graphs, false, &mut committed, label);
    }
    committed
}

/// The arm's sessions, greedy then sampled, its switches set and, for an arm
/// that replays graphs, the graphs captured under them.
fn run(model: &Model, pool: &SeqPool, load: &Load, prompts: &[Vec<i32>], arm: &Arm, label: &str) -> Vec<Committed> {
    apply(arm);
    if arm.graphs {
        let capture = capture_decode_graphs(model, pool).unwrap_or_else(|e| panic!("{label}: {}: capture: {e}", arm.name));
        let every = (1u32 << load.lanes) - 1;
        let verify = program_stats(model, pool).unwrap_or_else(|e| panic!("{label}: stats: {e}")).verify_graph_ready_mask;
        assert_eq!((capture.ready_mask, verify), (every, every), "{label}: {}: every width's decode and verify graph", arm.name);
        copy_free_graphs(model, pool, load, arm, label);
    }
    let sessions = KINDS
        .iter()
        .map(|&(kind, sampling)| session(model, pool, load, prompts, sampling, arm.graphs, &format!("{label} {} {kind}", arm.name)))
        .collect();
    apply(&REFERENCE);
    sessions
}

/// The first lane, and the first token of it, where `got` leaves `want`.
fn first_difference(want: &Committed, got: &Committed) -> Option<(usize, usize)> {
    want.iter().zip(got).enumerate().find_map(|(lane, (w, g))| {
        (w != g).then(|| (lane, w.iter().zip(g).position(|(a, b)| a != b).unwrap_or(w.len().min(g.len()))))
    })
}

/// Every selected arm against the reference on one load under one exception
/// setting: one materialize, one model load, one pool. The arms compared, or
/// `None` where the profile allows a skip.
fn hold(device: &mut CudaDevice, reader: &Reader, load: &Load, exceptions: ExceptionWeights, arms: &[&Arm], prompts: &[Vec<i32>]) -> Option<usize> {
    let label = format!("{} {exceptions}", load.name);
    let spec = load.speculation();
    let (plan, handles) = bind_model_scope_27b_with(reader, model_scope(Some(spec), load.vision(), exceptions))
        .unwrap_or_else(|e| panic!("{label}: bind: {e}"));
    let mut artifact = match materialize(reader, &plan, device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("{label}: materialize: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let converted = artifact.stats().fp8_device_bytes;
    assert_eq!(converted > 0, exceptions.is_fp8(), "{label}: {converted} bytes of exception tensors held in FP8");
    let compared = {
        let model = load_qwen38_27b_with_decode_lanes(
            reader,
            &artifact,
            &handles,
            CHUNK,
            load.context(),
            KvFormat::HqE8_2b,
            Some(spec),
            load.vision(),
            RopeScaling::NONE,
            PrefillAttention::default(),
            PrefillLinear::default(),
            load.lanes,
        )
        .unwrap_or_else(|e| panic!("{label}: model load: {e}"));
        let pool = SeqPool::create_with_speculation(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: KvFormat::HqE8_2b,
                kv_page_group_count: load.context().div_ceil(64) * load.lanes,
                max_context_tokens: load.context(),
                slot_count: load.lanes,
                retained_slot_count: 0,
                retained_host_slot_count: 0,
            },
            Some(load.backend),
        )
        .unwrap_or_else(|e| panic!("{label}: pool: {e}"));
        // An eager arm runs before the load's first capture.
        let eager: Vec<(&Arm, Vec<Committed>)> =
            arms.iter().filter(|arm| !arm.graphs).map(|&arm| (arm, run(&model, &pool, load, prompts, arm, &label))).collect();
        let reference = run(&model, &pool, load, prompts, &REFERENCE, &label);
        assert_ne!(reference[0], reference[1], "{label}: the sampled lanes commit the greedy lanes' tokens: their settings never reached the sampler");
        let mut compared = 0;
        let mut hold_arm = |arm: &Arm, sessions: Vec<Committed>| {
            for (((kind, _), want), got) in KINDS.iter().zip(&reference).zip(&sessions) {
                if let Some((lane, at)) = first_difference(want, got) {
                    let from = at.saturating_sub(4);
                    panic!(
                        "{label}: arm `{}`, {kind} lanes: lane {lane} leaves the reference at token {at}: \
                         reference {:?}, arm {:?}",
                        arm.name,
                        &want[lane][from..want[lane].len().min(at + 8)],
                        &got[lane][from..got[lane].len().min(at + 8)]
                    );
                }
            }
            let tokens: usize = sessions.iter().flatten().flatten().filter(|&&t| t >= 0).count();
            println!(
                "round_identity_gpu {label}: arm `{}` commits the reference's {tokens} tokens, widths 1..={}, \
                 verify rounds at {}, greedy and sampled lanes",
                arm.name, load.lanes, load.window
            );
            compared += 1;
        };
        for (arm, sessions) in eager {
            hold_arm(arm, sessions);
        }
        for &arm in arms.iter().filter(|arm| arm.graphs) {
            let sessions = run(&model, &pool, load, prompts, arm, &label);
            hold_arm(arm, sessions);
        }
        compared
    };
    artifact.release_arena(device).unwrap_or_else(|e| panic!("{label}: release: {e}"));
    Some(compared)
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn every_arm_commits_the_reference_rounds_tokens_at_every_width_and_the_draft_window() {
    let loads = selected("IGNIS_ROUND_IDENTITY_LOADS", &LOADS.iter().map(|load| load.name).collect::<Vec<_>>());
    let arm_names = selected("IGNIS_ROUND_IDENTITY_ARMS", &ARMS.iter().map(|arm| arm.name).collect::<Vec<_>>());
    let weights = selected("IGNIS_ROUND_IDENTITY_WEIGHTS", &ExceptionWeights::NAMES);
    let arms: Vec<&Arm> = ARMS.iter().filter(|arm| arm_names.contains(&arm.name)).collect();
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let (mut compared, mut skipped) = (0, false);
    for load in LOADS.iter().filter(|load| loads.contains(&load.name)) {
        let path = Path::new(load.artifact);
        if !path.exists() && gpu_profile::skip_or_fail(&format!("{}: artifact absent: {}", load.name, load.artifact)) {
            skipped = true;
            continue;
        }
        let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", load.artifact));
        let prompts = chat_prompts(&reader);
        for exceptions in [ExceptionWeights::Bf16, ExceptionWeights::Fp8] {
            if !weights.contains(&exceptions.as_str()) {
                continue;
            }
            if exceptions.is_fp8() && !load.exceptions {
                println!("round_identity_gpu {}: no BF16 exception tensor, so fp8 is its bf16 load: run once, as bf16", load.name);
                continue;
            }
            match hold(&mut device, &reader, load, exceptions, &arms, &prompts) {
                Some(arms_held) => compared += arms_held,
                None => skipped = true,
            }
        }
    }
    assert!(compared > 0 || skipped, "no arm was held to the reference: check the IGNIS_ROUND_IDENTITY_* filters");
}
