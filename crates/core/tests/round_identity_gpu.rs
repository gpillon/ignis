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
//! (no budget, no stop ids). Halfway through the verify rounds come a
//! constrained round, lane 0 held to a permitted set (the ten digits), which
//! runs eagerly whatever the arm, then a verify round whose every run the
//! host cuts one token short, as a grammar's check-first cut does (GitHub
//! #590): a sampled lane's count of the token cut off comes back to the host,
//! and the session fails if no run of two or more was cut; then a
//! teacher-forced verify round (GitHub #267), each lane forced to a run of
//! the digits, which runs eagerly where a lane samples under the sampled
//! draft (its recording is on the device, which no graph was captured with)
//! and so holds the eager verify traversal of each arm, the copy-free one
//! included, to the reference through the rounds after it. At the widest,
//! every odd lane is released and the rest ride verify rounds; each of them
//! is then snapshotted and restored from its snapshot, right after a verify
//! round that left its fold in flight where the arm defers, so that a
//! snapshot taken behind a fold still on the device has to carry the
//! committed state (the `sync-verify` arm's, taken after a synchronized fold,
//! is the comparison); then they ride plain rounds. So every width from 1 to
//! the load's lanes runs every kind of round,
//! from the graphs or eagerly as the arm says (every round asserts which), and
//! the transitions a lever that keeps work in flight has to survive all
//! happen: a prefill and a release between rounds, a width that grows and one
//! that shrinks, a plain or a constrained round after a verify round and the
//! reverse. Every round's tokens are compared, each verify run ended by -1 in
//! its lane's stream.
//!
//! GitHub #676: a DFlash2 load's verify rounds return with their fold in
//! flight once its graphs are captured, unless the arm sets
//! `IGNIS_DEFER_VERIFY=0`, and an MTP load's never do; so the `eager` arm's,
//! run before the load's first capture, return as `sync-verify`'s do. Each
//! arm's sessions assert which from the load's count of such rounds, and a
//! capture's drafter append graphs from their ready mask. The NVFP4 body
//! loads once per exception setting in one process, so a run of both settings
//! holds a second load's deferred rounds, on staging of its own, to main's
//! tokens: what the host-mapped memory following the load, not the process,
//! has to keep.
//!
//! On the vision load, lane 0's prompt is a multimodal span whose first four
//! tokens rotate as a 2x2 image at the prompt's start would, which leaves the
//! sequence a rope delta of [`IMAGE_ROPE_DELTA`]: every round after rotates
//! the lane at its positions plus that delta, the input a round stages for a
//! lane with an image.
//!
//! # Round shapes (#687)
//!
//! A DFlash2 load whose window is past [`BASE_WINDOW`] holds two round
//! shapes: the base shape, [`BASE_WINDOW`] + 1 columns a lane at every width,
//! and the extended shape, the load's window + 1 columns a lane at up to
//! [`EXTENDED_LANES`] lanes. `nvfp4-dflash2-w15`, `nvfp4-dflash2-w10` (a
//! window whose second attention chunk is 3 columns, not a whole tile) and
//! `nvfp4-vision-w15` (lane 0 under a rope delta) are such loads, and every
//! arm of them runs, the reference included, under [`ROUND_SHAPE`]`=extended`
//! ([`Load::round_shape`]): their verify rounds are extended at widths 1 to
//! [`EXTENDED_LANES`], and at the base shape past them and wherever the host
//! cuts a run (a round with a host-cut lane keeps the base shape). Every
//! verify round asserts the drafts its shape carries ([`round_drafts`]), so
//! each arm holds the extended round -- from its graphs, eager, deferred and
//! synchronized, timed, teacher-forced -- to the reference's tokens beside
//! the base one, and the program's stats report both shapes
//! ([`round_shape_stats`]).
//!
//! The base shape of such a load is the round of a load at [`BASE_WINDOW`]
//! drafts, token for token: the second test here
//! ([`a_fifteen_draft_loads_base_rounds_commit_a_seven_draft_loads_tokens`])
//! runs the session on `nvfp4-dflash2-w15` with the switch unset and on a load
//! at [`BASE_WINDOW`] that reserves its context ([`Load::sized_for`]), and
//! compares the two, greedy lanes and sampled ones, at every width; then a
//! verify round under a value the switch does not know is refused, naming it.
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
//! width-1 decode graph, which is the reference's. #681: each `gdn-gating-*`
//! arm's capture reports the home its switch names
//! (`ProgramStats::gdn_gating`), and its graphs copy as the reference's do.
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
    capture_decode_graphs, decode_program_batch_permitted, decode_program_batch_sampled, decode_program_verify_cut,
    decode_program_verify_forced, decode_program_verify_runs, prefill_program_multimodal, prefill_program_sampled, program_stats, MultimodalPrefill,
    PermittedLane, Redirect, SamplingParams, VerifyLane,
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
    Arm { name: "sync-verify", switches: &[("IGNIS_DEFER_VERIFY", "0")], graphs: true },

    // #677's arm, below this line.
    Arm { name: "decode-graph-copies", switches: &[("IGNIS_DECODE_GRAPH_COPIES", "1")], graphs: true },

    // #678's arm, below this line.
    // The FP8 q|k|gate|v projection's one-row GEMV: an `fp8` load's lever (`bf16` runs no FP8 GEMV).
    Arm { name: "fp8-gemv-one-row", switches: &[("IGNIS_FP8_GEMV_PAIRED", "0")], graphs: true },

    // #679's arm, below this line.

    // #680's arm, below this line.
    Arm { name: "decode-graph-copies-wide", switches: &[("IGNIS_DECODE_GRAPH_COPIES", "wide")], graphs: true },

    // #681's arms, below this line: both homes of the GDN gating, so that the
    // one the unset switch does not take is held to the one it takes, whichever
    // that is (kernel/src/gdn_gating_home.h `kUnset`); `chain` is main's.
    Arm { name: "gdn-gating-chain", switches: &[("IGNIS_GDN_GATING", "chain")], graphs: true },
    Arm { name: "gdn-gating-branch", switches: &[("IGNIS_GDN_GATING", "branch")], graphs: true },

    // #688's arm, below this line.

    // #689's arms, below this line.

    // #691's arms, below this line.

    // #693's arm, below this line.
    // The FP8 exception products' 128-token tile at 9-16 tokens, main's: an `fp8` load's lever
    // (`bf16` runs no W8A8 product). It reaches a one-lane extended round (16 columns, 11 on the
    // window of 10), a two-lane base round (16) and a prefill chunk of 9-16 tokens
    // (`a_prefill_whose_last_chunk_is_9_to_16_tokens_writes_the_same_bytes_on_both_fp8_tiles`);
    // an extended round of 2-4 lanes (32-64 columns) takes the 128-token tile either way.
    Arm { name: TILE128_ARM, switches: &[("IGNIS_FP8_W8A8_TILE16", "0")], graphs: true },

    // #694's arms, below this line.

    // #695's arms, below this line.
];

/// #687: the switch that names a two-shape load's round shape, read at each
/// verify round; the base shape's drafts a lane, and the most lanes a round
/// at the extended shape runs.
const ROUND_SHAPE: &str = "IGNIS_ROUND_SHAPE";
const BASE_WINDOW: u32 = 7;
const EXTENDED_LANES: usize = 4;
/// The load whose base shape the second test holds to a 7-draft load's.
const TWO_SHAPE_LOAD: &str = "nvfp4-dflash2-w15";

/// #687: the drafts a verify round of `width` lanes carries each lane on
/// `load`: the load's window on a load of one shape; on a load of two, the
/// window at the extended shape, where the load runs it
/// ([`Load::round_shape`]), the round has no more than [`EXTENDED_LANES`]
/// lanes and no lane the host cuts, and the base shape's [`BASE_WINDOW`]
/// otherwise.
fn round_drafts(load: &Load, width: usize, host_cut: bool) -> u32 {
    if !load.two_shapes() {
        return load.window;
    }
    if load.round_shape == Some("extended") && width <= EXTENDED_LANES && !host_cut { load.window } else { BASE_WINDOW }
}

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
    /// #687: the window the session's context is sized for where it is not
    /// the load's own: a load held to another load's tokens reserves that
    /// load's context, the envelope its attention splits by.
    sized_for: Option<u32>,
    /// #687: [`ROUND_SHAPE`]'s value on every arm of the load, the reference
    /// included.
    round_shape: Option<&'static str>,
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
        sized_for: None,
        round_shape: None,
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
        sized_for: None,
        round_shape: None,
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
        sized_for: None,
        round_shape: None,
    },
    // #687: DFlash2 at 15 drafts, both round shapes, every arm under
    // `IGNIS_ROUND_SHAPE=extended`: its rounds are extended at widths 1-4 (16
    // columns a lane) and at the base shape at 5-8 (8).
    Load {
        name: "nvfp4-dflash2-w15",
        artifact: NVFP4,
        backend: SpeculativeBackend::Dflash2,
        window: 15,
        draft_rule: DraftRule::Sampled,
        lanes: 8,
        vision: false,
        exceptions: true,
        sized_for: None,
        round_shape: Some("extended"),
    },
    // #687: a window between the two shapes' ends, 11 columns a lane: the
    // extended round's second attention chunk is 3 columns, not a whole tile.
    Load {
        name: "nvfp4-dflash2-w10",
        artifact: NVFP4,
        backend: SpeculativeBackend::Dflash2,
        window: 10,
        draft_rule: DraftRule::Sampled,
        lanes: 8,
        vision: false,
        exceptions: true,
        sized_for: None,
        round_shape: Some("extended"),
    },
    // #687: both round shapes with vision bound, so that lane 0's extended
    // rounds rotate at its positions plus a rope delta, staged at the extended
    // shape's stride.
    Load {
        name: "nvfp4-vision-w15",
        artifact: NVFP4,
        backend: SpeculativeBackend::Dflash2,
        window: 15,
        draft_rule: DraftRule::Sampled,
        lanes: 8,
        vision: true,
        exceptions: true,
        sized_for: None,
        round_shape: Some("extended"),
    },
    // #688's loads, below this line.

    // #689's loads, below this line.

    // #691's loads, below this line.

    // #693's loads, below this line.

    // #694's loads, below this line.

    // #695's loads, below this line.

    Load {
        name: "trellis-16gb",
        artifact: TRELLIS_16GB,
        backend: SpeculativeBackend::Dflash2,
        window: 7,
        draft_rule: DraftRule::Sampled,
        lanes: 8,
        vision: false,
        exceptions: false,
        sized_for: None,
        round_shape: None,
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
        sized_for: None,
        round_shape: None,
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
/// #681: the GDN gating's homes, as `ProgramStats::gdn_gating` reports them.
const GATING_HOMES: [(&str, u32); 2] = [("chain", 0), ("branch", 1)];
const MAIN_COPIES: u32 = 64 + 3 * 48;
const RESIDUAL_COPIES: u32 = 64;
const DRAFTER_COPIES: u32 = 5 + 1 + 16;

/// Plain rounds, then verify rounds, at each width of the session.
const PLAIN_ROUNDS: u32 = 4;
const VERIFY_ROUNDS: u32 = 4;
/// And, halfway through the verify rounds, one constrained round, one
/// verify round the host cuts and one teacher-forced verify round.
const CONSTRAINED_ROUNDS: u32 = 1;
const CUT_ROUNDS: u32 = 1;
const FORCED_ROUNDS: u32 = 1;
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
    /// at each width and once more after the release its plain and
    /// constrained tokens and whole verify runs, the forced one's included.
    fn context(&self) -> u32 {
        let window = self.sized_for.unwrap_or(self.window);
        let rounds = PLAIN_ROUNDS + CONSTRAINED_ROUNDS + (VERIFY_ROUNDS + CUT_ROUNDS + FORCED_ROUNDS) * (window + 1);
        let tokens = PROMPT_TOKENS + (self.lanes + 1) * rounds;
        tokens.div_ceil(128) * 128
    }

    /// #687: whether the load holds two round shapes: DFlash2 past
    /// [`BASE_WINDOW`] drafts.
    fn two_shapes(&self) -> bool {
        self.backend == SpeculativeBackend::Dflash2 && self.window > BASE_WINDOW
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

/// Unsets every switch an arm names and [`ROUND_SHAPE`], then sets the load's
/// round shape, if it names one (#687), and `arm`'s own switches.
fn apply(arm: &Arm, round_shape: Option<&str>) {
    for name in ARMS.iter().flat_map(|a| a.switches).map(|&(name, _)| name).chain([ROUND_SHAPE]) {
        // SAFETY (test-only): this binary's one test reads and writes the
        // environment on its own thread, where the leaf reads it too.
        unsafe { std::env::remove_var(name) };
    }
    if let Some(value) = round_shape {
        // SAFETY: as above.
        unsafe { std::env::set_var(ROUND_SHAPE, value) };
    }
    for &(name, value) in arm.switches {
        // SAFETY: as above.
        unsafe { std::env::set_var(name, value) };
    }
}

/// What every session reads: one prompt per lane, and the ids the
/// constrained round holds lane 0 to.
struct Script {
    prompts: Vec<Vec<i32>>,
    permitted: Vec<i32>,
}

fn script(reader: &Reader) -> Script {
    let frontend = FrontendSet::from_reader(reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let encode = |text: &str| -> Vec<i32> {
        frontend
            .tokenizer()
            .encode(text)
            .unwrap_or_else(|e| panic!("tokenize: {e}"))
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect()
    };
    let prompts = PROMPTS
        .iter()
        .map(|prompt| {
            let tokens = encode(&format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
            assert!(tokens.len() as u32 <= PROMPT_TOKENS, "a prompt of {} tokens", tokens.len());
            tokens
        })
        .collect();
    let permitted = (0..10)
        .map(|digit| match encode(&digit.to_string())[..] {
            [id] => id,
            ref ids => panic!("the digit {digit} is {} tokens", ids.len()),
        })
        .collect();
    Script { prompts, permitted }
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
/// same but the width-1 decode graph, the reference's. #681: and for the
/// `gdn-gating-*` arms, which copy as the reference does; on every load each
/// of them reports its switch's home on a card of compute capability 12 and
/// main's chain on any other.
fn copy_free_graphs(model: &Model, pool: &SeqPool, load: &Load, arm: &Arm, label: &str) {
    let named = arm.switches.iter().find(|&&(name, _)| name == "IGNIS_DECODE_GRAPH_COPIES").map(|&(_, value)| value);
    let gating = arm.switches.iter().find(|&&(name, _)| name == "IGNIS_GDN_GATING").map(|&(_, value)| value);
    let (_, (major, _)) = CudaDevice::identity(0).unwrap_or_else(|e| panic!("{label}: device identity: {e}"));
    let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("{label}: stats: {e}"));
    if let Some(home) = gating {
        let (_, value) = GATING_HOMES.iter().find(|&&(name, _)| name == home).expect("a home the switch names");
        let want = if major == 12 { *value } else { 0 };
        assert_eq!(stats.gdn_gating, want, "{label}: arm `{}`: the GDN gating's home", arm.name);
    }
    if load.name != COPY_FREE_LOAD || !(arm.switches.is_empty() || named.is_some() || gating.is_some()) {
        return;
    }
    let lanes = load.lanes as usize;
    let (nodes, copies) = (&stats.decode_graph_nodes[..lanes], &stats.decode_graph_copy_nodes[..lanes]);
    let (verify_nodes, verify_copies) = (&stats.verify_graph_nodes[..lanes], &stats.verify_graph_copy_nodes[..lanes]);
    println!(
        "round_identity_gpu {label}: arm `{}`: decode graph nodes {nodes:?}, copies {copies:?}; verify graph nodes \
         {verify_nodes:?}, copies {verify_copies:?}; GDN gating home {}",
        arm.name, stats.gdn_gating
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

/// #686 (spec #684): the round shapes a load builds, and the program's stats
/// report each apart. Its base shape, at the load's window and lanes (#687:
/// at [`BASE_WINDOW`] on a load of two shapes): its graphs are the load's
/// verify and drafter append graphs, and its VRAM the load's verify round and
/// drafter round lines. #687: on a DFlash2 load past [`BASE_WINDOW`] drafts,
/// its extended shape too, at the load's window and up to [`EXTENDED_LANES`]
/// lanes, which holds the base shape's buffers and none of its own; its
/// verify graphs are captured at its every width whenever the base shape's
/// are, and its append graphs where the base shape's are.
fn round_shape_stats(model: &Model, pool: &SeqPool, load: &Load, label: &str) {
    let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("{label}: stats: {e}"));
    let shapes = stats.shapes();
    assert_eq!(shapes.len(), if load.two_shapes() { 2 } else { 1 }, "{label}: the round shapes");
    let base = &shapes[0];
    let base_window = if load.two_shapes() { BASE_WINDOW } else { load.window };
    assert_eq!((base.window, base.lanes), (base_window, load.lanes), "{label}: the base shape's window and lanes");
    assert_eq!(
        (base.verify_graph_ready_mask, base.drafter_append_graph_ready_mask, base.verify_folds_in_flight),
        (stats.verify_graph_ready_mask, stats.drafter_append_graph_ready_mask, stats.verify_folds_in_flight),
        "{label}: the base shape's graphs and folds"
    );
    assert_eq!(
        (base.verify_graph_nodes, base.verify_graph_copy_nodes),
        (stats.verify_graph_nodes, stats.verify_graph_copy_nodes),
        "{label}: the base shape's graph nodes"
    );
    let reserved = model.stats().reserved;
    assert_eq!(base.vram_bytes, reserved.verify_round_bytes + reserved.drafter_round_bytes, "{label}: the base shape's VRAM");
    if let [_, extended] = shapes {
        let lanes = load.lanes.min(EXTENDED_LANES as u32);
        assert_eq!(
            (extended.window, extended.lanes, extended.vram_bytes),
            (load.window, lanes, 0),
            "{label}: the extended shape's window, lanes and own VRAM"
        );
        let every = (1u32 << lanes) - 1;
        let captured = base.verify_graph_ready_mask != 0;
        assert_eq!(extended.verify_graph_ready_mask, if captured { every } else { 0 }, "{label}: the extended shape's verify graphs");
        let appends = base.drafter_append_graph_ready_mask != 0;
        assert_eq!(extended.drafter_append_graph_ready_mask, if appends { every } else { 0 }, "{label}: the extended shape's append graphs");
        assert!(
            extended.verify_graph_nodes[..lanes as usize].iter().all(|&nodes| (nodes > 0) == captured),
            "{label}: the extended shape's graph nodes {:?}",
            extended.verify_graph_nodes
        );
    }
}

fn expect_replay(model: &Model, pool: &SeqPool, graphs: bool, what: &str) {
    let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("{what}: stats: {e}"));
    assert_eq!(stats.graph_launches, u64::from(graphs), "{what}: graph replay");
}

/// A kind of round the session rides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Round {
    Plain,
    Verify,
    /// A plain round, lane 0 held to [`Script::permitted`]: eager on every
    /// arm, as a constrained round always is.
    Constrained,
    /// A verify round whose every run of two or more the host cuts one
    /// token short.
    Cut,
    /// A teacher-forced verify round, each lane forced to a run of
    /// [`Script::permitted`]: eager where a lane samples under the sampled
    /// draft, from the graphs otherwise (the fixed draft forces greedy lanes).
    Forced,
}

/// `count` rounds of one kind over the riding lanes; a cut round counts the
/// runs it cut into `cuts`.
#[allow(clippy::too_many_arguments)]
fn ride(
    model: &Model,
    pool: &SeqPool,
    load: &Load,
    script: &Script,
    seqs: &mut [Option<Seq<'_>>],
    riding: &[usize],
    sampling: fn(usize) -> SamplingParams,
    graphs: bool,
    (round, count): (Round, u32),
    committed: &mut Committed,
    cuts: &mut u32,
    label: &str,
) {
    let width = riding.len();
    for at in 0..count {
        let what = format!("{label}: width {width} {round:?} round {at}");
        let mut refs = riders(seqs, riding);
        match round {
            Round::Verify | Round::Cut => {
                let lanes: Vec<VerifyLane<'_>> =
                    riding.iter().map(|&lane| VerifyLane { sampling: sampling(lane), ..VerifyLane::greedy(&[]) }).collect();
                let runs = if round == Round::Cut {
                    let mut cut = |_lane: usize, run: &[i32]| -> usize {
                        if run.len() < 2 {
                            return 1;
                        }
                        *cuts += 1;
                        run.len() - 1
                    };
                    decode_program_verify_cut(model, pool, &mut refs, &lanes, load.window, &vec![true; width], &mut cut)
                } else {
                    decode_program_verify_runs(model, pool, &mut refs, &lanes, load.window)
                }
                .unwrap_or_else(|e| panic!("{what}: {e}"));
                expect_replay(model, pool, graphs, &what);
                // #687: every lane drafts its round shape's whole window.
                let drafts = round_drafts(load, width, round == Round::Cut);
                for (&lane, run) in riding.iter().zip(runs) {
                    assert_eq!(run.extent, drafts, "{what}: lane {lane} drafts its round shape's whole window");
                    committed[lane].extend(run.tokens);
                    committed[lane].push(-1);
                }
            }
            Round::Plain => {
                let params: Vec<SamplingParams> = riding.iter().map(|&lane| sampling(lane)).collect();
                let drawn = decode_program_batch_sampled(model, pool, &mut refs, &params).unwrap_or_else(|e| panic!("{what}: {e}"));
                expect_replay(model, pool, graphs, &what);
                for (&lane, token) in riding.iter().zip(drawn) {
                    committed[lane].push(token);
                }
            }
            Round::Forced => {
                let sampled_draft = load.draft_rule == DraftRule::Sampled;
                let lanes: Vec<VerifyLane<'_>> = riding
                    .iter()
                    .map(|&lane| VerifyLane {
                        sampling: if sampled_draft { sampling(lane) } else { SamplingParams::greedy() },
                        ..VerifyLane::greedy(&[])
                    })
                    .collect();
                let recordings: Vec<Vec<i32>> = riding
                    .iter()
                    .map(|&lane| (0..=load.window as usize).map(|at| script.permitted[(lane + at) % script.permitted.len()]).collect())
                    .collect();
                let forced: Vec<&[i32]> = recordings.iter().map(Vec::as_slice).collect();
                let runs = decode_program_verify_forced(model, pool, &mut refs, &lanes, load.window, &forced)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let on_device = lanes.iter().any(|lane| lane.sampling.temperature > 0.0);
                expect_replay(model, pool, graphs && !on_device, &what);
                for (&lane, run) in riding.iter().zip(runs) {
                    committed[lane].extend(run.tokens);
                    committed[lane].push(-1);
                }
            }
            Round::Constrained => {
                let lanes: Vec<PermittedLane<'_>> = riding
                    .iter()
                    .map(|&lane| PermittedLane { sampling: sampling(lane), permitted: if lane == 0 { &script.permitted } else { &[] } })
                    .collect();
                let (drawn, _) =
                    decode_program_batch_permitted(model, pool, &mut refs, &lanes).unwrap_or_else(|e| panic!("{what}: {e}"));
                expect_replay(model, pool, false, &what);
                for (&lane, token) in riding.iter().zip(drawn) {
                    committed[lane].push(token);
                }
            }
        }
    }
}

/// The session (see the module's note) on `pool`, under the arm's graphs or
/// eagerly.
fn session(model: &Model, pool: &SeqPool, load: &Load, script: &Script, sampling: fn(usize) -> SamplingParams, graphs: bool, label: &str) -> Committed {
    let lanes = load.lanes as usize;
    let mut seqs: Vec<Option<Seq<'_>>> = Vec::with_capacity(lanes);
    let mut committed: Committed = vec![Vec::new(); lanes];
    let mut cuts = 0;
    let mut rides = |seqs: &mut [Option<Seq<'_>>], riding: &[usize], rounds: (Round, u32)| {
        ride(model, pool, load, script, seqs, riding, sampling, graphs, rounds, &mut committed, &mut cuts, label);
    };
    for width in 1..=lanes {
        let lane = width - 1;
        let mut seq = pool.alloc(load.context()).unwrap_or_else(|e| panic!("{label}: alloc lane {lane}: {e}"));
        prefill(model, pool, &mut seq, &script.prompts[lane], sampling(lane), load.vision && lane == 0, &format!("{label}: lane {lane}"));
        seqs.push(Some(seq));
        let riding: Vec<usize> = (0..width).collect();
        rides(&mut seqs, &riding, (Round::Plain, PLAIN_ROUNDS));
        rides(&mut seqs, &riding, (Round::Verify, VERIFY_ROUNDS / 2));
        rides(&mut seqs, &riding, (Round::Constrained, CONSTRAINED_ROUNDS));
        rides(&mut seqs, &riding, (Round::Cut, CUT_ROUNDS));
        rides(&mut seqs, &riding, (Round::Forced, FORCED_ROUNDS));
        rides(&mut seqs, &riding, (Round::Verify, VERIFY_ROUNDS - VERIFY_ROUNDS / 2));
    }
    if lanes > 1 {
        for seq in seqs.iter_mut().skip(1).step_by(2) {
            *seq = None;
        }
        let riding: Vec<usize> = (0..lanes).step_by(2).collect();
        rides(&mut seqs, &riding, (Round::Verify, VERIFY_ROUNDS));
        // GitHub #676: each lane's state through a snapshot taken right after
        // that verify round, and back.
        for &lane in &riding {
            let seq = seqs[lane].as_mut().expect("a riding lane");
            let blob = seq.snapshot().unwrap_or_else(|e| panic!("{label}: snapshot lane {lane}: {e}"));
            seq.restore(&blob).unwrap_or_else(|e| panic!("{label}: restore lane {lane}: {e}"));
        }
        rides(&mut seqs, &riding, (Round::Plain, PLAIN_ROUNDS));
    }
    assert!(cuts > 0, "{label}: no verify run of two or more was cut, so no cut took a count back");
    committed
}

/// GitHub #676: whether the arm's verify rounds return with their fold in
/// flight: a DFlash2 load's do once its graphs are captured, unless the arm
/// sets `IGNIS_DEFER_VERIFY=0`; an MTP load's never, nor the `eager` arm's,
/// which run before the load's first capture.
fn defers_verify(load: &Load, arm: &Arm) -> bool {
    load.backend == SpeculativeBackend::Dflash2 && arm.graphs && !arm.switches.contains(&("IGNIS_DEFER_VERIFY", "0"))
}

/// The arm's sessions, greedy then sampled, its switches set and, for an arm
/// that replays graphs, the graphs captured under them.
fn run(model: &Model, pool: &SeqPool, load: &Load, script: &Script, arm: &Arm, label: &str) -> Vec<Committed> {
    apply(arm, load.round_shape);
    let stats = || program_stats(model, pool).unwrap_or_else(|e| panic!("{label}: stats: {e}"));
    // #687: the folds left in flight by a round of either shape, and the
    // rounds at the extended shape, with those of them that ran eagerly.
    let folds_in_flight = || stats().shapes().iter().map(|shape| shape.verify_folds_in_flight).sum::<u64>();
    let extended_rounds = || stats().shapes().get(1).map(|shape| (shape.verify_rounds, shape.eager_verify_rounds));
    if arm.graphs {
        let capture = capture_decode_graphs(model, pool).unwrap_or_else(|e| panic!("{label}: {}: capture: {e}", arm.name));
        let every = (1u32 << load.lanes) - 1;
        let verify = stats().verify_graph_ready_mask;
        assert_eq!((capture.ready_mask, verify), (every, every), "{label}: {}: every width's decode and verify graph", arm.name);
        let append = stats().drafter_append_graph_ready_mask;
        let want = if defers_verify(load, arm) { every } else { 0 };
        assert_eq!(append, want, "{label}: {}: the drafter's append graphs", arm.name);
        copy_free_graphs(model, pool, load, arm, label);
    }
    round_shape_stats(model, pool, load, &format!("{label}: {}", arm.name));
    let folds = folds_in_flight();
    let extended_before = extended_rounds();
    let sessions = KINDS
        .iter()
        .map(|&(kind, sampling)| session(model, pool, load, script, sampling, arm.graphs, &format!("{label} {} {kind}", arm.name)))
        .collect();
    // Where the arm defers, every verify round but a cut or a forced one
    // leaves its fold in flight: the rest cut nothing, draft the whole window
    // and take no penalty count back. A cut round does too when its lanes are
    // greedy; a forced round never does.
    let left = folds_in_flight() - folds;
    let uncut = KINDS.len() as u64 * u64::from((load.lanes + u32::from(load.lanes > 1)) * VERIFY_ROUNDS);
    if defers_verify(load, arm) {
        assert!(left >= uncut, "{label}: {}: {left} verify rounds left their fold in flight, not the {uncut} uncut ones", arm.name);
    } else {
        assert_eq!(left, 0, "{label}: {}: verify rounds left their fold in flight", arm.name);
    }
    println!("round_identity_gpu {label}: arm `{}`: {left} verify rounds returned with their fold in flight", arm.name);
    // #687: a load under `extended` ran extended rounds, and an arm with no
    // graph ran every one of them eagerly; one under the base shape ran none.
    if let (Some((rounds_before, eager_before)), Some((rounds_after, eager_after))) = (extended_before, extended_rounds()) {
        let (rounds, eager) = (rounds_after - rounds_before, eager_after - eager_before);
        if load.round_shape == Some("extended") {
            assert!(rounds > 0, "{label}: {}: no verify round ran at the extended shape", arm.name);
        } else {
            assert_eq!(rounds, 0, "{label}: {}: verify rounds ran at the extended shape", arm.name);
        }
        if !arm.graphs {
            assert_eq!(eager, rounds, "{label}: {}: an extended round replayed a graph", arm.name);
        }
        println!("round_identity_gpu {label}: arm `{}`: {rounds} verify rounds at the extended shape, {eager} of them eager", arm.name);
    }
    apply(&REFERENCE, None);
    sessions
}

/// The first lane, and the first token of it, where `got` leaves `want`.
fn first_difference(want: &Committed, got: &Committed) -> Option<(usize, usize)> {
    want.iter().zip(got).enumerate().find_map(|(lane, (w, g))| {
        (w != g).then(|| (lane, w.iter().zip(g).position(|(a, b)| a != b).unwrap_or(w.len().min(g.len()))))
    })
}

/// One load under one exception setting, handed to `body` with its label: one
/// materialize, one model load, one pool, released before returning. What
/// `body` returns, or `None` where the profile allows a skip.
fn with_load<T>(
    device: &mut CudaDevice,
    reader: &Reader,
    load: &Load,
    exceptions: ExceptionWeights,
    body: impl FnOnce(&Model, &SeqPool, &str) -> T,
) -> Option<T> {
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
    let returned = {
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
        body(&model, &pool, &label)
    };
    artifact.release_arena(device).unwrap_or_else(|e| panic!("{label}: release: {e}"));
    Some(returned)
}

/// Every selected arm against the reference on one load under one exception
/// setting. The arms compared, or `None` where the profile allows a skip.
fn hold(device: &mut CudaDevice, reader: &Reader, load: &Load, exceptions: ExceptionWeights, arms: &[&Arm], script: &Script) -> Option<usize> {
    with_load(device, reader, load, exceptions, |model, pool, label| {
        // An eager arm runs before the load's first capture.
        let eager: Vec<(&Arm, Vec<Committed>)> =
            arms.iter().filter(|arm| !arm.graphs).map(|&arm| (arm, run(model, pool, load, script, arm, label))).collect();
        let reference = run(model, pool, load, script, &REFERENCE, label);
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
            let sessions = run(model, pool, load, script, arm, label);
            hold_arm(arm, sessions);
        }
        compared
    })
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
        let script = script(&reader);
        for exceptions in [ExceptionWeights::Bf16, ExceptionWeights::Fp8] {
            if !weights.contains(&exceptions.as_str()) {
                continue;
            }
            if exceptions.is_fp8() && !load.exceptions {
                println!("round_identity_gpu {}: no BF16 exception tensor, so fp8 is its bf16 load: run once, as bf16", load.name);
                continue;
            }
            match hold(&mut device, &reader, load, exceptions, &arms, &script) {
                Some(arms_held) => compared += arms_held,
                None => skipped = true,
            }
        }
    }
    assert!(compared > 0 || skipped, "no arm was held to the reference: check the IGNIS_ROUND_IDENTITY_* filters");
}

/// #687: a 15-draft load's rounds at the base shape commit what a load at
/// [`BASE_WINDOW`] drafts commits, token for token, at every width, with
/// greedy lanes and with sampled ones, under each exception setting: the load
/// [`TWO_SHAPE_LOAD`] with [`ROUND_SHAPE`] unset against a load at
/// [`BASE_WINDOW`] that reserves its context (the context is the envelope the
/// round's attention splits by, so a load at another context is another
/// summation order, not another round). Then, on the 15-draft load, a verify
/// round under a value the switch does not know is refused, naming it: the
/// proof that a round reads it, which every two-shape load's `extended` rests
/// on.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_fifteen_draft_loads_base_rounds_commit_a_seven_draft_loads_tokens() {
    let wide = LOADS.iter().find(|load| load.name == TWO_SHAPE_LOAD).expect("the two-shape load is in LOADS");
    let base = Load { round_shape: None, ..*wide };
    let seven = Load { name: "nvfp4-dflash2-w7-at-the-w15-context", window: BASE_WINDOW, sized_for: Some(wide.window), round_shape: None, ..*wide };
    assert_eq!(seven.context(), base.context(), "both loads reserve one context");
    let weights = selected("IGNIS_ROUND_IDENTITY_WEIGHTS", &ExceptionWeights::NAMES);
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let path = Path::new(base.artifact);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("{}: artifact absent: {}", base.name, base.artifact)) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", base.artifact));
    let script = script(&reader);
    let (mut compared, mut skipped) = (0, false);
    for exceptions in [ExceptionWeights::Bf16, ExceptionWeights::Fp8] {
        if !weights.contains(&exceptions.as_str()) {
            continue;
        }
        // The load at seven drafts first, released before the 15-draft one.
        let Some(want) = with_load(&mut device, &reader, &seven, exceptions, |model, pool, label| {
            run(model, pool, &seven, &script, &REFERENCE, label)
        }) else {
            skipped = true;
            continue;
        };
        let Some(got) = with_load(&mut device, &reader, &base, exceptions, |model, pool, label| {
            let sessions = run(model, pool, &base, &script, &REFERENCE, label);
            apply(&REFERENCE, Some("no-such-value"));
            let mut seq = pool.alloc(base.context()).unwrap_or_else(|e| panic!("{label}: alloc: {e}"));
            prefill(model, pool, &mut seq, &script.prompts[0], greedy(0), false, label);
            let mut refs = vec![&mut seq];
            let lanes = [VerifyLane::greedy(&[])];
            let Err(refused) = decode_program_verify_runs(model, pool, &mut refs, &lanes, base.window) else {
                panic!("{label}: a verify round under {ROUND_SHAPE}=no-such-value was not refused");
            };
            assert!(refused.contains(ROUND_SHAPE), "{label}: the round's refusal names the variable: {refused}");
            apply(&REFERENCE, None);
            sessions
        }) else {
            skipped = true;
            continue;
        };
        for (((kind, _), want), got) in KINDS.iter().zip(&want).zip(&got) {
            if let Some((lane, at)) = first_difference(want, got) {
                let from = at.saturating_sub(4);
                panic!(
                    "{} {exceptions}: at the base shape, {kind} lanes: lane {lane} leaves the load at {BASE_WINDOW} \
                     drafts at token {at}: that load {:?}, the base shape {:?}",
                    base.name,
                    &want[lane][from..want[lane].len().min(at + 8)],
                    &got[lane][from..got[lane].len().min(at + 8)]
                );
            }
        }
        let tokens: usize = got.iter().flatten().flatten().filter(|&&t| t >= 0).count();
        println!(
            "round_identity_gpu {} {exceptions}: the base shape commits the {tokens} tokens of the load at \
             {BASE_WINDOW} drafts, widths 1..={}, greedy and sampled lanes",
            base.name, base.lanes
        );
        compared += 1;
    }
    assert!(compared > 0 || skipped, "no exception setting was compared: check IGNIS_ROUND_IDENTITY_WEIGHTS");
}

/// #693: the arm that gives the FP8 exception products their 128-token tile
/// back at 9 to 16 tokens, and the load the prefill test below holds it to the
/// reference on.
const TILE128_ARM: &str = "fp8-w8a8-tile128";
const TILE128_PREFILL_LOAD: &str = "nvfp4-dflash2";

/// #693: a prefill whose last chunk is 9 to 16 tokens -- a prompt of one
/// [`CHUNK`] and that many more -- writes the same bytes on the FP8 exception
/// products' 16-token tile (the reference, every switch unset) as on
/// [`TILE128_ARM`]'s 128-token one: the span's last logits bit for bit, then
/// [`PLAIN_ROUNDS`] greedy plain rounds' tokens, which read the keys, values
/// and GDN state the chunk wrote. Eager, on `fp8` exception weights only
/// (`bf16` runs no W8A8 product). The first test's session never takes the
/// 16-token tile at a prefill: its prompts fit one chunk of more than 16
/// tokens. It runs where the `IGNIS_ROUND_IDENTITY_*` filters select its load,
/// its arm and `fp8`.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_prefill_whose_last_chunk_is_9_to_16_tokens_writes_the_same_bytes_on_both_fp8_tiles() {
    let load = LOADS.iter().find(|load| load.name == TILE128_PREFILL_LOAD).expect("the prefill test's load is in LOADS");
    let arm = ARMS.iter().find(|arm| arm.name == TILE128_ARM).expect("the 128-token tile's arm is in ARMS");
    let loads = selected("IGNIS_ROUND_IDENTITY_LOADS", &LOADS.iter().map(|load| load.name).collect::<Vec<_>>());
    let arms = selected("IGNIS_ROUND_IDENTITY_ARMS", &ARMS.iter().map(|arm| arm.name).collect::<Vec<_>>());
    let weights = selected("IGNIS_ROUND_IDENTITY_WEIGHTS", &ExceptionWeights::NAMES);
    if !(loads.contains(&load.name) && arms.contains(&arm.name) && weights.contains(&ExceptionWeights::Fp8.as_str())) {
        println!("round_identity_gpu: the IGNIS_ROUND_IDENTITY_* filters leave out {}, `{}` or fp8", load.name, arm.name);
        return;
    }
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let path = Path::new(load.artifact);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("{}: artifact absent: {}", load.name, load.artifact)) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {}: {e}", load.artifact));
    let script = script(&reader);
    let lengths: Vec<usize> = script.prompts.iter().map(Vec::len).collect();
    println!("round_identity_gpu {}: the session's prompts are {lengths:?} tokens", load.name);
    // One long prompt: the session's prompts end to end.
    let long: Vec<i32> = script.prompts.concat();
    let tails = 9..=16u32;
    assert!(long.len() as u32 >= CHUNK + tails.end(), "the prompts end to end are {} tokens", long.len());
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    // `None` is a skip the profile allows (`with_load`).
    let _ = with_load(&mut device, &reader, load, ExceptionWeights::Fp8, |model, pool, label| {
        let spans = |arm: &Arm| -> Vec<(Vec<u32>, Vec<i32>)> {
            apply(arm, None);
            let written = tails
                .clone()
                .map(|tail| {
                    let what = format!("{label}: arm `{}`, a last chunk of {tail} tokens", arm.name);
                    let prompt = &long[..(CHUNK + tail) as usize];
                    let mut seq = pool.alloc(load.context()).unwrap_or_else(|e| panic!("{what}: alloc: {e}"));
                    let mut logits = vec![0f32; vocab];
                    prefill_program_sampled(model, pool, &mut seq, prompt, 0, SamplingParams::greedy(), Some(&mut logits))
                        .unwrap_or_else(|e| panic!("{what}: prefill: {e}"));
                    let mut tokens = Vec::new();
                    for _ in 0..PLAIN_ROUNDS {
                        let mut refs = vec![&mut seq];
                        let drawn = decode_program_batch_sampled(model, pool, &mut refs, &[SamplingParams::greedy()])
                            .unwrap_or_else(|e| panic!("{what}: plain round: {e}"));
                        expect_replay(model, pool, false, &what);
                        tokens.extend(drawn);
                    }
                    (logits.iter().map(|v| v.to_bits()).collect(), tokens)
                })
                .collect();
            apply(&REFERENCE, None);
            written
        };
        let want = spans(&REFERENCE);
        let got = spans(arm);
        for ((tail, (want_logits, want_tokens)), (got_logits, got_tokens)) in tails.clone().zip(&want).zip(&got) {
            let moved = want_logits.iter().zip(got_logits).filter(|(a, b)| a != b).count();
            assert_eq!(moved, 0, "{label}: a last chunk of {tail} tokens: {moved} of {vocab} logits differ between the tiles");
            assert_eq!(want_tokens, got_tokens, "{label}: a last chunk of {tail} tokens: the plain rounds' tokens");
        }
        println!(
            "round_identity_gpu {label}: arm `{}`: prefills of {CHUNK} + 9..=16 tokens write the reference's logits bit \
             for bit and its {PLAIN_ROUNDS} plain rounds' tokens",
            arm.name
        );
    });
}
