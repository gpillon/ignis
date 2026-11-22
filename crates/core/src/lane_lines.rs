//! The 27B's VRAM plan lines that follow a load's decode lanes (GitHub
//! #336, spec 27b-16gb/04 solution 1).
//!
//! Sampling, the verify round, the drafter's round, the lanes' state and the
//! hq residual window, each a function of the lanes and the model's
//! dimensions: the arithmetic `kernel/src/model.cu` sizes the load by
//! (`LoadSizes::reservations`, `verify_round_sizes`,
//! `dflash2_round_scratch_bytes`) and the sequence pool's slot image, the
//! vendored layouts those ask included. So a load can be re-planned at fewer
//! lanes on the CPU, and `cargo test` holds every line to its formula; the
//! GPU plan test (`crates/runtime/tests/vram_plan_gpu.rs`) holds the
//! kernel's plan to these functions to the byte at 1, 2 and 8 lanes, as it
//! does the drafter's window against `speculation.rs`.
//!
//! GitHub #714 (spec #711): and the mixed rounds' graphs ([`mixed_graphs`]),
//! which follow the lanes and the pool's device retained slots, and which the
//! GPU test holds the kernel's plan to at every lane count it asks.
//!
//! The decode round's line (`decode_graph`) is not here. It is the widest
//! decoder layer's activations plus five workspace queries -- the hq
//! attention's split-K over the context envelope, and the W4A4 activation
//! codes each NVFP4 projection's route takes from four columns on -- whose
//! routes switch on every weight's qtype as the artifact stores it (and as
//! `--model-exception-weights` converts it). A copy here would be a second
//! implementation of the vendored route tables, wrong the first time one of
//! them moves; the GPU test pins the line at 1, 2 and 8 lanes instead, where
//! it is exactly affine in them. A load of two round shapes (GitHub #687)
//! sizes it for the larger of the base traversal and the extended one, the
//! latter at no fewer than 2 lanes, whose bytes follow the same routes: the
//! GPU test holds a one-lane load's line to the two-lane load's, which sizes
//! the same traversal, and pins the trellis load's at the 16 GB profile's 2
//! lanes.

// The leaf's arena alignment (`kArenaAlign`, `kernel/src/model.cu`), which
// the vendored layouts below align every tensor and their total to.
use crate::arena::{aligned, ALIGN as ARENA_ALIGN};
use crate::compute::{LayerKind, ModelConfig};
use crate::kv_format::KvFormat;
use crate::speculation::{RoundShape, Speculation};

/// `sizeof(ninfer::ops::SamplingConfig)` (`ninfer/ops/sampling.h`): a
/// temperature, a top-k, the top-p and min-p, two penalties, a seed and a
/// counts pointer -- 40 bytes on x64.
const SAMPLING_CONFIG_BYTES: u64 = 40;

/// The widest permitted set one lane's draw takes (`IGNIS_MAX_PERMITTED_TOKENS`,
/// `kernel/include/ignis_step.h`, GitHub #242).
const MAX_PERMITTED_TOKENS: u64 = 32;

/// The vendored sampler's multi-block geometry (`ops/common/sampling_workspace.h`):
/// a vocabulary wider than one tile is cut into partial blocks of
/// `kSamplerPartialTileItems` rows, merged in groups of
/// `kSamplerPartialsPerGroup`, each keeping `kSamplerCandidateCap`
/// candidates; at most `kSamplerMaxColumns` columns take that route.
const SAMPLER_TILE_ITEMS: u64 = 256;
const SAMPLER_PARTIAL_TILE_ITEMS: u64 = 512;
const SAMPLER_PARTIALS_PER_GROUP: u64 = 25;
const SAMPLER_CANDIDATE_CAP: u64 = 20;
const SAMPLER_GROUP_TILE_ITEMS: u64 = 512;
const SAMPLER_MAX_COLUMNS: u64 = 16;

/// The plan's lines that follow a load's decode lanes, in bytes: the
/// `ReservedBytes` / [`crate::VramLines`] lines of the same names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneLines {
    pub sampling: u64,
    pub verify_round: u64,
    pub drafter_round: u64,
    pub lane_state: u64,
    pub hq_residual_window: u64,
}

/// What a 27B load of `config` at `lanes` decode lanes reserves on the lines
/// that follow them: its KV format, its speculation, whether it binds the
/// vision tower, and the device retained slots that hold an hq residual
/// window beside the lanes.
pub fn lane_lines(
    config: &ModelConfig,
    kv_format: KvFormat,
    speculation: Option<Speculation>,
    vision: bool,
    lanes: u32,
    retained_device_slots: u32,
) -> LaneLines {
    let image = config.state_image(kv_format);
    // A lane's state is its pool slot (`kernel/src/seq.cu`): the GDN
    // recurrent state and conv taps, the penalty counts and the load's
    // drafter's state (DFlash2's window, the MTP head's continuation hidden).
    // The hq residual window rides every slot, lanes and device retained
    // slots alike, with the MTP head's layer in it on an MTP load (GitHub
    // #620).
    let slot = image.gdn_recurrent
        + image.gdn_conv_taps
        + image.penalty_counts
        + speculation.map_or(0, |speculation| speculation.drafter_state_bytes(1));
    let hq_window =
        image.hq_residual_window + speculation.map_or(0, |speculation| speculation.hq_residual_window_bytes(kv_format));
    LaneLines {
        sampling: sampling_bytes(config, lanes),
        verify_round: speculation.map_or(0, |speculation| verify_round_bytes(config, speculation, vision, lanes)),
        drafter_round: speculation.map_or(0, |speculation| speculation.round_scratch_bytes(lanes)),
        lane_state: u64::from(lanes) * slot,
        hq_residual_window: u64::from(lanes + retained_device_slots) * hq_window,
    }
}

/// GitHub #713, #714 (spec #711): the mixes a mixed verify round takes, `w`
/// extended lanes beside `n` base lanes at 64 columns or fewer, in the leaf's
/// order (`kMixTable`, `kernel/src/round_layout.h`; `IGNIS_ROUND_MIXES`).
pub const MIXES: [(u32, u32); 12] =
    [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2)];

/// The one draft window a mixed round runs at (#711, the owner's decision):
/// at any other a base group's GDN records do not share the extended ones'
/// stride over the backing both shapes share.
pub const MIXED_ROUND_WINDOW: u32 = 15;

/// The device memory a load's mixed graphs take (`kernel/src/mixed_graphs.h`
/// `bytes_of`): [`MIXED_GRAPH_BYTES`] a mix, each its verify graph and its
/// drafter append graph, and [`MIXED_GRAPHS_FIRST_BYTES`] beside them, in
/// whole pages of [`MIXED_GRAPHS_PAGE_BYTES`], as the RTX 5090 measured them:
/// 62 MiB for the twelve at capture, and a page more on the process's own
/// counter.
pub const MIXED_GRAPH_BYTES: u64 = 5 << 20;
pub const MIXED_GRAPHS_FIRST_BYTES: u64 = 4 << 20;
pub const MIXED_GRAPHS_PAGE_BYTES: u64 = 2 << 20;

/// The mixed graphs a load captures and plans (GitHub #714): the mixes, in
/// [`MIXES`]' order, and their bytes, the plan's `mixed_graphs` line.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MixedGraphs {
    pub mixes: Vec<(u32, u32)>,
    pub bytes: u64,
}

/// GitHub #715 (spec #711): whether a load of `speculation` asks for its mixed
/// graphs, at its plan, its load and each capture (`kernel/src/mixed_graphs.h`
/// `asked`): as the measurement switch `IGNIS_MIXED_ROUNDS` says where it is
/// set (`switch`, `Some(true)` for `1`), and otherwise where the shape policy
/// is `mixed`, never under `aggregate` or `all`.
pub fn mixed_rounds_asked(switch: Option<bool>, speculation: Option<Speculation>) -> bool {
    switch.unwrap_or_else(|| speculation.is_some_and(|speculation| speculation.shape_policy() == crate::speculation::ShapePolicy::Mixed))
}

/// The mixed graphs a 27B load of `speculation` at `lanes` decode lanes over a
/// pool with `retained_device_slots` device retained slots captures when
/// mixed rounds are asked (`mixed`; `kernel/src/mixed_graphs.h` `planned`): on a
/// DFlash2 load of the NVFP4 body (`trellis` false: the trellis body runs no
/// mixed round yet) at [`MIXED_ROUND_WINDOW`] drafts, every mix whose `w + n`
/// lanes fit the load's and whose base group's fold, behind its `2w` leading
/// rows that commit nothing, fits the pool's GDN slots, the lanes' and the
/// device retained slots'. None anywhere else. GitHub #715: `mixed` as
/// [`mixed_rounds_asked`] gives it.
pub fn mixed_graphs(
    speculation: Option<Speculation>,
    trellis: bool,
    mixed: bool,
    lanes: u32,
    retained_device_slots: u32,
) -> MixedGraphs {
    let captures = speculation.is_some_and(|speculation| speculation.holds_mixed_rounds(trellis)) && mixed;
    let slots = lanes + retained_device_slots;
    let mixes: Vec<(u32, u32)> = MIXES
        .into_iter()
        .filter(|&(extended, base)| captures && extended + base <= lanes && 2 * extended + base <= slots)
        .collect();
    let bytes = if mixes.is_empty() {
        0
    } else {
        (MIXED_GRAPHS_FIRST_BYTES + mixes.len() as u64 * MIXED_GRAPH_BYTES).div_ceil(MIXED_GRAPHS_PAGE_BYTES)
            * MIXED_GRAPHS_PAGE_BYTES
    };
    MixedGraphs { mixes, bytes }
}

/// Device sampling's line at `lanes` lanes (`LoadSizes::reservations`): the
/// single draw's config, position and output, each lane's, the permitted-set
/// staging (ids, counts and the committed probability, GitHub #242), the
/// decode logits and the sampler's workspace for every width up to the
/// lanes -- never under one arena block.
pub fn sampling_bytes(config: &ModelConfig, lanes: u32) -> u64 {
    let lanes = u64::from(lanes);
    let i32 = 4;
    SAMPLING_CONFIG_BYTES
        + 2 * i32
        + (SAMPLING_CONFIG_BYTES + 2 * i32) * lanes
        + i32 * lanes * MAX_PERMITTED_TOKENS
        + i32 * lanes
        + 4 * lanes
        + config.vocab * lanes * 2
        + sampling_workspace_capacity_bytes(config.vocab, 1, lanes).max(ARENA_ALIGN)
}

/// The verify round's line at `lanes` lanes under `speculation`'s window `k`
/// (`verify_round_sizes`, `kernel/src/model.cu`): its staging -- eight
/// per-lane words, the drafts, four (five with vision's rope positions)
/// words per column -- the logits and hidden of every column, each lane's
/// selected hidden, the ReplaySSM records of every GDN layer and lane, and
/// the greedy accept's workspace, never under one arena block. Its buffers
/// are separate allocations, so only the records and the workspace are
/// rounded.
///
/// GitHub #687: a load of two round shapes
/// ([`Speculation::round_shapes`]) holds one set of these buffers, each the
/// larger of the two shapes' (`larger_of`).
pub fn verify_round_bytes(config: &ModelConfig, speculation: Speculation, vision: bool, lanes: u32) -> u64 {
    if speculation.draft_tokens() == 0 {
        return 0;
    }
    let (base, extended) = speculation.round_shapes(lanes);
    let mut buffers = verify_round_buffers(config, base);
    if let Some(extended) = extended {
        for (bytes, extended) in buffers.iter_mut().zip(verify_round_buffers(config, extended)) {
            *bytes = (*bytes).max(extended);
        }
    }
    let [lane_words, drafts, column_words, logits, hidden, selected_hidden, records, accept] = buffers;
    let staging = 8 * lane_words + drafts + if vision { 5 } else { 4 } * column_words;
    staging + logits + hidden + selected_hidden + records + accept
}

/// The bytes of each buffer of the verify round at `shape`
/// (`VerifyRoundSizes`, `kernel/src/model.cu`), in its order: a per-lane
/// word buffer, the drafts, a per-column word buffer, the logits, the hidden,
/// the selected hidden, the ReplaySSM records and the accept's workspace.
fn verify_round_buffers(config: &ModelConfig, shape: RoundShape) -> [u64; 8] {
    let k = u64::from(shape.window);
    let columns = k + 1;
    let lanes = u64::from(shape.lanes);
    let (i32, bf16, fp32) = (4, 2, 4);
    // `verify_record_spec`: one record row per GDN layer and lane, `k + 1`
    // wide, laid out by the vendored `plan_gdn_replay_records`.
    let outer = config.layer_kinds.iter().filter(|&&kind| kind == LayerKind::Gdn).count() as u64 * lanes;
    let qk_heads = if config.gdn_head_dim == 0 { 0 } else { config.gdn_q_width / config.gdn_head_dim };
    let head = config.gdn_head_dim;
    let records = layout(&[
        config.gdn_conv_channels() * columns * outer * bf16,
        head * qk_heads * columns * outer * bf16,
        head * config.gdn_value_heads * columns * outer * bf16,
        2 * config.gdn_value_heads * columns * outer * fp32,
    ]);
    // `speculative_accept_greedy_drafts_workspace_capacity_bytes`: one
    // sampler workspace of `k + 1` columns per lane.
    let accept = (sampling_workspace_capacity_bytes(config.vocab, columns, columns) * lanes).max(ARENA_ALIGN);
    [
        lanes * i32,
        k * lanes * i32,
        columns * lanes * i32,
        config.vocab * columns * lanes * bf16,
        config.hidden * columns * lanes * bf16,
        config.hidden * lanes * bf16,
        records,
        accept,
    ]
}

/// `ninfer::ops::sampling_workspace_capacity_bytes`: the multi-block
/// sampler's workspace for every column count in `min..=max` over a
/// `vocab`-row domain, its widest; none where one block serves.
fn sampling_workspace_capacity_bytes(vocab: u64, min_columns: u64, max_columns: u64) -> u64 {
    if vocab <= SAMPLER_TILE_ITEMS || min_columns > SAMPLER_MAX_COLUMNS {
        return 0;
    }
    let columns = max_columns.min(SAMPLER_MAX_COLUMNS);
    let partial_blocks = vocab.div_ceil(SAMPLER_PARTIAL_TILE_ITEMS);
    let groups = partial_blocks.div_ceil(SAMPLER_PARTIALS_PER_GROUP);
    if groups * SAMPLER_CANDIDATE_CAP > SAMPLER_GROUP_TILE_ITEMS {
        return 0;
    }
    // `make_sampling_workspace_layout`: the partial keys, the speculative
    // distribution's indices, probabilities and support, the group counters
    // and the finalize counter.
    let stride = partial_blocks + groups;
    layout(&[
        8 * SAMPLER_CANDIDATE_CAP * stride * columns,
        4 * SAMPLER_CANDIDATE_CAP * columns,
        4 * SAMPLER_CANDIDATE_CAP * columns,
        4 * columns,
        4 * columns,
        4,
    ])
}

/// `ninfer::LayoutBuilder` at the arena's alignment: each tensor starts on a
/// block, and the whole is rounded up to one.
fn layout(tensors: &[u64]) -> u64 {
    aligned(tensors.iter().fold(0, |cursor, &bytes| aligned(cursor) + bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculation::SpeculativeBackend;

    const MB: f64 = 1e6;
    const MIB: u64 = 1 << 20;

    fn dflash2() -> Speculation {
        Speculation::new(SpeculativeBackend::Dflash2, 7).expect("dflash2-7")
    }

    /// The `rtx5090` profile's 27B (hq-e8-2b, DFlash2 at 7, vision) with two
    /// device retained slots, at `lanes`.
    fn served(lanes: u32, vision: bool) -> LaneLines {
        lane_lines(&ModelConfig::qwen38_27b(), KvFormat::HqE8_2b, Some(dflash2()), vision, lanes, 2)
    }

    /// Spec 27b-16gb/04: at eight lanes every line is main 75210360's plan to
    /// the byte, as its leaf planned it at these options (with vision, and
    /// the 262,144-token text load without it).
    #[test]
    fn eight_lanes_are_mains_plan_to_the_byte() {
        let main = LaneLines {
            sampling: 4_623_344,
            verify_round: 152_139_488,
            drafter_round: 78_027_552,
            lane_state: 1_575_043_072,
            hq_residual_window: 356_516_480,
        };
        assert_eq!(served(8, true), main);
        assert_eq!(served(8, false), LaneLines { verify_round: 152_139_232, ..main }, "vision's rope positions, 8 x 8 words");
    }

    /// GitHub #687: a DFlash2 load past 7 drafts holds two round shapes on one
    /// set of buffers, each the larger of the two shapes'. At eight lanes both
    /// top out at 64 columns (8 x 8 and 4 x 16), so the verify line is the
    /// 7-draft load's plus the 16 bytes the extended shape's drafts take past
    /// the base shape's (15 x 4 words against 7 x 8), and no other lane line
    /// moves; under the sampled draft the drafter's line holds 512 bytes more
    /// of draws. Below eight lanes the extended shape has the more columns and
    /// sizes the line: at four lanes, its 4 x 16 columns to the byte.
    #[test]
    fn two_round_shapes_hold_the_seven_draft_loads_buffers_at_eight_lanes() {
        let config = ModelConfig::qwen38_27b();
        let at = |window: u32, lanes: u32, vision: bool| {
            let speculation = Speculation::new(SpeculativeBackend::Dflash2, window).expect("dflash2");
            lane_lines(&config, KvFormat::HqE8_2b, Some(speculation), vision, lanes, 2)
        };
        for vision in [false, true] {
            let main = at(7, 8, vision);
            assert_eq!(at(15, 8, vision), LaneLines { verify_round: main.verify_round + 16, ..main }, "vision {vision}");
            // 4 x 10 columns and 4 x 9 drafts: under the base shape's on every buffer.
            assert_eq!(at(9, 8, vision), main, "vision {vision}");
        }
        assert_eq!(at(15, 8, true).verify_round, 152_139_504);
        let sampled = |window: u32| {
            let speculation = Speculation::new(SpeculativeBackend::Dflash2, window).expect("dflash2");
            let speculation = speculation.with_draft_rule(crate::DraftRule::Sampled, crate::DrafterSampling::default());
            lane_lines(&config, KvFormat::HqE8_2b, Some(speculation), true, 8, 2)
        };
        assert_eq!(sampled(15), LaneLines { verify_round: sampled(7).verify_round + 16, drafter_round: 78_035_232, ..sampled(7) });
        assert_eq!(sampled(7).drafter_round, 78_034_720);
        // Four lanes: 4 x 16 columns against 4 x 8, so every buffer is the
        // extended shape's, and the line what that shape takes alone.
        let extended = verify_round_buffers(&config, RoundShape { window: 15, lanes: 4 });
        let base = verify_round_buffers(&config, RoundShape { window: 7, lanes: 4 });
        assert!(extended.iter().zip(base).all(|(&extended, base)| extended >= base));
        assert_eq!(at(15, 4, true).verify_round, 152_093_296);
        // An MTP load keeps its one shape at every window it takes.
        for window in 1..=7 {
            let mtp = Speculation::new(SpeculativeBackend::Mtp, window).expect("mtp");
            let lines = lane_lines(&config, KvFormat::HqE8_2b, Some(mtp), true, 8, 2);
            assert_eq!(lines.verify_round, verify_round_bytes(&config, mtp, true, 8), "mtp {window}");
            let (shape, none) = mtp.round_shapes(8);
            assert_eq!((shape, none), (RoundShape { window, lanes: 8 }, None), "mtp {window}");
        }
    }

    /// GitHub #687 (the owner's requirement): a load past 7 drafts keeps a
    /// 7-draft load's KV pages. At eight lanes, the `rtx5090` profile's, the
    /// verify and drafter lines of every window 8..15 pass a 7-draft load's by
    /// less than one KV page (589,824 B under hq-e8-2b), so the pool, the
    /// whole pages left of the budget, loses none, or one where the remainder
    /// sat on a page's edge. What the extended shape cannot share at 15 drafts
    /// is its 60 draft columns against the base shape's 56: 16 B of drafts
    /// and, under the sampled draft, 512 B of kept candidates and
    /// probabilities. A plan that shared the buffers but summed the drafter's
    /// scratch over both shapes' forwards would be 2,142,736 B and three pages
    /// over.
    #[test]
    fn a_two_shape_load_at_eight_lanes_keeps_a_seven_draft_loads_kv_pages() {
        let config = ModelConfig::qwen38_27b();
        let page = KvFormat::HqE8_2b.page_bytes(config.kv_geometry());
        assert_eq!(page, 589_824, "one KV page of the 27B under hq-e8-2b: 64 tokens of 9,216 B");
        let drafting = |window: u32, vision: bool| {
            let speculation = Speculation::new(SpeculativeBackend::Dflash2, window)
                .expect("dflash2")
                .with_draft_rule(crate::DraftRule::Sampled, crate::DrafterSampling::default());
            lane_lines(&config, KvFormat::HqE8_2b, Some(speculation), vision, 8, 2)
        };
        for vision in [false, true] {
            for window in 8..=15 {
                let (extended, main) = (drafting(window, vision), drafting(7, vision));
                let over = (extended.verify_round + extended.drafter_round) - (main.verify_round + main.drafter_round);
                assert!(
                    over < page,
                    "window {window}, vision {vision}: the load reserves {over} B past a 7-draft load's verify and drafter lines, {} whole KV pages",
                    over / page
                );
            }
        }
    }

    /// The sampling and verify lines the leaf planned at one and two lanes
    /// when the lanes became a load setting (`vram_plan_gpu`, 2026-10-23, with
    /// vision and without), which nothing but the kernel had computed then.
    #[test]
    fn one_and_two_lanes_are_what_the_kernel_planned() {
        for (lanes, sampling, with_vision, without) in [(1, 579_048, 19_017_436, 19_017_404), (2, 1_156_768, 38_034_872, 38_034_808)] {
            assert_eq!((served(lanes, true).sampling, served(lanes, true).verify_round), (sampling, with_vision), "{lanes} lanes");
            assert_eq!(served(lanes, false).verify_round, without, "{lanes} lanes without vision");
        }
    }

    /// At one and two lanes each line costs a lane what PIANO-VRAM's table
    /// says, within its rounding (`.scratch/27b-16gb-study-2026-10-20`,
    /// fitted to 39 measured plans): a lane's state 233 MB -- GDN fp32
    /// 151 MB, conv 3 MB, penalty 1 MB, the DFlash2 window 40 MiB, the hq
    /// window 36 MB -- and the round buffers ~35 MB a lane beside ~34 MB
    /// that does not move at DFlash2 7. Here are sampling's ~0.58 MB, the
    /// verify round's ~19.02 MB and the drafter's ~5.56 MB a lane; the decode
    /// round's ~10.27 MB, the rest of the 35, is the GPU test's.
    #[test]
    fn one_and_two_lanes_cost_what_piano_vram_says_a_lane() {
        let (one, two) = (served(1, true), served(2, true));
        let image = ModelConfig::qwen38_27b().state_image(KvFormat::HqE8_2b);
        let near = |bytes: u64, mb: f64, line: &str| {
            assert!((bytes as f64 / MB - mb).abs() <= 0.5, "{line}: {bytes} B against PIANO-VRAM's {mb} MB");
        };
        near(image.gdn_recurrent, 151.0, "GDN fp32");
        near(image.gdn_conv_taps, 3.0, "conv");
        near(image.penalty_counts, 1.0, "penalty");
        assert_eq!(dflash2().drafter_state_bytes(1), 40 * MIB, "the DFlash2 window");
        near(image.hq_residual_window, 36.0, "hq window");
        for (lanes, lines) in [(1u32, one), (2, two)] {
            let slots = u64::from(lanes + 2);
            assert_eq!(lines.hq_residual_window, slots * image.hq_residual_window, "{lanes} lanes");
            near(lines.lane_state / u64::from(lanes) + image.hq_residual_window, 233.0, "a lane's state");
        }
        let rounds = |lines: LaneLines| lines.sampling + lines.verify_round + lines.drafter_round;
        let step = |line: fn(&LaneLines) -> u64| line(&two) - line(&one);
        near(step(|l| l.sampling), 0.58, "sampling a lane");
        near(step(|l| l.verify_round), 19.02, "the verify round a lane");
        near(step(|l| l.drafter_round), 5.56, "the drafter's round a lane");
        near(rounds(one) - (rounds(two) - rounds(one)), 34.0, "the round buffers that do not move");
    }

    /// Each line is the lanes' own: the state, the hq window and the verify
    /// round exactly in proportion, sampling and the drafter's round affine
    /// within the arena's rounding of their smallest buffers -- 1, 2 and 8
    /// lanes, as the GPU test asks the kernel.
    #[test]
    fn every_line_follows_the_lanes() {
        let config = ModelConfig::qwen38_27b();
        for vision in [false, true] {
            let at = |lanes| served(lanes, vision);
            let (one, two, eight) = (at(1), at(2), at(8));
            assert_eq!(eight.lane_state, 8 * one.lane_state);
            assert_eq!(eight.verify_round, 8 * one.verify_round, "the verify round has no fixed part");
            assert_eq!(two.verify_round, 2 * one.verify_round);
            for (line, one, two, eight) in [
                ("sampling", one.sampling, two.sampling, eight.sampling),
                ("drafter_round", one.drafter_round, two.drafter_round, eight.drafter_round),
            ] {
                let affine = one + 7 * (two - one);
                assert!(eight.abs_diff(affine) <= 7 * ARENA_ALIGN, "{line}: {eight} against {affine}");
            }
        }
        // Without speculation, no round is verified or drafted and no lane
        // keeps a drafter's window; a BF16 load keeps no hq window.
        let plain = lane_lines(&config, KvFormat::Bf16, None, false, 2, 0);
        assert_eq!((plain.verify_round, plain.drafter_round, plain.hq_residual_window), (0, 0, 0));
        assert_eq!(plain.lane_state, 2 * config.state_image(KvFormat::Bf16).total_bytes());
        assert_eq!(plain.sampling, served(2, false).sampling, "sampling is the lanes' alone");
    }

    /// GitHub #714 (spec #711): the mixed graphs a load plans are the mixes it
    /// captures, and only there: a DFlash2 load of the NVFP4 body at 15
    /// drafts with mixed rounds asked, each mix whose lanes fit the load's
    /// and whose base group's fold, its 2w leading rows beside its n lanes,
    /// fits the pool's GDN slots (the lanes and the device retained slots).
    #[test]
    fn the_mixed_graphs_are_planned_for_the_mixes_a_load_captures() {
        let at = |window: u32, lanes: u32, retained: u32| {
            let speculation = Speculation::new(SpeculativeBackend::Dflash2, window).expect("dflash2");
            mixed_graphs(Some(speculation), false, true, lanes, retained).mixes
        };
        // The `rtx5090` profile's 8 lanes and no device retained slot hold all twelve.
        assert_eq!(
            at(15, 8, 0),
            [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2)]
        );
        assert_eq!(at(15, 1, 2), [], "one lane makes no mix");
        assert_eq!(at(15, 2, 0), [], "two lanes and no retained slot: (1,1)'s fold wants three slots");
        assert_eq!(at(15, 2, 1), [(1, 1)]);
        assert_eq!(at(15, 4, 0), [(1, 1), (1, 2)]);
        assert_eq!(at(15, 4, 2), [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]);
        assert_eq!(at(15, 7, 0), [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (3, 1)]);
        for window in [7, 8, 10, 14] {
            assert_eq!(at(window, 8, 2), [], "{window} drafts: a mixed round runs at 15 alone");
        }
        let fifteen = Speculation::new(SpeculativeBackend::Dflash2, 15).expect("dflash2-15");
        let mtp = Speculation::new(SpeculativeBackend::Mtp, 7).expect("mtp-7");
        assert_eq!(mixed_graphs(Some(fifteen), false, false, 8, 0), MixedGraphs::default(), "mixed rounds not asked");
        assert_eq!(mixed_graphs(Some(fifteen), true, true, 8, 2), MixedGraphs::default(), "the trellis body");
        assert_eq!(mixed_graphs(Some(mtp), false, true, 8, 2), MixedGraphs::default(), "the MTP head");
        assert_eq!(mixed_graphs(None, false, true, 8, 2), MixedGraphs::default(), "no speculation");
        // What the RTX 5090 measured the graphs to take, and a page more: the
        // twelve 64 MiB (62 at capture, 64 MiB and 4 KiB left after the
        // teardown), the first mix alone 10 MiB (8 at capture); two, 14 MiB.
        assert_eq!(mixed_graphs(Some(fifteen), false, true, 8, 0).bytes, 67_108_864);
        assert_eq!(mixed_graphs(Some(fifteen), false, true, 2, 1).bytes, 10 << 20);
        assert_eq!(mixed_graphs(Some(fifteen), false, true, 4, 0).bytes, 14 << 20);
    }

    /// The sampler's workspace the leaf asks the vendored query for: one
    /// 82,176-byte layout at one column, 648,704 at eight, as the 8-lane
    /// sampling line and the verify round's accept (8 columns a lane) hold.
    #[test]
    fn the_samplers_workspace_is_the_vendored_layout() {
        let vocab = ModelConfig::qwen38_27b().vocab;
        assert_eq!(sampling_workspace_capacity_bytes(vocab, 1, 1), 82_176);
        assert_eq!(sampling_workspace_capacity_bytes(vocab, 1, 8), 648_704);
        assert_eq!(sampling_workspace_capacity_bytes(vocab, 8, 8), 648_704);
        assert_eq!(sampling_workspace_capacity_bytes(vocab, 1, 32), sampling_workspace_capacity_bytes(vocab, 1, 16));
        assert_eq!(sampling_workspace_capacity_bytes(vocab, 17, 17), 0, "past sixteen columns, one block");
        assert_eq!(sampling_workspace_capacity_bytes(256, 1, 8), 0, "a domain of one tile");
    }
}
