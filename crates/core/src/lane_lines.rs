//! The 27B's VRAM plan lines that follow a load's decode lanes (GitHub
//! #336, spec 27b-16gb/04 solution 1).
//!
//! Sampling, the verify round, the drafter's round, the lanes' state and the
//! hq residual window, each a function of the lanes and the model's
//! dimensions: the arithmetic `kernel/src/model.cu` sizes the load by
//! (`LoadSizes::reservations`, `verify_round_bytes`,
//! `dflash2_round_activation_bytes`) and the sequence pool's slot image, the
//! vendored layouts those ask included. So a load can be re-planned at fewer
//! lanes on the CPU, and `cargo test` holds every line to its formula; the
//! GPU plan test (`crates/runtime/tests/vram_plan_gpu.rs`) holds the
//! kernel's plan to these functions to the byte at 1, 2 and 8 lanes, as it
//! does the drafter's window against `speculation.rs`.
//!
//! The decode round's line (`decode_graph`) is not here. It is the widest
//! decoder layer's activations plus five workspace queries -- the hq
//! attention's split-K over the context envelope, and the W4A4 activation
//! codes each NVFP4 projection's route takes from four columns on -- whose
//! routes switch on every weight's qtype as the artifact stores it (and as
//! `--model-exception-weights` converts it). A copy here would be a second
//! implementation of the vendored route tables, wrong the first time one of
//! them moves; the GPU test pins the line at 1, 2 and 8 lanes instead, where
//! it is exactly affine in them.

use crate::compute::{LayerKind, ModelConfig};
use crate::kv_format::KvFormat;
use crate::speculation::Speculation;

/// The leaf's arena alignment (`kArenaAlign`, `kernel/src/model.cu`), which
/// the vendored layouts below align every tensor and their total to.
const ARENA_ALIGN: u64 = 256;

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
    // drafter's state (DFlash2's window). The hq residual window rides every
    // slot, lanes and device retained slots alike.
    let slot = image.gdn_recurrent
        + image.gdn_conv_taps
        + image.penalty_counts
        + speculation.map_or(0, |speculation| speculation.drafter_state_bytes(1));
    LaneLines {
        sampling: sampling_bytes(config, lanes),
        verify_round: speculation.map_or(0, |speculation| verify_round_bytes(config, speculation, vision, lanes)),
        drafter_round: speculation.map_or(0, |speculation| speculation.round_scratch_bytes(lanes)),
        lane_state: u64::from(lanes) * slot,
        hq_residual_window: u64::from(lanes + retained_device_slots) * image.hq_residual_window,
    }
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
/// (`verify_round_bytes`, `kernel/src/model.cu`): its staging -- eight
/// per-lane words, the drafts, four (five with vision's rope positions)
/// words per column -- the logits and hidden of every column, each lane's
/// selected hidden, the ReplaySSM records of every GDN layer and lane, and
/// the greedy accept's workspace, never under one arena block. Its buffers
/// are separate allocations, so only the records and the workspace are
/// rounded.
pub fn verify_round_bytes(config: &ModelConfig, speculation: Speculation, vision: bool, lanes: u32) -> u64 {
    let k = u64::from(speculation.draft_tokens());
    if k == 0 {
        return 0;
    }
    let columns = k + 1;
    let lanes = u64::from(lanes);
    let (i32, bf16, fp32) = (4, 2, 4);
    let staging = 8 * lanes * i32 + k * lanes * i32 + if vision { 5 } else { 4 } * columns * lanes * i32;
    let activations =
        config.vocab * columns * lanes * bf16 + config.hidden * columns * lanes * bf16 + config.hidden * lanes * bf16;
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
    staging + activations + records + accept
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
    let aligned = |bytes: u64| bytes.div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
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
