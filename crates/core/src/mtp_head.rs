//! The 27B's MTP head on the VRAM plan (GitHub #620, spec #618): what a 27B
//! load built for its MTP head reserves, computed once on the CPU and held to
//! the leaf's own plan to the byte (`kernel/src/mtp_head.h`,
//! `kernel/src/seq.cu`; the GPU pin is `crates/core/tests/mtp_head_gpu.rs`).
//!
//! The head is the checkpoint's one MTP layer, which every 27B artifact
//! carries: its weights are on the plan's `weights` line, its attention state
//! is one more attention layer of the KV pool (the pool's bytes per token
//! grow by one layer in sixteen, [`crate::seq::kv_geometry`]), and the rest
//! is here: the continuation hidden every pool slot keeps (the `lane_state`
//! and `retained_slots` lines), the head layer's own ring words under
//! hq-e8-2b beside its residual planes (the `hq_residual_window` line), and
//! the head's buffers in a prefill chunk (`workspace`) and a round
//! (`decode_graph`). Nothing of DFlash2.

/// The head's 12 objects in the container: five W8G32 matrices and seven BF16
/// gains, the same bytes on every 27B artifact
/// (`ignis_artifact::mtp_scope_27b`).
pub const WEIGHT_BYTES: u64 = 451_267_584;

/// The target's hidden width, which the head's stem and continuation hidden
/// share.
const HIDDEN: u64 = 5120;

/// The head's attention layer: 24 query heads of 256 and 4 KV heads, the
/// fused projection's rows query, key, gate and value, and its dense MLP.
const QUERY_WIDTH: u64 = 24 * 256;
const KV_WIDTH: u64 = 4 * 256;
const PROJECTION_ROWS: u64 = 2 * QUERY_WIDTH + 2 * KV_WIDTH;
const FFN: u64 = 17408;

/// The most rows one head call takes in a prefill chunk
/// (`ignis::mtp::kPrefillRows`): what the head adds to a chunk's scratch does
/// not grow past this many rows.
pub const PREFILL_ROWS: u64 = 128;

/// The hq-e8-2b residual window's rows per layer, KV head and slot (32 sink
/// keys and a 512-key ring) and its ring words per slot
/// (`kIgnisHqResidualRows`, `kIgnisHqRingWords`).
const HQ_RESIDUAL_ROWS: u64 = 32 + 512;
const HQ_RING_WORDS: u64 = 512 / 32;

/// The leaf's scratch-arena allocation alignment.
const ARENA_ALIGN: u64 = 256;

fn bf16(elements: u64) -> u64 {
    (elements * 2).div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

fn i32(elements: u64) -> u64 {
    (elements * 4).div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

/// The continuation hidden one pool slot keeps: the target's post-final-norm
/// hidden at the sequence's last position, BF16.
pub fn continuation_bytes() -> u64 {
    HIDDEN * 2
}

/// What the head's layer adds to one slot's hq-e8-2b residual window: its K
/// and V planes (every KV head's 544 exact rows) and its own ring words.
pub fn hq_residual_window_bytes() -> u64 {
    2 * HQ_RESIDUAL_ROWS * KV_WIDTH * 2 + HQ_RING_WORDS * 4
}

/// The buffers one head call of `rows` rows takes from its arena, beside its
/// attention's workspace, which the target's GQA layers' reservation at the
/// same widths already holds (`ignis::mtp::call_bytes`): the stem's two normed
/// inputs, their concatenation and the residual; the attention's input, the
/// fused projection, its four parts, the two rotated ones and the
/// attention's output; the output projection, the MLP's input, gate and up,
/// the SwiGLU and down.
pub fn call_bytes(rows: u64) -> u64 {
    2 * bf16(HIDDEN * rows)
        + bf16(2 * HIDDEN * rows)
        + bf16(HIDDEN * rows)
        + bf16(HIDDEN * rows)
        + bf16(PROJECTION_ROWS * rows)
        + 3 * bf16(QUERY_WIDTH * rows)
        + 3 * bf16(KV_WIDTH * rows)
        + bf16(QUERY_WIDTH * rows)
        + 2 * bf16(HIDDEN * rows)
        + bf16(2 * FFN * rows)
        + bf16(FFN * rows)
        + bf16(HIDDEN * rows)
}

/// What a `prefill_chunk_tokens`-wide chunk adds to the prefill scratch
/// (`ignis::mtp::prefill_scratch_bytes`): one call of at most
/// [`PREFILL_ROWS`] rows, its input hiddens, embeddings and head hiddens, its
/// positions, rotation positions and table row.
pub fn prefill_scratch_bytes(prefill_chunk_tokens: u32) -> u64 {
    let rows = u64::from(prefill_chunk_tokens).min(PREFILL_ROWS);
    3 * bf16(HIDDEN * rows) + 2 * i32(rows) + i32(1) + call_bytes(rows)
}

/// What a round of `columns` columns adds to the decode-round scratch
/// (`ignis::mtp::round_scratch_bytes`): one call over every column, its
/// continuation hiddens, embeddings and head hiddens and its positions. A load
/// sizes it for its widest round, `lanes * (k + 1)` columns.
pub fn round_scratch_bytes(columns: u64) -> u64 {
    3 * bf16(HIDDEN * columns) + i32(columns) + call_bytes(columns)
}

/// The route of the head's five W8G32 matrices (GitHub #627), which no load
/// option names: the leaf decides it once per MTP load from the card, or the
/// measurement switch `IGNIS_MTP_HEAD_LINEAR` names ours on Blackwell, and
/// reports it (`Model::mtp_head_linear`; `enum ignis_mtp_head_linear`,
/// `kernel/include/ignis_model.h`). The VRAM plan is the same on either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadLinear {
    /// The vendored `ops::linear`: a Blackwell card's MTP load.
    Vendored,
    /// ignis's (`kernel/src/w8_imma.h`): the vendored SIMT route up to 8
    /// tokens, the integer MMA above; an RTX 30 or RTX 40 card's MTP load, or
    /// a Blackwell card's under `IGNIS_MTP_HEAD_LINEAR=ours`.
    Ours,
}

impl HeadLinear {
    /// The leaf's code: `Some(None)` for a load without the head (0),
    /// `Some(Some(..))` for 1 (vendored) and 2 (ours), `None` for any other.
    pub const fn from_abi(code: u32) -> Option<Option<Self>> {
        match code {
            0 => Some(None),
            1 => Some(Some(Self::Vendored)),
            2 => Some(Some(Self::Ours)),
            _ => None,
        }
    }

    /// The switch's own spellings, which the load's
    /// `ignis.runtime.decode_routes` line reports (`none` without the head).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vendored => "vendored",
            Self::Ours => "ours",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The leaf's codes, and the switch's spellings.
    #[test]
    fn the_heads_matrix_routes_are_the_headers_codes() {
        assert_eq!(HeadLinear::from_abi(0), Some(None));
        assert_eq!(HeadLinear::from_abi(1), Some(Some(HeadLinear::Vendored)));
        assert_eq!(HeadLinear::from_abi(2), Some(Some(HeadLinear::Ours)));
        assert_eq!(HeadLinear::from_abi(3), None);
        assert_eq!((HeadLinear::Vendored.as_str(), HeadLinear::Ours.as_str()), ("vendored", "ours"));
    }

    /// The container's 12 objects at their stored sizes: five W8G32 matrices
    /// (an int8 code per weight, one FP16 scale per 32 along K) and seven BF16
    /// gains. The head's weights are its plan's `weights` line.
    #[test]
    fn the_heads_weights_are_its_twelve_objects() {
        let w8 = |rows: u64, cols: u64| rows * cols + rows * (cols / 32) * 2;
        let matrices = w8(5120, 10240) + w8(14336, 5120) + w8(5120, 6144) + w8(34816, 5120) + w8(5120, 17408);
        let gains = 5 * 5120 * 2 + 2 * 256 * 2;
        assert_eq!(matrices + gains, WEIGHT_BYTES);
    }

    /// One slot's continuation hidden is a BF16 vector of the hidden; its hq
    /// window is one more layer's planes, 2,228,224 bytes, and 64 bytes of
    /// ring words.
    #[test]
    fn a_slot_keeps_one_hidden_and_the_heads_layer_of_the_hq_window() {
        assert_eq!(continuation_bytes(), 10_240);
        assert_eq!(hq_residual_window_bytes(), 2_228_224 + 64);
    }

    /// The prefill line stops growing at the head's call width: a 1,024-token
    /// chunk adds what a 128-token one does.
    #[test]
    fn the_prefill_line_is_one_call_at_most_the_call_width_wide() {
        assert_eq!(prefill_scratch_bytes(1024), prefill_scratch_bytes(128));
        assert!(prefill_scratch_bytes(64) < prefill_scratch_bytes(128));
        // 140,288 BF16 elements a row through the call's buffers.
        assert_eq!(call_bytes(128), 140_288 * 2 * 128);
        assert_eq!(prefill_scratch_bytes(128), 3 * 5120 * 2 * 128 + 2 * 512 + 256 + call_bytes(128));
    }

    /// The round line follows the round's columns.
    #[test]
    fn the_round_line_follows_the_rounds_columns() {
        for columns in [1u64, 8, 64] {
            assert_eq!(round_scratch_bytes(columns), 3 * bf16(5120 * columns) + i32(columns) + call_bytes(columns));
        }
        assert!(round_scratch_bytes(64) > round_scratch_bytes(8));
    }
}
