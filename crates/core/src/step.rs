//! The degenerate step ABI call (ADR 0009, GitHub #54, P1-18): embedding ->
//! final RMSNorm -> W8G32 output head -> argmax, with every decoder layer
//! skipped.
//!
//! Scope (P1-18): `skip_layers` is a test-only switch on the leaf's
//! `ignis_prefill` / `ignis_decode` (`kernel/include/ignis_step.h`) -- no
//! production caller sets it, since the GQA/GDN layer bodies do not exist
//! yet (P1-21 / #57, P1-22 / #58). The functions here always set it, so
//! their names say what they do: [`prefill_degenerate`] and
//! [`decode_degenerate_batch`] are for the acceptance test only.

#![cfg(feature = "cuda")]

use std::ffi::CStr;

use std::marker::PhantomData;

use crate::model_load::Model;
use crate::seq::{Seq, SeqPool, ffi::IgnisSeq};
use crate::vision::{Grid, VisionItemControl};

mod ffi {
    use std::os::raw::c_char;

    use crate::model_load::ffi::IgnisModel;
    use crate::seq::ffi::{IgnisSeq, IgnisSeqPool};

    /// 1:1 with `struct ignis_sampling_params` (ADR 0016's size-prefixed
    /// options struct, P3-03, GitHub #99). `size` must be
    /// `sizeof(struct ignis_sampling_params)`; the leaf rejects a size it
    /// does not recognize.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct IgnisSamplingParams {
        pub size: u32,
        pub greedy: i32,
        pub temperature: f32,
        pub top_k: i32,
        pub top_p: f32,
        pub presence_penalty: f32,
        pub frequency_penalty: f32,
        pub seed: u64,
        /// P5-04 (GitHub #153): read only by the verify round. The lane's
        /// remaining budget in tokens, this round's anchor included; 0 = none.
        pub remaining_tokens: u32,
        /// P5-04: entries in `stop_ids`; 0 with a null pointer means none.
        pub stop_id_count: u32,
        /// P5-04: caller-owned, valid for the call.
        pub stop_ids: *const i32,
    }

    /// 1:1 with `struct ignis_decode_options` (ADR 0016; P5-04, GitHub
    /// #153). `size` must be `sizeof(struct ignis_decode_options)`; a null
    /// options pointer is today's round.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct IgnisDecodeOptions {
        pub size: u32,
        pub speculative_window: u32,
        pub drafts: *const i32,
        pub draft_counts: *const u32,
        pub out_committed_counts: *mut i32,
        /// P5-05 (GitHub #155): each lane's extent this round, or null.
        pub out_extents: *mut u32,
    }

    /// 1:1 with `struct ignis_prefill_options` (ADR 0016, P2-02, GitHub
    /// #84). `size` must be `sizeof(struct ignis_prefill_options)`; the
    /// leaf rejects a size it does not recognize.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct IgnisPrefillOptions {
        pub size: u32,
        pub route: i32,
        pub compute_policy: i32,
        /// GitHub #178: the span's axis-major `[3, T]` positions, or null
        /// for a text span (whose media fields are then empty).
        pub mrope_positions: *const i32,
        pub rope_delta: i32,
        pub media_column_count: u32,
        pub media: *const IgnisMediaEmbedding,
        pub media_scatter_indices: *const i32,
        pub media_first_column: u32,
    }

    /// Opaque `struct ignis_media_embedding` (GitHub #178).
    #[repr(C)]
    pub struct IgnisMediaEmbedding {
        _private: [u8; 0],
    }

    /// 1:1 with `struct ignis_media_encode_input` (GitHub #178).
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct IgnisMediaEncodeInput {
        pub size: u32,
        pub grid_t: u32,
        pub grid_h: u32,
        pub grid_w: u32,
        pub patches: *const u16,
        pub position_ids: *const i32,
        pub cu_seqlens: *const i32,
        pub position_table_indices: *const i32,
        pub position_table_weights: *const f32,
    }

    /// `enum ignis_prefill_route`.
    pub const IGNIS_PREFILL_ROUTE_CHUNKED: i32 = 0;
    pub const IGNIS_PREFILL_ROUTE_PER_TOKEN: i32 = 1;

    /// `enum ignis_prefill_compute_policy`. `ENGINE_DEFAULT` leaves the
    /// route decision to the vendored dispatch's own per-projection token
    /// thresholds; `A16_ONLY` forces the pre-existing W8A16 path on every
    /// projection (P2-03, GitHub #85) so a test can compare the two routes
    /// on identical inputs.
    pub const IGNIS_PREFILL_COMPUTE_POLICY_ENGINE_DEFAULT: i32 = 0;
    pub const IGNIS_PREFILL_COMPUTE_POLICY_A16_ONLY: i32 = 1;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct IgnisProgramStats {
        pub vram_bytes: u64,
        pub last_step_micros: u64,
        pub kernel_count: u64,
        /// P3-05 (GitHub #102): `cudaGraphLaunch` calls the most recent
        /// `ignis_program_decode` call made -- 1 if it replayed a decode
        /// graph, 0 if it ran the eager per-lane loop.
        pub graph_launches: u64,
        /// P3-05 (GitHub #102): bit (w-1) set when a decode graph for exact
        /// width w (1..IGNIS_DECODE_MAX_BATCH) is captured and replayable.
        pub decode_graph_ready_mask: u32,
        /// P5-04 (GitHub #153): the same for the verify graphs at the load's
        /// draft window; 0 on a load without one.
        pub verify_graph_ready_mask: u32,
    }

    unsafe extern "C" {
        pub fn ignis_prefill(
            model: *mut IgnisModel,
            token_ids: *const i32,
            num_tokens: u64,
            start_position: u64,
            skip_layers: i32,
            sampling: *const IgnisSamplingParams,
            out_token_id: *mut i32,
            out_logits: *mut f32,
        ) -> i32;

        pub fn ignis_decode(
            model: *mut IgnisModel,
            token_ids: *const i32,
            batch_size: u64,
            skip_layers: i32,
            sampling: *const IgnisSamplingParams,
            out_token_ids: *mut i32,
            out_logits: *mut f32,
        ) -> i32;

        pub fn ignis_step_last_error() -> *const c_char;

        pub fn ignis_program_prefill(
            model: *mut IgnisModel,
            pool: *mut IgnisSeqPool,
            seq: *mut IgnisSeq,
            token_ids: *const i32,
            num_tokens: u64,
            start_position: u64,
            sampling: *const IgnisSamplingParams,
            options: *const IgnisPrefillOptions,
            out_logits: *mut f32,
        ) -> i32;

        pub fn ignis_program_decode(
            model: *mut IgnisModel,
            pool: *mut IgnisSeqPool,
            sequences: *const *mut IgnisSeq,
            batch_size: u64,
            sampling: *const IgnisSamplingParams,
            out_token_ids: *mut i32,
            options: *const IgnisDecodeOptions,
        ) -> i32;

        pub fn ignis_program_stats(
            model: *const IgnisModel,
            pool: *const IgnisSeqPool,
            out_stats: *mut IgnisProgramStats,
        ) -> i32;

        pub fn ignis_decode_graph_capture(
            model: *mut IgnisModel,
            pool: *mut IgnisSeqPool,
            out_capture_micros: *mut u64,
            out_ready_mask: *mut u32,
        ) -> i32;

        pub fn ignis_decode_graph_last_error() -> *const c_char;

        pub fn ignis_media_encode(
            model: *mut IgnisModel,
            input: *const IgnisMediaEncodeInput,
            out_embedding: *mut *mut IgnisMediaEmbedding,
        ) -> i32;

        pub fn ignis_media_embedding_columns(embedding: *const IgnisMediaEmbedding) -> u32;

        pub fn ignis_media_embedding_release(embedding: *mut IgnisMediaEmbedding);

        pub fn ignis_media_last_error() -> *const c_char;
    }
}

/// Greedy sampling: bit-identical to argmax by the vendored sampler's own
/// contract (P3-03, GitHub #99), regardless of the (unread) fields below.
const GREEDY: ffi::IgnisSamplingParams = ffi::IgnisSamplingParams {
    size: std::mem::size_of::<ffi::IgnisSamplingParams>() as u32,
    greedy: 1,
    temperature: 0.0,
    top_k: 0,
    top_p: 1.0,
    presence_penalty: 0.0,
    frequency_penalty: 0.0,
    seed: 0,
    remaining_tokens: 0,
    stop_id_count: 0,
    stop_ids: std::ptr::null(),
};

/// A sequence's real sampling parameters for one program-layer call (P3-03,
/// GitHub #99) -- temperature, top-k (an ignis extension over the
/// OpenAI-compatible surface), top-p, presence/frequency penalties and a
/// seed. `top_k <= 0` or `>= 20` keeps the vendored sampler's own top-20
/// cap; `top_p >= 1.0` disables it. RNG is counter-based (keyed by `seed`
/// and the sequence's own logical position, not by which lane of a decode
/// round it occupies), so what a sequence generates depends only on its own
/// seed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: u64,
}

impl SamplingParams {
    /// `temperature <= 0` selects the leaf's greedy (argmax) branch, so
    /// every other field here is unread.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
        }
    }

    fn to_ffi(self) -> ffi::IgnisSamplingParams {
        ffi::IgnisSamplingParams {
            size: std::mem::size_of::<ffi::IgnisSamplingParams>() as u32,
            greedy: 0,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            remaining_tokens: 0,
            stop_id_count: 0,
            stop_ids: std::ptr::null(),
        }
    }
}

/// One lane's inputs to a verify round (P5-04, GitHub #153): its sampling
/// settings, its remaining generation budget, its stop ids and the drafts
/// proposed for it -- the internal seam a fake drafter fills in tests and
/// the DFlash2 drafter fills from P5-05 on.
///
/// `remaining_tokens` counts the anchor this round emits, so the round
/// proposes at most `remaining_tokens - 1` drafts; 0 means no budget.
/// `drafts` holds at most the load's window; an empty slice proposes nothing
/// and the lane runs at extent 0, one committed token. The committed run is
/// cut at the first id in `stop_ids`, inclusive.
#[derive(Debug, Clone, Copy)]
pub struct VerifyLane<'a> {
    pub sampling: SamplingParams,
    pub remaining_tokens: u32,
    pub stop_ids: &'a [i32],
    pub drafts: &'a [i32],
}

impl<'a> VerifyLane<'a> {
    /// A greedy lane with no budget, no stop ids and these drafts.
    pub fn greedy(drafts: &'a [i32]) -> Self {
        VerifyLane {
            sampling: SamplingParams::greedy(),
            remaining_tokens: 0,
            stop_ids: &[],
            drafts,
        }
    }
}

/// Device footprint and most-recent-step telemetry from the real program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProgramStats {
    pub vram_bytes: u64,
    pub last_step_micros: u64,
    pub kernel_count: u64,
    /// `cudaGraphLaunch` calls the most recent [`decode_program_batch`] /
    /// [`decode_program_batch_sampled`] call made (P3-05, GitHub #102): 1 if
    /// it replayed a captured decode graph, 0 if it ran the eager per-lane
    /// loop.
    pub graph_launches: u64,
    /// Bit `w - 1` set when a decode graph for exact batch width `w`
    /// (1..=8) is captured and replayable (P3-05, GitHub #102) -- see
    /// [`capture_decode_graphs`].
    pub decode_graph_ready_mask: u32,
    /// The same for the verify graphs at the load's draft window (P5-04,
    /// GitHub #153): captured by [`capture_decode_graphs`] on a load with a
    /// window, 0 without one.
    pub verify_graph_ready_mask: u32,
}

/// The result of [`capture_decode_graphs`]: how long capture took and which
/// exact batch widths (1..=8, bit `w - 1`) ended up with a replayable graph.
/// A width whose bit is clear always falls back to the eager per-lane loop
/// (P3-05, GitHub #102, ADR 0019: a capture failure degrades performance and
/// never refuses service) -- [`last_decode_graph_error`] names the most
/// recent one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeGraphCapture {
    pub capture_micros: u64,
    pub ready_mask: u32,
}

impl DecodeGraphCapture {
    /// Whether exact batch width `width` (1..=8) has a replayable graph.
    pub fn is_ready(&self, width: u32) -> bool {
        (1..=8).contains(&width) && (self.ready_mask & (1 << (width - 1))) != 0
    }

    /// How many of the 8 exact widths captured successfully.
    pub fn ready_count(&self) -> u32 {
        self.ready_mask.count_ones()
    }
}

fn last_error() -> String {
    let message = unsafe { CStr::from_ptr(ffi::ignis_step_last_error()) };
    message.to_string_lossy().into_owned()
}

/// The message from the most recent width whose capture failed inside the
/// latest [`capture_decode_graphs`] call (P3-05, GitHub #102) -- its own
/// channel, separate from the program-layer error [`last_error`] reads.
pub fn last_decode_graph_error() -> String {
    let message = unsafe { CStr::from_ptr(ffi::ignis_decode_graph_last_error()) };
    message.to_string_lossy().into_owned()
}

/// Captures one CUDA graph per exact decode batch width 1..=8 (P3-05,
/// GitHub #102, ADR 0019). Call once, after `pool` is created and before any
/// concurrent [`decode_program_batch`] / [`decode_program_batch_sampled`]
/// call -- capture is not thread-safe with replay. Every width is attempted
/// independently: this call always returns `Ok` (the leaf never fails model
/// availability over a capture failure), and [`DecodeGraphCapture::ready_mask`]
/// reports which widths actually got a graph -- a width whose bit is clear
/// always runs the eager per-lane loop, exactly as it did before this call.
pub fn capture_decode_graphs(model: &Model, pool: &SeqPool) -> Result<DecodeGraphCapture, String> {
    let mut capture_micros: u64 = 0;
    let mut ready_mask: u32 = 0;
    let rc = unsafe {
        ffi::ignis_decode_graph_capture(
            model.handle(),
            pool.handle(),
            &mut capture_micros,
            &mut ready_mask,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(DecodeGraphCapture {
        capture_micros,
        ready_mask,
    })
}

/// Runs the degenerate program (GitHub #54) over a token span for one
/// sequence: embedding -> final RMSNorm -> output head -> argmax, with
/// every decoder layer skipped. Returns the last position's argmax token
/// id. `out_logits`, if `Some`, is filled with that position's full
/// vocab-length logits (promoted from the leaf's BF16 storage) -- the
/// caller sizes it to the model's vocab (`ModelConfig::qwen38_27b().vocab`).
///
/// Test-only (`skip_layers` is hardcoded here): no production caller exists
/// until P1-21 / P1-22 (#57 / #58) land the GQA/GDN layer bodies.
pub fn prefill_degenerate(
    model: &Model,
    token_ids: &[i32],
    start_position: u64,
    out_logits: Option<&mut [f32]>,
) -> Result<i32, String> {
    let mut out_token_id: i32 = -1;
    let logits_ptr = match out_logits {
        Some(buf) => buf.as_mut_ptr(),
        None => std::ptr::null_mut(),
    };
    let rc = unsafe {
        ffi::ignis_prefill(
            model.handle(),
            token_ids.as_ptr(),
            token_ids.len() as u64,
            start_position,
            1, // skip_layers: test-only (GitHub #54)
            &GREEDY,
            &mut out_token_id,
            logits_ptr,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(out_token_id)
}

/// Runs the degenerate program (GitHub #54) once per token id in
/// `token_ids` (batch decode: one id per sequence, no cross-token state --
/// the leaf's `ignis_decode`). Returns one argmax token id per input.
/// `out_logits`, if `Some`, must be exactly `token_ids.len() * vocab`
/// entries long; sequence `i`'s logits land at `out_logits[i * vocab ..
/// (i + 1) * vocab]`.
///
/// Test-only (`skip_layers` is hardcoded here): see [`prefill_degenerate`].
pub fn decode_degenerate_batch(
    model: &Model,
    token_ids: &[i32],
    out_logits: Option<&mut [f32]>,
) -> Result<Vec<i32>, String> {
    let mut out_token_ids = vec![-1i32; token_ids.len()];
    let logits_ptr = match out_logits {
        Some(buf) => buf.as_mut_ptr(),
        None => std::ptr::null_mut(),
    };
    let rc = unsafe {
        ffi::ignis_decode(
            model.handle(),
            token_ids.as_ptr(),
            token_ids.len() as u64,
            1, // skip_layers: test-only (GitHub #54)
            &GREEDY,
            out_token_ids.as_mut_ptr(),
            logits_ptr,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(out_token_ids)
}

/// Which internal path [`prefill_program_with_route`] takes over a span's
/// chunks (`enum ignis_prefill_route`, ADR 0016, P2-02, GitHub #84).
/// [`prefill_program`] always uses [`PrefillRoute::Chunked`] (`NULL`
/// options, the production default); [`PrefillRoute::PerToken`] is
/// test-only, the self-oracle for the chunk loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillRoute {
    Chunked,
    PerToken,
}

/// The compute policy passed through the prefill options (`enum
/// ignis_prefill_compute_policy`, ADR 0016, P2-03, GitHub #85).
///
/// [`ComputePolicy::EngineDefault`] (the production default, what `NULL`
/// options select) leaves the route decision to the vendored dispatch's
/// own per-projection token thresholds -- no engine-introduced threshold.
/// [`ComputePolicy::A16Only`] forces the pre-existing W8A16 path on every
/// projection, so a test can compare both routes on identical inputs and
/// confirm the A4 route's divisors are the only thing that changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputePolicy {
    EngineDefault,
    A16Only,
}

impl PrefillRoute {
    fn to_options(self, policy: ComputePolicy) -> ffi::IgnisPrefillOptions {
        let route = match self {
            PrefillRoute::Chunked => ffi::IGNIS_PREFILL_ROUTE_CHUNKED,
            PrefillRoute::PerToken => ffi::IGNIS_PREFILL_ROUTE_PER_TOKEN,
        };
        let compute_policy = match policy {
            ComputePolicy::EngineDefault => {
                ffi::IGNIS_PREFILL_COMPUTE_POLICY_ENGINE_DEFAULT
            }
            ComputePolicy::A16Only => ffi::IGNIS_PREFILL_COMPUTE_POLICY_A16_ONLY,
        };
        ffi::IgnisPrefillOptions {
            size: std::mem::size_of::<ffi::IgnisPrefillOptions>() as u32,
            route,
            compute_policy,
            mrope_positions: std::ptr::null(),
            rope_delta: 0,
            media_column_count: 0,
            media: std::ptr::null(),
            media_scatter_indices: std::ptr::null(),
            media_first_column: 0,
        }
    }
}

/// Prefill one span through the complete 64-layer program, via the
/// production chunked route (`NULL` options -- ADR 0016, P2-02, GitHub
/// #84): the leaf cuts the span into `ignis_model_load`'s
/// `prefill_chunk_tokens`-wide chunks internally, the caller never needs to
/// know the chunk width. The sequence's KV pages, GDN slot, conv taps and
/// position land exactly where the per-token route would have left them.
///
/// `out_logits`, if `Some`, is filled with the span's *last* position's full
/// vocab-length logits (promoted from the leaf's BF16 storage) -- the same
/// position whose argmax becomes the successor the next
/// [`decode_program_batch`] round emits. Debug-only (GitHub #72: confirming
/// a near-tie argmax flip on the canary suite needs the real logits, not
/// just the winning id); the caller sizes the buffer to the model's vocab
/// (`ModelConfig::qwen38_27b().vocab`).
pub fn prefill_program(
    model: &Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
    token_ids: &[i32],
    start_position: u64,
    out_logits: Option<&mut [f32]>,
) -> Result<(), String> {
    let logits_ptr = match out_logits {
        Some(buf) => buf.as_mut_ptr(),
        None => std::ptr::null_mut(),
    };
    // NULL options: the production defaults (chunked route, engine default
    // compute policy) -- ADR 0016 keeps this call site exactly as simple as
    // it was before the options struct existed.
    let rc = unsafe {
        ffi::ignis_program_prefill(
            model.handle(),
            pool.handle(),
            sequence.handle(),
            token_ids.as_ptr(),
            token_ids.len() as u64,
            start_position,
            &GREEDY,
            std::ptr::null(),
            logits_ptr,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(())
}

/// [`prefill_program`], with real sampling parameters (P3-03, GitHub #99)
/// instead of the hardcoded greedy default: the span's last position's
/// successor -- what [`decode_program_batch_sampled`] will first emit for
/// this sequence -- is drawn under `sampling` instead of forced to argmax.
pub fn prefill_program_sampled(
    model: &Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
    token_ids: &[i32],
    start_position: u64,
    sampling: SamplingParams,
    out_logits: Option<&mut [f32]>,
) -> Result<(), String> {
    let logits_ptr = match out_logits {
        Some(buf) => buf.as_mut_ptr(),
        None => std::ptr::null_mut(),
    };
    let params = sampling.to_ffi();
    let rc = unsafe {
        ffi::ignis_program_prefill(
            model.handle(),
            pool.handle(),
            sequence.handle(),
            token_ids.as_ptr(),
            token_ids.len() as u64,
            start_position,
            &params,
            std::ptr::null(),
            logits_ptr,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(())
}

/// [`prefill_program`], with the route selectable (ADR 0016, P2-02, GitHub
/// #84). Test-only entry point: [`PrefillRoute::PerToken`] exists so the
/// chunked route has a self-oracle (the same prompt prefilled both ways
/// must agree) that needs no reference engine. The compute policy stays
/// [`ComputePolicy::EngineDefault`]; use [`prefill_program_with_policy`]
/// to force [`ComputePolicy::A16Only`] as well (P2-03, GitHub #85).
pub fn prefill_program_with_route(
    model: &Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
    token_ids: &[i32],
    start_position: u64,
    route: PrefillRoute,
    out_logits: Option<&mut [f32]>,
) -> Result<(), String> {
    prefill_program_with_policy(
        model,
        pool,
        sequence,
        token_ids,
        start_position,
        route,
        ComputePolicy::EngineDefault,
        out_logits,
    )
}

/// [`prefill_program`], with both the route and the compute policy
/// selectable (ADR 0016, P2-03, GitHub #85). Test-only entry point:
/// [`ComputePolicy::A16Only`] forces the pre-existing W8A16 path on every
/// projection -- the pre-#85 behaviour -- so a test can compare the two
/// routes on identical inputs and pin down that the A4 route's divisors
/// are the only difference.
pub fn prefill_program_with_policy(
    model: &Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
    token_ids: &[i32],
    start_position: u64,
    route: PrefillRoute,
    policy: ComputePolicy,
    out_logits: Option<&mut [f32]>,
) -> Result<(), String> {
    let logits_ptr = match out_logits {
        Some(buf) => buf.as_mut_ptr(),
        None => std::ptr::null_mut(),
    };
    let options = route.to_options(policy);
    let rc = unsafe {
        ffi::ignis_program_prefill(
            model.handle(),
            pool.handle(),
            sequence.handle(),
            token_ids.as_ptr(),
            token_ids.len() as u64,
            start_position,
            &GREEDY,
            &options,
            logits_ptr,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(())
}

/// Emit one greedy token for each sequence and prepare the following round.
pub fn decode_program_batch(
    model: &Model,
    pool: &SeqPool,
    sequences: &mut [&mut Seq<'_>],
) -> Result<Vec<i32>, String> {
    let mut handles: Vec<*mut IgnisSeq> = sequences.iter_mut().map(|seq| seq.handle()).collect();
    // ignis_program_decode reads one ignis_sampling_params per sequence
    // (P3-03, GitHub #99) -- an array parallel to `handles`, not one shared
    // struct.
    let params = vec![GREEDY; handles.len()];
    let mut tokens = vec![-1; handles.len()];
    let rc = unsafe {
        ffi::ignis_program_decode(
            model.handle(),
            pool.handle(),
            handles.as_mut_ptr(),
            handles.len() as u64,
            params.as_ptr(),
            tokens.as_mut_ptr(),
            std::ptr::null(),
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(tokens)
}

/// [`decode_program_batch`], with one real [`SamplingParams`] per sequence
/// (P3-03, GitHub #99): `sampling[i]` draws sequence `i`'s successor, so
/// lanes sharing this round may carry independent temperatures, seeds and
/// penalty histories. `sampling.len()` must equal `sequences.len()` and must
/// not exceed the leaf's `IGNIS_DECODE_MAX_BATCH` (8, `ignis_step.h`).
pub fn decode_program_batch_sampled(
    model: &Model,
    pool: &SeqPool,
    sequences: &mut [&mut Seq<'_>],
    sampling: &[SamplingParams],
) -> Result<Vec<i32>, String> {
    if sampling.len() != sequences.len() {
        return Err(format!(
            "decode_program_batch_sampled: {} sampling params for {} sequences",
            sampling.len(),
            sequences.len()
        ));
    }
    let mut handles: Vec<*mut IgnisSeq> = sequences.iter_mut().map(|seq| seq.handle()).collect();
    let params: Vec<ffi::IgnisSamplingParams> = sampling.iter().map(|p| p.to_ffi()).collect();
    let mut tokens = vec![-1; handles.len()];
    let rc = unsafe {
        ffi::ignis_program_decode(
            model.handle(),
            pool.handle(),
            handles.as_mut_ptr(),
            handles.len() as u64,
            params.as_ptr(),
            tokens.as_mut_ptr(),
            std::ptr::null(),
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(tokens)
}

/// One verify round over `sequences` at the load's draft `window` (P5-04,
/// GitHub #153, spec 05): every lane's anchor and drafts are verified in one
/// traversal, the vendored accept kernel licenses the accepted prefix, the
/// leaf cuts each run at the lane's first stop id and commits it -- KV
/// frontier, GDN slot and conv taps (the ReplaySSM fold), pending token.
/// Returns, per lane in order, the committed run: the anchor followed by the
/// accepted drafts (1..=window+1 tokens, fewer after a cut). A lane whose
/// budget or context leaves no room for drafts, or whose `drafts` is empty,
/// runs at extent 0 and commits its anchor alone -- today's round.
///
/// `window` must be the window the model was loaded with
/// ([`crate::Speculation::draft_tokens`]); the leaf rejects any other, naming
/// both. `lanes.len()` must equal `sequences.len()` and each lane's `drafts`
/// must not exceed `window`.
pub fn decode_program_verify(
    model: &Model,
    pool: &SeqPool,
    sequences: &mut [&mut Seq<'_>],
    lanes: &[VerifyLane<'_>],
    window: u32,
) -> Result<Vec<Vec<i32>>, String> {
    Ok(decode_program_verify_runs(model, pool, sequences, lanes, window)?
        .into_iter()
        .map(|run| run.tokens)
        .collect())
}

/// One lane's run through a verify round (P5-05, GitHub #155): the committed run, and how
/// many drafts the round verified for the lane -- its extent, whether the
/// caller proposed them or the DFlash2 drafter did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneVerifyRun {
    pub tokens: Vec<i32>,
    pub extent: u32,
}

/// [`decode_program_verify`], reporting each lane's extent with its run.
///
/// On a load with the DFlash2 drafter ([`crate::SpeculativeBackend::Dflash2`])
/// the leaf proposes every lane's drafts from the lane's own window, so every
/// lane's `drafts` must be empty; the leaf refuses a proposal it would
/// otherwise ignore.
pub fn decode_program_verify_runs(
    model: &Model,
    pool: &SeqPool,
    sequences: &mut [&mut Seq<'_>],
    lanes: &[VerifyLane<'_>],
    window: u32,
) -> Result<Vec<LaneVerifyRun>, String> {
    if lanes.len() != sequences.len() {
        return Err(format!(
            "decode_program_verify: {} lanes for {} sequences",
            lanes.len(),
            sequences.len()
        ));
    }
    if window == 0 {
        return Err(
            "decode_program_verify: window 0 is today's round, not a verify round -- use decode_program_batch_sampled"
                .to_string(),
        );
    }
    let width = window as usize;
    for (index, lane) in lanes.iter().enumerate() {
        if lane.drafts.len() > width {
            return Err(format!(
                "decode_program_verify: lane {index} proposes {} drafts for a window of {window}",
                lane.drafts.len()
            ));
        }
    }
    let mut handles: Vec<*mut IgnisSeq> = sequences.iter_mut().map(|seq| seq.handle()).collect();
    let params: Vec<ffi::IgnisSamplingParams> = lanes
        .iter()
        .map(|lane| ffi::IgnisSamplingParams {
            remaining_tokens: lane.remaining_tokens,
            stop_id_count: lane.stop_ids.len() as u32,
            stop_ids: if lane.stop_ids.is_empty() {
                std::ptr::null()
            } else {
                lane.stop_ids.as_ptr()
            },
            ..lane.sampling.to_ffi()
        })
        .collect();
    // Row-major `[batch][window]`, each lane's proposals first, the rest
    // unread past its count.
    let mut drafts = vec![0i32; handles.len() * width];
    let mut draft_counts = vec![0u32; handles.len()];
    for (index, lane) in lanes.iter().enumerate() {
        drafts[index * width..index * width + lane.drafts.len()].copy_from_slice(lane.drafts);
        draft_counts[index] = lane.drafts.len() as u32;
    }
    let mut tokens = vec![-1i32; handles.len() * (width + 1)];
    let mut committed = vec![0i32; handles.len()];
    let mut extents = vec![0u32; handles.len()];
    // No proposal on any lane is a null seam: every lane at extent 0 under a
    // caller-fed load, the drafter's own proposals under DFlash2.
    let proposes = lanes.iter().any(|lane| !lane.drafts.is_empty());
    let options = ffi::IgnisDecodeOptions {
        size: std::mem::size_of::<ffi::IgnisDecodeOptions>() as u32,
        speculative_window: window,
        drafts: if proposes { drafts.as_ptr() } else { std::ptr::null() },
        draft_counts: if proposes { draft_counts.as_ptr() } else { std::ptr::null() },
        out_committed_counts: committed.as_mut_ptr(),
        out_extents: extents.as_mut_ptr(),
    };
    let rc = unsafe {
        ffi::ignis_program_decode(
            model.handle(),
            pool.handle(),
            handles.as_mut_ptr(),
            handles.len() as u64,
            params.as_ptr(),
            tokens.as_mut_ptr(),
            &options,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(committed
        .iter()
        .zip(&extents)
        .enumerate()
        .map(|(index, (&count, &extent))| {
            let start = index * (width + 1);
            LaneVerifyRun {
                tokens: tokens[start..start + count as usize].to_vec(),
                extent,
            }
        })
        .collect())
}

/// A media item's device-resident encoder output (GitHub #178): the
/// `[hidden, columns]` merged columns the media encode step wrote into the
/// load's vision reservation. The reservation holds one at a time, so an
/// embedding is dropped (released) as soon as its last placeholder column is
/// prefilled.
pub struct MediaEmbedding<'m> {
    handle: *mut ffi::IgnisMediaEmbedding,
    _model: PhantomData<&'m Model>,
}

// The handle is a plain leaf-owned record; like `Seq`, it is driven from the
// single scheduler thread.
unsafe impl Send for MediaEmbedding<'_> {}

impl MediaEmbedding<'_> {
    /// The embedding's merged columns.
    pub fn columns(&self) -> u32 {
        unsafe { ffi::ignis_media_embedding_columns(self.handle) }
    }

    /// Detach the embedding from its model's borrow.
    ///
    /// # Safety
    ///
    /// The model must outlive the returned embedding.
    pub unsafe fn into_static(self) -> MediaEmbedding<'static> {
        let handle = self.handle;
        std::mem::forget(self);
        MediaEmbedding {
            handle,
            _model: PhantomData,
        }
    }
}

impl Drop for MediaEmbedding<'_> {
    fn drop(&mut self) {
        unsafe { ffi::ignis_media_embedding_release(self.handle) }
    }
}

fn media_last_error() -> String {
    let message = unsafe { CStr::from_ptr(ffi::ignis_media_last_error()) };
    message.to_string_lossy().into_owned()
}

/// Run the media encode step (GitHub #178) over one item: its row-major BF16
/// patch rows on `grid` and the encoder control
/// ([`crate::vision::vision_item_control`]) computed for that grid.
pub fn encode_media<'m>(
    model: &'m Model,
    grid: Grid,
    patches: &[u16],
    control: &VisionItemControl,
) -> Result<MediaEmbedding<'m>, String> {
    let raw = grid.raw_patches() as usize;
    if patches.len() != raw * ignis_artifact::vision::PATCH_FEATURES {
        return Err(format!(
            "encode_media: {} patch values for a {}x{}x{} grid",
            patches.len(),
            grid.t,
            grid.h,
            grid.w
        ));
    }
    if control.patches as usize != raw
        || control.position_ids.len() != 2 * raw
        || control.position_table_indices.len() != 4 * raw
        || control.position_table_weights.len() != 4 * raw
        || control.cu_seqlens.len() != grid.t as usize + 1
    {
        return Err("encode_media: the control does not describe the grid".to_string());
    }
    let input = ffi::IgnisMediaEncodeInput {
        size: std::mem::size_of::<ffi::IgnisMediaEncodeInput>() as u32,
        grid_t: grid.t,
        grid_h: grid.h,
        grid_w: grid.w,
        patches: patches.as_ptr(),
        position_ids: control.position_ids.as_ptr(),
        cu_seqlens: control.cu_seqlens.as_ptr(),
        position_table_indices: control.position_table_indices.as_ptr(),
        position_table_weights: control.position_table_weights.as_ptr(),
    };
    let mut handle = std::ptr::null_mut();
    let rc = unsafe { ffi::ignis_media_encode(model.handle(), &input, &mut handle) };
    if rc != 0 {
        return Err(media_last_error());
    }
    Ok(MediaEmbedding {
        handle,
        _model: PhantomData,
    })
}

/// One span of a multimodal prompt for [`prefill_program_multimodal`]
/// (GitHub #178).
#[derive(Clone, Copy)]
pub struct MultimodalPrefill<'a> {
    /// Axis-major `[3, tokens]` positions the span rotates at.
    pub positions: &'a [i32],
    /// The prompt's rope delta, which every later decode round applies.
    pub rope_delta: i32,
    /// The media columns the span's placeholders take, if it covers any.
    pub media: Option<SpanMediaColumns<'a>>,
}

/// A media item's columns placed over a span's placeholder rows.
#[derive(Clone, Copy)]
pub struct SpanMediaColumns<'a> {
    pub embedding: &'a MediaEmbedding<'a>,
    /// The embedding column of the first placeholder.
    pub first_column: u32,
    /// Span-relative placeholder positions, strictly increasing.
    pub scatter_indices: &'a [i32],
}

/// [`prefill_program_sampled`] over a span of a multimodal prompt (GitHub
/// #178): the chunked route, rotated at the span's three-axis positions, with
/// the media columns scattered over its placeholder rows. The sequence keeps
/// the span's rope delta for every later decode round.
pub fn prefill_program_multimodal(
    model: &Model,
    pool: &SeqPool,
    sequence: &mut Seq<'_>,
    token_ids: &[i32],
    start_position: u64,
    sampling: SamplingParams,
    span: MultimodalPrefill<'_>,
    out_logits: Option<&mut [f32]>,
) -> Result<(), String> {
    if span.positions.len() != 3 * token_ids.len() {
        return Err(format!(
            "prefill_program_multimodal: {} positions for {} tokens",
            span.positions.len(),
            token_ids.len()
        ));
    }
    let (media, scatter, count, first_column) = match span.media {
        Some(columns) => (
            columns.embedding.handle as *const ffi::IgnisMediaEmbedding,
            columns.scatter_indices.as_ptr(),
            columns.scatter_indices.len() as u32,
            columns.first_column,
        ),
        None => (std::ptr::null(), std::ptr::null(), 0, 0),
    };
    let options = ffi::IgnisPrefillOptions {
        mrope_positions: span.positions.as_ptr(),
        rope_delta: span.rope_delta,
        media_column_count: count,
        media,
        media_scatter_indices: scatter,
        media_first_column: first_column,
        ..PrefillRoute::Chunked.to_options(ComputePolicy::EngineDefault)
    };
    let logits_ptr = match out_logits {
        Some(buf) => buf.as_mut_ptr(),
        None => std::ptr::null_mut(),
    };
    let params = sampling.to_ffi();
    let rc = unsafe {
        ffi::ignis_program_prefill(
            model.handle(),
            pool.handle(),
            sequence.handle(),
            token_ids.as_ptr(),
            token_ids.len() as u64,
            start_position,
            &params,
            &options,
            logits_ptr,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(())
}

/// Read full-program telemetry without exposing a device pointer or stream.
pub fn program_stats(model: &Model, pool: &SeqPool) -> Result<ProgramStats, String> {
    let mut stats = ffi::IgnisProgramStats::default();
    let rc = unsafe { ffi::ignis_program_stats(model.handle(), pool.handle(), &mut stats) };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(ProgramStats {
        vram_bytes: stats.vram_bytes,
        last_step_micros: stats.last_step_micros,
        kernel_count: stats.kernel_count,
        graph_launches: stats.graph_launches,
        decode_graph_ready_mask: stats.decode_graph_ready_mask,
        verify_graph_ready_mask: stats.verify_graph_ready_mask,
    })
}
