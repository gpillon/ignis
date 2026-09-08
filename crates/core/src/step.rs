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

use crate::model_load::Model;
use crate::seq::{Seq, SeqPool, ffi::IgnisSeq};

mod ffi {
    use std::os::raw::c_char;

    use crate::model_load::ffi::IgnisModel;
    use crate::seq::ffi::{IgnisSeq, IgnisSeqPool};

    /// 1:1 with `struct ignis_sampling_params`.
    #[repr(C)]
    pub struct IgnisSamplingParams {
        pub greedy: i32,
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
        ) -> i32;

        pub fn ignis_program_stats(
            model: *const IgnisModel,
            pool: *const IgnisSeqPool,
            out_stats: *mut IgnisProgramStats,
        ) -> i32;
    }
}

/// Greedy sampling (G1) -- the only supported mode until G3 adds
/// temperature / top-p / top-k / penalties / seed.
const GREEDY: ffi::IgnisSamplingParams = ffi::IgnisSamplingParams { greedy: 1 };

/// Device footprint and most-recent-step telemetry from the real program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProgramStats {
    pub vram_bytes: u64,
    pub last_step_micros: u64,
    pub kernel_count: u64,
}

fn last_error() -> String {
    let message = unsafe { CStr::from_ptr(ffi::ignis_step_last_error()) };
    message.to_string_lossy().into_owned()
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
    let mut tokens = vec![-1; handles.len()];
    let rc = unsafe {
        ffi::ignis_program_decode(
            model.handle(),
            pool.handle(),
            handles.as_mut_ptr(),
            handles.len() as u64,
            &GREEDY,
            tokens.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(tokens)
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
    })
}
