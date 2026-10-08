//! Flash-Next's mixture-of-experts ops in the kernel leaf (spec flash-next/02, GitHub #300):
//! the flat C ABI of `kernel/include/ignis_moe.h`, 1:1, and thin wrappers that turn its return
//! codes into `Result`s carrying the leaf's own message.
//!
//! The ops are program-side ops (ADR 0009): the Flash-Next layer program (spec 04) calls them
//! inside the leaf, and these bindings exist for Rust-side tests and tools that drive them
//! directly. Every pointer argument is a device pointer the caller vouches for, so the
//! wrappers are `unsafe`; the shapes and counts they take are checked by the leaf.
//!
//! All of them are our own implementation, with no port claim (ADR 0010 / ADR 0043).

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::os::raw::c_void;

/// The MoE block's hidden width.
pub const HIDDEN: u32 = 2560;
/// Routed experts per layer.
pub const EXPERTS: u32 = 512;
/// Experts the router selects per token.
pub const TOP_K: u32 = 10;
/// Each expert's (and the shared expert's) intermediate width.
pub const INTERMEDIATE: u32 = 640;
/// Tokens the decode route serves in one launch.
pub const DECODE_MAX_TOKENS: u32 = 8;

/// The fused gate/up expert projection, (in, out) = (2560, 1280).
pub const PROJ_GATE_UP: u32 = 0;
/// The down expert projection, (in, out) = (640, 2560).
pub const PROJ_DOWN: u32 = 1;

/// The decode route of a workspace (`MoeWorkspace::decode_route`, `IGNIS_MOE_DECODE_*`): one
/// persistent launch of work items taken by ticket (up to 4 tokens streamed through shared
/// memory, GitHub #306), one thread-block cluster per selected expert, or the ticket launch
/// before #306 whose units hold their weights in registers.
pub const DECODE_TICKETS: u32 = 0;
pub const DECODE_CLUSTERS: u32 = 1;
pub const DECODE_REGISTERS: u32 = 2;

/// One slot-table entry, 1:1 with `struct ignis_moe_slot`: the device address of an expert
/// projection's record (layout.md §3) and its bit width as `k2 = 2 K`. A layer's table is
/// `EXPERTS * 2` entries indexed `expert * 2 + projection`; a selected entry with a NULL
/// record traps the kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeSlot {
    pub record: *const c_void,
    pub k2: u32,
    pub reserved: u32,
}

impl MoeSlot {
    /// The entry of a projection that is not resident.
    pub const ABSENT: MoeSlot = MoeSlot { record: std::ptr::null(), k2: 0, reserved: 0 };
}

/// A MoE workspace, 1:1 with `struct ignis_moe_workspace`: its device base, the two capacities it
/// was sized for (`workspace_bytes`) and the decode route of the model instance that owns it,
/// passed to every routed-expert call.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeWorkspace {
    pub base: *mut c_void,
    pub decode_tokens: u32,
    pub prefill_tokens: u32,
    pub decode_route: u32,
}

/// The device buffers one MoE block needs, 1:1 with `struct ignis_moe_plan`: the plan lines a
/// Flash-Next load reserves (ADR 0030), each rounded up to 256 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoePlan {
    pub workspace: u64,
    pub acc: u64,
    pub router: u64,
    pub shared: u64,
    pub total: u64,
}

/// 1:1 with `kernel/include/ignis_moe.h`.
pub mod ffi {
    use super::{MoePlan, MoeSlot, MoeWorkspace};
    use std::os::raw::{c_char, c_void};

    unsafe extern "C" {
        pub fn ignis_moe_record_bytes(projection: u32, k2: u32, bytes: *mut u64) -> i32;
        pub fn ignis_moe_trellis_reconstruct(
            trellis: *const c_void,
            k2: u32,
            in_features: u32,
            out_features: u32,
            w_f16: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_moe_router(
            x: *const c_void,
            tokens: u32,
            w_router: *const c_void,
            ids: *mut i32,
            weights: *mut f32,
            logits: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_moe_experts_decode(
            x: *const c_void,
            tokens: u32,
            ids: *const i32,
            weights: *const f32,
            slots: *const MoeSlot,
            workspace: *const MoeWorkspace,
            acc: *mut i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_moe_experts_prefill(
            x: *const c_void,
            tokens: u32,
            ids: *const i32,
            weights: *const f32,
            slots: *const MoeSlot,
            workspace: *const MoeWorkspace,
            acc: *mut i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_fp8_linear_prepare() -> i32;
        pub fn ignis_fp8_linear(
            weight: *const c_void,
            rows: u32,
            cols: u32,
            x: *const c_void,
            tokens: u32,
            y: *mut c_void,
            y_f32: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_fp8_linear_swiglu(
            gate: *const c_void,
            up: *const c_void,
            rows: u32,
            cols: u32,
            x: *const c_void,
            tokens: u32,
            h: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_moe_shared_expert(
            gate: *const c_void,
            up: *const c_void,
            down: *const c_void,
            x: *const c_void,
            tokens: u32,
            h: *mut c_void,
            shared: *mut f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_moe_combine(
            acc: *mut i64,
            shared: *const f32,
            x: *const c_void,
            w_gate: *const c_void,
            tokens: u32,
            out: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_moe_prepare() -> i32;
        pub fn ignis_moe_decode_cluster_size() -> i32;
        pub fn ignis_moe_workspace_bytes(decode_tokens: u32, prefill_tokens: u32) -> u64;
        pub fn ignis_moe_workspace_init(
            workspace: *const MoeWorkspace,
            acc: *mut i64,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_moe_plan_bytes(decode_tokens: u32, prefill_tokens: u32, plan: *mut MoePlan) -> i32;
        pub fn ignis_moe_stream_sync(stream: *mut c_void) -> i32;
        pub fn ignis_moe_last_error() -> *const c_char;
    }
}

/// The leaf's message for the most recent failed MoE call on this thread.
pub fn last_error() -> String {
    unsafe { CStr::from_ptr(ffi::ignis_moe_last_error()) }.to_string_lossy().into_owned()
}

fn check(rc: i32) -> Result<(), String> {
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// Bytes of one expert-projection record, padding included (layout.md's class table).
pub fn record_bytes(projection: u32, k2: u32) -> Result<u64, String> {
    let mut bytes = 0u64;
    check(unsafe { ffi::ignis_moe_record_bytes(projection, k2, &mut bytes) })?;
    Ok(bytes)
}

/// Prepares the current device for every MoE op and the FP8 linear (kernel attributes, launch
/// geometry); once per device at load, outside any stream capture. The ops refuse to run on a
/// device that was not prepared.
pub fn prepare() -> Result<(), String> {
    check(unsafe { ffi::ignis_moe_prepare() })
}

/// The CTAs per expert cluster the current device runs `DECODE_CLUSTERS` with (16 or 8), or 0 if
/// the device was not prepared or runs no such cluster.
pub fn decode_cluster_size() -> i32 {
    unsafe { ffi::ignis_moe_decode_cluster_size() }
}

/// The routed-expert workspace for `decode_tokens` lanes and prefill chunks of `prefill_tokens`
/// (the `acc` buffer is the caller's own plan line).
pub fn workspace_bytes(decode_tokens: u32, prefill_tokens: u32) -> u64 {
    unsafe { ffi::ignis_moe_workspace_bytes(decode_tokens, prefill_tokens) }
}

/// Zeroes a fresh workspace and `acc` and prepares the device; once at load.
///
/// # Safety
/// `workspace.base` and `acc` are device buffers of the sizes the plan gives.
pub unsafe fn workspace_init(workspace: &MoeWorkspace, acc: *mut i64, stream: *mut c_void) -> Result<(), String> {
    check(unsafe { ffi::ignis_moe_workspace_init(workspace, acc, stream) })
}

/// The MoE block's plan lines for `decode_tokens` lanes and prefill chunks of `prefill_tokens`.
pub fn plan_bytes(decode_tokens: u32, prefill_tokens: u32) -> Result<MoePlan, String> {
    let mut plan = MoePlan::default();
    check(unsafe { ffi::ignis_moe_plan_bytes(decode_tokens, prefill_tokens, &mut plan) })?;
    Ok(plan)
}

/// Blocks until `stream` (null: the legacy default stream) is idle.
pub fn stream_sync(stream: *mut c_void) -> Result<(), String> {
    check(unsafe { ffi::ignis_moe_stream_sync(stream) })
}

/// Decodes a trellis tensor to its fp16 `[in][out]` inner weight.
///
/// # Safety
/// `trellis` and `w_f16` are device pointers of the sizes `in`, `out` and `k2` imply.
pub unsafe fn trellis_reconstruct(
    trellis: *const c_void,
    k2: u32,
    in_features: u32,
    out_features: u32,
    w_f16: *mut c_void,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_moe_trellis_reconstruct(trellis, k2, in_features, out_features, w_f16, stream) })
}

/// The router: ids `[tokens][10]`, weights `[tokens][10]` and fp32 logits `[tokens][512]`.
///
/// # Safety
/// Every pointer is a device pointer of the size `tokens` implies.
pub unsafe fn router(
    x: *const c_void,
    tokens: u32,
    w_router: *const c_void,
    ids: *mut i32,
    weights: *mut f32,
    logits: *mut f32,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_moe_router(x, tokens, w_router, ids, weights, logits, stream) })
}

/// The routed experts on the decode route (1..=8 tokens), into the fixed-point `acc`.
///
/// # Safety
/// Every pointer is a device pointer of the size `tokens` and the workspace plan imply.
pub unsafe fn experts_decode(
    x: *const c_void,
    tokens: u32,
    ids: *const i32,
    weights: *const f32,
    slots: *const MoeSlot,
    workspace: &MoeWorkspace,
    acc: *mut i64,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_moe_experts_decode(x, tokens, ids, weights, slots, workspace, acc, stream) })
}

/// The routed experts on the prefill route, into the fixed-point `acc`.
///
/// # Safety
/// Every pointer is a device pointer of the size `tokens` and the workspace plan imply.
pub unsafe fn experts_prefill(
    x: *const c_void,
    tokens: u32,
    ids: *const i32,
    weights: *const f32,
    slots: *const MoeSlot,
    workspace: &MoeWorkspace,
    acc: *mut i64,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_moe_experts_prefill(x, tokens, ids, weights, slots, workspace, acc, stream) })
}

/// The FP8 row-scale linear; `y_f32` selects fp32 output, else BF16.
///
/// # Safety
/// Every pointer is a device pointer of the size the shapes imply.
pub unsafe fn fp8_linear(
    weight: *const c_void,
    rows: u32,
    cols: u32,
    x: *const c_void,
    tokens: u32,
    y: *mut c_void,
    y_f32: bool,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_fp8_linear(weight, rows, cols, x, tokens, y, y_f32 as u32, stream) })
}

/// The shared expert into fp32 `shared` `[tokens][2560]`, `h` its BF16 `[tokens][640]` scratch.
///
/// # Safety
/// Every pointer is a device pointer of the size `tokens` implies.
pub unsafe fn shared_expert(
    gate: *const c_void,
    up: *const c_void,
    down: *const c_void,
    x: *const c_void,
    tokens: u32,
    h: *mut c_void,
    shared: *mut f32,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_moe_shared_expert(gate, up, down, x, tokens, h, shared, stream) })
}

/// The combine into BF16 `out`, zeroing the `acc` rows it read.
///
/// # Safety
/// Every pointer is a device pointer of the size `tokens` implies.
pub unsafe fn combine(
    acc: *mut i64,
    shared: *const f32,
    x: *const c_void,
    w_gate: *const c_void,
    tokens: u32,
    out: *mut c_void,
    stream: *mut c_void,
) -> Result<(), String> {
    check(unsafe { ffi::ignis_moe_combine(acc, shared, x, w_gate, tokens, out, stream) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_matches_the_c_struct() {
        // struct ignis_moe_workspace { void *base; uint32_t decode_tokens, prefill_tokens, decode_route; }
        assert_eq!(std::mem::size_of::<MoeWorkspace>(), 24);
        let ws = MoeWorkspace { base: std::ptr::null_mut(), decode_tokens: 3, prefill_tokens: 8192, decode_route: DECODE_CLUSTERS };
        let base = &ws as *const MoeWorkspace as usize;
        assert_eq!(&ws.decode_tokens as *const u32 as usize - base, 8);
        assert_eq!(&ws.prefill_tokens as *const u32 as usize - base, 12);
        assert_eq!(&ws.decode_route as *const u32 as usize - base, 16);
    }

    #[test]
    fn slot_entry_matches_the_c_struct() {
        // struct ignis_moe_slot { const void *record; uint32_t k2; uint32_t reserved; }
        assert_eq!(std::mem::size_of::<MoeSlot>(), 16);
        assert_eq!(std::mem::align_of::<MoeSlot>(), 8);
        let slot = MoeSlot { record: std::ptr::null(), k2: 5, reserved: 0 };
        let base = &slot as *const MoeSlot as usize;
        assert_eq!(&slot.k2 as *const u32 as usize - base, 8);
        assert_eq!(&slot.reserved as *const u32 as usize - base, 12);
    }

    #[test]
    fn record_bytes_are_layout_mds_class_table() {
        // docs/specs/flash-next/layout.md §3, record bytes with padding, k2 = 4, 5, 6, 8.
        let gate_up = [827_392u64, 1_032_192, 1_236_992, 1_646_592];
        let down = [417_792u64, 520_192, 622_592, 827_392];
        for (i, k2) in [4u32, 5, 6, 8].into_iter().enumerate() {
            assert_eq!(record_bytes(PROJ_GATE_UP, k2), Ok(gate_up[i]));
            assert_eq!(record_bytes(PROJ_DOWN, k2), Ok(down[i]));
        }
        let refused = record_bytes(PROJ_DOWN, 7).unwrap_err();
        assert!(refused.contains("k2"), "{refused}");
    }

    #[test]
    fn plan_lines_at_a_4096_token_chunk() {
        // By hand from the layout (kernel/src/moe_workspace.cuh) at 4096 tokens: decode counters
        // 2,048 + gate/up sums 6,553,600 + h 1,638,400; prefill chunk histograms 131,072, expert
        // offsets 2,304, item count 256, 1,152 items 9,216, sorted rows 163,840, h 104,857,600.
        let plan = plan_bytes(8, 4096).unwrap();
        assert_eq!(plan.workspace, 113_358_336);
        assert_eq!(plan.acc, 4096 * 2560 * 8);
        assert_eq!(plan.router, 163_840 + 163_840 + 8_388_608);
        assert_eq!(plan.shared, 5_242_880 + 41_943_040);
        assert_eq!(plan.total, 253_146_624);
        assert!(plan_bytes(0, 4096).is_err());
        assert!(plan_bytes(9, 4096).is_err());
        assert!(plan_bytes(8, 0).is_err());
    }

    #[test]
    fn workspace_grows_with_the_chunk() {
        let decode_only = workspace_bytes(1, 1);
        let chunk = workspace_bytes(1, 4096);
        assert!(decode_only > 0);
        // The prefill rows dominate: at least the SwiGLU outputs of 4096 tokens x 10 experts.
        assert!(chunk - decode_only >= 4096 * 10 * 640 * 4 - 640 * 10 * 4);
    }

    #[test]
    fn decode_regions_follow_the_lane_count() {
        // gate/up sums int64 [10 D][D][1280] and h f32 [10 D][D][640]: 8 lanes against 3.
        let eight = workspace_bytes(8, 4096);
        let three = workspace_bytes(3, 4096);
        assert_eq!(eight - three, (6_553_600 - 921_600) + (1_638_400 - 230_400));
    }
}
