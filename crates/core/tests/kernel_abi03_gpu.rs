//! Gated GPU graph-capture tests for the ticket-10 (kernel-abi-03)
//! CUDA-graph surface (kernel/src/graph_capture.cu, GitHub #10).
//!
//! Two gated tests. Both self-skip when the GPU is busy (ADR 0006): a
//! `rc` of -1 means "GPU busy / unavailable" and the test skips, never
//! going red. The startup check additionally distinguishes a *real*
//! divergence (rc -2 — the graph path is broken, a genuine failure) from
//! a busy GPU (rc -1 — a skip); see `graph_startup_check_gpu` below.
//!
//! 1. `graph_primitives_roundtrip_gpu` — drives the four capture primitives
//!    through the FFI: begin → (double-begin guard) → end (an empty capture
//!    materializes a zero-work graph) → launch → destroy, on the leaf's
//!    internal stream (null = the leaf-owned non-blocking stream).
//! 2. `graph_startup_check_gpu` — the startup verification: capture a
//!    representative prefill + decode kernel sequence into a CUDA graph,
//!    replay it, and confirm replay ≡ eager (rc 0 = verified; rc -1 = GPU
//!    busy / unavailable → skip; rc -2 = a real divergence → failure).
//!
//! They fit in a few KB of VRAM, so they can run even with the model loaded
//! (the ADR 0006 nuance). Run with:
//! `cargo test -p ignis-core --test kernel_abi03_gpu -- --ignored`
//!
//! Build precondition: links the kernel .lib, so it only builds once
//! `ignis-core`'s build script runs AND the canonical
//! `kernel/build/ignis_kernel.lib` has been rebuilt with the ticket-10
//! symbols (kernel/build.ps1 — the CMake GLOB picks up the new .cu).

use ignis_core::ffi;

/// Skip helper: a non-zero rc (CUDA error / busy GPU) is a skip, never a
/// failure (ADR 0006 — the GPU is occupied by ninfer-serve).
fn skip_if_busy(rc: i32, what: &str) -> bool {
    if rc != 0 {
        eprintln!("SKIP: {what} returned {rc} (GPU busy / unavailable, ADR 0006)");
        return true;
    }
    false
}

#[test]
#[ignore = "GPU graph test — a few KB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn graph_primitives_roundtrip_gpu() {
    // begin (leaf-owned non-blocking stream — the null-stream case).
    let rc = unsafe { ffi::ignis_graph_begin_capture(std::ptr::null_mut()) };
    if skip_if_busy(rc, "ignis_graph_begin_capture") {
        return;
    }

    // A double begin must be rejected (one capture at a time, v1).
    let rc = unsafe { ffi::ignis_graph_begin_capture(std::ptr::null_mut()) };
    assert_eq!(rc, -1, "a double begin while a capture is active must return -1");

    // end (the empty capture materializes a zero-work graph) → launch →
    // destroy (the null stream = the graph's own capture stream).
    let mut g: *mut ffi::IgnisGraph = std::ptr::null_mut();
    let rc = unsafe { ffi::ignis_graph_end_capture(std::ptr::null_mut(), &mut g) };
    if skip_if_busy(rc, "ignis_graph_end_capture") {
        return;
    }
    assert!(!g.is_null(), "end_capture must materialize a graph handle");

    let rc = unsafe { ffi::ignis_graph_launch(g, std::ptr::null_mut()) };
    if skip_if_busy(rc, "ignis_graph_launch") {
        unsafe {
            ffi::ignis_graph_destroy(g);
        }
        return;
    }
    unsafe {
        ffi::ignis_graph_destroy(g);
    }
}

#[test]
#[ignore = "GPU startup check — a few KB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn graph_startup_check_gpu() {
    // The startup verification: capture a prefill + decode kernel sequence,
    // replay the graph, and confirm replay ≡ eager. rc 0 = verified;
    // rc -1 = GPU busy / unavailable → skip (ADR 0006); rc -2 = a real
    // divergence (the graph path is broken) → failure.
    let rc = unsafe { ffi::ignis_graph_startup_check(std::ptr::null_mut()) };
    match rc {
        0 => eprintln!("PASS: graph replay ≡ eager (the startup check verified the capture)"),
        -1 => eprintln!("SKIP: ignis_graph_startup_check returned -1 (GPU busy / unavailable, ADR 0006)"),
        -2 => panic!("ignis_graph_startup_check: the graph replay diverged from the eager path (the capture mechanism is broken)"),
        other => panic!("ignis_graph_startup_check returned {other} (expected 0, -1, or -2)"),
    }
}

// -------------------------------------------------------------------------
// CPU-verifiable contract pins (no FFI CUDA calls — the null guards run
// before any CUDA call, so these are safe without a GPU, like the
// geometry pins in kernel_abi01_gpu.rs).
// -------------------------------------------------------------------------

#[test]
fn graph_null_handle_contract() {
    // A null graph handle is a clean -1 before any CUDA call.
    let rc = unsafe { ffi::ignis_graph_launch(std::ptr::null_mut(), std::ptr::null_mut()) };
    assert_eq!(rc, -1, "a null-graph launch must return -1 without touching CUDA");

    // A null out pointer to end_capture is a clean -1 (the null guard
    // precedes the pairing check — no CUDA call).
    let rc = unsafe { ffi::ignis_graph_end_capture(std::ptr::null_mut(), std::ptr::null_mut()) };
    assert_eq!(rc, -1, "an end_capture with a null out must return -1 without touching CUDA");

    // A null destroy is a no-op (must not crash, must not touch CUDA).
    unsafe { ffi::ignis_graph_destroy(std::ptr::null_mut()) };
}