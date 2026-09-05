# KERNEL-WIP — GDN (linear-attention) step: investigation findings

Status: investigation + parked minimal patch (worktree, uncommitted). No
changes in the main tree. 2026-09-05.

## 1. E2E status of the real-model forward

Verified on this box (RTX 5090 32 GB, clean GPU, serial runs):

- The full pipeline is wired: HTTP → real tokenizer/chat-template →
  scheduler → prefill/decode (eager + CUDA-graph) → greedy sample →
  detokenize.
- The real model **loads correctly**: `from_artifact` materializes the
  19 GB arena in VRAM with the real `qwen38_27b()` topology
  (`vram_resident()` passes).
- The **synthetic** forward passes (small geometries, incl. the self-
  consistency check).
- The **real-model forward is blocked**: every run of the real forward
  (prefill + decode through `from_artifact`) fails with
  `CUDA error: invalid argument`, and the GPU-gated e2e test
  misattributes it to "GPU busy / OOM" and self-skips (ADR 0006
  contract). So a real, "sane" 27B completion has never actually been
  produced on this machine — it has always been a self-skip.

## 2. Root cause: the GDN step launch is invalid at 27B geometry

`gdn_step_kernel` is launched **one thread per d_k column**:
`blockDim.x = state_cols`, with a per-dv-row block reduce over the d_k
columns (smem array of `state_cols` floats).

- Real `qwen38_27b()` geometry: `gdn_state_cols = 2048`
  (= 16 key heads × 128). **2048 > 1024** (the CUDA threads-per-block
  limit) → `cudaErrorInvalidValue` ("invalid argument").
- Synthetic geometry: `state_cols = 8` → legal. This is why every
  synthetic/CPU test passes and only the real forward fails.

Failure chain (observed on a clean GPU):

1. The decode-graph capture (built at `from_artifact`) replays the
   representative GDN step during capture → invalid launch → capture
   invalidated → the graph is dropped (eager fallback) and the leaf
   logs "operation failed due to a previous error during capture".
2. The eager prefill/decode then hits the same invalid GDN launch →
   `ComputeError::Kernel(-1)` → the test treats that as "GPU busy"
   (self-skip). The skip is misleading: it is a launch-geometry bug,
   not GPU availability.

The GDN step is exercised on **both** paths (eager prefill and the
graph-captured decode), so the graph hot path is equally blocked.

## 3. The deeper problem: the GDN state ABI is not the model's structure

The current ABI is a **flat** representation:

- state: bf16 `[batch][48 linear layers][6144][2048]`, where
  `6144 = 48 value heads × 128` (all value-head rows flattened into one
  matrix) and `2048 = 16 key heads × 128` (all key-head columns
  flattened into one wide d_k).
- inputs: a single flat feature block `x` (k, v, g, beta packed).

The reference (ninfer, `src/ops/linear_attention/gated_delta_net/`)
does not flatten heads:

- state: **FP32**, one **128×128 matrix per value head**
  (`[48][128][128]` per layer — the `mamba_ssm_dtype = float32`
  contract), slot-indexed pool.
- d_k is **128 per key head** (48 value heads → 16 key heads, 3:1 via
  fast-div head mapping). The dot product `y = S·k` is a **128-dim**
  sum per value head — not a 2048-dim sum over all key heads at once.
- separate q/k/v/g/beta tensors (not a flat `x`), Q/K L2 normalization
  and a 1/√128 readout scale.
- kernel tiling: 128 d_k columns fit **exactly in one warp**
  (32 lanes × 4), 4 dv rows per warp, 4 warps per 128-thread block
  (16 dv rows per block), register-resident state tile, **warp-local**
  reduce (shuffles — no shared-memory block reduce, no cross-warp
  sync), `__launch_bounds__(128, 2)`.

Two consequences:

1. **Fidelity**: the flat 2048-wide dot mixes 16 key heads' d_k into a
   single reduction, which is structurally different from the model's
   per-(value head, key head) 128-dim recurrence. Even a "sane"
   output from the flat ABI is not comparable to the reference's
   (the bf16 state + shared scalar g/beta is part of the same gap).
2. **Performance**: the flat wide-d_k block-reduce is the *opposite* of
   a performance-oriented structure. The reference layout is what can
   be optimized toward the perf gate: register-tiled per-head state,
   warp-local reduces, vectorized loads, and (downstream) the
   reference's chunked-prefill / replay-fold kernels. A flat 2048-wide
   bf16 state cannot align with any of that, and a 1024-thread
   smem-reduced block is a perf dead end.

## 4. What was tried, and where it stopped

- A **minimal launch patch** (tile the d_k reduce to `T = min(state_cols,
  1024)`: per-thread partials + block reduce; smem shrunk from
  `state_cols` to `T` floats) was implemented in worktree
  `gdn-step-dk-tiling` (kernel + the 3 launch sites in
  `prefill_gdn_surface.cu`, `decode_graph_surface.cu`,
  `graph_capture.cu`, plus wide-d_k regression tests
  `state_cols ∈ {1025, 2048}` vs an f32 CPU reference). The kernel
  `.lib` was rebuilt in the worktree.
  - With the patch, the real forward **does run** (the capture no
    longer invalidates; no more self-skip).
  - Verification was not driven to a green result: a full debug-build
    e2e run (2× `from_artifact` 19 GB arenas + 2× full 27B forward +
    the host-side W8 endpoint dequant, ~5.1 G elements) saturated this
    machine (32 GB VRAM + ~28 GB WDDM shared memory) and was killed
    on request. The heavy part is the debug build + WDDM spillover,
    not a hang in the kernel.
  - **Decision (2026-09-05): do not land the minimal patch to close v1.**
    It unblocks a forward whose GDN state is structurally not the
    model's, and it locks in the flat-ABI dead end. The worktree is
    parked (uncommitted).
- The host-side W8 endpoint dequant (`dequant_w8_endpoints`, ~5.1 G
  element ops at load, one-shot) was investigated as the "slow e2e"
  suspect: it is a deliberate A1 normalization choice (only the two W8
  text-scope endpoints; NVFP4 planes stay device-resident). In release
  builds it is a few seconds; in debug builds it is minutes. Not the
  forward blocker, just a load-path cost.

## 5. First sketch proposal (to be reviewed)

Direction: **replace** the flat GDN state ABI with the reference's
per-head structure (not a patch of it). The d_k=128 per-head
structure makes the launch-geometry problem disappear *by
construction* (128 d_k fits in a warp; a 128-thread block needs no
smem reduce at all) and is the structure the perf work can target.

1. **State layout**: FP32 per-value-head state `[48][128][128]` per
   linear layer, slot-indexed pool (the `mamba_ssm_dtype=float32`
   contract); head mapping 48→16 via 3:1 (fast-div). Storage decision
   needed: FP32 state (reference) vs bf16 state + FP32 accum (a
   perf/storage trade the reference does not make).
2. **Step kernel**: warp-tiled per the reference — 128 d_k per warp
   (4 per lane), 16 dv rows per 128-thread block, register-resident
   state tile, warp-local reduce (shuffles), `__launch_bounds__(128,
   2)`. Separate q/k/v/g/beta; Q/K L2 norm + 1/√128 readout.
3. **Hot path**: per-token recurrent step (decode) + chunked prefill
   (the reference has a separate chunked kernel; the current
   single-token loop is the prefill blocker on long prompts).
   Re-wire the decode-graph staging (the graph's representative GDN
   node changes shape) and the `compute.rs` routing.
4. **Perf headroom**: this is the structure that can carry
   vectorized loads / swizzled smem, cp.async + PDL, and eventually
   tensor-core GEMM for the `S·k` / `S·q` reads — i.e., the path to
   the 99% gate. The flat-ABI patch has none of that.

Open questions for the review:

- FP32 vs bf16 state storage (precision contract vs 2× state VRAM:
  48 heads × 128 × 128 × 4 B × 48 layers ≈ 150 MB FP32 — small).
- How the decode-graph (B2) fixed-address staging maps onto per-head
  state (the graph's GDN node becomes a per-head warp-tiling launch).
- Whether the Q/K L2 norm belongs in the step kernel (reference does
  it in-kernel) or stays a separate op.
- Prefill: chunked vs single-token loop for long prompts (the
  reference splits this into a separate chunked kernel; a v1
  single-token prefill is slow but simple).

This is a medium/large refactor (kernel + C-ABI surface + `compute.rs`
routing + state layout + graph staging), not a patch. The parked
worktree (the minimal patch) can serve as a reference for the
wide-d_k test vectors (the f32 CPU reference already exists and can
be reused for the per-head geometry).