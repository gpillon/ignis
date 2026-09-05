> **SUPERSEDED (2026-09-05)** by `.scratch/runtime/specs/01-device-resident-forward.md` (GitHub #36): the per-op host-pointer ABI and the host-resident forward this spec describes are being deleted. Kept for history only; do not implement against it.

# 04 — compute-adapter: wire the kernel leaf as the production `Compute` backend

GitHub: #23

The `Compute` trait (`crates/core/src/scheduler.rs`) is the *only*
GPU-coupled step in the engine (ADR 0006: the scheduler's logic —
admission, lanes, batched prefill, eviction — never touches the GPU and
stays CPU-testable). Today it is implemented only by `MockCompute`
(deterministic, CPU-only) and test doubles, and the server entrypoint
(`crates/server/src/main.rs`) drives the scheduler with that mock — so
`ignis-server` cannot serve a real model, and a mock-driven server
cannot pass the 99% performance gate (ADR 0007). This ticket closes
that gap: a production `Compute` implementation backed by the
kernel-leaf C ABI (ADR 0001), reusing the surface already built:

- the decode GEMM + GQA + GDN step (kernel-abi 01, ticket #5),
- the norms / embedding / greedy-sampling surface (kernel-abi 02,
  ticket #6),
- the eager CUDA-graph primitives + startup check (kernel-abi 03,
  ticket #10),
- the artifact `CudaDevice` (feature `cuda`, artifact-01, ticket #4)
  for VRAM weight materialization (ADR 0002: `.ninfer` artifacts load
  directly, no converter),
- the multi-token NVFP4 GEMM for the prefill (FFN) projections
  (kernel-abi 05 — a *prerequisite*, see `05-multitoken-gemm.md`).

**Seam:** `Compute` — `prefill_step(jobs: &[PrefillJob])` and
`decode_step(jobs: &[DecodeJob]) -> Vec<Option<TokenId>>`
(`crates/core/src/scheduler.rs`). The implementation lives in
`crates/core` (a new module) and is injected into
`ConcreteScheduler::with_config` through the existing constructor —
the seam the server-01 spec reserves: "the kernel-leaf adapter replaces
it via the same scheduler-constructor injection". The server entrypoint
selects the backend: `IGNIS_ARTIFACT` set → production adapter; no
artifact → `MockCompute` (dev mode, ADR 0006, unchanged).

**Scope:**
- A production `Compute` implementation (`CudaCompute`) in
  `crates/core`:
  - device lifecycle: CUDA context/stream via the artifact `CudaDevice`
    (feature `cuda`); the kernel-abi 03 startup check
    (`ignis_graph_startup_check`) at construction.
  - `prefill_step`: batched prefill (core-04 groups several jobs into
    one GPU batch) → `ignis_gqa_attention_prefill` (attention) +
    `ignis_nvfp4_gemm_prefill` (the multi-token NVFP4 GEMM from
    kernel-abi 05) for the FFN projections; a request that reuses a
    cached sibling prefix (core-07) carries only its *tail* — the
    leading shared prefix's blocks are bound read-only by the kernel
    leaf (`concrete.rs`), so only the tail warms the KV.
  - `decode_step`: per-lane decode (`ignis_gqa_attention_decode` /
    `ignis_nvfp4_gemm_decode` — the single-token GEMV, correct for
    decode) + `ignis_gdn_step` for the linear-attention layers +
    `ignis_rmsnorm` / `ignis_embedding`; `ignis_greedy_sample`
    (greedy + fixed seed, ADR 0007). Per-job `None` = that request
    finished this step (max_tokens / EOS — the soft-stop semantics of
    `ComputeError::Stopped`).
  - FFI error mapping: a kernel return code → `ComputeError::Kernel(rc)`.
  - The CUDA-graph fast path (kernel-abi 03): capture the
    representative sequence at startup, launch the graph per decode
    step.
- Server entrypoint wiring (`crates/server/src/main.rs`):
  `IGNIS_ARTIFACT` → verified loader (server-03) + `CudaDevice`
  materialization → `CudaCompute` → `ConcreteScheduler::with_config`;
  empty artifact → `MockCompute` (the existing dev mode, unchanged).
- **Prefill GEMM (depends on kernel-abi 05)**: the kernel-abi 01–03
  surface has *no* multi-token GEMM (only the single-token GEMV
  `ignis_nvfp4_gemm_decode`). The prefill FFN projections therefore use
  `ignis_nvfp4_gemm_prefill`, added by **kernel-abi 05** (a
  prerequisite — see `05-multitoken-gemm.md`). If the two tickets are
  implemented together, 05 lands first.

## Acceptance

- **The production `Compute` passes against a live device**: a
  GPU-gated test (self-skip on a busy GPU, per the `kernel_abi02_gpu`
  convention) drives a `prefill_step` + `decode_step` sequence through
  `CudaCompute`: the emitted token ids are in vocabulary range, and a
  greedy + fixed-seed run is reproducible (same input → same tokens —
  self-consistency, ADR 0007). The prefill path uses
  `ignis_nvfp4_gemm_prefill` (kernel-abi 05) for the FFN projections.
- **The server runs a real model**: `ignis-server` started with
  `IGNIS_ARTIFACT` set (the graph startup check passes) serves
  `/v1/chat/completions` and returns a *real* model completion —
  output is **self-checked** (a sane completion for the prompt; ADR
  0007: *not* token-agreement with the reference) — GPU-gated.
- **Dev mode unchanged**: without `IGNIS_ARTIFACT` the server drives
  `MockCompute`; the existing CPU tests pass unmodified (ADR 0006).
- **Error mapping**: a kernel failure return code surfaces as
  `ComputeError::Kernel(rc)`; a request that hits max_tokens / EOS
  completes with a per-job `None` in `decode_step` (a soft stop, not a
  fault).
- **`cargo test --workspace` green** (AGENTS.md: every code change
  ships with a test, and the task is not complete until it passes
  workspace-wide).

Depends on kernel-abi 05 (multi-token NVFP4 GEMM). This is the last
code prerequisite of the v1 gate run (bench-03, ADR 0007): the 99%
performance gate can only be measured once the server serves a real
model.