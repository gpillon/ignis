# 09 — CUDA-graph decode replay (the decode hot path)

GitHub: #32 (the #25 follow-up — B2, "CUDA-graph decode replay")

The compute-adapter (kernel-abi 04) captures a *representative* decode graph at
construction (the `new` path) but **never launches it** — every `decode_step`
runs the eager sequence (the kernel-abi 03 graph primitives are captured but
unused). The production `from_artifact` path is fully eager (`new_eager`). This
ticket makes the CUDA graph the **decode hot path**: capture the decode
sequence into a graph that reads/writes **persistent device staging buffers**
(ADR 0008) and replay it via `ignis_graph_launch` per decode step.

**Why a staging-buffer model (ADR 0008):** a CUDA graph is a DAG of kernel
nodes bound to *fixed* device pointers/sizes as captured between
`ignis_graph_begin_capture` / `ignis_graph_end_capture`; `ignis_graph_launch`
replays that exact DAG, and the leaf has **no** per-node update primitive. So
"the graph reads the latest activation each decode step" is achieved by
**stable staging buffers**, not by re-capture or node updates: the graph
reads/writes pre-allocated device buffers at fixed addresses; each step H2D's
the new activation into the input buffer, launches the graph (every node
operates on the fixed buffers), and D2H's the logits (ADR 0008).

**Seam:** the `Compute::decode_step` hot path in `crates/core/src/compute.rs`
(`CudaCompute::decode`), + the construction-time capture (`CudaCompute::new` and
the `from_artifact` path). Reuses the kernel-abi 03 graph primitives
(`ignis_graph_begin_capture` / `ignis_graph_end_capture` / `ignis_graph_launch`
/ `ignis_graph_destroy`).

**Scope:**
- **The staging-buffer decode graph (ADR 0008):** at construction, allocate the
  fixed device buffers for one representative decode batch (the
  `GraphGeometry`): the input activation, the per-layer KV/GDN writeback, and
  the logits buffer. Capture the decode sequence (embed → the per-layer stack →
  final rmsnorm → lm-head GEMV → logits) so every kernel node reads/writes the
  *fixed* buffers. The `from_artifact` (production) path builds the same graph
  (replacing the `new_eager` fully-eager construction).
- **The replay:** `decode_step` (the single-token, representative-batch case)
  does: H2D the current activation into the input buffer → `ignis_graph_launch`
  → D2H the logits → `ignis_greedy_sample`. No per-step capture, no node
  update (ADR 0008).
- **The eager fallback (ADR 0003):** a decode step whose batch does not match
  the captured `GraphGeometry` runs the eager sequence; a busy/absent GPU
  (capture self-skip, ADR 0006) leaves the graph `None` (the eager fallback).
- **The `uses_graph` flag** reports whether the graph hot path is active.

**Non-goals (v1, per ADR 0008):** per-batch-size graph variants; per-step
graph-node updates; lazy re-capture; a *prefill* graph (the prefill path is
B1 / batched prefill, a separate ticket — not the decode graph).

**Acceptance:**
- **The decode graph is captured at construction** (the `new` + the
  `from_artifact` path): the startup capture succeeds on a free GPU (self-skip
  on a busy GPU, ADR 0006) and the `GraphGeometry` is set.
- **Graph replay ≡ the eager decode path:** a GPU-gated test captures the
  decode graph, launches it, and compares its logits against the eager `decode`
  path — **bit-exact** (same model, same weights, greedy + fixed seed — ADR
  0007 self-consistency). This is the kernel-abi 03 "replay ≡ eager" invariant
  applied to the *actual* staging-buffer decode graph (the current capture is
  empty, so today's check is a warm-up, not a real decode graph).
- **The hot path uses the graph:** with the graph active, `decode_step`
  launches via `ignis_graph_launch` (not the eager sequence) — verifiable via
  the `uses_graph()` flag + a test that the eager fallback engages only on a
  batch-size mismatch / a busy GPU.
- **`cargo test --workspace` green** (AGENTS.md). The graph tests are
  GPU-gated and self-skip on a busy GPU (the `kernel_abi0N_gpu` convention,
  ADR 0006).

**Blocked by:** A3 (the full-correct 27B forward pass — the *real-model*
decode graph is only meaningful once A3 lands; the *mechanism* — staging
buffers + graph launch — is testable on the synthetic model).

**References:** ADR 0008 (the staging-buffer decode-graph model — the design
decision for this ticket), ADR 0003 (the eager fallback), ADR 0006 (exclusive
GPU), ADR 0007 (self-consistency, not parity). kernel-abi 03 (the graph
primitives), kernel-abi 04 (the compute-adapter), kernel-abi 01 (the GQA + GDN
decode kernels the graph composes).