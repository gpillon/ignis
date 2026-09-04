# ADR 0008 — CUDA-graph decode replay: persistent device staging buffers

## Status

Accepted (2026-09-04, the #25 performance-material planning session).

## Context

The kernel-leaf exposes eager CUDA-graph primitives (kernel-abi 03, ADR
0001): `ignis_graph_begin_capture` / `ignis_graph_end_capture` /
`ignis_graph_launch` / `ignis_graph_destroy`, plus a startup check
(`ignis_graph_startup_check`). A CUDA graph is a DAG of kernel nodes bound
to **fixed** device pointers, sizes, and parameters as captured between
`begin_capture` and `end_capture`; `ignis_graph_launch` then replays that
exact DAG. The leaf has **no** per-node parameter-update primitive (no
`cudaGraphExecKernelNodeSetParams` / `cudaGraphExecUpdate` wrapper).

The compute-adapter (kernel-abi 04) currently captures a *representative*
decode graph at construction (`new_eager` path only — the `from_artifact`
path is fully eager) but **never launches it**: every `decode_step` runs the
eager sequence. The #25 follow-up is to make the graph actually the decode
hot path.

Two models achieve "the graph reads the latest activation each decode
step":

- **(a) per-step node updates** — after capture, update each node's input
  pointers per step. Requires a *new* C-ABI primitive in the leaf (the
  `cudaGraphExecUpdate` / `KernelNodeSetParams` family) plus per-step
  parameter writes on the hot path.
- **(b) persistent staging buffers** — capture the graph reading/writing a
  set of pre-allocated device buffers at **stable addresses** (allocated
  once at construction, lifetime of the backend). Per decode step:
  H2D the new activation into the fixed input buffer → `ignis_graph_launch`
  (every node in the graph reads/writes the same fixed buffers) → D2H the
  logits. The graph is invariant after capture; no node updates, no
  re-capture.

## Decision

**(b).** Persistent, pre-allocated device staging buffers with stable
addresses are the CUDA-graph model for decode replay. Per-step mutable state
lives in those buffers; the hot path never updates graph nodes.

## Rationale

- A single `ignis_graph_launch` is the cheapest possible replay (one launch
  over the whole captured decode DAG). Node updates (a) add per-step CUDA
  API calls to the hot path and need a new, non-trivial leaf primitive.
- The staging buffers are small — one decode batch's worth of per-layer
  activations + the logits buffer, not the weights. The H2D/D2H around each
  launch is negligible against the GEMM/attention work (and the
  weights stay in the artifact's VRAM arena, ADR 0002, untouched).
- The buffers are allocated once at backend construction, so their device
  addresses are stable for the lifetime of the backend — exactly the
  invariant CUDA graphs require. This is the standard "eager graph +
  fixed-address staging" pattern (the reference stack runs eager graphs the
  same way, design §4).
- The kernel-leaf C-ABI is **unchanged**: replay reuses the existing
  `ignis_graph_launch` (kernel-abi 03). The staging-buffer design is a
  compute-adapter (Rust) concern, not a leaf concern.

## Consequences

- `decode_step` becomes: H2D the current activation into the input buffer →
  `ignis_graph_launch` → D2H the logits → `ignis_greedy_sample`. The graph
  itself is invariant after construction-time capture.
- The graph is captured for the **representative geometry** (the
  `GraphGeometry` dims, ADR 0003 eager fallback): a decode step of a
  different batch size falls back to the eager sequence — no per-size
  recapture in v1.
- **Non-goals (v1):** per-batch-size graph variants, per-step node updates,
  and lazy re-capture. If a later perf gate shows the single-geometry graph
  underperforms, that is a later decision — re-gated per ADR 0005 / 0007.
- The B2 spec (`kernel-abi/specs/09-cuda-graph-replay.md`) implements this
  model and references this ADR.