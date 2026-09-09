# ADR 0019 — decode CUDA graph replay via device-resident slot indirection

## Status

Accepted (2026-09-09) — P3-05, GitHub #102. Supersedes ADR 0008's staging-buffer
model (already superseded once by ADR 0009) with a design that actually fits
the batched decode round P3-03 (GitHub #99) built.

## Context

`ignis_program_decode` runs one decode round as a host-side loop: for each of
up to `IGNIS_DECODE_MAX_BATCH` (8) lanes, a full single-sequence forward pass
(embedding through the output head) is enqueued, then one batched
`ninfer::ops::sample` call draws every lane's successor together. No CUDA
graph exists anywhere in the leaf; every kernel is launched eagerly.

A CUDA graph replays a fixed DAG of kernel nodes bound to the exact device
addresses captured. The decode round's composition — which physical sequence
pool slots hold this round's decode-ready lanes — changes every round: slots
are assigned at `ignis_seq_alloc` and stay fixed for a sequence's lifetime,
but which subset of the pool's (up to 8) slots is decode-ready in any given
round is arbitrary and non-contiguous in general (one slot may be mid-prefill,
others free, others decode-ready in no particular order). A graph captured
against one specific set of slots' addresses would silently replay against
the wrong sequence's KV pages and GDN state the moment a different slot
combination fills the same width — exactly the class of silent,
batch-width-dependent bug #96 cost this project a gate run.

Two facts make a correct design possible without touching vendored kernels
(ADR 0010 pins them) or fusing the layer math across lanes into new batched
GEMM call sites (high-risk rewrite, not this ticket's scope):

1. **The vendored ops already carry a device-resident row-selector for
   exactly this case.** `ninfer::ops::gqa_attention` (A1) takes
   `kv_table_rows`, a device I32 array selecting which row of a *shared*
   block-table matrix each batch row reads —
   `PagedKVPool::block_tables()` returns that whole-pool matrix (all slots,
   fixed address, exists from pool construction). `gated_delta_net_snapshot`
   and `causal_conv1d_silu_snapshot` take `initial_state_slots` /
   `snapshot_base_slots`, device I32 arrays selecting which pool slot's GDN
   state each batch row reads and writes (in place when
   `snapshot_base_slots == initial_state_slots`). All three read the
   selector's *value* from device memory at kernel execution time — the
   host never bakes a specific slot into the launch.
2. **`ninfer::ops::offset_i32_positions`** computes `destination[i] =
   source[i] + delta[0]` where `delta` is a device I32 scalar, unlike
   `fill_i32_positions`'s host-scalar `start`. RoPE's per-round position can
   therefore be read from a stable device address instead of baked at
   capture time.

Both facts point at the same mechanism P3-03 already scaffolded: stable,
model-owned staging buffers refreshed by an H2D copy before each call,
exactly the ADR 0008 "persistent staging buffer" pattern, just applied to
the layer bodies' slot addressing and RoPE position instead of only to
sampling.

## Decision

Keep the round's existing per-lane *sequential* structure (one forward pass
per lane, in program order, then one batched sample) — do not fuse the GEMM
projections across lanes. Capture that sequential structure as one CUDA
graph per exact batch width 1..8 at startup (right after the sequence pool
is created, before any sequence is ever allocated), using:

- A dedicated `decode_graph_scratch` `DeviceArena`, separate from the
  prefill/eager-decode `scratch` arena, sized once for one lane's per-layer
  peak (T=1) and reused (via its own `Scope`) sequentially across lanes
  inside one graph, exactly like `scratch` is reused across lanes in the
  eager loop today. Being a distinct allocation, a prefill chunk running
  between two replays can never alias the addresses the graph rereads.
- A `decode_graph_token_ids` and `decode_graph_slots` `DeviceBuffer`
  (I32 × 8 each), refreshed by one H2D copy per round before replay:
  `decode_graph_slots[lane]` is that lane's *physical pool slot* for this
  round, read at kernel-execution time by GQA's `kv_table_rows` and GDN's
  `initial_state_slots`/`snapshot_base_slots` (same value serves both — a
  physical slot is a physical slot). `block_tables` for the GQA call and the
  conv/recurrent state tensors for the GDN calls are built once, spanning
  the *whole pool* (all 8 slots), not one sequence's private view — fixed
  addresses that exist independent of round composition.
- RoPE position sourced via `offset_i32_positions(zero, delta =
  sampling_decode_positions[lane], positions)` instead of
  `fill_i32_positions(positions, host_scalar)` — reusing P3-03's existing
  `sampling_decode_positions` staging buffer, just refreshed before the
  layer loop instead of only before the sample call.
- A fixed, conservative `GqaExecutionEnvelope` (`max_visible_keys =
  model->max_context_tokens`) for every graph capture. The envelope is
  documented as a host launch-resource promise for workspace sizing and
  kernel-route selection, not the causal mask itself (the mask comes from
  the `positions` device tensor, read per row inside the kernel) — so a
  fixed worst-case envelope is always a safe over-approximation, never a
  correctness hazard, and lets one capture serve every round regardless of
  the lane's actual current position.
- Graph replay never pads: a round of width W only ever launches the
  width-W graph (or, if that width failed to capture, the eager per-lane
  loop). A width whose graph failed to build at startup falls back to eager
  for every round at that width — model load still succeeds; only that
  width's decode throughput degrades.

`ignis_program_decode`'s existing signature is unchanged. Internally it
dispatches to the width's graph when ready, else the untouched eager path.
No ABI break, no new Rust call sites for the hot path.

## Consequences

- Row `i` of a captured width-W graph always means "whichever sequence
  `decode_graph_slots[i]` names this round," not a fixed physical slot —
  this is what makes one graph per width correct for *any* combination of W
  decode-ready lanes, not just a canonical low prefix.
- The eager per-lane loop is untouched (still reads `seq->slot` / `seq->kv`
  directly); the graph path is new, parallel code that must independently
  match it, which is exactly what the width 1..8 replay-vs-eager test
  verifies.
- No vendored kernel is modified. The `_snapshot` op forms already existed
  in the vendored surface for this purpose (or one very like it); this
  ticket is the first caller.
- A future true batched-decode redesign (fusing the GEMM projections
  themselves across lanes, using the same `_snapshot`/`kv_table_rows`
  machinery at B=W instead of calling it W times at B=1) remains available
  later without another ABI or staging-buffer change — the row-selector
  buffers this ADR introduces are already shaped `[IGNIS_DECODE_MAX_BATCH]`
  and pool-wide.
