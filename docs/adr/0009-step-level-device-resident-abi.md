# ADR 0009 — Step-level, device-resident C ABI (the forward pass lives in the leaf)

## Status

Accepted (2026-09-05, project review). **Revises ADR 0001** — the ABI
granularity, not the two-language split. **Supersedes ADR 0008.**

Sources: `.scratch/REVIEW-2026-09-05.md` §2, §3, §5.3, §6 (the review that
found the defect); `.scratch/runtime/specs/01-device-resident-forward.md`
(GitHub #36, the spec that implements this ADR).

## Context

ADR 0001 put the boundary at "Rust owns everything above compute": the kernel
leaf exposes one flat C entry point *per operator* with explicit host pointers
and sizes, and Rust drives the forward pass op by op.

Built out, that boundary produced an engine that cannot work:

- **The activations never reach VRAM.** Per request the hidden state, the
  paged KV cache and the GDN state live in host `Vec<u16>`; every op H2Ds its
  inputs and D2Hs its outputs synchronously, with `stream = null`. The GDN
  state alone is 2.4 GB of PCIe traffic per decode token; residual, SiLU·mul
  and the state readout are host loops. The floor of this design is seconds
  per token against the reference's 13.1 ms — two to three orders of
  magnitude, before any kernel quality question (review §3.1).
- **The per-op boundary cannot carry the program's invariants.** Layer kind,
  fused-plane row order, per-head geometry, RoPE position, gating parameters
  and state shape all live on the Rust side of an ABI that only sees flat
  buffers, so each of them was gotten wrong independently: NVFP4 scales read
  as row-major instead of `blockscale-k16-m128x4-v1`, the `weight_divisor`
  never applied, GDN gating collapsed to one scalar for all 48 value heads,
  GDN state one flat bf16 matrix instead of 48 fp32 128×128 heads, RoPE
  advanced per layer instead of per token, the gated RMSNorm and the GQA
  output gate missing (review §2).
- **CUDA graphs have nothing real to capture.** ADR 0008's staging-buffer
  model exists because the graph could not see the sequence's device state;
  what got captured was a five-kernel toy over zeroed state with no layers
  and no projections, and batch-1 decode was routed through it (review §0.4).
  The real object to capture is one batched decode round over per-slot state
  views — which only exists if the state is device-resident and owned by the
  leaf.
- Rust orchestrating the reference's ~630 launches per decode round through
  FFI is possible, but buys nothing and costs the graph-capture ergonomics.

Options considered:

- **(a) Keep the per-op ABI, make it device-resident** — pass device pointers
  instead of host pointers, keep the forward in Rust. Removes the copies but
  leaves the program's invariants split across the boundary and leaves graph
  capture without a natural unit.
- **(b) Move the boundary to the step** — the leaf owns a *program* (arena,
  streams, per-layer op sequence, sequence state) and Rust calls it once per
  prefill span / decode round.
- **(c) Move the boundary to the request** — the leaf owns scheduling too.
  Discards the Rust core (scheduler, admission, KV accounting, host tier,
  prefix index), which is the part of ignis that works.

## Decision

**(b).** The C ABI is **step-level, device-resident, and opaque-handled.**

- **Rust owns everything above the step**: HTTP and the OpenAI surface, the
  tokenizer and chat template, the scheduler and admission state machine, KV
  page accounting, the prefix index, the host tier policy, telemetry, the
  bench. **C++ owns the step.**
- The leaf gains a **program** layer: a device arena and tensor views, CUDA
  streams, the per-layer op sequence for the Qwen 3.8-27B text model, a
  sequence-state store (paged KV pages, a slot-indexed fp32 GDN state pool,
  conv taps, position), and the sampling epilogue.
- **No host activation pointer crosses the boundary.** Activations, KV and
  GDN state live in VRAM for the lifetime of a sequence. Streams,
  synchronization and op dispatch are internal to the leaf.
- The surface is flat C with opaque handles and integer return codes:
  model load from the artifact's device arena plus a topology descriptor;
  sequence allocate / release with a context reservation; sequence snapshot
  to a caller-provided pinned host region and restore from it; prefill of a
  token span from a starting position; **one decode round over a batch of
  sequence handles**; a sampling-parameters struct; runtime statistics (VRAM,
  page geometry, per-step time, kernel and graph counters).
- The **batch parameter and the span+position parameter are present from day
  one** even though this milestone runs batch 1 and per-token prefill, so
  batched decode rounds (G3) and chunked / tail prefill (G2, G4) are leaf
  changes, not ABI changes.
- The program layer is **ours**, written on top of vendored ops (ADR 0010);
  the reference's own program is not vendored (it is entangled with MTP,
  vision, graphs and its engine).

## Consequences

- ADR 0001's decision (a) stands — Rust core, C++/CUDA leaf, flat C ABI, two
  toolchains, provenance in `NOTICE`. Only its "Rust owns everything above
  compute" is revised to "above the **step**", and its "explicit pointers +
  sizes, no shared state" becomes "opaque handles over leaf-owned device
  state; no host activation pointers".
- ADR 0008 is superseded: there is no staging-buffer graph. A decode graph is
  captured per batch width over per-slot state views (G3 work).
- The per-op host-pointer surfaces, the host-resident forward in
  `crates/core/src/compute.rs`, the toy decode graph and the graph-capture
  surfaces are deleted rather than migrated (spec 01, GitHub #39).
- The `Compute` trait is implemented by a thin safe Rust wrapper
  (`crates/runtime`) over the step ABI, with model and sequence handles that
  release on `Drop` and integer codes mapped to `ComputeError`. `MockCompute`
  stays for CPU tests.
- The scheduler's KV pool counts **real device pages** reported by the
  runtime, so admission capacity math reflects VRAM.
- The artifact crate gains a device-view export (device pointer, storage
  layout, format, shape per bound tensor) so the program binds planes
  directly — no host dequant of the W8 endpoints.
- The FFI boundary stays the only place cross-language state crosses, and it
  gets *smaller*: a handful of step calls instead of one call per operator.
- Correctness is no longer checkable op-by-op from Rust. It is established by
  the canary oracle and the f64 layer references of spec 01, run under the
  GPU test profile (ADR 0006).
