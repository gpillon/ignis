---
status: accepted
---

# Step-level, device-resident C ABI (the forward pass lives in the leaf)

The C ABI between Rust and the kernel leaf (ADR 0001) is step-level, device-resident and opaque-handled: Rust owns
everything above the step, and C++ owns the step. The boundary first sat at the operator, one flat entry point per
operator with host pointers and Rust driving the forward pass op by op, and that engine could not work at any speed:

- **The activations never reached VRAM.** Hidden state, KV cache and GDN state lived in host memory and every op
  copied its inputs in and its outputs out, synchronously: a floor of seconds per token against the reference's
  13.1 ms, two to three orders of magnitude, before any question of kernel quality.
- **A per-operator boundary cannot carry the program's invariants.** Layer kind, fused-plane row order, per-head
  geometry, RoPE position, gating parameters and state shape lived on the Rust side of an ABI that sees only flat
  buffers, and each was gotten wrong independently.
- **A CUDA graph had nothing real to capture.** The object to capture is one batched decode round over per-slot
  state views, which exists only when the state is device-resident and owned by the leaf.

What holds:

- **Rust owns everything above the step**: HTTP and the OpenAI surface, the tokenizer and chat template, the
  scheduler and admission state machine, KV page accounting, the prefix index, the host tier policy, telemetry, the
  bench.
- **The leaf has a program layer**: a device arena and tensor views, CUDA streams, the model's per-layer op
  sequence, a sequence-state store (paged KV pages, a slot-indexed fp32 GDN state pool, conv taps, position) and
  the sampling epilogue. It is ours, written on top of vendored ops (ADR 0010); the reference's own program is not
  vendored, being entangled with MTP, vision, graphs and its engine.
- **No host activation pointer crosses the boundary.** Activations, KV and GDN state live in VRAM for the lifetime
  of a sequence. Streams, synchronization and op dispatch are internal to the leaf.
- **The surface is flat C with opaque handles and integer return codes**: model load from the artifact's device
  arena plus a topology descriptor; sequence allocate and release with a context reservation; sequence snapshot to
  a caller-provided pinned host region and restore from it; prefill of a token span from a starting position; one
  decode round over a batch of sequence handles; a sampling-parameters struct; runtime statistics (VRAM, page
  geometry, per-step time, kernel and graph counters).
- **The batch parameter and the span and start-position parameters are there from the first version**, so that
  batched decode rounds and chunked or tail prefill are leaf changes and not changes of the ABI's shape. That holds
  for the shape only: how a call is modulated goes in the options structs of ADR 0016.

## Considered Options

- **Keep the per-operator ABI and pass device pointers**, the forward pass still in Rust: rejected. It removes the
  copies but leaves the program's invariants split across the boundary and graph capture without a natural unit;
  Rust driving the reference's roughly 630 launches of a decode round through FFI is possible, buys nothing and
  costs the ergonomics of graph capture.
- **Move the boundary to the request**, the leaf owning scheduling too: rejected. It discards the Rust core
  (scheduler, admission, KV accounting, host tier, prefix index), the part of ignis that worked.
- **A decode graph over staging buffers that the Rust adapter fills and reads back around each launch, captured
  for one representative geometry**, the decode replay this ADR replaced: it existed because the graph could not
  see the sequence's device state, and what it captured was a five-kernel toy over zeroed state, with no layers and
  no projections. The decode graph is ADR 0019's, which keeps the one principle of it that holds: a graph is
  invariant after capture, and what changes from step to step lives in buffers at stable device addresses.

## Consequences

- The `Compute` trait is implemented by a thin safe Rust wrapper over the step ABI (`crates/runtime`); `MockCompute`
  stays for CPU tests.
- The scheduler's KV pool counts real device pages reported by the runtime, so admission's capacity math reflects
  VRAM.
- The artifact crate exports a device view (device pointer, storage layout, format and shape per bound tensor), so
  the program binds planes directly, with no host dequantization of the W8 endpoints.
- The FFI boundary stays the only place cross-language state crosses, and it is smaller: a handful of step calls
  instead of one call per operator.
- Correctness is no longer checkable op by op from Rust. It is established by the canary oracle and the f64 layer
  references, run under the GPU test profile (ADR 0006).
