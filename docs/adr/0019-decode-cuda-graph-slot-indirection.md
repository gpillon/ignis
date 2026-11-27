---
status: accepted
---

# decode CUDA graph replay via device-resident slot indirection

A decode round is replayed from a CUDA graph, one captured per exact batch width at startup, and a graph never
names a sequence: each row reads its token, its position and its physical pool slot from staging buffers the model
owns, at stable device addresses, refreshed from the host before each replay and read by the kernels when they
execute. A graph replays a fixed DAG bound to the device addresses it captured, while the round's composition
changes every round: a sequence keeps the pool slot it was allocated for its lifetime, but which of the pool's
slots are decode-ready is arbitrary and in general not contiguous. A graph captured against particular slots'
addresses would silently replay against another sequence's KV pages and GDN state as soon as a different
combination of slots filled the same width, a silent bug that depends on the batch width, of the kind that had
already cost a gate run.

- **A graph is invariant after capture.** What changes from round to round lives in pre-allocated buffers at
  stable addresses, and the hot path never updates graph nodes: it is the standard pattern of an eager graph over
  fixed-address staging, and the reference stack runs its graphs the same way.
- **Row `i` of a width-W graph means whichever sequence the slot buffer names this round**, not a fixed physical
  slot: that is what makes one graph per width correct for any combination of W decode-ready lanes, not only a
  canonical low prefix. The one slot value serves the attention's block-table row selector and the recurrence's
  state selectors, and the block tables and the recurrent and conv state tensors a graph binds span the whole pool,
  at fixed addresses that exist whatever the round's composition.
- **A position is read from device staging too**, never baked in at capture.
- **The graphs have their own scratch arena**, apart from the one prefill uses, so a prefill chunk running between
  two replays can never alias the addresses a graph rereads.
- **Every capture takes one fixed, conservative attention envelope**, the model's maximum context as the most
  visible keys. The envelope is a host promise of launch resources, for workspace sizing and the choice of kernel
  route, not the causal mask, which comes from the positions each row reads inside the kernel: the worst case is
  always a safe over-approximation, never a correctness hazard, and it lets one capture serve every round whatever
  a lane's position.
- **Replay never pads.** A round of width W launches the width-W graph only, or, when that width failed to
  capture, the eager path. A width whose graph failed to build at startup runs eagerly in every round at that
  width: the model load still succeeds and only that width's decode throughput degrades. A capture failure degrades
  performance and never refuses service.
- **The graphs are captured right after the sequence pool is created, before any sequence is allocated.**
- **The decode call's signature is unchanged**: inside, it dispatches to the width's graph when one is ready and
  to the eager path otherwise, with no ABI break and no new Rust call site on the hot path. Capture and its
  last-error query are entry points of their own, a one-time startup action being a lifecycle call like model load
  and pool creation, not per-call modulation (ADR 0016).
- **The round a graph captures is one batch-wide traversal of the model** (ADR 0020), the lanes being rows of one
  call and not separate forward passes.

No vendored kernel is modified (ADR 0010): the vendored attention and recurrence ops already take device-resident
row selectors, read at execution time, and a position can be read from a device address.

## Considered Options

- **Update each node's input pointers before every step**: rejected. It adds CUDA API calls and parameter writes to
  the hot path and needs a new primitive in the leaf, where one launch of an invariant graph is the cheapest replay
  there is.
- **One graph for a representative geometry, every other batch size running eagerly**: the earlier model, replaced
  by a graph per exact width.
