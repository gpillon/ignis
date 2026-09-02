# Core engine — spec

The Rust core (`ignis-core`) owns everything above the kernel leaf: the
scheduler, the KV cache, the GDN state, and the request state machine. It
sits between the HTTP server (above) and the flat C ABI kernel leaf (below).
This feature builds the v1 core, layer by layer, per
`docs/design/ignis-v1.md` §2 (Scheduler, Paged KV, KV-RAM host tier, GDN
state).

The core is the highest-risk module of v1 (the design doc says so explicitly),
so it gets the most test coverage. It is the dogfood target: good enough to
run the developer's own coding agent (1 main + ~10 subagents, ~98% shared
prefix → sibling-prefix reuse).

## v1 scope (priority order)

1. **Paged KV cache + block table (VRAM)** — `core-01`. KV pool auto-sized
   from free VRAM (~12–13 GB budget after weights + graphs); block table maps
   logical blocks to physical.
2. **GDN state management** — `core-02`. The recurrent state of the linear
   attention (GDN) layers; resumable **only** at checkpoint / frontier
   boundaries. The KV-RAM host tier must respect this boundary (a snapshot
   mid-prefill is invalid for GDN layers).
3. **Request state machine + basic admission** — `core-03`. Request
   lifecycle (admitted → prefilling → running → done) + basic lane
   assignment.
4. **Scheduler: N=8 decode lanes + batched prefill** — `core-04`. 8
   resident decode lanes (N=8) with host-tier overflow; concurrent / batched
   prefill to saturate the GPU and cut burst TTFT (an experiment to verify —
   we may be compute-bound, in which case it is useless).
5. **Full admission state machine** — `core-05`. The fairness machinery:
   protection, backfill class, temporal credit, frontier distance — decides
   which request gets which lane.
6. **KV-RAM host tier** — `core-06`. Snapshots GPU lanes to host RAM so
   sibling requests restore instead of re-prefilling; two-tier eviction
   (probation → protected).
7. **Prefix reuse (sibling prefix caching)** — `core-07`. Concurrent
   requests sharing a prefix skip the redundant prefill.

## Acceptance

- The scheduler holds N=8 resident decode lanes with host-tier overflow.
- Paged KV + block table auto-sizes from free VRAM; no OOM under the N=8
  load.
- GDN state is resumable at frontier / checkpoint boundaries; the host tier
  honors the boundary.
- The full admission state machine (protection / backfill / credit /
  frontier) drives lane assignment.
- Under a "1 main + ~10 subagents" trace, sibling-prefix reuse is active and
  evictions are bounded.

## References

- Design: `docs/design/ignis-v1.md` §2 (Scheduler, Paged KV, KV-RAM, GDN),
  §3 (milestones).
- Glossary: `CONTEXT.md` (Scheduling & cache, Engine & process).
- Upstream: `artifact-01` (device materialization) for KV tensor geometry.
- Kernel side: `kernel-abi-01` (prefill + GDN C ABI) for the scheduler to
  drive.