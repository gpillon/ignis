# runtime 03 — serving loop: chunk-level interleaving, batched decode rounds, per-width graphs, sampling (gate G3)

GitHub: #64 (phase 3 master; blocked by #63, gate G2 — closed 2026-09-09)

Source: `.scratch/REVIEW-2026-09-05.md` §6 (Phase 3), `.scratch/ROADMAP.md`
(the G3 row and the phase 3–5 candidate decomposition), and the grilling
session of 2026-09-09 that turned them into this spec. Builds directly on
`.scratch/runtime/specs/02-real-prefill.md` (G2).

ADRs respected: 0005 (performance-first above a sane-output floor), 0006
(exclusive GPU testing), 0007 (performance gates, correctness self-checked),
0009 (step-level device-resident ABI), 0010 (vendored reference kernels),
0011 (tracing for structured logging), 0012 (`request.id` as trace id), 0014
(teacher-forced canary floor), 0015 (live/live gate on cold samples), 0016
(extensible options structs at the step ABI), 0017 (Prometheus metrics).
ADR introduced by this work: **0018** (chunk-level prefill/decode
interleaving, scheduler-driven, single stream).

Reference: ninfer at the manifest's pinned commit
(`kernel/vendor/manifest.json`).

## Problem Statement

G2 made prefill real; decode is still what G1 left. Three gaps sit between
that and a serving loop.

**The loop stops the world for a prefill.** `ConcreteScheduler::advance()`
runs a prefill phase to completion and then a decode phase, on one thread,
and a prefill call returns only when the whole span is done. At the measured
G2 numbers — a 1,024-token chunk costs ~110 ms (94–119 ms across runs), a
32K span is 32 of them, a reference decode step is 13.2 ms — a 32K prompt
inserts a ~3,820 ms gap into every decoding lane's token stream. The
glossary already promised otherwise: `CONTEXT.md` said decoding on other
lanes continues while the prefill lane runs, and it did not.

**Decode is eager and greedy.** `ignis_program_decode` walks its batch one
sequence at a time, `IgnisSamplingParams` has a single field (`greedy`), and
`DecodeParams` carries only `temperature` and `seed`. No CUDA graph exists
anywhere in the leaf or the runtime: ADR 0008's staging-buffer model was
superseded by ADR 0009 and explicitly deferred the decode graph to this
phase. The gate asks for ≥ 99% of the reference's C=1 rate, where the whole
margin is 0.13 ms on a 13.2 ms step.

**The scheduler's capacity numbers are not the GPU's.** `KvPool` counts
abstract pages, admission's capacity math does not use real bytes, and the
C=4 cell is the first time admission must queue or refuse on capacity that
exists. A gate measured against invented accounting measures a policy that
will not ship.

There is also a measurement gap, the same one G2 had: no reference number
exists for C=4 or for inter-token latency under a concurrent prefill, and
`ignis-bench` cannot measure either.

## Solution

The scheduler drives the prefill chunk loop. `advance()` performs at most
one prefill chunk and one decode round, so a decode-ready lane never waits
longer than one chunk for its turn. Nothing runs concurrently on the GPU:
prefill and decode take turns on the one model stream. This is
**prefill/decode interleaving**, and it is deliberately not **true
prefill/decode overlap**, which stays a north-star item (ADR 0018).

It needs no ABI change. Rust passes a span one chunk wide, the leaf's own
loop runs a single iteration, and the serving chunk width is a Rust-side
number bounded above by the width the program scratch was reserved for at
model load. The leaf keeps its internal loop for long spans: the GPU tests
and the per-token self-oracle use it.

Because a request now lives in `Prefilling` for tens of ticks, that state
becomes durable and carries **prefill progress**. Resuming is the absence of
anything special: the next chunk gets scheduled. Cancel is abort, not
suspend — the in-flight chunk finishes, then the sequence is released. Each
completed chunk boundary is recorded as a GDN resumable boundary, which is
correct because the leaf advances the sequence's persistent conv slot and
GDN recurrent slot in place and moves `seq->position` only after the chunk's
synchronization returns.

Decode becomes a real batched round: one call over every decode-ready lane,
sampled device-side in the leaf with per-sequence parameters and
per-sequence RNG state, replayed from a CUDA graph captured per **exact**
batch width 1..8. Widths are never padded: GDN slot traffic is per sequence
(144 MiB each), so padding width 1 up to width 8 moves 1.15 GB instead of
144 MB and spends several times the C=1 gate's entire margin. The graph's
staging buffers are a reservation separate from the prefill scratch, because
a prefill chunk landing between two replays would otherwise clobber the
fixed addresses a graph rereads.

The gate is measured, not asserted, on the same live/live rule as G2
(ADR 0015). `ignis-bench` gains the three G3 cells, measured over HTTP/SSE
against either engine — the reference emits none of ignis's internal events,
so the request log is observability and diagnosis, never the comparative
oracle.

## User Stories

1. As the engine owner, I want a long prompt's prefill to cost the decoding lanes one chunk of latency instead of the whole span, so that a subagent starting up does not stall every session already running.
2. As the engine owner, I want the scheduler to drive the chunk loop, so that the decision of when to decode lives above the step boundary where scheduling belongs.
3. As the engine owner, I want the leaf to keep its own loop over a long span, so that the per-token self-oracle and the GPU tests keep working unchanged.
4. As the engine owner, I want interleaving to require no ABI change, so that this phase spends its ABI budget on sampling alone.
5. As the engine owner, I want a serving prefill chunk width bounded by the width reserved at load, so that narrowing under load is free and widening is refused rather than silently corrupting scratch.
6. As the engine owner, I want the serving chunk width chosen from the gate measurement rather than argued in advance, so that the one dial that moves p95 is set by evidence.
7. As the engine owner, I want no adaptive chunk-width policy in this phase, so that the loop stays explainable while its constants are still unmeasured.
8. As the engine owner, I want one decode round per prefill chunk as the initial policy, so that there is a baseline to move rather than a tuned constant nobody can justify.
9. As the engine owner, I want the gate's functional property stated independently of that ratio, so that improving it does not fail the gate.
10. As the engine owner, I want exactly one request at a time holding device-resident prefill progress, so that VRAM no tier can yet reclaim is not multiplied across half-prefilled sequences.
11. As the engine owner, I want `Prefilling` to be a durable state carrying how far the prompt is prefilled, so that admission can see a half-prefilled request instead of a momentary one.
12. As the engine owner, I want resuming a prefill to need no suspend/resume primitive, so that the loop does not carry machinery for a case this phase does not have.
13. As the engine owner, I want cancel to finish the in-flight chunk and then abort the sequence, releasing its KV pages, GDN slot and conv taps, so that cancellation is a definite point rather than a race with the GPU.
14. As the engine owner, I want each completed chunk boundary recorded as a GDN resumable boundary, so that the old "mid-prefill is never resumable" rule stops being wrong now that mid-chunk and mid-prefill are different positions.
15. As the engine owner, I want the boundary bookkeeping to stop being a per-token vector scanned linearly, so that the resumability check is not O(context) on the hot path.
16. As the engine owner, I want a sequence that has not finished prefilling to be impossible to decode, so that a scheduling bug is a refused call rather than corrupted output.
17. As the engine owner, I want one decode call spanning every decode-ready lane, so that eight lanes stream the model's weights once rather than eight times.
18. As the engine owner, I want sampling to happen device-side in the leaf, so that no logits cross the ABI and ADR 0009's device-resident step is preserved.
19. As the engine owner, I want temperature, top-p, top-k, presence penalty, frequency penalty and seed supported, so that the engine serves the parameters coding clients actually send.
20. As the engine owner, I want the sampling parameters passed as an array parallel to the sequences, so that lanes with different settings still share one decode round.
21. As the engine owner, I want RNG state carried per sequence, so that what a request generates depends on its own seed and never on which lanes shared its round.
22. As the engine owner, I want presence and frequency penalties backed by per-sequence count state on the device, so that a penalty is a property of the request rather than of the batch.
23. As the engine owner, I want that new per-sequence state added to the sequence handle's inventory, so that G4's snapshot checklist is complete when it is written.
24. As the engine owner, I want the sampling ABI extended through the size-prefixed options struct, so that ADR 0016's mechanism is used rather than a second ABI break.
25. As the engine owner, I want `top_k` documented as an ignis extension, so that the OpenAI-compatible surface stays honest about what is standard.
26. As the engine owner, I want a parameter the engine does not support to be refused rather than silently ignored, so that a client's settings are never quietly dropped.
27. As the engine owner, I want a decode CUDA graph captured for every exact batch width 1..8, so that no round pays for lanes it does not have.
28. As the engine owner, I want batch widths never padded up to a captured width, so that the C=1 cell is not spending its gate margin on GDN traffic for absent sequences.
29. As the engine owner, I want the graph's staging buffers reserved separately from the prefill scratch, so that an interleaved chunk cannot clobber what a replay rereads.
30. As the engine owner, I want those staging buffers sized once for the widest batch and shared across the widths, so that eight graphs cost graph nodes rather than eight buffer sets.
31. As the engine owner, I want an eager fallback for any width without a graph, so that a capture failure degrades performance instead of refusing service.
32. As the engine owner, I want `KvPool` pages to be the device pages the leaf actually built, so that admission counts the resource that runs out.
33. As the engine owner, I want admission's capacity math in real bytes, so that a request is refused when the GPU is full and not before or after.
34. As the engine owner, I want the scheduler's capacity view and the leaf's pool checked against each other, so that a disagreement is a test failure rather than a load-time surprise.
35. As the engine owner, I want the request log to be the canonical `ignis.request.*` events emitted as JSONL, so that logs and gate diagnosis read the same stream.
36. As the engine owner, I want no third parallel telemetry stream, so that what the logs say and what the gate measures cannot drift apart.
37. As the engine owner, I want per-phase fields covering chunks consumed, prefilled tokens and per-lane inter-token latency, so that a failing cell is attributable without another run.
38. As the engine owner, I want interval counters to stay in Prometheus, so that metrics-shaped data does not leak into the event stream (ADR 0017).
39. As the engine owner, I want the gate measured live/live in one session against the reference, so that a committed record can never decide it (ADR 0015).
40. As the engine owner, I want the bench to measure TTFT and inter-token latency from the HTTP/SSE side, so that the same instrument works against an engine that emits none of ignis's internal events.
41. As the engine owner, I want the C=1 cell expressed as a percentage of the live reference, so that the historical 75–76 tok/s is a note and not the oracle.
42. As the engine owner, I want each cell's fixture stated as reserved `context_tokens`, so that a fixture is sized against what `ignis_seq_alloc` actually takes from the pool.
43. As the engine owner, I want the ITL cell to repeat its cold prefill ten times, so that the percentile it reports has a distribution behind it.
44. As the engine owner, I want those ten prefillers to be sequential, each released before the next is allocated, so that the cell measures one active prefill and not a concurrency this phase does not have.
45. As the engine owner, I want the four decode lanes to stay alive across the whole series, so that their inter-token latency is sampled continuously rather than restarted.
46. As the engine owner, I want p50, p95, p99 and max all recorded even though p95 decides, so that a later reader can see the shape and not only the verdict.
47. As the engine owner, I want each generation cap derived from the measured window, so that reserved pages are not wasted on tokens no cell will generate.
48. As the engine owner, I want the gate to state that N=8 at long context is unreachable with BF16 KV, so that nobody reads G3's concurrency as a promise G4 has not yet delivered.
49. As the engine owner, I want the teacher-forced canary floor and the chunked-vs-per-token self-oracle re-run on this phase's tree, so that the serving loop is proven not to have changed the forward pass.
50. As the engine owner, I want any correctness regression found at G3 filed as its own ticket, so that no gap is waived to make a gate pass.
51. As a coding-agent user, I want my tokens to keep arriving while another agent's session starts up, so that one long prompt does not freeze the others.
52. As a coding-agent user, I want my sampling settings honoured regardless of who else is being served in the same round, so that reproducibility is mine and not the batch's.

## Implementation Decisions

**Scope boundary.** This phase makes the *serving loop* real. It does not
pack several requests' prefill into one traversal, does not run prefill and
decode concurrently on the GPU, does not hold more than one active prefill,
does not snapshot or restore a sequence, and does not change the KV format.

**Interleaving (ADR 0018).** One `advance()` performs at most one prefill
chunk and at most one decode round. The scheduler passes
`ignis_program_prefill` a span one serving-chunk wide; the leaf's internal
loop runs one iteration and returns after its single synchronization. The
serving chunk width is a scheduler-side value constrained to be at most the
width the program scratch was reserved for at model load (P2-01); its G3
default is that load width. No adaptive policy.

**K, the decode rounds per chunk, is policy and not invariant.** K=1 is the
shipped default. The gate's functional property is stated K-agnostically:
while a prefill is active, no decode-ready lane waits more than one prefill
chunk between two of its decode opportunities. Raising K must not fail the
gate. K does not move p95 over any affordable range: per cycle a lane sees
`K-1` short gaps and exactly one crossing a chunk, so the long gaps are the
top `1/K` of the distribution and p95 stays among them until `K >= 20`, a
240% TTFT inflation.

**One active prefill.** Exactly one request at a time holds device-resident
prefill progress and consumes chunks; the rest queue. Multi-prefill chunk
interleaving buys fairness for a short prompt queued behind a long one,
never throughput, and is deferred to G4 with burst scheduling and packed
prefill.

**Prefill progress and cancel.** `Prefilling` carries the position reached.
No suspend/resume primitive is built: this phase has no case that needs one,
and priority preemption of a prefill is G4. Cancel finishes the in-flight
chunk, then aborts the sequence and releases its KV pages, GDN slot and conv
taps.

**Chunk boundaries are GDN resumable boundaries.** The loop calls
`GdnState::checkpoint(position)`, not `advance(position)`, at each completed
chunk. `GdnState`'s boundary set stops being a per-token `Vec<usize>`
scanned with `contains()`.

**Sampling.** Device-side in the leaf; the ABI keeps returning token ids.
`ignis_sampling_params` is extended through ADR 0016's size-prefixed struct
with temperature, top-p, top-k, presence penalty, frequency penalty and
seed, and `ignis_program_decode` takes an array of them parallel to its
sequences. RNG state and penalty count state are per sequence and live in
the sequence handle, which makes them new sections on G4's snapshot
checklist. Parameters are staged in device buffers at stable addresses, so a
captured graph reads them by replay.

**Decode graphs.** Captured per exact width 1..8 at startup, eager fallback
for any width without one. Staging buffers reserved separately from the
prefill scratch, sized for width 8, shared across the widths.

**Core rewiring.** `KvPool` pages become the device pages the leaf built;
admission's capacity math uses real bytes; scheduler and leaf capacity views
are cross-checked. GDN state needs no wiring: it is already device-resident
and consistent, and the boundary recording belongs to the interleaving
ticket.

**Request log.** The canonical `ignis.request.*` events (ADR 0011, 0012)
emitted as JSONL, extended with the phase fields this loop creates. No third
stream. Interval counters stay in Prometheus (ADR 0017).

## Gate G3

Three cells, all live/live in one measurement session against the reference
in the owner's production profile (ADR 0015), all measured over HTTP/SSE.
Fixtures are stated as reserved `context_tokens`, because `ignis_seq_alloc`
reserves, materializes and zeroes pages for the full prompt-plus-cap
entitlement at allocation.

| cell | fixture | reserved `context_tokens` | pool 65,536 | verdict |
|---|---|---|---|---|
| C=1 | prompt 8,192, cap 256 | 8,448 | headroom 57,088 | at least 99% of the live reference's tok/s (historically ~75–76) |
| C=4 | 4 × (prompt 8,192, cap 256) | 33,792 | headroom 31,744 | aggregate at least 99% of the live reference's aggregate |
| ITL | 4 × (prompt 4,096, cap 512) with a prefiller at prompt 32,768, cap 64 | 51,264 peak | headroom 14,272 | p95 within the live reference's envelope |

The ITL cell runs **ten sequential cold prefillers**: each 32,768-token
prefiller is allocated, prefilled, and released before the next is
allocated, every prompt cold and distinct under ADR 0015's rule, while the
four decode lanes stay alive across the whole series. The lanes' inter-token
intervals are sampled for the entire series (~1,240 samples at K=1); p50,
p95, p99 and max are all recorded and p95 decides. The 512-token cap covers
the ~320 tokens a lane generates across the series.

Alongside the three cells, two non-negotiables: the functional
anti-serialization property above, proven by a CPU test rather than inferred
from a permissive envelope; and the G2 correctness checks (teacher-forced
canary floor, chunked-vs-per-token self-oracle) still green on this phase's
tree.

**Recorded inequality.** N=8 at long context is not reachable with BF16 KV.
At this geometry a sequence-token costs 64 KiB (16 GQA layers × 4 KV heads ×
256 × K+V × 2 bytes), so the 65,536-token pool is ~4 GiB and eight lanes at
full context would be ~20 GiB next to ~19 GB of weights. N=8 in BF16 exists
at short contexts; G4's hq-e8-2b makes it practical at long ones. Nothing in
G3 should be read as a promise otherwise.

## Testing Decisions

- The interleaving property is a CPU test against `MockCompute`: with a
  prefill active and decode-ready lanes present, the recorded call sequence
  must never place two prefill chunks between two decode rounds.
- Prefill progress, cancel-as-abort and the one-active-prefill rule are CPU
  tests on the scheduler seam.
- Rust-driven chunking is checked against leaf-driven chunking: the same
  prompt as one long span and as a sequence of chunk-wide spans must leave
  the same state and predict the same next token. The per-token route is a
  third, independent oracle and is retained for this phase.
- Sampling is tested in the leaf's own test binary at real geometry, plus a
  determinism test: the same seed on the same request produces the same
  tokens regardless of which other lanes shared its rounds.
- Graph replay is tested against the eager path at every width 1..8, and the
  staging-buffer separation is tested by running an interleaved prefill
  chunk between two replays and requiring identical output.
- Capacity accounting is tested by driving admission to refusal and
  comparing its view against the leaf's reported pool.
- The bench cells are CPU-testable end to end against a stub endpoint; only
  the gate run itself needs the GPU (ADR 0006).

## Out of Scope

- Packed prefill: several requests' prefill tokens in one traversal. Its
  phase is decided after #92 reports.
- True prefill/decode overlap on separate streams (roadmap phase 6).
- More than one active prefill, multi-prefill chunk interleaving, and
  priority preemption of a prefill (G4).
- Snapshot and restore of a sequence, and eviction of a half-prefilled one:
  the KV-RAM host tier is G4. This phase only establishes that a chunk
  boundary is a valid capture point for the state that exists.
- hq-e8-2b KV, device prefix reuse, tagged lanes (G4).
- MTP, DFlash2, ReplaySSM (G5). Vision.
- Deleting the per-token prefill route: retained here, reconsidered after G3.
- Adaptive chunk-width policy, and any kernel work of our own.

## Further Notes

- The gate clause as #64 originally worded it ("within the reference
  envelope") was satisfiable by building nothing, because the reference does
  not overlap prefill and decode at all. That is why the phase carries a
  functional property alongside the measured one.
- The p95 floor is the chunk time plus a decode round and nothing else.
  Chunk width is the only dial that moves it, which is why the serving width
  knob exists and why K does not.
- `ignis_seq_alloc` materializes and zeroes the whole reservation at
  allocation, so the 32,768-token prefiller zeroes ~2 GiB inside its own
  TTFT. Expected, and not a defect to chase when reading that number.
- Work splits into eight tracer-bullet tickets: the interleaving loop, the
  capacity rewiring, leaf sampling, the HTTP sampling surface, decode
  graphs, the request log, the measurement instrument, and the gate run.
  Four of them start in parallel; the critical path is leaf sampling into
  graphs into the gate, because a graph captured before sampling would have
  to be captured again.
