# ADR 0018 — Prefill and decode interleave at chunk granularity, driven by the Rust scheduler

## Status

Accepted (2026-09-09, grilling session for GitHub #64 / G3). Builds on ADR
0009 (step-level device-resident ABI) and ADR 0016 (extensible per-call
options). Does not amend either.

Sources: `.scratch/ROADMAP.md` phase 3, `.scratch/REVIEW-2026-09-05.md` §6.

## Context

Until G3 the serving loop had no opinion about what decode does while a
prefill runs, because there was nothing to have an opinion about: the
scheduler's `advance()` ran a prefill phase to completion and then a decode
phase, on one thread, and a prefill call returned only when the whole span
was done. A 32K prompt therefore stopped every decoding lane for the length
of the prefill.

The measured cost of that at G2 is not marginal. A 1024-token prefill chunk
costs ~110 ms (94–119 ms across runs), a 32K span is 32 of them, and a
reference decode step is 13.2 ms. Serialized, a 32K prefill inserts a
~3820 ms gap into every decoding lane's token stream. The G3 gate asks for
p95 inter-token latency under a concurrent prefill to sit inside the
reference's envelope.

The reference does not overlap prefill and decode at all. So the gate clause
as written could be satisfied by building nothing: whatever ignis does, it
would be inside an envelope that is itself a full-length stall. That is
precisely why this is a design decision and not an optimization — the
question is not whether the gate passes but what shape the serving loop is
frozen into, since G3 is the phase that defines it.

Three options:

- **(a) Serialize, and pass the clause literally.** No work now, the whole
  concern deferred to the north-star phase. But the loop shape gets frozen
  around "a prefill call runs to completion", and the request state machine
  around "a request is never observably half-prefilled". Undoing both later
  is a rewrite of the scheduler, not an addition to it.
- **(b) Interleave at chunk granularity on the one model stream.** The
  scheduler hands the leaf one chunk per call and puts a decode round between
  chunks. Nothing runs concurrently on the GPU; the two take turns. Costs the
  decode lanes one chunk of latency (~123 ms) instead of a whole span
  (~3820 ms), and costs the prefill one decode round per chunk (+11% TTFT at
  one round per chunk).
- **(c) True overlap on separate streams.** Prefill and decode resident
  together, contending for SMs. The largest win and the largest unknown:
  SM contention with no partitioning story, a workspace arena that is not
  built for two writers, and a direct conflict with capturing decode CUDA
  graphs while another stream is live.

## Decision

- **(b).** Prefill and decode interleave at **chunk granularity on the single
  model stream**. One `advance()` performs at most one prefill chunk and one
  decode round.
- **The Rust scheduler drives the chunk loop.** It calls the leaf with
  one chunk-wide span at a time. The leaf keeps its own loop over a long
  span — the per-token self-oracle and the GPU tests use it — but it is no
  longer the only thing that loops. There is **no callback from the leaf into
  Rust**: that would put scheduling below the step boundary, against ADR 0009.
- **True prefill/decode overlap is explicitly out of scope for G3**, and stays
  a roadmap phase-6 item. It is excluded on grounds of risk and of conflict
  with the decode graph model, not because it is unwanted — it is the
  differentiator the north star names.
- **A serving prefill chunk width** is introduced, constrained to `<=` the
  width the program scratch was reserved for at model load (P2-01 / #83).
  Shrinking at serving time is free; growing is not. Its default is the load
  width, and its value is chosen empirically from the live/live gate run, not
  argued in advance. No adaptive policy in G3.
- **One decode round per prefill chunk (K=1)** is the G3 default. K is a
  measured baseline, not a claim: it is turned only if the C=4 cell fails.
- **Exactly one active prefill.** One request at a time holds device-resident
  prefill progress and consumes chunks; others queue. Multi-prefill chunk
  interleaving — which buys fairness for a short prompt stuck behind a long
  one, never throughput — is deferred to G4 alongside burst scheduling and
  packed prefill.
- **A completed chunk boundary is recorded as a GDN resumable boundary**
  (`GdnState::checkpoint`, not `advance`). That module's rule was written when
  prefill was atomic, so "mid-prefill" and "mid-chunk" were the same position;
  chunked prefill separates them, and a chunk boundary is as consistent as a
  decode round's. Whether the *whole* sequence state may be captured there is
  a separate property — see **snapshot point** in `CONTEXT.md` — and it holds
  today for the state that exists.
- **`Prefilling` becomes a durable state carrying prefill progress**, since a
  request now lives in it for tens of ticks. Resumption needs no primitive:
  it is the absence of a scheduled next chunk. **Cancel is abort, not
  suspend** — the in-flight chunk finishes, then the sequence is aborted and
  its KV pages, GDN slot and conv taps are released. G3 ships no explicit
  suspend/resume; priority preemption of a prefill is G4.

## Consequences

- The p95 inter-token latency under a prefill is floored by the chunk time
  plus a decode round, and by nothing else. K, the number of decode rounds
  between two chunks, does not move it over any affordable range: per cycle a
  lane sees `K-1` short gaps and exactly one long one, so the long gaps are
  the top `1/K` of the distribution and the p95 stays in that group until
  `K >= 20` — a 240% TTFT inflation. Chunk width is the only dial that moves
  the gate metric, which is why it is the knob G3 introduces.
- A half-prefilled sequence can be paused but not evicted, because the KV-RAM
  host tier is a G4 feature — **not** because its state is uncapturable.
  Verified in the leaf: `causal_conv1d_silu` advances the sequence's own
  persistent conv slot in place, the GDN recurrent slot likewise, KV pages are
  appended, and `seq->position` moves only after the chunk's synchronization
  returns. All four are consistent together at a completed chunk boundary, so
  a chunk boundary is a valid whole-sequence snapshot point for the state that
  exists today. G4's hq-e8-2b exact-key side store is new state and must be
  re-checked against this property when it lands. Meanwhile pausing holds the
  KV pages, GDN slot and conv taps, which is the constraint behind "exactly
  one active prefill": N active prefills would multiply VRAM that no tier can
  yet reclaim.
- A sequence cannot accidentally decode mid-prefill: `pending_token` is
  written only by the last chunk, and `ignis_program_decode` rejects a
  sequence without one.
- Decode CUDA-graph staging buffers must be a reservation **separate** from
  the prefill scratch. A graph replays fixed device addresses; a prefill chunk
  landing between two replays would otherwise clobber what the graph expects
  to reread. The failure would be silent and batch-width dependent, the same
  class of bug #96 cost this project a gate run.
- Prefix-cache registration moves to the end of the last chunk rather than the
  return of a single prefill call.
- The scheduler acquires a second reason to be woken that is not a command:
  a request mid-prefill. `model_thread_loop` already ticks while anything is
  in flight, so this is a state question, not a threading change.
