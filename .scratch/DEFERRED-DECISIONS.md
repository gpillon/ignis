# Deferred decisions

Things considered and deliberately **not** built, with the reasoning that
put them off and what would change the answer. Written so that a later phase
re-opens a question knowing what was already argued, instead of arguing it
again from zero.

Status and blocking live on GitHub (AGENTS.md). This file records *why*, not
*when*.

---

## From the G3 grilling session (2026-09-09, GitHub #64)

Source conversation: the design session that produced
`.scratch/runtime/specs/03-serving-loop.md` and ADR 0018. Ten items below;
each was raised, argued and closed in that session. Nothing here elaborates
on the later phases beyond what was actually decided.

### 1. Packed prefill — several requests' tokens in one traversal

**Deferred to:** decided after #92 reports; candidate G4, possibly phase 6.

Today `prefill_step(&[PrefillJob])` takes a group of requests but
`RuntimeCompute` walks it and calls `ignis_program_prefill` once per
sequence. The batched shape is real, the batched traversal is not.

The case *for* building it is strong and comes from #92: if the per-chunk
cost is dominated by synchronization and dispatch rather than by the math —
which is what #86 landing faster kernels with no change in per-chunk time
suggests — then packing N requests into one traversal amortizes that
overhead across N.

It was still deferred, because the G3 gate is decode at C=1 and C=4 and
packed prefill does not move it. It buys throughput on the subagent burst
workload, which is the G4 trace. Putting the phase's largest kernel-side
risk (varlen attention, per-sequence GDN state side by side, a new ABI
signature) on the critical path of a gate that cannot measure it was the
trade rejected.

**What would change the answer:** #92 concluding that the sync/dispatch
overhead is both dominant and irreducible any other way. Then packing is the
remaining lever and its phase moves up.

### 2. True prefill/decode overlap on separate streams

**Deferred to:** roadmap phase 6 (the north star). Excluded from G3 by
ADR 0018.

Three options were weighed for G3: serialize and pass the gate clause
literally; interleave at chunk granularity on the one stream; or run prefill
and decode concurrently on separate streams. The third is the largest win
and the largest unknown — SM contention with no partitioning story, a
workspace arena not built for two writers, and a direct conflict with
capturing decode CUDA graphs while another stream is live.

The second was chosen, and the reasoning matters for whoever picks this up:
it was chosen as the *shape of the serving loop*, not as an optimization. A
loop frozen around "a prefill call runs to completion" would have to be
rewritten, not extended, to reach either of the other two.

**Note for the future reader:** the reference does not overlap prefill and
decode at all. This is named in `.scratch/REVIEW-2026-09-05.md` §6 as the
real differentiator for subagent bursts.

### 3. More than one active prefill, and multi-prefill chunk interleaving

**Deferred to:** G4, alongside burst scheduling and packed prefill.

G3 holds exactly one request with device-resident prefill progress; the rest
queue. Two arguments closed it. First, N in-flight prefills do not add
prefill throughput: the chunks stay serial on the one stream, so two 32K
prefills finish in the same total time but both with doubled TTFT, instead
of one fast and one normal. Second, and decisive, a half-prefilled sequence
holds KV pages, a GDN slot and conv taps that no tier can reclaim before G4,
so N active prefills multiply exactly the VRAM that cannot be freed.

The thing it *does* buy is fairness: a short prompt queued behind a long one.
That is a real want, and it is why this is G4 rather than never.

### 4. Priority preemption of a prefill, and an explicit suspend/resume primitive

**Deferred to:** G4.

G3 builds no suspend/resume primitive. Resuming a prefill needs no
mechanism — it is the absence of a scheduled next chunk — and with only one
active prefill there is no beneficiary of suspending it: pausing frees no
VRAM, so suspending the only prefill to start another leaves two
half-prefilled sequences and neither advancing.

Cancel is a different thing and *is* built: the in-flight chunk finishes,
then the sequence is aborted and its KV pages, GDN slot and conv taps
released. Cancel is abort, not suspend.

### 5. Snapshot, restore, and eviction of a half-prefilled sequence

**Deferred to:** G4 (the KV-RAM host tier).

The finding worth carrying forward is that the blocker is the tier, not the
state. Verified in the leaf during the session: `causal_conv1d_silu`
advances the sequence's own persistent conv slot in place, the GDN recurrent
slot likewise, KV pages are appended, and `seq->position` moves only after
the chunk's synchronization returns. All four are consistent together at a
completed chunk boundary, so **a chunk boundary is a valid whole-sequence
snapshot point for the state that exists today**.

This corrected a rule in `crates/core/src/gdn.rs` that was written when
prefill was atomic, so "mid-prefill" and "mid-chunk" were the same position.
G3 therefore records each completed chunk boundary with
`GdnState::checkpoint`, not `advance`, and G4 inherits the ability to evict
a half-prefilled sequence rather than having to re-open the invariant.

Two terms were separated in `CONTEXT.md` to keep this honest: **chunk
boundary** is a consistency property of the serving loop; **snapshot point**
is a permission granted to the host tier. They coincide today without being
the same property.

**Caveat that must be honoured:** G4's hq-e8-2b exact-key side store is
*new* sequence state. The snapshot-point property must be re-verified
against it when it lands; it is not inherited.

### 6. hq-e8-2b KV, and N=8 at long context

**Deferred to:** G4, as already planned. What the session added is the
number that makes it non-optional.

At this geometry a sequence-token costs 64 KiB of BF16 KV (16 GQA layers × 4
KV heads × 256 × K+V × 2 bytes). The pool is 65,536 tokens, about 4 GiB, and
it is a budget shared across all lanes rather than a per-lane reservation.
Eight lanes at full context would be ~20 GiB next to ~19 GB of weights.

So **N=8 in BF16 exists only at short contexts**, and the `N-lane
concurrency` term in `CONTEXT.md` becomes real at long contexts only with
hq-e8-2b. This is written into the G3 gate as a recorded inequality so that
nobody reads G3's concurrency as a promise G4 has not yet delivered.

Anticipating hq-e8-2b into G3 was explicitly considered and rejected: it is
the largest piece of G4, and pulling it forward repeats the mistake that was
just avoided with packed prefill.

### 7. Raising the KV pool above 65,536 tokens

**Considered and declined for G3.**

There is headroom: 32 GB of card, less ~19 GB of weights, 1.15 GiB of GDN
slots and the prefill scratch, leaves roughly 9–10 GB, so the pool could
reach ~140K tokens. It was declined because the G3 cells do not need it —
C=4 at 8,192-token prompts reserves 33,792 of 65,536 — and raising it would
spend VRAM margin without moving the gate metric.

One correction from the session is worth keeping: decode throughput is *not*
independent of context length. Attention rereads the whole KV cache per
token, so per-token time grows with context. The reason 8,192 was chosen for
the throughput cells is that it is a representative fixture that fits
comfortably and allows a clean live/live comparison, not that context is
free.

### 8. Deleting the per-token prefill route

**Deferred to:** reconsidered after G3. ADR 0016 had already named this "a
G3 decision".

Retained. G3 changes *who drives* the chunk loop, which is a bad moment to
retire an oracle. The route is not the only check available — leaf-driven
chunking can be compared against Rust-driven chunking directly — but it is a
third, independent oracle, and independence is what it is for.

### 9. Adaptive chunk-width policy

**Deferred to:** after the G3 measurement.

G3 introduces a serving prefill chunk width, constrained to be at most the
width the program scratch was reserved for at load. Narrowing at serving
time is free; widening is refused. Its default is the load width and its
real value comes from the gate run.

An adaptive policy driven by a latency budget was considered and rejected as
superstructure: the curve relating chunk width to chunk time has not been
measured, and #92 suggests it is not linear, so a controller would be tuned
against a guess.

Chunk width matters because it is the **only** dial that moves the gate's
p95: the p95 floor under a prefill is one chunk time plus one decode round.

### 10. Raising K, the decode rounds per prefill chunk, above 1

**Deferred to:** turned only if the C=4 cell fails.

K=1 ships. The session first claimed K cannot move p95 at all; that was
overstated and corrected. The exact behaviour: per cycle a lane sees `K-1`
short gaps and exactly one gap crossing a chunk, so the long gaps are the
top `1/K` of the distribution and p95 stays among them until `K >= 20` —
at which point it drops to a decode round, but at a 240% TTFT inflation. It
is a cliff, not a gradient, and the cliff is unaffordable.

So over the payable range K trades mean inter-token latency against TTFT and
leaves the gate metric alone. It is the first number to turn if C=4 misses.

**Consequence for the gate wording:** the functional anti-serialization
clause is stated K-agnostically — while a prefill is active, no decode-ready
lane waits more than one prefill chunk between two of its decode
opportunities — precisely so that raising K cannot fail the gate.
