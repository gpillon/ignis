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
`docs/specs/runtime/03-serving-loop.md` and ADR 0018. Ten items below;
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
decode at all. This is named in `docs/REVIEW-2026-09-05.md` §6 as the
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

---

## From the G5 grilling session (2026-09-13, GitHub #66)

Source conversation: the design session that produced
`docs/specs/runtime/05-speculative-decoding.md` and closed #65 on the
G4 run-2 verdict. Four items below.

### 1. MTP as a drafter — deferred behind DFlash2

**Deferred to:** built only if short calls (classifier, tool-call turns) pay
the DFlash2 drafter's TTFT penalty visibly in real agent use.

The roadmap said "MTP first, then DFlash2". The session reversed it. The
reference's own clean depth discriminator (`SEPT-OPTIMIZAZION.md` §1,
2026-09-01, greedy, one request at a time, 512 tokens) reads MTP7-adaptive
150 / 104 / 79 tok/s against DFlash2-7 147 / 132 / 144 tok/s at 24K / 98K /
196K: level at 24K, +27% at 98K, +82% at 196K for DFlash2, which pays 0.5–5 s
of TTFT for it. Agentic prompts live at the deep end. Most of the work is the
shared substrate either drafter needs (verify traversal of k+1 columns per
lane, the accept kernel with its greedy and distribution-preserving branches,
KV rollback, the ReplaySSM fold, one graph per batch width at the verify
window, prefill feature taps); the drafter itself is the small part. Building
MTP first would exercise that substrate on the drafter that loses where it
matters. The G5 reference lane is therefore ninfer DFlash2-7 alone; MTP7 is
not measured.

What MTP would still buy: no drafter TTFT (the head is one layer fed by the
target's own hidden state), `--lm-head-draft`, and the adaptive width
controller, which DFlash2 in the reference does not have. The glossary keeps
the term; `CONTEXT.md` marks it deferred.

### 2. Rebuilding the DFlash2 drafter state on restore, instead of carrying it

**Deferred to:** re-opened if the host tier's restore and prefix-clone rate
in real agent use makes the +80 MiB per sequence the thing that limits how
many sequences the tier holds — the owner expects ignis to checkpoint and
restore far more often than the reference does.

The owner's first preference was to rebuild: no new snapshot sections, the
drafter re-derives its state on restore. The session recorded the fact that
decides it and chose to carry the state (option a), the reference's own
layout (sections `dflash_local` and `dflash_checkpoint`, header
`dflash_context_frontier`).

**The fact.** The drafter does not consume tokens. It consumes the target's
hidden states at layers 5, 19, 33, 47 and 61 (five 5120-wide features,
projected by `dflash2/feature_projection [5120, 25600]`) and keeps a
2048-token sliding BF16 KV window over them, plus a rewrite checkpoint copy of
that window. The reference keeps those features as chunk-scoped scratch and
never persists them (`program_impl.h:1111`). So "rebuild on restore" is not a
5-layer pass over the tail: it is a **64-layer target pass over the last 2048
tokens** to regenerate the features, then the drafter. That is a real
re-prefill on the restore path and on every device-side prefix clone.

**The numbers** (snapshot blob from
`docs/findings/2026-09-12-sequence-snapshot-transfer-cost.md`, hq-e8-2b,
9 KiB/token; drafter window from `src/core/cyclic_kv_cache.cpp`, BF16,
5 layers × 2048 × 8 heads × 128 × K+V = 40 MiB, ×2 with the checkpoint):

| context | blob today | of which GDN slot | blob carrying the drafter | delta |
|---|---:|---:|---:|---:|
| 128 tokens (floor) | 149 MiB | 144 MiB | 229 MiB | +54% |
| 24K | ~365 MiB | 144 MiB | ~445 MiB | +22% |
| 40,960 (measured) | 508 MiB | 144 MiB | 588 MiB | +16% |
| 98K | ~1.03 GiB | 144 MiB | ~1.11 GiB | +8% |
| 196K | ~1.92 GiB | 144 MiB | ~2.0 GiB | +4% |

| cost | carry (a) | rebuild (b) |
|---|---:|---:|
| per restore | +7 ms at the measured ~11 GB/s | ~0.3 s at 24K, ~0.7 s at 196K (a 2048-token target re-prefill at the reference's prefill rate for that depth) |
| per device prefix clone | ~0.1 ms | the same re-prefill |
| host tier | +80 MiB per resident snapshot against the 2 GiB default (`--kv-host-pool-bytes`; the machine has 64 GB) | nothing |
| VRAM | identical: 8 lanes × 80 MiB = 640 MiB for the drafter's own window either way | identical |

Today's 2 GiB default holds ~5 snapshots at 24K, 1 at 98K, 0 at 196K; with
the drafter carried, ~4 at 24K. The default budget is the actual limit on how
many sequences the tier holds, not the drafter's share of a blob.

**Why carry.** The ratio is about 100× on every restore and clone, and (b)
puts its cost exactly on the subagent-burst path where restores and clones are
the norm. The owner's principle for ignis — agentic work first, small raw-speed
sacrifices are acceptable when they buy a lot for that use — points at (a)
here, not (b): the ~0.3 s is paid per restore by the agent waiting, and the
80 MiB is paid by a host budget that can be raised.

**What re-opens it.** If a later measurement shows the tier evicting because
of blob size rather than the budget, the trade is ~0.3 s of restore latency
against 80 MiB of host RAM per snapshot, and the snapshot blob is versioned
(ADR 0024), so dropping the two sections is a format bump, not a redesign.
The number to write down first is the real restore-per-minute rate under an
agent session, which nobody has measured.

### 3. The G4 `sub` cell at 0.969 — accepted, not fixed before G5

**Deferred to:** an optimization ticket, not blocking #66.

Run 2 of the G4 gate (`.scratch/g4-run2/`, session `g4-20260913T163557Z`)
returned `main` 1.264, `sub` 0.969, aggregate 1.013, both needles retrieved,
the G3 cells inside their band. The `sub` miss is real (both launches agree,
0.975 and 0.962, spreads 2–3%) and the owner accepted it as optimization on
the decode round, which G5 rewrites anyway. #65 closed on the recorded
verdict with the gap filed as **#148**, an optimization ticket that blocks
nothing, the way #64 closed with #110. Where the 3% most likely lives is in
#148's own text: admission and the single prefill lane under a burst, not
raw decode speed (`main` wins by 26% in the same run).

### 4. Gate sessions per ticket

**Deferred to:** never; the phase gate runs once, at the end of the phase.

The G3 and G4 sessions (two launches per engine, pooled cells, thresholds
inside launch noise, eight harness issues) cost more than the engine work
they checked. The ≥ 99% objective stays as the phase gate, pooled over two
launches per engine (ADR 0021 stands). Per ticket, the check is the smallest
instrument that catches a large error: the op test, and for the speculative
substrate the greedy equivalence spec-on == spec-off on the existing
canaries. No gate cell, no extra launch, no sub-1% comparison inside a phase.
