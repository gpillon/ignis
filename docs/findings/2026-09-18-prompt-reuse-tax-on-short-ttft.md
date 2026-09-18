# Cross-request reuse costs 37% of a 1,024-token TTFT, because it cuts the prompt into three traversals

- Kind: experiment
- Status: current
- Observed: 2026-09-18
- Last verified: 2026-09-18
- Scope: serving / prefill chunking, cross-request state reuse, the G2 TTFT cell
- Related: [ADR 0018](../adr/0018-chunk-level-prefill-decode-interleaving.md),
  [ADR 0029](../adr/0029-cross-request-state-reuse.md),
  [ADR 0015](../adr/0015-g2-live-live-cold-prefix-gate.md),
  [#126](https://github.com/gpillon/ignis/issues/126) (prefix publish point),
  [#186](https://github.com/gpillon/ignis/issues/186) (prompt checkpoint),
  [Vision TTFT live/live](2026-09-16-vision-ttft-live-live.md) (the 1.85x this
  bears on),
  [Decode round anatomy](2026-09-18-decode-round-anatomy.md)
- Superseded by: none

**Hardware:** RTX 5090, exclusive card (ADR 0006).
**Engine:** production defaults otherwise — `hq-e8-2b`, `--prefill-chunk 1024`,
`--spec dflash2 --draft-tokens 7`.
**Instrument:** `ignis-bench ttft --cells 1024 --samples 12` (all cold), with
`IGNIS_CHUNK_PROFILE` on. **Raw data:** `.scratch/prefill-2026-09-18/`.

## Question

TTFT at 1,024 tokens is where ignis is reported 1.85x the reference. The model
traversal for 1,024 tokens is about 30 ms of kernels. Where does a 204 ms TTFT
go?

## Evidence

The chunk profiler answers it directly: **the request is prefilled as three
separate leaf calls**, not one.

| | prompt-reuse **on** (default) | prompt-reuse **off** |
|---|---|---|
| prefill calls per request | **3** — 960, 60, 4 tokens | **1** — 1,024 tokens |
| device time per request | 159.4 ms | 111.8 ms |
| **TTFT, median of 12 cold samples** | **204.3 ms** | **129.6 ms** |

Each call is a full traversal of all 64 layers, so each re-streams the weights.
That is why the tails are not cheap: 119.3 ms for 960 tokens, **26.8 ms for 60,
19.0 ms for 4**. Fitting those three points gives a fixed cost of about **19 ms
per traversal** and about 0.1 ms per token — so the 64 tokens in the two tails
cost 45.8 ms, 29% of the prefill's device time for 6% of its tokens.

The cut is deliberate and the code says so, at `crates/core/src/concrete.rs`
(the `take = cut_at(...)` in the prefill-job builder):

> P4-10 (GitHub #126): a request that will publish a prefix is cut at its
> publish point, even mid-prompt. […] GitHub #186: and cut again at the
> generation opener, for the same reason […] **this second cut costs one short
> chunk (typically a few dozen tokens)** and only on the tick that actually
> takes the checkpoint.

The token count in that comment is right — the tails are 60 and 4. **The cost
estimate behind it is not.** It reads as though a short chunk is a cheap chunk;
a chunk's cost is dominated by streaming the weights, so a 4-token chunk costs
19 ms, 16% of what a 960-token chunk costs, for 0.4% of its tokens.

TTFT falls by more than the device time does (−74.7 ms against −47.6 ms). The
remaining ~27 ms is the host side of the same cause: three scheduler advances
instead of one, each with its own synchronization and bookkeeping.

## Finding

**Observed.** With cross-request reuse on — the default — a 1,024-token request
pays **74.7 ms of extra TTFT, 37% of the total**, to be reusable. The mechanism
is not overhead in the usual sense: it is the prompt being cut at the publish
point and again at the generation opener, which turns one traversal of the
model into three.

**Observed.** The tax is a fixed cost per extra traversal (~19 ms), not a
proportional one, so it is invisible on a long prompt and dominant on a short
one. On the 70,368-token prompt the same three-way cut appears (68 chunks of
1,024, then 704, 30, 2) and is under 2%.

**Inference.** The reuse machinery is a bet: pay ~75 ms now so a later sibling
can skip ~160 ms. For the agentic load the north star names — many requests
sharing a system prompt — that bet plausibly pays. **The G2 TTFT cell is
cold-prefix by construction (ADR 0015), so it is a measurement in which the
bet can never pay and the premium is always charged.** Part of the reported
1.85x is therefore not a deficiency but a configuration the benchmark is built
to penalise.

**Not established.** How much of the 1.85x this accounts for. The reference's
TTFT was not re-measured here, and no live/live run was made with reuse off.

## The other side of the bet, measured

Reuse anchors on the **system block**: `Request::publish_point`
(`crates/core/src/request.rs:294-305`) walks the retained-prefix point and the
publish point in prompt order, and the comment there says why — "the system
block ends before the last generation opener". A first attempt at an "agentic"
arm put the shared text inside one big `user` message and collected nothing;
**that was the test being wrong, not the engine.** Modelled correctly — one
shared system block, a new user turn each time — it collects, and handsomely:

| shape | TTFT |
|---|---|
| cold system, nothing to claim | 288.4 ms |
| shared system, first turn (pays the premium) | 275.8 ms |
| **shared system, every later turn** | **124.0 ms** (−57%) |
| identical prompt repeated | 54.1 ms |

So the bet pays after a single reuse, and for a conversation it is not close.

**But the premium is charged on every request, including the ones that
collect.** The chunk profiler on that run:

| request | prefill calls | tokens per call | GPU |
|---|---|---|---|
| cold system | **5** | 1024, 320, 64, 43, 2 | 231 ms |
| claiming the prefix | **3** | 64, ~15, 2 | **66 ms for ~81 tokens** |

A claimant skips the 960-token system block — that is the win — and then pays
three full traversals for an 81-token tail, 66 ms where one traversal is about
21 ms. At ~19 ms of fixed cost per traversal, **every request carries roughly
two extra traversals, about 40 ms, whether it reuses or not**: 36% of a
reusing turn's 124 ms, and the same again on top of a cold one.

## Implications

- Any TTFT comparison against the reference should state which side of
  `--prompt-reuse` it was measured on, or measure both.
- **The optimisation is the cut, not the reuse.** Reuse earns its keep; what
  does not is paying two extra fixed-cost traversals to take the cuts, on
  every request, including the claimants that are supposed to be the cheap
  ones. Sized: about 40 ms per request, 36% of a reusing turn.
- Directions, none yet costed: align the publish point with the chunk boundary
  so the cut lands where a traversal was ending anyway; let the leaf snapshot
  mutable state at an interior offset so one traversal can publish at a point
  inside it; or refuse the cut when what remains is too short to be worth its
  own traversal.
- The comment in `concrete.rs` should carry the measured cost of a short chunk
  rather than its token count, so the next reader prices the cut correctly.

## Limits and unknowns

- One prompt length (1,024 tokens), one sample set of 12, one launch per
  configuration. The effect is 4x the spread between the two configurations'
  medians, but no variance was established within either.
- The 19 ms fixed cost per traversal is fitted from three points of one
  measurement, not measured directly.
- Whether the tails are the publish point and the opener specifically was read
  from the code, not confirmed by instrumenting which cut produced which span.
- The hit side is measured at one system-block length (~960 tokens) and one
  tail length (~64), sequentially, on one launch. Concurrency was tried only
  on the wrongly-shaped arm.
- Which cut produced which span is still read from the code rather than
  instrumented; the five-call shape of a cold system request is not fully
  accounted for by the two cuts this finding names.

## Follow-ups

- Account for all five prefill calls of a cold system request; this finding
  names two cuts and sees four.
- Cost the three directions above against the ~40 ms they would recover.
- Re-measure the reference live/live at 1,024 tokens, on both sides of
  `--prompt-reuse`, before attributing any of the 1.85x.
