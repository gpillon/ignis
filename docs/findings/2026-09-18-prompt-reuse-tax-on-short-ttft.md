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

## All four cuts, named

`take` in the prefill-job builder is `chunk_take` narrowed by `publish_point`
and then by `checkpoint_point`, and `publish_point` returns **a different
boundary on successive advances** — it walks `[retained_prefix_point,
publish_tokens]` in prompt order and yields the first one past what the
request already shares. So two functions produce three points, and with the
chunk width that is four cuts and five pieces. Production `kv_page_tokens` is
64 (`crates/runtime/src/cuda_leaf.rs:530`), which is why the boundaries below
land on multiples of 64:

| cut | at | why | piece measured |
|---|---|---|---|
| 1 | `serving_chunk_tokens` = 1,024 | ADR 0018, so a long prefill costs the decode lanes one chunk of latency | 1024 |
| 2 | `retained_prefix_point` — the system block end, floored to a page | #126/#188: publish the block a burst of siblings shares | 320 |
| 3 | `publish_tokens` — the opener floored to a page | #187: chain a second prefix over the block, so a conversation past its first page can still take a checkpoint | 64 |
| 4 | `checkpoint_point` — the generation opener itself | #186: the prompt checkpoint's state is the state *there* | 43 |
| — | the rest of the prompt | | 2 |

A claimant already shares cut 2, and still pays cuts 3 and 4 — which is the
three-call, 66 ms shape above.

**Cuts 3 and 4 are always less than one page apart**, by construction: one is
the opener floored to a 64-token page, the other is the opener. The piece
between them is 0..63 tokens and costs a full traversal.

## Pricing each cut by switching it off

Every cut behind an environment switch (`IGNIS_CUT_OFF`, an experiment kept on
this branch and marked not for merge), one server launch per configuration.
Two workloads, because they disagree:

**A stable system block with independent short queries** — parallel subagents
sharing tools, asking unrelated things:

| cuts off | cold system | **shared system, steady** | prefill calls per reusing request |
|---|---|---|---|
| none | 342.2 ms | **138.1 ms** | 3 |
| `capture` | 240.6 ms | **74.5 ms** (−46%) | 2 |
| `opener` | 229.8 ms | **57.7 ms** (−58%) | **1** |
| `opener` + `capture` | 226.2 ms | 57.8 ms | 1 |
| all three | 218.0 ms | **183.0 ms** — reuse stops working | 1024-token calls |

**A conversation that grows** — each turn carries every previous one, which is
what the chained opener publish (#187) says it exists for:

| cuts off | first turn | **turns 2+, median** |
|---|---|---|
| none | 438.2 ms | **206.0 ms** |
| `opener` | 246.0 ms | **232.5 ms** — worse |
| `capture` | 279.9 ms | **245.1 ms** — worse |

The two tables are the whole answer. On the first workload cuts 3 and 4 cost
58% of every reusing turn and buy nothing, because the shared prefix *is* the
system block and cut 2 alone carries it. On the second they pay for
themselves twice over, because the prefix grows past the block and only they
let turn N reuse turn N−1. **Every cut is load-bearing for a real shape**, and
removing any of them trades one workload against another.

## What the real trace says: 6.8%, not 58%

The two synthetic workloads above have ~81-token tails, where a cut is most of
the work. A recorded 157-request qwen-code session
(`.scratch/vram-analysis/trace-merged-system.jsonl`) does not look like that.
Over its first 60 requests, each prompt's longest shared prefix with an
earlier one is a **median 78.2%** — reuse has plenty to bite on — but the tail
left over is a **median 5,205 tokens** (p25 3,571, p75 12,645), and **not one
of the 59 tails is under a single 1,024-token serving chunk**.

Replayed in order with `max_tokens` forced to 1, so what is timed is the
prefill (`.scratch/prefill-2026-09-18/replay_prefill.py`):

| | calls | tokens | GPU | share |
|---|---|---|---|---|
| full chunks (≥512 tokens) | 490 | 496,176 | 120.6 s | **93.2%** |
| short calls (<512 tokens) — the cuts | 162 | 12,501 | **8.8 s** | **6.8%** |

**10.9 prefill calls per request: 8.2 full chunks and 2.7 cut tails.** A cut
tail is a median of 19 tokens and costs a median of 32.8 ms — more than the
19 ms fitted at 1K context, because a traversal at 95K context carries its
attention too (p90 112.7 ms).

So the cut tax on the load this engine is built for is **6.8% of prefill**,
about 146 ms per request. The 58% in the table above is an artifact of tails
short enough that the fixed cost is everything; real agent turns add thousands
of tokens, and the fixed cost is amortised over eight full chunks.

## Implications

- Any TTFT comparison against the reference should state which side of
  `--prompt-reuse` it was measured on, or measure both.
- **The optimisation is not removing a cut — it is making a cut cheap.** The
  switch-off table settles that: each cut earns its keep on at least one real
  workload, so dropping one is a trade, not a win. What is not earned is that
  *taking* a cut costs a whole extra traversal of the model.
- That leaves one direction rather than three: let the leaf publish or capture
  at an interior offset **without ending the traversal** — snapshot the mutable
  state there and keep going. **The prize on the real load is 6.8% of prefill**,
  ~146 ms a request, which is a poor trade against the leaf work it needs. The
  58% figure belongs to a synthetic shape, not to this engine's traffic.
- The `--prompt-reuse off` comparison at the top of this finding therefore
  measures the premium, not an available saving. The saving available without
  giving anything up is the traversal, not the reuse.
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

- Cost the interior-snapshot direction: what the leaf would have to expose for
  a prefill call to publish or capture mid-traversal, and whether GDN's chunked
  prefill materializes a usable intermediate state at all.
- Both workloads here are synthetic and sequential, one launch each. Before
  building anything, replay a real captured agent trace and see which of the
  two shapes it actually is — the answer decides how much the 38 ms is worth.
- Re-measure the reference live/live at 1,024 tokens, on both sides of
  `--prompt-reuse`, before attributing any of the 1.85x.
