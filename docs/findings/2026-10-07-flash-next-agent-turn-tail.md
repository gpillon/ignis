# A reused Flash-Next agent turn is three copy-bound traversals; a prefill lookahead at the router's top-k cuts it 12%

- Kind: experiment
- Status: current
- Observed: 2026-10-07
- Last verified: 2026-10-07
- Scope: serving / Flash-Next prompt reuse, the reused tail's traversals, the prefill lookahead width, expert copies
- Related: https://github.com/gpillon/ignis/issues/306 (item 5), spec [flash-next/05](../specs/flash-next/05-prompt-reuse.md) acceptance 6, spec [flash-next/03](../specs/flash-next/03-expert-residency.md), [the prefill chunk finding](2026-10-07-flash-next-prefill-chunk-and-decode-share.md), [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md), [the prompt reuse tax](2026-09-18-prompt-reuse-tax-on-short-ttft.md)
- Superseded by: none

## Question

A reused agent turn (a ~9K-token conversation plus a ~1.0K-token follow-up) took 2.2 s to its first token, against spec 05 acceptance 6's 1.6 s. Where does the tail's time go: expert copies, compute, the n-gram gather, or host gaps? Which cheap lever cuts the copies?

## Evidence

**Setup.**
- Build: release, branch `short-tail-306` from main 5548f55.
- Server: `make config MODEL=flash-next`'s flags (hq-e8-2b, 262,144 tokens, 3 lanes, 8192-token chunks, MTP off), with one change: `--kv-host-pool-bytes 1G`. The host plan refused make's 2G by 0.66 GB of free RAM. The arena is not on this path: every reused turn here restored from the device.
- Client: `.scratch/flash-next-306-307/harness/shorttail/` in the main checkout, verifyGPU's method. Thinking off, greedy, `max_tokens` 24, eight seeded conversations per leg.
  - Turn 1 is a cold ~8.3-8.7K-token prompt.
  - Turn 2 adds the reply and a ~1.0K-token follow-up: the reused tail.
  - Turn 3 adds ~100 tokens. It proves turn 2 left its checkpoint.
  - Metrics are scraped around turn 2.
- A/B: one load per leg, legs back to back, the same prompts in every leg.
- Profile: `nsys --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node`, one 35 s window. A traversal starts at each eager `embed_rows` launch (`traversals.py`).

**The tail is three traversals.** Reuse cuts a prompt twice:
- at its publish point, the opener's page floor (#126, #187);
- at the generation opener (#186).

So a tail runs [publish − cached, opener − publish, 4] tokens, e.g. [941, 63, 4]. Each piece is a whole 48-layer forward that streams its own experts.

**One tail before, [931, 22, 4] tokens, client TTFT 2.03 s** (ms):

| traversal | span | compute | expert copies | copies exposed |
|---|---:|---:|---:|---:|
| 931 tokens | 1,414 | 274 | 1,360 | 1,137 |
| 22 tokens | 394 | 98 | 380 | 292 |
| 4 tokens | 56 | 24 | 48 | 28 |

- 27 ms between the traversals, ~135 ms before the first. The latter holds HTTP, the template, the 10 ms restore and the n-gram gather.
- The publish-point piece costs more the wider it is:

  | width (tokens) | 7 | 13 | 20-22 | 36-42 | 51-63 |
  |---|---:|---:|---:|---:|---:|
  | span (ms) | 128 | 237-255 | 323-394 | 472-594 | 542-665 |

  Turn 2's TTFT sorts by this width, in verifyGPU's six reps and in these: 1.72-1.76 s at 1 token, 2.36-2.40 s at 63.
- Each tail moved 18-26 GB host-to-device (`ignis_expert_bytes_moved_total{phase="prefill"}`): almost the whole non-resident pool (a 37.8 GB pool, a 14.4 GB cache at this load).
- 26-31% of the prefetched projections were never used: 23-33K issued, 17-23K used.

**The lever: the prefill lookahead's width.** A chunk prefetches each token's top 16 of the next router, unbudgeted, though that router selects 10. Three legs, the same eight conversations:

| prefill lookahead width | turn 2 TTFT, median (range) | GB per tail, median | turn 3 TTFT | cold turn 1 TTFT |
|---|---|---:|---:|---:|
| 16 (before) | 2.12-2.13 s (1.72-2.40) | 23.1 | 1.14 s | 4.44 s |
| **10, the router's top-k** | **1.85-1.86 s (1.55-2.11)** | 19.8 | 0.90 s | 3.96 s |
| 0 (demand copies only) | 1.86 s (1.53-2.06) | 16.3 | 0.85 s | 4.86 s |

- 10 beats 16 on all eight paired conversations, by 0.17-0.30 s. Width 10 was measured twice:
  - first as a program-side variant;
  - then as residency's `prefill_lookahead_width`, the committed form.

  Both moved the same bytes per conversation. The committed A/B gives 2.120 → 1.863 s.
- Width 0 moves the fewest bytes. It loses the link/compute overlap, so a cold 8192-token chunk waits on its copies: the cold turn is 0.4 s slower than at 16.
- `ignis_expert_residency_stall_seconds_total{phase="prefill"}` rises at width 10, from 0.13-0.21 s per tail to 0.23-0.36 s. It counts only demand copies, and more of the copies are demand now. The copy time the card waits on is what fell.
- In every leg, every turn 3 claimed turn 2's checkpoint: cached = turn 2's prompt − 4.

## Finding

Observed:

1. **The reused tail is copy-bound.** Expert copies the card waits on are ~1.46 s of 2.03 s. Compute is ~0.4 s, mostly hidden under copies. The host takes ~0.16 s: gather, restore, HTTP and the gaps between traversals.
2. **A reused turn runs three traversals, and each re-streams its experts.**
   - The publish-point piece costs more the wider it is: 0.13 s at 7 tokens, up to 0.66 s at 60 (width 16).
   - The opener piece costs ~50-70 ms.
   - On the 27B a traversal cost ~19 ms fixed ([the prompt reuse tax](2026-09-18-prompt-reuse-tax-on-short-ttft.md)). On Flash-Next it costs its expert stream, so the same cuts take up to a third of the tail. That finding's "not worth the leaf-side interior-snapshot work" does not carry over.
3. **A width-16 prefill lookahead over-streams.**
   - Ranks 11-16 are mostly experts the next layer never reads.
   - Taking the router's own top-k cuts the tail's bytes by 14%, turn 2 by 0.26 s at the median, turn 3 by 0.24 s and a cold 8.5K prompt by 0.48 s.
4. **Spec 05 acceptance 6 is not met.**
   - 1.86 s at the median with a ~9K history, against 1.6 s: a gap of +0.26 s, down from +0.52 s on this harness.
   - The criterion's 30K history was not run. It can only add attention and indexer work to the tail.
   - A tail whose publish-point piece is short meets it: 1.55 s at 1 token.

## Implications

- **The lever left is the traversal count.** Folding the publish-point piece into the first traversal saves its cost: ~0.35 s at a median width at prefill width 10, inferred from turn 2's spread (1.55 s at 1 token, 2.11 s at 63). That brings the median near 1.5 s.
  - The cut exists because a publish needs the state image at the prefix's end (`seq_prefix.cu`).
  - The chained prefix at the opener's page floor (#187) is there to hold pages under the checkpoint. Its image serves only a claimant that diverges in the last ≤ 63 tokens.
  - A pages-only chained prefix, claimable only through the checkpoint above it, would drop the cut. That is an ADR 0029 amendment, not a tuning.
- **The opener piece** (4 tokens, ~50-70 ms) is not worth an interior snapshot of the GDN state.
- **Residency now has a prefill width.** It is the load parameter `prefill_lookahead_width` (policy `prefill_prefetch_width`), and Flash-Next loads 10. Decode keeps 16 under its byte budget.

## Limits and unknowns

- One harness: README-word prompts, thinking off, a ~9K history, not 30K. No swarm. One host, PCIe Gen 3 x16.
- Widths between 0 and 10 were not measured. 0 wins on short chunks (turn 3: 0.85 against 0.90 s) and loses on a cold 8192-token chunk. A width of 6-8, or one by chunk length, may sit between.
- The profile was taken before the change only. After it, the per-traversal split is inferred from the counters.
- KV-RAM arena 1 GiB instead of make's 2 GiB; no turn restored from it.

## Follow-ups

1. Publish the chained prefix at the opener's page floor without a state image, and drop the publish-point cut. This is a spec 05 / ADR 0029 change, worth ~0.35 s on a median reused turn.
2. A one-hold sweep of the prefill width (6, 8), and of a width by chunk length.
3. Acceptance 6 at the spec's 30K history once 1. lands.
