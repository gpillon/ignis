# DFlash2 on real agent turns commits 3.5 tokens a round and no lever buys +8%: the drafter loses 1.5 tokens a round, one-hot drafting 3.1, and a token tree's +8-22% a round nets out near zero

- Kind: experiment
- Status: current
- Observed: 2026-09-24
- Last verified: 2026-09-24
- Scope: speculative decoding / DFlash2 acceptance under agent sampling, draft selection, verify width cost
- Related: [spec runtime/08](../specs/runtime/08-dflash2-sampled-acceptance.md) (GitHub #267),
  [copy drafting on agent traces](2026-09-24-copy-drafting-on-agent-traces.md) (the turns),
  [reasoning effort on coding tasks](2026-09-24-reasoning-effort-on-coding-tasks.md),
  [decode round anatomy](2026-09-18-decode-round-anatomy.md),
  [hq attention at long context](2026-09-24-hq-attention-at-long-context.md) (GitHub #268 is
  making that attention cheaper; the width-cost curve below predates it),
  the replay's seams, driver and scripts (`tools/dflash2-replay/`, the examples
  `dflash2_replay` and `verify_width_cost`, `dflash2_replay_seams_gpu.rs`) on branch
  `dflash2-sampled-acceptance-267`, from 8e818f3
- Superseded by: none

## Question

DFlash2 commits 5.43 tokens per verify round on greedy coding prompts and ~3.3 on real
agent traffic, which samples at T = 1.0, top_p 0.95, top_k 20. How much of the drop is
**inherent** to sampling (a one-hot draft is accepted with probability p_target(draft),
so even the target's own argmax is often rejected), and how much is the **drafter**? And
which of the spec's three levers clears **+8% aggregate tok/s** at the lane count it
targets?

1. Adaptive per-lane extent, at 8 lanes.
2. A sampling-aware selector, at 1 lane.
3. Multi-candidate (tree) verify, at 1 lane.

## Evidence

### Method: a teacher-forced replay through real verify rounds

- **Turns.** 200 turns from the copy-drafting dataset, drawn with seed 267. Each stratum
  (source x kind) has 25 or 37-38 turns. "Think" means at least 50% of the turn's output
  is thinking.
  - The sources are A (the opencode #191 trace, complete request bodies), B (qwen-code main
    chats) and C (qwen-code subagents). B and C lack the qwen-code system prompt and tool
    schemas.
  - Every turn is joined to its own ninfer request-log DFlash2 stats. Every one of them ran
    at T = 1.0, top_p 0.95, top_k 20.
  - Contexts: median 49K tokens, mean 58K, up to 148K (turns past 150K were left out).
    In all: 116 token streams, 6.47 M prefilled context tokens and 208,860 replayed output
    tokens (61,692 rounds).
- **Load.** DFlash2-7, hq-e8-2b KV, 1024-token prefill chunks, verify graphs. Branch
  `dflash2-sampled-acceptance-267` over main 81fdc37, whose only additions are test-only.
  RTX 5090, exclusive (the swarm's GPU lock).
- **The walk.**
  - Everything before a turn is prefilled, and its last draw is forced to the turn's first
    recorded token.
  - Each round, the drafter's proposals, its 16 candidates per column and the selector's
    transition scores are read (`dflash2_readout::propose`).
  - A **teacher-forced** verify round (`step::decode_program_verify_forced`) then runs the
    ordinary pass, but commits the recorded text: the leading drafts it agrees with, then
    the recorded token.
  - The round's target logits are read for every column (`dflash2_readout::verify_logits`).
- **Why it is exact.** The vendored accept takes a one-hot draft exactly when the token it
  samples equals it. A recorded text therefore fixes a live run's rounds, and replaying it
  gives them back. `crates/core/tests/dflash2_replay_seams_gpu.rs` holds forced rounds to
  live rounds **bit for bit**, greedy and sampled, at widths 1 and 2.
  - The readout names the drafts the round verifies: 0 disagreements in 61,692 rounds.
  - It writes nothing: rounds with reads equal rounds without, and the sequence snapshot
    does not move.
- **Estimators.** Along the recorded chain, the probability that any one-hot proposal x is
  accepted is Rao-Blackwellized:

      E[accepted] = sum_i prod_{j<i} 1[r_j == x_j] * p_rec(x_i)

  Here p_rec is the target's sampling distribution at the recorded position, taken from
  the vendored sampler's rule: top 20, inclusive top-p, renormalized. The round's own
  drafts additionally get the **exact** expectation, from the draft-conditioned verify
  columns.
- **Aggregation.** Per-cell values are pooled with the traffic's weights (each cell's share
  of the dataset's 18.26 M output tokens).

### The replay reproduces ninfer's live rounds on the same turns

| source | turns | replay tok/round | ninfer tok/round (live) | ratio |
|---|---:|---:|---:|---:|
| A (complete contexts) | 50 | 3.016 | 3.030 | 0.995 |
| B | 75 | 3.508 | 3.543 | 0.990 |
| C | 75 | 3.400 | 3.326 | 1.022 |
| all | 200 | 3.386 | 3.361 | 1.007 |

P(accepted >= i), i = 1..7:

| source | engine | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| A | ninfer, live | .682 | .458 | .311 | .218 | .157 | .115 | .088 |
| A | replay | .678 | .450 | .310 | .218 | .155 | .115 | .089 |
| B | ninfer, live | .739 | .546 | .405 | .303 | .230 | .178 | .141 |
| B | replay | .733 | .538 | .396 | .297 | .227 | .177 | .141 |

Per turn: median |log ratio| 0.03-0.04; 60% of turns within 5%, 85% within 10%.

The missing system prompts of B and C do not show at this level. The target puts 0.9% of
recorded tokens outside its own sampling support, which reflects the engines' numerics
and the rebuilt contexts.

### Round level: what a round commits, and the ceilings

Committed tokens per round, traffic-weighted:

| rule | tok/round | vs DFlash2 |
|---|---:|---:|
| DFlash2 as served (realized walk) | 3.455 | — |
| DFlash2, exact expectation from the verify columns | 3.470 | +0.4% |
| the same drafts under a **greedy** verify | 3.851 | +11% |
| selector walk without its pairwise term (candidate 0 per column) | 3.341 | −3% |
| **one-hot oracle**: propose the target's argmax | 4.924 | +42% |
| best single candidate of the lattice (oracle selection) | 4.308 | +25% |
| sampling-aware selection: expected-length walk, drafter T = 0.5 / 1 / 2 | 3.468 / 3.475 / 3.479 | +0.4 / +0.6 / +0.7% |
| verify every top-2 / top-4 / all 16 candidates per column (unbounded tree) | 3.978 / 4.565 / 5.568 | +15 / +32 / +61% |

- **The split.** Out of the 8 tokens a round could commit, 4.53 are lost.
  - The drafter accounts for 1.47 of them: the argmax oracle's 4.92 minus DFlash2's 3.46.
  - One-hot drafting under this sampling accounts for the other 3.08. A perfect
    argmax drafter still has its first draft rejected 18% of the time.
Per stratum (DFlash2 as served / greedy verify of the same drafts / one-hot oracle):

| source | think | content |
|---|---|---|
| A | 2.89 / 3.36 / 4.08 | 3.38 / 3.68 / 5.03 |
| B | 3.38 / 3.77 / 4.89 | 4.08 / 4.37 / 5.65 |
| C | 3.27 / 3.69 / 4.62 | 4.13 / 4.45 / 5.76 |

- **The greedy verdict.** On the same rounds, a greedy verify would commit 3.85. The step
  from 5.43 (canary coding prompts, greedy) to ~3.4 is mostly the traffic, not the
  sampling rule: the sampled accept costs 10% against greedy verification of the same
  drafts.

### Per draft column and per output kind (columns reached, recorded chain)

| column | reached | p(draft) | p(argmax) | inherent 1 − p(argmax) | drafter p(argmax) − p(draft) | draft = argmax | top-2 mass | top-4 | top-16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d1 | 61,692 | .720 | .819 | .181 | .099 | .790 | .835 | .906 | .968 |
| d2 | 44,111 | .718 | .846 | .154 | .128 | .769 | .803 | .876 | .952 |
| d3 | 31,570 | .729 | .869 | .131 | .140 | .766 | .791 | .865 | .945 |
| d4 | 22,990 | .749 | .890 | .110 | .141 | .780 | .796 | .864 | .942 |
| d5 | 17,193 | .757 | .900 | .100 | .143 | .783 | .797 | .864 | .942 |
| d6 | 13,025 | .780 | .917 | .083 | .137 | .802 | .811 | .872 | .945 |
| d7 | 10,153 | .796 | .928 | .072 | .132 | .814 | .819 | .881 | .948 |

| kind | columns | p(draft) | p(argmax) | inherent | drafter | draft = argmax |
|---|---:|---:|---:|---:|---:|---:|
| thinking | 146,312 | .705 | .830 | .170 | .125 | .763 |
| prose | 14,096 | .735 | .886 | .114 | .151 | .767 |
| tool-call syntax | 9,253 | .944 | .987 | .013 | .043 | .948 |
| tool arguments | 12,947 | .784 | .939 | .061 | .155 | .797 |
| edit/write arguments | 17,936 | .844 | .955 | .045 | .111 | .856 |

"Top-k mass" is the target's probability on the drafter's top-k candidates of that
column.

- Thinking, 73.5% of agent output, is where both losses are largest.
- The drafter's loss is largest on tool arguments and prose. It is not concentrated in
  late columns: from d2 on, its share is about flat at 0.13-0.14.

### The verify's width-cost curve (lanes each at the same context)

Median round in ms, two independent launches per cell, which agree within ~2% (ADR 0021).
Width 1 is the plain decode round, and width w = k + 1 is DFlash2-k. Measured on main's
kernels, before #268.

| lanes | context | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 K | 13.1 | 15.3 | 15.8 | 16.7 | 16.3 | 16.7 | 16.8 | 17.0 |
| 1 | 16 K | 15.6 | 17.9 | 18.3 | 19.2 | 18.9 | 19.6 | 19.5 | 19.5 |
| 1 | 32 K | 16.0 | 18.2 | 18.8 | 19.6 | 19.4 | 20.0 | 20.0 | 20.0 |
| 8 | 4 K | 17.7 | 23.0 | 25.0 | 27.1 | 29.6 | 31.2 | 34.0 | 35.8 |
| 8 | 16 K | 29.8 | 35.7 | 36.9 | 39.1 | 40.7 | 43.2 | 45.1 | 46.5 |
| 8 | 32 K | 41.3 | 47.1 | 48.8 | 51.9 | 53.3 | 55.8 | 57.9 | 59.5 |

At one lane, widths 2 to 8 differ by 10%. At 8 lanes, width 8 costs 1.56x width 2 at
4K, but only 1.26x at 32K, where attention dominates the round.

### Lever 1: adaptive extent at 8 lanes

The inputs are the exact per-round survival curves and the curve above, with 8 lanes of
independent rounds drawn with the traffic weights. Gains are against every round at
window 7.

| policy | 4 K | 16 K | 32 K |
|---|---:|---:|---:|
| best fixed window for everyone | +4.3% (k = 5) | 0 (k = 7) | 0 (k = 7) |
| **oracle**, one window per round for all lanes (a verify graph per window) | +8.0% | +3.1% | +2.1% |
| recent acceptance, one window per round (the widest lane's) | −4.0% | — | −5.7% |
| oracle, a ragged verify (each lane pays its own columns) | +27.7% | +16.2% | +12.8% |
| recent acceptance, ragged | +5.9% | — | −0.7% |

- **The fixed window.** At 4K and 8 lanes window 5 beats 7 by 4%; from 16K on, 7 is the
  best fixed window, and agent contexts are longer than that.
- **The lever as the spec states it** (one more term in the host's extent clamp) changes
  no cost. The verify graph computes every lane's k + 1 columns whatever its extent.
- **What it could save.** Only the round's width can save anything. Even an oracle that
  knows every lane's exact survival buys +2-3% at agent contexts and +8% at 4K.
- **The ragged verify.** Its large oracle numbers belong to a different kernel design,
  per-lane column counts inside the verify graph. A recent-acceptance policy recovers
  none of them at 32K.

### Lever 2: sampling-aware selection

The expected-length walk is a dynamic program over the lattice with the drafter's own
conditional q(succ | pred) = softmax(scores / T). It is the rule that maximizes expected
accepted length under the drafter's belief, and it is exactly the greedy walk as T → 0.

It changes the path in 7-28% of rounds (the first draft in 1.4-6%) and gains +0.4-0.7%.
The +25% headroom of oracle selection within the lattice exists, but only with the
target's knowledge. The drafter's scores do not carry it.

### Lever 3: a token tree from the drafter's own lattice (priced, not built)

The tree is built best-first by the drafter's path probability: children are the next
column's 16 candidates, q from the transition scores. It is scored along the recorded
chain; multi-draft rejection accepts a child with the target's mass on the child set.
12,000 rounds, traffic-weighted.

| tree | T = 1 | T = 2 |
|---|---:|---:|
| chain of 7 (today) | 3.453 | — |
| 7 nodes (the same 8 columns) | 3.682 (+6.6%) | 3.737 (+8.2%) |
| 15 nodes (16 columns) | 4.116 (+19.2%) | 4.205 (+21.8%) |
| 31 nodes (32 columns) | 4.435 (+28.4%) | 4.534 (+31.3%) |

- **Net of width, 15-node tree at 1 lane.** The width cost comes from the curve, bounded
  below by the 8-lane per-column slope and above by the 2-lane width-8 round. The net is
  about +8% at 4K and −6% to +11% at 32K.
- **Net of width, 31-node tree.** Net ≤ +2%.
- **Net of width, 7-node tree.** +7-8% before its own costs.
- **Per-branch GDN state, a rough estimate (not measured).** Every branch point needs its
  own copy of the lane's recurrent state. A device clone of the lane's 148 MiB of mutable
  state costs ~0.25 ms ([device prefix clone cost](2026-09-12-device-prefix-clone-cost.md)),
  and the chain's ReplaySSM records cost 0.48 ms per 8-column round
  ([decode round anatomy](2026-09-18-decode-round-anatomy.md)), a floor the tree's own
  nodes also pay. Assuming 1-3 branch points for 7 nodes and 4-8 for 15, that is about
  0.5-1 ms and 1-2 ms per round: 2.5-5% and 5-10% of a ~20 ms one-lane round. Tree masks
  in the hq verify attention come on top.
- **Rough net.**

  | tree | net of width | net of width and GDN state |
  |---|---|---|
  | 7 nodes | +7-8% | about +3-5% |
  | 15 nodes, 4K | about +8% | about −2 to +3% |
  | 15 nodes, 32K | −6 to +11% | about −16 to +6% |

  All below the gate.
- **T = 2 was picked on the same rounds**, so its extra point is slightly optimistic.

## Finding

Observed:

- **DFlash2 on real agent turns.** It commits 3.46 tokens per round, traffic-weighted
  (3.39 unweighted over the 200 turns). The teacher-forced replay reproduces ninfer's live
  rounds on the same turns to within 1-2% per source.
- **The split of the loss** from a perfect 8:
  - 3.08 tokens are inherent to one-hot drafting under T = 1 / top-p 0.95 / top-k 20.
    Even the target's argmax is accepted only 82-93% of the time per column.
  - 1.47 tokens are the drafter's. Its draft is the target's argmax 77-81% of the time.
- **Greedy verification** of the same drafts would commit 3.85. The drop from the 5.43
  greedy canary figure is mostly the traffic, not the sampled accept.
- **Lever 1** (adaptive extent at 8 lanes): its oracle ceiling is +2-3% at agent contexts
  and +8% at 4K, and the recent-acceptance policy loses throughput. Per-lane extents alone
  save nothing: the verify computes every lane's columns.
- **Lever 2** (sampling-aware selection): +0.4-0.7%.
- **Lever 3**, token trees from the existing lattice: gross +7-8% a round at today's
  width and +19-22% at 16 columns. Net of the 16-column width, that is about +8% at 4K
  and between −6% and +11% at 32K. With a rough estimate of per-branch GDN state, the
  7-node tree nets about +3-5% and the 15-node tree about −16% to +6%.

Inferred:

- **Gate result.** No lever clears the +8% bar as the spec defines it. Levers 1 and 2 fail
  outright. Lever 3's gross ceiling clears it, but once width and branch state are paid it
  nets roughly zero to +5%, and it is the expensive lever.
- **What would move the number.** The remaining headroom is the drafter's 1.5 tokens a
  round (a better drafter or a head retrained under sampling; out of scope) and more
  verified candidates per column (lever 3). The host-side levers leave nothing to take.

## Implications

- #267 closes on this finding: none of the three levers is built (owner, 2026-09-24).
- **Lever 3 stays an owner decision for later**, on these numbers. Were it revisited:
  - The 7-node tree is the variant to price first: the same verify width, +7-8% a round
    gross.
  - Its prerequisites are per-branch GDN state in the verify (ReplaySSM along tree paths)
    and a tree mask in the hq verify attention, under a spec amendment.
  - The width cost should be re-measured after #268, which makes the 16-column tree
    cheaper at long context.
- **A ragged verify** (per-lane column counts) is worth +13% at 32K and +28% at 4K only
  to a clairvoyant policy. It is not a lever to fund on this evidence.
- The replay on branch `dflash2-sampled-acceptance-267` pairs any future drafter or
  verify change against these same 61,692 rounds: rebase it, rerun the driver and compare
  `per_round.npz`.

## Limits and unknowns

- **Context.** B and C contexts lack the qwen-code system prompt and tool schemas (~18-36K
  tokens). Their per-source match with ninfer's live rounds (0.99, 1.02) says it barely
  matters to acceptance. Source A is complete.
- **The recorded text** was sampled by ninfer, not ignis, so the coupling is exact only up
  to the engines' numerics. 0.9% of recorded tokens fall outside ignis's sampling support.
- **Lever 1 approximations:**
  - The first k drafts are assumed not to depend on the window k. At a smaller window the
    drafter's block is shorter; this was not measured.
  - Lanes are independent draws.
  - The ragged cost is interpolated on the 8-lane curve, so it charges the drafter as if
    every lane's columns could shrink.
- **Lever 3:**
  - The trees use only the lattice's per-column candidates, ranked by the pairwise scores.
  - A tree node's children at a column are the same 16 candidates for every parent.
  - Real tree costs (GDN branch state, masks, commit of the winning branch) are
    unmeasured; the GDN figure above is an estimate from two other findings' numbers and
    an assumed count of branch points.
- **Width cost.** Measured once per context at 1 and 8 lanes with every lane at the same
  context, on main's kernels. #268's hq verify attention work will lower the long-context
  rows.
- **Scope.** The replay runs one lane. The 8-lane numbers combine independent single-lane
  rounds with the measured 8-lane cost.

## Follow-ups

- None filed. Lever 3 is recorded here for a later owner decision.
- If the width cost matters again after #268: rerun `verify_width_cost` (two launches)
  from the branch, then `lever1.py` and the tree net-of-width estimate.
- Any drafter change (retraining is out of scope here): measure it with the same replay.
  The drafter's 1.47 tokens a round is its ceiling.
