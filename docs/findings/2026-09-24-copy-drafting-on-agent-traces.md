# Copy drafting at width 8 buys 0–8% on real agent traffic: thinking dominates the output and DFlash2 already rides the copyable spans

- Kind: experiment
- Status: current
- Observed: 2026-09-24
- Last verified: 2026-09-24
- Scope: speculative decoding / draft source selection (DFlash2 vs prompt-lookup / suffix copy)
- Related: the 2026-09-19 n-gram study (conditional memory in hidden states, not
  prompt-lookup drafting; branch `ngram-study`,
  `docs/findings/2026-09-19-codebase-as-ngram-memory.md` there),
  [decode round anatomy](2026-09-18-decode-round-anatomy.md),
  `.scratch/sota-research-2026-09-24/SINTESI.md` §2.3 (items A3/B1),
  raw material in `.scratch/copy-drafting-2026-09-24/`
- Superseded by: none

## Question

Say a model-free copy drafter (prompt-lookup or suffix match on the lane's own context)
**replaces** DFlash2's 7 proposals inside the same width-8 verify, one source per round.
The copy source is taken when its match is confident, and DFlash2 is used otherwise.
How much would that raise committed tokens per round and tok/s on the coding-agent
traffic Ignis actually serves?

## Evidence

**Traces.** 13,102 real assistant turns with 18,256,821 output tokens, each with its
full rendered context (43.9 M stream tokens):

| source | turns | output tokens |
|---|---|---|
| opencode session recorded for #191 (`bench/traces/reuse-191-trace.jsonl`; outputs recovered from later requests' history) | 166 | 87 K |
| qwen-code main chats produced by the local qwen3.8-27b (172 sessions, 2026-08-27..09-19) | 6,570 | 9.61 M |
| their subagent transcripts | 6,366 | 8.56 M |

- Rendering follows the artifact's chat template (thinking preserved, XML tool calls);
  tokenization uses the artifact tokenizer.
- Reconstructed outputs have exactly the server-reported completion token count in
  ≥95% of turns (median ratio 1.000).
- 3,838 turns (5.02 M tokens) are joined to their own ninfer request-log DFlash2 stats
  (rounds, accepted, accepted-per-position). The join key is completion tokens, prompt
  tokens and a timestamp within 15 s.

**Output composition.**

| category | share |
|---|---|
| thinking | 73.5% |
| edit/write arguments | 10.2% |
| other tool arguments | 6.9% |
| prose | 5.8% |
| tool-call syntax | 3.6% |

**DFlash2 on this traffic** (ninfer, sampled T=1.0 / top_p 0.95 / top_k 20):
3.34 tok/round pooled over the joined turns, against 5.43 greedy on coding prompts.

**Method.**

- For each output position the drafter proposes the up-to-7 tokens that followed the
  most recent earlier occurrence of the longest matching suffix (n up to 64).
  Acceptance is the prefix match against the recorded output.
- Recorded outputs are samples and the vendored verify accepts a deterministic draft
  with probability p(draft), so this is an exact coupling: no greedy/sampled bias for
  the copy source.
- Each turn is walked in rounds under pick-one rules (min match length n, a
  SuffixDecoding-style frequency score, and an oracle).
- Round cost: 15.8 ms per verify, with the assumption that a copy round saves 2.5 ms
  of drafter work. A saving of 0 is also reported.

**Per position.**

- Some earlier match exists for 97% of positions. The first copied token is right in
  50%, and 2.04 drafts are accepted on average.
- With L≥8 (26% of positions): 4.99 accepted, 7/7 in 59% of them.
- With L≥16 (16%): 5.90 accepted, 7/7 in 75%.
- By category, mean accepted:

  | category | mean accepted |
  |---|---|
  | thinking | 1.46 |
  | prose | 1.78 |
  | tool arguments | 3.79 |
  | tool-call syntax | 4.00 |
  | edit/write | 4.49 |

- The frequency (suffix-tree) variant is indistinguishable from most-recent
  prompt-lookup (2.02 vs 2.04).

**Walk, assuming copyability is independent of DFlash2's speed (optimistic).**

| DFlash2 model | best rule | tok/round | speedup | speedup, no drafter saving | oracle |
|---|---|---|---|---|---|
| D = 5.43 | pl:12–16 | 5.43 → 5.53–5.56 | ×1.04 | ×1.02 | ×1.10 |
| D = 4.5 | pl:12 / sfn | — | ×1.07 | — | — |
| D = 6.0 | pl:16 | — | ×1.035 | — | — |
| measured per-turn D | pl:8 | 3.32 → 3.59 | ×1.11 | ×1.08 | ×1.22 |

- Edit turns, measured D: ×1.24. Other turns: ×1.08.
- Copy commits 22–24% of the output tokens under pl:8. When it fires, it accepts 7/7
  in 48.5% of rounds and 0 in 17.7%.

**Overlap with DFlash2.**

- Per turn, DFlash2's tok/round rises with copy coverage: 2.6–2.7 below 5% coverage,
  4.6 above 50%. Pearson 0.68 over 3,825 turns.
- In turns 80–90% covered, DFlash2 measures 5.10 tok/round with P(7/7) = 0.39. The copy
  rounds on the same turns have P(7/7) = 0.86, and the hybrid is 6.86 tok/round.
- A per-turn bound where DFlash2's best rounds fall exactly on the copied spans gives:

  | rule | pessimistic | optimistic |
  |---|---|---|
  | pl:8 | ×0.98 | ×1.08 |
  | pl:16 | ×1.00 | ×1.06 |
  | oracle | ×1.005 | ×1.19 |

- A per-class regression of DFlash2 rounds does not identify: it fits more than 8
  tok/round on covered spans, which is impossible, so it is discarded.

**Wider copy-only verify** (16 or 32 columns, optimistic): measured D reaches ×1.12–1.15
if the wide round costs the same as a width-8 one, and ×1.10 at 1.5× the cost.

## Finding

Observed:

- On real qwen-code and opencode traffic, only 10–24% of output tokens are copyable
  with a confident match. Thinking is 73.5% of the output, and copy recovers almost
  nothing there.
- The copyable tokens concentrate in edit/write arguments, tool arguments and tool-call
  syntax.
- DFlash2 is already much faster than its average on copy-heavy turns, yet slower there
  than copy (5.1 vs ~6.9 tok/round in 80–90%-covered turns).
- At width 8 the hybrid gains:

  | traffic | range |
  |---|---|
  | sampled (DFlash2 ≈ 3.3 tok/round) | ×0.98–1.11 |
  | greedy (DFlash2 at 5.43) | ×1.00–1.04 |

  The low end is the worst case of overlap, the high end is independence.

Inferred:

- The realistic gain at 1 lane is probably 0–8%, and less at 8 lanes: a copy round in a
  batch still pays the drafter for the other lanes, which gives the "no saving" column,
  about ×1.08 at best.
- The community's +68% (llama.cpp, 18-turn session) does not transfer to this workload:
  our output is dominated by thinking, and the verify caps a copy round at 7 drafts.
- DFlash2's own drop from 5.43 (greedy) to ~3.3 tok/round under agent sampling is a
  larger lever than copy drafting.

## Implications

- Do not build copy drafting as a width-8 DFlash2 replacement now. The expected payoff
  does not cover an M-sized change: a new draft source, a selector, and drafter-window
  bookkeeping on copy rounds.
- If it is revisited, the selector should use a long minimum match (n ≥ 12–16, or a
  frequency score together with n ≥ 8). Short-match rules (n ≤ 6) lose to DFlash2.
- A copy round cannot simply skip DFlash2. The drafter's 2048-token sliding window is
  fed by target taps and must still receive the committed tokens (cf. the #157
  extent-0 window hole), so only its forward, not its window append, is saved.

## Limits and unknowns

- DFlash2's per-position acceptance on these texts is unknown, hence the range. Its
  stats are ninfer's (2026-09-01..03 and 09-16), not current Ignis.
- The qwen-code transcripts omit the system prompt and tool schemas (~20–30K tokens),
  so copy context is slightly understated. The opencode trace is complete.
- Lookup is per lane. A global cross-request suffix tree (subagents reading the same
  files) was not measured.
- The 2.5 ms saved per copy round is an assumption, not a measurement.
- The cost of a wide copy-only verify is unmeasured.

## Follow-ups

- GPU: a teacher-forced DFlash2 replay over ~200 of these turns with per-position
  acceptance, through the VerifyOnly / `decode_program_verify` seam, turns the range
  into a number. Build only if it shows ≥ +8% at 1 lane.
- GPU: DFlash2 acceptance at T=0 vs T=1.0 on the same prompts, to explain 5.43 vs 3.3.
- Revisit if thinking gets budgeted or turned down (`reasoning_effort` low), or if the
  workload shifts to edit/rewrite-heavy turns, where the optimistic figure is ×1.24.
