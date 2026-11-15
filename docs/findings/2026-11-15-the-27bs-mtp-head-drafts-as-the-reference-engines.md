# The 27B's MTP head drafts in ignis as in the reference engine: 4.241 against 4.207 tokens a round under BF16 KV and 4.175 against 4.164 under hq-e8-2b, on the same sixteen prompts

- Kind: experiment
- Status: current
- Observed: 2026-11-15
- Last verified: 2026-11-15
- Scope: the 27B's MTP head as ignis's verify round drafts with it (spec #618, the integration branch at
  b2a56397, #621's round) against the reference engine's MTP with the same head, on the RTX 5090: the NVFP4
  artifact, one lane, a fixed draft window of 7, the full proposal head, greedy, BF16 and hq-e8-2b KV
- Topic: decode-27b
- Related: https://github.com/gpillon/ignis/issues/625,
  https://github.com/gpillon/ignis/issues/618,
  https://github.com/gpillon/ignis/issues/621,
  [ADR 0095](../adr/0095-the-27b-has-two-drafters-and-its-mtp-head-keeps-attention-state-over-the-whole-context-and-a-continuation-hidden.md)
- Superseded by: none
- Summary: RTX 5090, 27B NVFP4, MTP at a fixed window of 7, one lane, greedy, 16 prompts: ignis commits 4.241 tokens a round to the reference engine's 4.207 (BF16 KV) and 4.175 to 4.164 (hq-e8-2b); where the texts match, so do the counters.

## Question

The verify round keeps the target's text whatever the drafter proposes, so a head that ignis reads wrongly
(a concatenation in the wrong order, the wrong hidden, a misplaced rope position) shows only as drafts the
target refuses. #620's double-precision recomputation proves the kernels compute what ignis takes the head
to be, not that ignis took the checkpoint's head as it was trained. The reference engine drafts the 27B with
the same head from the same artifact. On the same prompts, at the same settings, does ignis commit as many
tokens a round, within 5%? If not, where do the two part? This is #625's gate, placed right after the round
drafts (#621) so that a head read wrongly is found before the rest of the spec is built on it.

## Evidence

**The prompts.** Sixteen greedy prompts, each a single user turn with thinking closed and no system message:
the four canaries (`crates/bench/src/canary.rs`) and twelve more, which are code (Python, Rust, SQL, a bash
script, a regular expression), JSON, a short story, two explanations, a five-point summary, a translation, a
list of tips and a word problem. The test holds them as `ACCEPTANCE_PROMPTS`. Both engines render a prompt as
`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, and the test asserts
that every prompt's token count under the artifact's own tokenizer equals the reference engine's (22 to 57
tokens).

**The reference engine** (run first, alone on the card after `make gpu-status`): its build of 2026-09-02, one
commit older than the commit ignis vendors from (`kernel/vendor/manifest.json`). It ran as its CLI, one
process per prompt: the 27B NVFP4 artifact (`qwen3_8_27b_nvfp4full-v2`, the file ignis serves),
`--no-thinking --greedy --spec mtp --draft-tokens 7 --kv-dtype bf16|hq-e8-2b --max-context 4096
--max-new 256`. That is the full proposal head (its default; the shortlist flag was not given) and the fixed
policy (the adaptive flag was not given). One request; it stops at its stop token. Its per-request counters
are `mtp accepted by pos` and `mtp drafted by pos`. A round at extent 0 is a fallback step, not a round.
`drafted[i]` counts the rounds with an extent past i, and `accepted[i]` the rounds whose accepted leading run
reaches i + 1. Tokens a round are 1 + accepted / rounds. Its figures are the fixture
`crates/core/tests/fixtures/mtp_reference_acceptance.json`, which the test checks pools to the reference's
own totals.

**ignis**: `mtp_round_gpu.rs`,
`the_heads_acceptance_is_within_five_percent_of_the_reference_engines_on_the_same_prompts`, under the GPU
profile (`cargo test -p ignis-core --features cuda --target x86_64-pc-windows-msvc --test mtp_round_gpu
the_heads_acceptance -- --ignored --nocapture`). It ran the integration branch at b2a56397, the same artifact,
and one MTP load per KV format at window 7, context 4096 and one lane, with its verify graph captured. Each
prompt is prefilled and then run through the leaf's verify rounds (`decode_program_verify_runs`), greedy.
The leaf has no loop guard, so it is off. ignis's `SpecCounters` (`drafted_at`, `accepted_at`) use the same
leading-run definition.

**The accounting, fixed before the run.** The reference's round licenses and counts the whole run the target
accepts and only then ends the request at a stop token, so its counters include drafts accepted past the
stop. On its capped prompts, generated = 1 + rounds + accepted (+ fallback steps); on its stopped prompts that
sum overshoots by up to 4 tokens. So the test counts ignis the same way:
- Each round is given what is left of the 256-token budget and no stop ids, and its extent is asserted to be
  the reference's, `min(7, remaining − 1)`.
- The run ends after the round whose generated tokens (its accepted drafts and the target's next token)
  hold a stop id, or at the cap.
- The test also prints ignis's figure with each round cut at its stop, as the served counters count it.

**Pooled, the 16 prompts:**

| KV | engine | rounds | at extent 0 | accepted | tokens a round | ignis / reference |
|---|---|---:|---:|---:|---:|---:|
| BF16 | ignis | 663 | 1 | 2,149 | **4.241** (4.219 cut at the stop) | **1.0082** |
| BF16 | reference | 634 | 1 | 2,033 | **4.207** | |
| hq-e8-2b | ignis | 651 | 2 | 2,067 | **4.175** (4.146 cut at the stop) | **1.0028** |
| hq-e8-2b | reference | 654 | 2 | 2,069 | **4.164** | |

**Accepted over drafted at each position** (pooled; the leading-run definition):

| position | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 ignis | 0.848 | 0.654 | 0.515 | 0.435 | 0.354 | 0.254 | 0.202 |
| BF16 reference | 0.853 | 0.676 | 0.525 | 0.396 | 0.329 | 0.252 | 0.194 |
| ignis / reference | 0.993 | 0.967 | 0.982 | 1.098 | 1.075 | 1.009 | 1.042 |
| hq-e8-2b ignis | 0.839 | 0.663 | 0.499 | 0.397 | 0.323 | 0.256 | 0.216 |
| hq-e8-2b reference | 0.856 | 0.670 | 0.515 | 0.400 | 0.317 | 0.239 | 0.196 |
| ignis / reference | 0.979 | 0.990 | 0.970 | 0.993 | 1.020 | 1.072 | 1.106 |

**Per prompt**: tokens a round, ignis against the reference. "Parts at" is the first generated token where
the two texts differ ("same" when they match to the end).

| prompt | BF16 ignis | BF16 ref | parts at | hq ignis | hq ref | parts at |
|---|---:|---:|---:|---:|---:|---:|
| rust-hello | 5.000 | 5.000 | same (22) | 5.000 | 5.000 | same (22) |
| rust-sort | 4.333 | 5.667 | 3 | 5.273 | 4.692 | 0 |
| math-greedy | 5.682 | 5.722 | 0 | 5.208 | 5.150 | 0 |
| explain-reverse | 5.143 | 4.000 | 18 | 4.875 | 5.571 | same (36) |
| py-fib | 6.568 | 6.541 | 21 | 6.722 | 6.694 | 42 |
| rust-struct | 6.071 | 5.795 | 65 | 6.220 | 5.543 | 6 |
| sql-join | 5.242 | 4.317 | 4 | 5.000 | 4.263 | 20 |
| json-convert | 7.667 | 7.667 | same (69) | 7.778 | 7.778 | same (69) |
| prose-story | 2.589 | 2.490 | 29 | 2.288 | 2.356 | 27 |
| prose-explain | 3.228 | 3.493 | 3 | 3.148 | 3.269 | 11 |
| history | 3.269 | 3.299 | 9 | 3.188 | 3.098 | 29 |
| translate | 4.600 | 4.600 | same (20) | 4.600 | 3.833 | same (20) |
| math-word | 5.204 | 5.667 | 31 | 5.907 | 5.543 | 32 |
| list-tips | 3.110 | 3.228 | 9 | 3.148 | 3.577 | 9 |
| regex | 5.404 | 4.636 | 0 | 5.312 | 5.426 | 23 |
| bash-script | 5.054 | 5.500 | 23 | 5.400 | 5.000 | 23 |

**Where the texts are the same.** Under BF16 three prompts (rust-hello, json-convert, translate: 19 rounds)
give the same text on both engines, and on them the two engines' counters are identical at every position,
round count and accepted drafts included. Under hq-e8-2b four prompts give the same text. Two of them
(rust-hello, json-convert) have identical counters. On the other two (explain-reverse, translate) the rounds
differ: 4.875 against 5.571 and 4.600 against 3.833 tokens a round. The four pool to 27 rounds and 130
accepted on both engines.

**The reference against itself.** Its own BF16 and hq-e8-2b runs part on 12 of the 16 texts and differ by
1.0% pooled (4.207 and 4.164); ignis's two differ by 1.6% (4.241 and 4.175).

The ignis run took 39 s for both KV formats.

## Finding

Observed:
- ignis's pooled tokens a round are 0.8% above the reference engine's under BF16 KV and 0.3% above under
  hq-e8-2b, on the same prompts at the same settings.
- At no draft position is ignis's rate under the reference's by more than 5%. The lowest ratios are position 2
  under BF16 (0.967) and position 3 under hq-e8-2b (0.970).
- Where the two engines write the same BF16 text, they draft the same tokens, round for round.
- Cut at the stop, as the served counters are, ignis's figure stays within 1.5% of the reference's (4.219 and
  4.146).

Inferred:
- ignis reads the head as the reference engine does: the stem's concatenation order, the target's
  post-final-norm hidden, the rope position, the chain's feedback, and scoring with the target's output head.
- A head read wrongly would show from position 1 on, the draft that depends on nothing but the head's reading
  of the target's hidden and the anchor. Position 1 agrees within 0.7% (BF16) and 2.1% (hq-e8-2b).
- The per-prompt gaps (−24% to +29%) are not the head:
  - Under BF16 they come only on prompts whose texts part, often within the first few tokens, and they cancel
    when pooled.
  - Under hq-e8-2b two prompts with the same text still differ (−12% and +20%), and the four same-text prompts
    pool to the same totals: most likely the head's quantized state drafting otherwise on a near tie. Either
    way a single prompt's figure carries noise of that size.

The gate of #625 is met. The ticket's figures are both engines' pooled tokens a round and the rates at each
position, above.

## Implications

- With the full proposal head and a fixed window of 7, the head commits about 4.2 tokens a round on these
  prompts on both engines. The spec's "about 3" came from a different configuration: the reference engine
  served with the shortlist head and the adaptive policy under hq-e8-2b. It is not the figure for this
  configuration, so it is no reason to suspect ignis's head.
- The measurement of step 2 (#628: acceptance on the recorded agent turns, the cost of a round, the default
  window) can take the head as read right. Its acceptance numbers measure the head, not a defect in how ignis
  reads it.
- The fixture and the test stay as the regression check on the head's reading: a change to the stem, the
  chain or the commit that moves pooled acceptance past 5% fails it.

## Limits and unknowns

- Sixteen prompts and 2,655 to 2,808 generated tokens per engine and format. The texts part early on most prompts (on some at the
  first token), so the per-prompt and per-position figures compare different texts. The pooled figure, and
  the prompts whose texts match, are the comparison. Positions 4 to 7 carry about ±10% of that noise.
- On three prompt-format pairs the texts part at the first generated token: math-greedy under both formats,
  regex under BF16, rust-sort under hq-e8-2b. That token is the target's pick from the prefill, before any
  draft, so the difference lies in the target's prefill numerics, not in the head. Whether each is a near tie
  was not probed.
- One card, the NVFP4 artifact only (not the 16 GB artifact), one lane, greedy only, the fixed draft, and a
  context of at most about 313 tokens (the hq-e8-2b ring never wraps). Sampled lanes and the other lane
  counts were not compared with the reference.
- The reference ran one process per prompt. ignis ran one load per KV format, with its prompts one after
  another on one pool.
- The reference's counters carry no per-round record, so the round where the same text under hq-e8-2b drafts
  differently (explain-reverse, translate) was not located. Inferred: it is the head's hq-e8-2b state, which
  is quantized in both engines and which neither engine computes bit for bit like the other.

## Follow-ups

- Step 2's measurement: https://github.com/gpillon/ignis/issues/628.
