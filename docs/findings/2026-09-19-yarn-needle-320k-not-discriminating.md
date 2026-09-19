# An exact-string needle at 320K tokens is recalled with or without YaRN

- Kind: experiment
- Status: current
- Observed: 2026-09-19
- Last verified: 2026-09-19
- Scope: serving / RoPE scaling, long-context evaluation method
- Related: [#227](https://github.com/gpillon/ignis/issues/227) (YaRN as a load
  option), `.scratch/rope-scaling/specs/01-yarn.md`
- Superseded by: none

**Hardware:** RTX 5090, exclusive card (ADR 0006).
**Engine:** `qwen3_8_27b_nvfp4full-v2`, `hq-e8-2b`, `--prefill-chunk 1024`,
`--spec dflash2 --draft-tokens 7`, `--max-context 380000`. The two runs differ
only in `--rope-scaling`.
**Instrument:** one `/v1/chat/completions` call per run, greedy, thinking off,
320,076 prompt tokens: filler drawn from a 25-word vocabulary with one literal
record (`the vault combination for locker 4417 is TANGERINE-90210`) at ~66%
depth, then a question asking for the code.

## Question

#227 adds YaRN so a context past the checkpoint's trained 262,144 positions
means something. Does the obvious probe — a needle past that envelope — show
the difference?

## Evidence

| `--rope-scaling` | Prompt tokens | Answer |
|---|---|---|
| `yarn:4` | 320,076 | `TANGERINE-90210` |
| `none` | 320,076 | `TANGERINE-90210` |

Both exact, both first try. The two loads are genuinely different — the same
short prompt comes back worded differently under each, and the `kv_pool` log
line carries `rope_scaling` — so this is not a flag that failed to take
effect.

## What it means

**The single literal needle does not discriminate.** At 320K, 1.22x the
trained envelope, the unscaled table still places the record well enough for
exact retrieval. Two things plausibly hold it up and neither is the rotary
table: the record is a *literal string* the model can match rather than
reason about, and hq-e8-2b keeps the first 32 and last 512 K/V rows exact
(`kGqaHqSinkKeys` / `kGqaHqRecentKeys`) — though the needle sits in neither
window here.

So this probe cannot be used as the acceptance for a rotary-table change, in
either direction: it would have passed a build that ignored `--rope-scaling`
entirely. What proves the table is live is the unit-level check
(`ignis_kernel_rope_scaling_test` pins the table against independently
computed HF reference vectors) plus the two loads wording the same short
answer differently.

**What would discriminate** is a probe whose answer depends on the *relative*
positions of things far apart rather than on one string: multi-hop retrieval
across two distant records, ordering questions ("which of these three notes
came first"), or a summary scored against a reference. Also untried here: a
depth sweep, and a context far past 320K (the engine's ceiling is 1,048,576
visible keys under hq-e8-2b). None of that is filed yet — file it before
claiming YaRN buys quality, rather than the position range it demonstrably
buys.
