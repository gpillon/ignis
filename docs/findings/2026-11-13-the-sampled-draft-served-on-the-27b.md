# The sampled draft served on the 27B: +10.9% tokens a round on the recorded agent turns, sampled decode 1.05-1.27 times the fixed draft's, greedy text and prefill unchanged

- Kind: experiment
- Status: current
- Observed: 2026-11-13
- Last verified: 2026-11-13
- Scope: speculative decoding / DFlash2's sampled draft (spec #574, ADR 0090) on the 27B, on the RTX 5090:
  the integration branch at 288f41b7 under each draft rule against main at 9c240a4d, and the teacher-forced
  replay of #267's recorded rounds at 666988cf
- Topic: decode-27b
- Related: https://github.com/gpillon/ignis/issues/578,
  https://github.com/gpillon/ignis/issues/574,
  [ADR 0090](../adr/0090-the-27bs-sampled-lanes-draw-their-drafts-and-take-them-by-an-accept-of-ignis-s-own.md),
  [DFlash2 sampled acceptance](2026-09-24-dflash2-sampled-acceptance.md) (#267: the replay and its rounds)
- Superseded by: none
- Summary: RTX 5090, 27B DFlash2: the sampled draft commits 3.825 tokens a round on 200 replayed agent turns against 3.450 (+10.9%); served, sampled decode is 1.05-1.27x the fixed draft's, greedy text and prefill unchanged.

## Question

Spec #574's sampled draft (ADR 0090) was priced offline at +8.0% tokens a
round on #267's recorded agent rounds. Built (#575-#577) and served on the
RTX 5090: does the teacher-forced replay of the same rounds clear the gate's
+5%? Is any decode cell, at one and eight lanes, short and long context,
greedy and sampled, slower than under the fixed draft? Does a request at
temperature 0.6 commit no fewer tokens a round? Are greedy text and prefill
unchanged? And what does the branch as it would ship change against main?
This is #578's gate, which keeps or drops the default flip (288f41b7).

## Evidence

**Carried forward** (spec #574, #575, #577; the method and the detail are
there):
- **The offline study**, on #267's 61,692 recorded rounds: 3.461 tokens a
  round under the fixed draft, 3.738 under the sampled draft, +8.0% (95%
  interval +7.1% to +9.1%), and +8.0% on a second set of 61,739 rounds. At
  request temperature 1 the drafter temperature factor is flat between 0.8
  (3.730) and 1.0 (3.736); 0.9 gives 3.738. On the 16 GB artifact's 61,978
  recorded rounds: 3.459 to 3.745, +8.3% (+7.4% to +9.3%). The study's report
  and scripts are in the 27B speed line's scratch folder
  (`report-1-coupled-draft.md`, `coupled-draft/`).
- **The oracle**: the host reference `ignis_core::sampled_draft` against a
  double-precision oracle that sweeps the accept's and the refusal's uniforms
  on a 1024 × 1024 grid and the draft's own on 2^20 points, for six drafter
  distributions chosen to break a wrong rule (q = p, one-hot on the target's
  top token and on a 1/64 token, no mass on five of the target's eight
  tokens, half outside the support, flat over sixteen): worst deviation
  5.0e-4 against a tolerance of 3.9e-3. A refusal drawn from p instead of
  max(p − q, 0) fails 5 of the 6. The replay's forced verdict holds to its
  own oracle within 1.8e-4. The device kernels pass the same oracles on the
  card (`sampled_draft_gpu`).

**Setup** (2026-11-13): RTX 5090 (Windows 11, WDDM), `make start UI=0
LOOP_GUARD=off IDLE_UNLOAD=0`: the 27B NVFP4 artifact
(`qwen3_8_27b_nvfp4full-v2`), KV `hq-e8-2b`, DFlash2 with 7 draft tokens, 8
decode lanes. The host's user configuration file (FP8 exception weights,
1,500 MB headroom) applied to every launch alike. Three builds:
- *main*: 9c240a4d (`ignis-server.exe` sha256 `37fbeb2c...`);
- *fixed*: the integration branch at 288f41b7 (#575-#577, the bench's
  sampling flags and the default flip), `IGNIS_SPEC_DRAFT_RULE=fixed`;
- *sampled*: the same exe (sha256 `2657c1b1...`) at its default, `sampled`.

One bench binary for every launch (`ignis-bench`, built from 4c21da3d,
sha256 `bedd8f67...`). The protocol was fixed before the first launch. Every
launch read the speculative counters before and after each cell; tokens a
round are (accepted drafts + rounds) / rounds. Decode tok/s is decoded tokens
over decode time, each request's total less its TTFT, a build's two launches
pooled; a build's spread is its two launches' difference over their mean.
The round's code is the same at the branch's tip, 3fe97beb: the later
commits change comments, the configuration patch, tests and the bench.

Every record, script and server log is in `.scratch/574/`: `ab/protocol.md`,
`ab/run-launch.sh`, `ab/analyze.py` and `ab/summary.json` (the short-context
campaign), `ab/ttft-confirm/`, `long/` (the long-context campaign),
`replay-577/` (the replay) and `final/data.md`, one directory per launch.

**The replay**: #267's 200 recorded agent turns, sampled at T 1, top-p 0.95,
top-k 20, 7 drafts; the replay driver at 666988cf, one run per rule on the
same turns (45 and 48 minutes). Tokens a round, traffic-weighted as #267's
finding weights them:

| | fixed draft | sampled draft | gain |
|---|---:|---:|---:|
| rounds | 61,778 | 55,373 | |
| realized | 3.450 | 3.825 | **+10.9%** |
| expected (exact over the coins) | 3.456 | 3.833 | +10.9% |
| offline prediction (#574's study) | 3.461 | 3.738 | +8.0% |

- **The fixed run is #267's instrument**: 3.450 against #267's 3.461, on
  61,778 rounds against 61,692. 25 of the 200 turns walk #267's rounds
  exactly; the rest differ by a few rounds, from the drafter's numerics on
  today's build. Readout disagreements: 0 under both rules.
- **Per draft column**, taken given reached, fixed to sampled: d1 0.714 to
  0.797, d2 0.716 to 0.750, d3 0.727 to 0.748, d4 0.748 to 0.746, d5 0.759
  to 0.755, d6 0.777 to 0.769, d7 0.798 to 0.782.
- **Per kind of output**: thinking 0.702 to 0.745, prose 0.733 to 0.751,
  tool syntax 0.947 to 0.943, tool arguments 0.777 to 0.782, edits and
  writes 0.843 to 0.848. Per source, thinking: A 2.886 to 3.342, B 3.367 to
  3.722, C 3.268 to 3.700 tokens a round; the content cells gain 6 to 9%.
- **The short replay** (#267's jobs 0-6, 7 turns, 3,011 tokens): 3.673 to
  3.991, +8.7%.

**Decode, the agentic trace** (16:03-17:29Z): `ignis-bench replay` of
`bench/traces/g4-load-trace.jsonl` (11 recorded agent chat bodies)
`--as-recorded`, at `--conc 8` (c8) and `--conc 1` (c1). The greedy cells send
the trace's own greedy bodies; s1 samples at T 1 and s06 at T 0.6, both with
top-p 0.95, top-k 20 and sample seed 574 (a seed per request derived from
it). Six launches alternated: main 1, fixed 1, sampled 1, sampled 2, fixed 2,
main 2. Decode tok/s pooled, tokens a round in parentheses:

| cell | main | fixed | sampled | sampled / fixed | fixed's spread | sampled / main |
|---|---:|---:|---:|---:|---:|---:|
| greedy-c8 | 119.56 (3.609) | 118.65 (3.457) | 118.98 (3.511) | 1.0028 | 0.004 | 0.9952 |
| greedy-c1 | 202.51 (3.584) | 204.40 (3.584) | 204.43 (3.584) | 1.0001 | 0.001 | 1.0095 |
| s1-c8 | 111.57 (2.798) | 108.25 (2.800) | 137.38 (3.478) | **1.2690** | 0.003 | 1.2313 |
| s1-c1 | 153.95 (2.716) | 153.95 (2.716) | 177.18 (3.140) | **1.1509** | 0.001 | 1.1509 |
| s06-c1 | 182.76 (3.256) | 182.67 (3.256) | 191.52 (3.400) | **1.0485** | 0.001 | 1.0480 |

- **greedy-c1 did the same work on every launch**: 3.584 tokens a round and
  the same tokens. main's own spread there is 0.018 (204.37 and 200.69
  tok/s): sampled / main's 1.0095 is main's second launch, the session's
  last, reading low.
- **greedy-c8 is not bit-stable across launches, on main too**: 3.490 and
  3.755 tokens a round, 24,126 and 21,336 tokens on main's two launches. Its
  builds compare different tokens.
- **main and fixed run the same rule**, and their sampled cells commit the
  same tokens a round (2.798 and 2.800, 2.716, 3.256). main's s1-c8 spread
  is 0.054 (108.43 and 114.42).
- **The canaries** (4 prompts × 2, greedy, 64 tokens, each launch's warm-up):
  the text and the reasoning are byte-identical on all six launches, main,
  fixed and sampled; every check sane and deterministic.

**Decode, long context** (17:32-17:57Z, six launches in the same order):
prompts of about 64K tokens of the repository's code and a question,
`max_tokens` 3000, sampled at T 1, top-p 0.95, top-k 20, seed 574;
`ignis-bench replay` of eight prompts at `--conc 8` (long-c8) and three at
`--conc 1` (long-c1):

| cell | main | fixed | sampled | sampled / fixed | fixed's spread | sampled / main |
|---|---:|---:|---:|---:|---:|---:|
| long-c8 | 36.09 (2.827) | 36.82 (2.843) | 39.33 (3.182) | **1.0682** | 0.046 | 1.0900 |
| long-c1 | 144.61 (2.891) | 144.87 (2.891) | 169.04 (3.403) | **1.1668** | 0.0004 | 1.1689 |

long-c8's decode time overlaps the other lanes' 64K prefills, so its absolute
rate is low; the instrument is the same for every build.

**Prefill and TTFT**: `ignis-bench ttft`, cold prefixes at exact lengths,
three samples a cell, 8 tokens. Medians pooled over main's 4 launches,
fixed's 2 and sampled's 4: the six launches above and a TTFT-only
confirmation of four, alternated ABBA (sampled 3, main 3, main 4, sampled 4,
17:57-18:07Z):

| prompt | main | fixed | sampled | sampled / main |
|---|---:|---:|---:|---:|
| 8K | 703.3 ms (11,648 tok/s) | 702.7 ms | 701.1 ms (11,684 tok/s) | 0.997 |
| 32K | 2,750.0 ms (11,915 tok/s) | 2,757.4 ms | 2,749.2 ms (11,919 tok/s) | 1.000 |
| 128K | 16,016.9 ms (8,183 tok/s) | 16,032.5 ms | 16,032.1 ms (8,176 tok/s) | 1.001 |

- **One launch, sampled 2, read 32K and 128K about 4% slower** on all three
  samples, while sampled 1 matched main; its exe is fixed's, and the card's
  clocks and power matched (575 W cap). The ABBA confirmation read sampled /
  main 0.995, 0.999 and 1.002.
- greedy-c8's cold TTFT median: main 1,580.9 ms, fixed 1,597.5, sampled
  1,615.0: agentic prompts, eight lanes on cold prefixes, a noisy figure.

## Finding

- **Observed: on the recorded agent turns the sampled draft commits +10.9%
  tokens a round**, 3.825 against 3.450, above the offline +8.0% and the
  gate's +5%. The replay's realized and expected figures agree under both
  rules (3.825 and 3.833, 3.450 and 3.456).
- **Observed: the gain is on the first draft column and on thinking text**:
  the first draft is taken 0.797 of the time against 0.714, thinking text's
  columns 0.745 against 0.702; tool syntax, already at 0.947, does not move.
  The later columns are taken a little less often given reached (d7 0.782
  against 0.798), and a round takes all seven drafts more often (0.151
  against 0.131 of rounds).
- **Observed: served, no decode cell of the sampled draft is slower than the
  fixed draft's**: the lowest ratio is greedy-c1's 1.0001. Sampled requests
  decode at 1.15 times the fixed draft's rate at one lane and 1.27 at eight on
  the agentic trace at T 1, 1.05 at T 0.6, and 1.17 and 1.07 at 64K tokens of
  context.
- **Inferred: a round costs what it cost.** At one lane the decode ratio
  follows the tokens-a-round ratio within 1%: 1.151 for 1.156 at T 1, 1.049
  for 1.044 at T 0.6, 1.167 for 1.177 at 64K.
- **Observed: at T 0.6 the sampled draft commits 3.400 tokens a round
  against 3.256**, the gate's cell for a temperature the recorded traffic
  does not hold.
- **Observed: greedy is unchanged.** The canaries are byte-identical on six
  launches of three builds, and greedy-c1 commits the same tokens at the same
  rate under both rules.
- **Observed: prefill and TTFT are unchanged**, sampled at 0.997 to 1.001 of
  main pooled. The one slow launch did not repeat in the ABBA confirmation.
  Inferred from the code: the draft rule runs in the verify round alone, and
  its device memory is two buffers of top-k × window × lanes, so nothing on
  the prefill path moved.
- **#578's gate passes**: the replay's +10.9% against +5%; no decode cell
  slower than the fixed draft beyond the fixed draft's spread; 3.400 against
  3.256 at T 0.6; greedy unchanged. The default flip (288f41b7) stays.

## Implications

- The sampled draft stays the 27B's default with DFlash2, and `fixed`, the
  rule of earlier releases, stays one option away
  (`--spec-draft-rule fixed`) for a comparison on someone's own traffic.
- On the recorded agent turns the default now commits about 3.8 tokens a
  round, where the guide's sampled-traffic figure was #267's 3.46.
- The replay, with `--draft-rule` and `tools/dflash2-replay/draft_rule.py`,
  is the instrument for the next drafter or accept change on the same
  61,692 rounds; its fixed-rule run reproduces #267's figure within 0.3%.
- A served comparison across builds compares one-lane greedy cells and the
  canaries: eight-lane greedy batches change their tokens from launch to
  launch on main as well.

## Limits and unknowns

- One card and one artifact, the NVFP4 27B on the RTX 5090. The 16 GB
  artifact and the RTX 30 and 40 cards have the offline estimate (+8.3%) and
  the GPU tests, not a served measurement.
- Two launches per build, one sample seed (574). The trace cells are 11
  requests and the long cells 8 and 3 prompts, so their gains (+15.6% and
  +24.2% tokens a round at T 1) are those requests', not the replay's
  population. A sampled request's text differs between the two rules by
  design, so the sampled cells compare different tokens drawn from the same
  distribution.
- Why the replay gains more than the offline estimate is not established.
  The offline study priced the rule at the anchors the fixed draft's rounds
  start from (3.741 at this replay's own), while a run under the rule starts
  its rounds where its own runs end.
- The recorded traffic is all at T 1; T 0.6 was measured at one lane on the
  trace alone, and other temperatures not at all.
- long-c8's decode overlaps the other lanes' prefills; it compares builds, it
  is not a decode rate.
- The protocol's `g5` long-context cells, run in the short campaign's
  launches, are void as a measurement: `g5` sends no EOS suppression, and two
  runs in one launch share prompt 0's cache (#578). The long-context figures
  above are the separate `replay` campaign.
- The host's user configuration file put the exception weights in FP8 on
  every launch; the absolute rates are for that configuration.

## Follow-ups

- `g5`'s sampled long-context cells: the bench follow-up #578 names for its
  close-out.
