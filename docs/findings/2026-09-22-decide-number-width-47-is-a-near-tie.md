# "47 invoices" at two digits is a near-tie, not an hq read of the future

- Kind: experiment
- Status: current
- Observed: 2026-09-22
- Last verified: 2026-09-22
- Scope: serving / `/v1/decide` `number`; kernel / hq-e8-2b residual window, short prompts; prompt reuse
- Related: [#259](https://github.com/gpillon/ignis/issues/259), [#254](https://github.com/gpillon/ignis/issues/254), [#257](https://github.com/gpillon/ignis/issues/257), [#258](https://github.com/gpillon/ignis/issues/258); spec [runtime/07](../specs/runtime/07-hq-prefill-ring-read-before-append.md) § As built; [the number prompt declares an alignment](2026-09-21-the-number-prompt-declares-an-alignment.md); [hq prefill reads the ring before the append](2026-09-22-hq-prefill-reads-the-ring-before-the-append.md)
- Superseded by: none

## Question

Since #257 wired the hq-e8-2b residual window, `decide_number_width_gpu`
reads **4** for "47 invoices" at `digits: 2` (first digit p = 0.676); before
it read 47 (p = 0.907). The prompt is shorter than the 512-key ring, so under
the window every key it attends is read exact. Is this the #254 alignment
clause holding only by the codec's noise, a fault in the hq path's
short-history regime, or something else?

## Evidence

All on the RTX 5090, one GPU process at a time, main at `c7ceaf7` (#257 and
#258 in). The server test run as it is, and as temporary copies that change
only its `EngineShape` (the `cuda_scheduler` argument): `kv_format:
KvFormat::Bf16` for BF16, `prompt_reuse: false, retained_slots: 0` for reuse
off, `prefill_chunk: 128` for the chunk control, each over
`..EngineShape::default()`. Raw logs in `.scratch/issue-259/`.

**The cell under five engine shapes** (`invoices`, truth 47, `digits: 2`;
p is the first digit's probability):

| KV format | prompt reuse | reads | first digit p |
|---|---|---|---|
| hq-e8-2b | on (the test's default) | **4** | 0.676 |
| hq-e8-2b | off | 47 | 0.752 |
| BF16 | on | 47 | 0.939 |
| BF16 | off | 47 | 0.846 |
| hq-e8-2b, codec only (before #257; quoted from spec runtime/07's run on `57fbef4`, not re-run) | on | 47 | 0.907 |

`prefill_chunk: 128` changes nothing in either format with reuse off: the
prompt is under 128 tokens, one chunk at any width. Every other cell the test
asserts reads its truth in the four shapes run here (and passed on `57fbef4`); the one-digit field of the same
question (not asserted — 47 cannot fit) flips between 0 and 4 across them
at p = 0.52–0.75.

**The same question after ~800 tokens of neutral filler** in the evidence,
so the history is past the ring: hq reads 47 at p = 0.965, BF16 at 0.985.

**The short-history regime at the op level.** `test_hq_route_agreement.cu`
held its short-history arms to the codec's bound only. Held to the
exact-window bound #258 introduced (median relative L2 ≤ 0.01, every row
≤ 0.05) — the first chunk (W=200 at 0), the decode round over a 200-key
history (B=1..8), and two new chunks after a history shorter than the ring
(W=40 at 60, W=12 at 100, where the ring's bound before the chunk is
negative) — every arm passes at median 0.0022–0.0033, max 0.0040–0.0056:
the same one rotated rounding as the arms after a 512-key history (0.0028).
The two new arms are bit-exactly causal.

**The claim path** (cited, not re-measured here) is already held bit-exact to a cold prefill split at the
same boundary under hq (`retained_prefix_gpu`, #188/#257).

## Finding

Observed: it is not the codec's noise
holding the clause up: lossless BF16 reads 47, in both reuse modes. It is
not a fault in how hq reads a short prompt: the op agrees with BF16 to one
rotated rounding there, as it does past the ring, and the claim reproduces a
split cold prefill exactly.

Inferred from those: the cell is a near-tie between two renderings of the
same number, and the failing shape lands on the other side of it. What moves
the answer is the combination of two
perturbations that are each a rounding — the rotated frame (hq vs BF16) and
the prefix split prompt reuse makes (on vs off, which moves BF16 by 0.09 on
its own, and hq from BF16 by 0.09 with reuse off) — on a cell whose winner
never exceeds 0.94 in any shape. Low confidence alone is not the tie: `days`
at one digit reads 3 at 0.65 under BF16 and 0.85 under hq, and holds. The
codec-only engine read 47 by the same chance.

## Implications

- `decide_number_width_gpu` failed on a correct engine: it asserted an exact
  answer on a cell whose margin is smaller than two roundings. Making it
  green was a decision about the test or the prompt, not a kernel fix; the
  owner chose the test (Follow-ups).
- #254's alignment clause does hold on this cell (BF16, both reuse modes;
  hq with reuse off). What it does not have is margin: where 47 wins, the
  other first digits together still take 0.06–0.25, and in the failing shape
  another digit wins at 0.68.
- An hq-vs-BF16 or reuse-on-vs-off comparison on a single decision can flip
  on a correct engine. A regression test on decisions needs a margin (or a
  first-digit floor) that a rounding cannot cross, or a cell far from the tie.

## Limits and unknowns

- One prompt, one cell. The probability of the competing first digit was not
  read out, only the winner's; whether the hq reuse-on first digit is a
  leading zero is inferred from the reading, not observed.
- The reference engine was not run on this cell.
- The filler control changes the prompt, not only its length; it shows the
  cell is prompt-sensitive, not which property of the prompt matters.

## Follow-ups

- #259 item 1, decided by the owner 2026-09-22: the test keeps the served
  shape (hq, prompt reuse) and asserts #254's signature instead of every
  cell's value - no fitting cell wrong with its first digit at p >= 0.9, at
  most 2 of 21 wrong in all. The prompt is unchanged.
