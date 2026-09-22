# hq prefill reads the ring before the append: causal again, at no cost, and the tool-call count cannot tell

- Kind: experiment
- Status: current
- Observed: 2026-09-22
- Last verified: 2026-09-22
- Scope: kernel / hq-e8-2b prompt route, residual window, vendored correctness patch; serving / tool-call decisions
- Related: [#258](https://github.com/gpillon/ignis/issues/258), [#257](https://github.com/gpillon/ignis/issues/257), [#173](https://github.com/gpillon/ignis/issues/173); spec [runtime/07](../specs/runtime/07-hq-prefill-ring-read-before-append.md); [ADR 0037](../adr/0037-a-vendored-file-may-carry-a-correctness-patch.md); [the residual window was the tool-call gap](2026-09-22-the-residual-window-was-the-tool-call-gap.md)
- Superseded by: none

## Question

With the hq-e8-2b residual window wired (#257), the vendored prompt route
appended a prefill chunk before it read the 512-slot ring for the keys before
the chunk, so those keys were served the rows of chunk keys up to ~1,000
positions later. #258 swaps the two launches when the window is on — the first
Ignis-patched behaviour (ADR 0037). Does that make hq prefill causal on every
shape of the route, what does it cost, and what does it do to #173's
tool-call decisions?

## Evidence

All on the RTX 5090, one GPU process at a time, branch
`hq-residual-window-257`. Raw logs in that worktree's `.scratch/hq258/` and
`.scratch/diag-160/twenty/` (`patched-*`, `patched-tp2-*`, `bf16-*`).

**The kernel test** (`kernel/tests/test_hq_route_agreement.cu`, the
prefill-after-history arms): a chunk at position 512 after a 512-key history
sees only sinks, ring rows and its own fresh rows, so hq must agree with BF16
to one rotated rounding; and replacing the chunk's rows after token `t` must
not move one bit of columns `0..t`.

| Arm | Reference's order: median / worst rel. L2 | moved (must be 0) | Patched: median / worst | moved |
|---|---|---|---|---|
| W=200 at 512 | 0.351 / 4.41 | 528,778 | 0.0028 / 0.0110 | 0 |
| W=512 at 512 | 1.958 / 30.15 | 1,488,400 | 0.0028 / 0.0115 | 0 |
| W=200 at 512, masked 150 | 0.260 / 3.92 | 483,532 | 0.0028 / 0.0110 | 0 |
| W=200 across the 262,144-key band | (no BF16 twin) | 108,320 | — | 0 |

Two patched runs identical; the whole CTest suite 63/63.

**What attention reads** (`crates/server/tests/attn_tap_hq_consumed_gpu.rs`,
the two pointing inputs, query chunk 122 wide at 16,384 / 1,024): fresh 122,
sink 32, **ring 512, clobbered 0** (#257's build: ring 390, clobbered 122),
every exact row within 0.0020. Against #257's capture: L3's codec rows
byte-identical (15,840 of 15,840; 480 of 480); at L39 exactly the first
chunk's codec positions are still #257's bytes (992 of 15,840 at 4096 px; all
480 at 1024 px, whose codec rows all sit in its first chunk).

**Cost.** The twenty tool prompts' prefill (`ignis.request.admitted`
`duration_ms`, same prompt tokens), patched over #257's build: median ratio
0.9945 per request (0.94-1.16), 7,253 against 7,315 tokens/s over all twenty.
Decode is not on the patched path: the throughput prompt generates the same
text on both builds (172 rounds, 228 accepted drafts at one lane), and its one-lane
rate was 126.6-129.2 then 132.9-137.2 tokens/s in two launches of the patched
build against 133.9-145.6 in #257's two — launch-to-launch spread, not the
patch.

**#173, re-run** (the recorded 20 prompts with tools, greedy, thinking off,
`max_tokens` 1024, Makefile defaults; prompt tokens identical to the
reference on 20 of 20):

| Build | Ends in a tool call | Agrees with ignis BF16's decision |
|---|---|---|
| reference (ninfer `a00648cb`, hq-e8-2b, recorded 2026-09-15) | 18 / 20 | 7 / 20 |
| ignis #257, window at reference parity | 17 / 20 | 10 / 20 |
| ignis #258, Ignis patched | 16 / 20 | 7 / 20 |
| ignis, **BF16 KV** (`KV_FORMAT=bf16 MAX_CONTEXT=32768`, same build) | **7 / 20** | — |

Patched against #257's build, five prompts flip, both ways (three from a tool
call to the length cap, two the other way). The BF16 leg's 13 others run to
the 1,024-token cap writing the analysis the prompt asks for.

## Finding

1. **Observed: attending before appending makes hq prefill causal on every
   shape of the prompt route** — dense, masked and banded — and serves all
   512 keys before a chunk their own rows. The reference's order fails the same
   checks by two orders of magnitude.
2. **Observed: it costs nothing measurable.** The same kernels run in another
   order on one stream; prefill time per request is within noise, and decode
   never takes the route.
3. **Observed: a chunk at position 0 cannot see the order.** Nothing precedes
   it, so the first chunk of a fresh prompt is computed identically either
   way; the L39 bytes show it. A prefill that continues a claimed prefix or a
   checkpoint starts past 0, so even its first chunk can.
4. **Observed: #173's count moved 17 → 16, and lossless KV gives 7.** Inferred:
   whether a 24K-token agentic turn ends in a tool call is decided by near-ties
   that any change to KV numerics moves, in both directions, and lossless KV
   ends fewer of these turns in a tool call than either hq engine. The count is
   therefore not a measure of how right the hq path is, and the reference's
   18/20 is the reference's hq numerics rather than a target: #257 closed the
   gap *to the reference*, not to the model.

## Implications

- Any comparison with the reference that includes an hq prefill chunk starting
  past position 0 — every chunk after a prompt's first, and every chunk of a
  prefill continuing a claimed prefix or a checkpoint — crosses this
  Ignis-patched behaviour and must say so (ADR 0037). Only a fresh prompt that
  fits one chunk does not.
- #173 needs an oracle before another hq change is judged by it: the BF16
  decisions (ADR 0022) — or a reference BF16 leg — not the reference's hq count.
- A prefill-route change needs a chunk that starts past the ring (512) to be
  tested at all; a first-chunk arm cannot see a ring bug. The bit-exact
  causality check needs no oracle and no exact window, so it reaches shapes a
  BF16 twin cannot afford.

## Limits and unknowns

- One BF16 leg, ignis only, 20 prompts, greedy: the 7/20 is not a reference
  BF16 measurement, and no leg repeats.
- The banded arm is checked for causality only, not for agreement.
- Only the 27B head geometry (24 query / 4 KV heads) is tested; the patch's
  second branch, for the reference's 35B geometry, is the same two lines and
  runs on no model ignis serves.
- Decode throughput across launches moves ±5% on its own here; one-lane
  comparisons between builds need several launches.

## Follow-ups

- #173 carries these numbers; the next step there is an oracle (the BF16 leg
  above, or the reference's own BF16), not the last hq tool call.
