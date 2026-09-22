# The hq residual window was most of the tool-call gap, and the prompt route read it after its own append (until #258)

- Kind: experiment
- Status: current
- Observed: 2026-09-22
- Last verified: 2026-09-22
- Scope: kernel / hq-e8-2b attention, residual window, sequence state; serving / tool-call decisions
- Related: [#257](https://github.com/gpillon/ignis/issues/257), [#173](https://github.com/gpillon/ignis/issues/173), [#160](https://github.com/gpillon/ignis/issues/160), [#161](https://github.com/gpillon/ignis/issues/161); spec [runtime/06](../specs/runtime/06-hq-residual-window.md); [hq attention route agreement](2026-09-12-hq-attention-route-agreement.md), [the codec costs the head its read](2026-09-22-the-codec-costs-the-head-its-read.md), [hq prefill reads the ring before the append](2026-09-22-hq-prefill-reads-the-ring-before-the-append.md) (#258)
- Superseded by: none

## Question

Under hq-e8-2b ignis read every key through the codec: the vendored kernels
keep the 32 sink keys, a 512-key recent ring and the current chunk exact only
when the cache view carries a residual window, and ignis never passed one. The
reference (ninfer `a00648cb`) does. Wired the way the reference wires it,
what does attention read, does the window survive every state transition, and
is it what separated ignis from the reference on #173 (18/20 prompts end in a
tool call on the reference, 7/20 on ignis)?

## Evidence

All on the RTX 5090, the qwen3.8-27B NVFP4 artifact, hq-e8-2b, one GPU process
at a time. Raw logs in `.scratch/hq257/` and `.scratch/diag-160/twenty/` of the
`hq-residual-window-257` worktree.

**What the prompt route reads** (`crates/server/tests/attn_tap_hq_consumed_gpu.rs`,
consumed-key tap, the committed pointing scene served at 4096 px and box-filtered
to 1024 px, GQA ordinals 0 and 9). Each consumed row against the rotated key the
layer produced, classified by `ignis_core::hq_ring::prompt_source`:

| Input | Tokens / chunks | Query chunk | fresh | sink | ring | clobbered | codec | image through codec |
|---|---|---|---|---|---|---|---|---|
| 4096 px | 16,506 / 17 | 16,384 (122 wide) | 122 | 32 | 390 | 122 | 15,840 | 96.3% |
| 1024 px | 1,146 / 2 | 1,024 (122 wide) | 122 | 32 | 390 | 122 | 480 | 40.1% |

- Fresh, sink and ring rows: exact, worst relative L2 0.0020 (one rotated BF16
  rounding). Before the window every row of both prompts sat at the codec's
  0.33-0.78, median 0.3696.
- **Clobbered rows**: the first 122 keys of the ring window `[p0 - 512, p0)`
  come back exact — to the chunk key 512 positions later (0.0019-0.0020 to
  it). The vendored `gqa_attention_prompt_launch` appends the chunk (the fill
  kernel dual-writes the chunk's last `min(w, 512)` keys into their ring slots
  and sets their bits) and only then runs the scratch decode that serves the
  512 keys before the chunk from the ring. The reference launches the same
  order.
- Codec rows at GQA ordinal 0 (L3, only GDN layers above it): 15,840 of 15,840
  positions (480 of 480 at 1024 px) have the pre-change key bit for bit and
  decode to the pre-change bytes — the dual write left the codec's own write
  path untouched. At L39 no codec key is the pre-change one: every GQA layer
  above it now attends over exact rows.

**Route agreement** (`kernel/tests/test_hq_route_agreement.cu`, hq against
ignis's own BF16 route on identical keys): the arms' histories (≤ 216 keys)
fit inside the window, so the production view agrees with BF16 to one rounding
— prefill W=200 median relative L2 **0.0023** (was 0.2280), W=9..16
0.0028-0.0029 (was 0.36-0.38), decode W=1 B=1..8 **0.0032-0.0033** (was
0.174-0.190). The same arms over a view with the window taken off reproduce
the 2026-09-12 numbers to six digits (0.227955; 0.174162-0.190225).

**Lifecycle** (GPU tests run under hq as well as BF16): a retained prefix's
claimant holds the publisher's window at the prefix's end; after the same tail
its whole snapshot is the split cold control's byte for byte and it decodes
the same 8 tokens (`retained_prefix_gpu.rs`); the same for a checkpoint claim,
a KV-RAM restore of the materialized checkpoint blob and a chained turn
(`prompt_checkpoint_gpu.rs`); a sequence restored into the slot another
sequence used continues to the same tokens, the re-allocated slot's window
reading all zero first (`seq_snapshot_gpu.rs`); and the ring words read out of
the snapshot after a prefill, after six one-token decode rounds, and after
every one of 15 DFlash2 verify rounds over 4 lanes — two of them on prompts
longer than the ring, so 82 of the 114 rejected columns cleared a slot an
older key inside the window named — are exactly the host rule's
(`seq_snapshot_gpu.rs`, `dflash2_round_gpu.rs`).

**#173, re-run** (`.scratch/diag-160/twenty/`: the recorded 20 prompts with
tools, greedy, thinking off, `max_tokens` 1024, served with the Makefile's
defaults — hq-e8-2b, `--prefill-chunk 1024`, DFlash2/7). Prompt tokens identical
to the reference on 20 of 20 in both builds:

| Build | Ends in a tool call | Runs to the length cap |
|---|---|---|
| reference (ninfer `a00648cb`, recorded 2026-09-15) | 18 / 20 | 2 |
| ignis `c81b6e0`, no window (same session) | 4 / 20 | 16 |
| ignis `c81b6e0` + #257, window wired | **17 / 20** | 3 |
| (ignis 2026-09-15, `ignis172-tools.json`) | 7 / 20 | 12 |

**Cost.** One slot's window is 34 MiB (32 + 512 rows x 4 KV heads x 256 x BF16,
K and V, 16 layers, plus 16 ring words): 570,426,368 bytes at the Makefile's 8
lanes and 8 retained slots, its own VRAM-plan line, about 967 KV pages less.
The load's plan total (30,179,378,468 B) and NVML's delta (30,166,384,640 B)
still agree within 13 MiB. Decode per round, same prompt, same session: one
lane 15.8 ms (window) against 16.1 ms (no window), eight lanes 32.8 against
33.0-33.5 ms. Tokens per second at one lane moved from 157.7 to 145.1 only
because the two builds generate different text and the drafter accepts less
of this one (2.33 against 2.58 tokens per round).

## Finding

1. **The residual window accounts for almost all of the #173 gap.** On the
   same code in the same session, wiring it takes the prompts that end in a
   tool call from 4 to 17 of 20, against the reference's 18 — the remaining
   one is within what a single near-tie moves (see *Limits*). Reading the 512
   most recent keys and the sinks through the codec was enough to change the
   greedy decision of a 24K-token agentic turn.
2. **The ring was served after the chunk's own append — a read of the
   future** (#257's build, and the reference; ignis attends first since #258). For a prefill chunk `[p0, p0 + w)`, the keys `[p0 - 512, p0 - 512
   + min(w, 512))` are read as the exact rows of the chunk keys 512 positions
   later that share their ring slots. A query early in the chunk thus attends,
   at a past position the causal mask allows, to the row of a key up to ~1,000
   positions *ahead* of it: hq prefill with the window is not causal there. At
   the serving chunk of 1,024 every full chunk after the first reads its whole
   pre-chunk ring this way; a short last chunk loses `w` of it. Decode is not
   affected (its window ends at the round's own last append). The reference
   does the same, so the #173 comparison above is like for like.
   **Update (#258, 2026-09-22):** this describes ignis as #257 built it and
   the reference as it still is. ignis now attends a chunk before it appends
   it (an Ignis-patched behaviour, ADR 0037), and its ring window is served
   its own rows — see [hq prefill reads the ring before the
   append](2026-09-22-hq-prefill-reads-the-ring-before-the-append.md).
3. **The window is slot state, not page state.** It survives a prefix claim, a
   checkpoint claim, a snapshot and a KV-RAM restore only because it is a CLONE
   section of the state table; and a verify round's rejected columns leave
   ring bits naming rows the sequence did not keep until the round clears them.

## Implications

- Every hq measurement taken before 2026-09-22 describes the codec-only engine:
  #160's first-token divergence, #161's spec-on/spec-off gap, the pointing
  study's set C and C4096 hq arms, and G5's hq cells. They are candidates for
  re-measurement, not results about the format.
- The clobbered ring is a correctness gap — a causality leak inside every
  prefill chunk after the first — that the reference shares. Swapping the two
  launches in `gqa_attention_prompt_launch` when the window is on would serve
  the 512 pre-chunk keys their own rows; it is a patch to a vendored file that
  ADR 0031's bottleneck exemption does not cover, and it would make ignis read
  differently from the reference it is compared against. **Done in #258**
  under ADR 0037, which admits a correctness patch; comparisons with the
  reference now cross that departure.
- Codec coverage inside attention now needs a history longer than the window
  (or the codec-only arm the route agreement test keeps).

## Limits and unknowns

- 20 prompts, one reference build, greedy only. 17/20 against 18/20 is within
  what one different near-tie moves.
- What the clobbered ring costs in quality is not measured here: that needs a
  build with the launch order swapped, measured the same way. (Measured in
  #258's finding: 16/20 against 17/20, within the near-tie noise, and the
  count itself turned out not to measure correctness.)
- The decode route has no tap; its reads are observed only through the ring
  words and through the tokens the lifecycle tests compare.
- The byte-identical codec check needs a capture from a build without the
  window (`IGNIS_TAP_BASELINE`); it is a measurement taken once, recorded
  above, not something the GPU profile re-checks.

## Follow-ups

- The launch order: the owner decided to patch it (ADR 0037, a recorded
  vendored patch with a test that fails without it) — GitHub #258, spec
  runtime/07; patched, see [hq prefill reads the ring before the
  append](2026-09-22-hq-prefill-reads-the-ring-before-the-append.md).
- #160 and #161: re-measure their hq cells with the window on (both issues
  track it; #173 carries the numbers above).
- The pointing study's hq arms (set C, C4096): its own follow-up, per the spec.
