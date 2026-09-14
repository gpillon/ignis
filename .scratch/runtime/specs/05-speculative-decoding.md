# runtime 05 — speculative decoding: the verify round, then the DFlash2 drafter (gate G5)

GitHub: #66 (phase 5 master; blocked by #65, gate G4 — closed 2026-09-13)

Source: `.scratch/REVIEW-2026-09-05.md` §6 (Phase 5), `.scratch/ROADMAP.md`
(the G5 row), the G3 constraints recorded on #66 (2026-09-09), and the
grilling session of 2026-09-13 whose decisions are in
`.scratch/DEFERRED-DECISIONS.md` ("From the G5 grilling session"). Builds
directly on `.scratch/runtime/specs/04-reference-feature-floor.md` (G4).

ADRs respected: 0005 (performance-first above a sane-output floor; the
north-star is the coding engine, not raw tok/s), 0006 (exclusive GPU testing),
0007 (performance gate, correctness self-checked), 0009 (step-level
device-resident ABI), 0010 (vendored reference kernels), 0015 (live/live gate
on cold samples), 0016 (steps grow by options structs), 0018 (chunk-level
interleaving), 0019 / 0020 (decode graph slot indirection, batch-wide round),
0021 (a live/live gate pools at least two launches per engine), 0022 (BF16
stays the correctness oracle), 0024 (sequence state transfer through the
section table).
No ADR is introduced: every decision below is reversible through a versioned
blob, an options struct or a load flag.

Reference: ninfer at the manifest's pinned commit `a00648cb`
(`kernel/vendor/manifest.json`; the manifest names the branch
`feat/dflash2-local`, the tree is on `gpillon/coding` at the same SHA).

## Problem Statement

Every decode round today produces exactly one token per lane, and the cost of
a round is the cost of streaming 15.85 GB of weights through the card once
(`REVIEW` §8: a 9.47 ms floor, 13.2 ms measured at C=1). The reference stops
paying that per token: with its DFlash2 drafter it verifies seven proposed
tokens in one traversal and commits 3.4–5.75 of them per round, which at the
depths agentic prompts live at is the difference between 79 and 144 tok/s
(`SEPT-OPTIMIZAZION.md` §1, 196K). ignis at 196K is a plain decode loop.

Three facts shape the work.

**The substrate is most of it; the drafter is the small part.** Whatever
proposes the tokens, a speculative round needs: verify inputs of k+1 columns
per lane, a target traversal over those columns that records instead of
advancing the GDN state, an accept kernel with a greedy branch and a
distribution-preserving branch, KV commit and trim to the accepted length, a
ReplaySSM fold that rebuilds the GDN slot and the conv taps from the accepted
prefix, and one CUDA graph per batch width at the verify window. Of that,
ignis has vendored only the pieces that were never speculative: the hq
width-8 verify attention tile (`gqa_attention_decode_hq.cuh`), the ReplaySSM
records and fold (`gdn_replay*`, `replay.cpp`) and the sampling device
primitives. Nothing above them exists.

**The drafter consumes hidden states, not tokens.** DFlash2 is fed the
target's hidden states at layers 5, 19, 33, 47 and 61, concatenated and
projected (`dflash2/feature_projection [5120, 25600]`), into a 2048-token
sliding BF16 KV window of its own, with a rewrite checkpoint of that window.
So the target's prefill must export those features, the drafter has its own
per-sequence state, and "rebuild the drafter on restore" would mean a
64-layer target pass over 2048 tokens. That fact decided the state question
(`DEFERRED-DECISIONS.md`, G5 item 2).

**The Rust seam emits one token per lane.** `Compute::decode_step` returns a
`DecodeOutcome::Token(TokenId)` per job (`crates/core/src/scheduler.rs:73`),
and the server, the stop rules, the request log and the prefix-publish path
all assume a round moves a sequence by one. A round that commits 1..k+1
tokens changes that contract at exactly one seam, and the sequence state must
never run ahead of the text a request emitted, or a published prefix carries
tokens past an EOS.

## Solution

**One verify round, one drafter, chosen at load.** The leaf gains a
speculative round behind the existing `ignis_program_decode` entry point
(ADR 0016: an options struct, not an `_ex` variant). With no speculative
option the call is today's width-1 round on today's graphs; with one, every
decode-ready lane goes through the verify round at a batch-wide window of k,
and a lane whose remaining budget or context is shorter than k runs at a
per-lane extent down to 0 (a fallback step inside the same round, as the
reference does). Widths are never padded (G3 constraint 1): graphs are
captured per exact batch width 1..8 at the one window the load fixed, eight
graphs, not fifty-six, because DFlash2 has no adaptive width.

**The accept rule is vendored, both branches.** The reference's one accept
kernel serves greedy (longest matching prefix, target argmax at the
divergence) and temperature > 0 (accept `d_i` with probability
`p_target(d_i)`, resample the rejected column from the masked residual, draw a
bonus token when every draft survives), with a stateless per-sequence RNG
keyed by seed, position and purpose. That is what keeps G3's property — a
request's output depends on its own seed alone, never on its round-mates —
without ignis inventing a rule. Greedy speculation is lossless by
construction, which is the correctness oracle of this phase.

**The commit is stop-aware on the device.** The per-lane sampling params gain
the remaining token budget and the stop ids; the accepted prefix is cut at
the first stop before KV commit and fold. A sequence therefore never holds
state past its emitted text, and prefix publish after a turn stays correct
without the Rust side reaching into the leaf.

**The drafter's state is carried, not rebuilt (ADR 0024).** Two new sections
in the leaf's table — the drafter's window and its checkpoint, clone
semantics — plus the drafter frontier in the progress image. Snapshot,
restore and device clone carry them through the machinery G4 built; the blob
version bumps. The chunk-boundary snapshot-point permission is re-earned for
them by the round-trip test, per the G3 caveat.

**The Rust seam carries a run.** `DecodeOutcome` carries the committed tokens
of the round for that lane, in order, and the same finish reasons as today;
the server streams the run as it streams tokens now, one SSE delta per token
or one per run — the protocol is unchanged either way. The mock `Compute`
keeps emitting runs of one, so every CPU test of the scheduler stays valid.

**The reference lane is DFlash2-7 alone.** The gate compares ignis with its
speculation on against the reference launched with `--spec dflash2
--draft-tokens 7` and hq-e8-2b, live/live, at three depths. MTP is not
measured and not built (`DEFERRED-DECISIONS.md`, G5 item 1).

## User Stories

1. As the engine owner, I want speculation chosen at load with a backend and a draft window, so that the serving profile is a flag and the spec-off engine is the same binary.
2. As the engine owner, I want a load without the speculative option to bind nothing of the drafter, so that today's VRAM footprint and today's graphs are untouched until asked.
3. As the engine owner, I want the verify round behind the existing decode entry point through an options struct, so that the step ABI does not grow a second decode.
4. As the engine owner, I want the accept rule vendored with both its branches, so that temperature > 0 stays distribution-preserving and a request's output still depends on its seed alone.
5. As the engine owner, I want greedy speculation to produce exactly the tokens the spec-off engine produces, so that one equivalence test is the correctness oracle for verify, accept, rollback and fold together.
6. As the engine owner, I want the accepted prefix cut at the first stop on the device, so that no sequence state ever runs ahead of the text a request emitted.
7. As the engine owner, I want the target prefill to export the drafter's feature taps for its window only, so that the drafter's TTFT cost is bounded by 2048 tokens and not by the prompt.
8. As the engine owner, I want the drafter's window and checkpoint to be sections in the leaf's table, so that snapshot, restore and clone carry them or refuse the blob, never silently drop them.
9. As the engine owner, I want the drafter sections' snapshot-point permission re-earned by a test, so that the G3 caveat is honoured as an act.
10. As the engine owner, I want graphs captured per exact batch width at the load's window, so that a narrow round never pays a wide round's GDN slot traffic.
11. As the engine owner, I want a lane whose budget or context is shorter than the window to run at its own extent inside the round, so that the last tokens of a request do not need a second decode path.
12. As the engine owner, I want the Rust seam to carry a run of committed tokens per lane, so that the change to "one token per round" lives at one seam and the mock stays a run of one.
13. As the engine owner, I want the request log to carry rounds, drafted and accepted counts per request, so that acceptance in real agent traffic is a number and not a guess.
14. As the engine owner, I want committed tok/s counted the reference's way — first token excluded, rejected tokens excluded, decode phase only — so that the ratio compares one metric on both sides.
15. As the engine owner, I want the G5 cells cut from the corpus at three depths, cold, greedy, 512 committed tokens, so that decode dominates the measurement and TTFT does not enter the ratio.
16. As the engine owner, I want the verify substrate testable with a fake drafter before the real one exists, so that the accept and rollback paths are proven on their own seam.
17. As the engine owner, I want the gate run once at the end of the phase and nothing gate-shaped per ticket, so that the phase's cost is the engine work and not the benchmark.

## Implementation Decisions

**Scope boundary.** This phase builds the verify substrate and the DFlash2
drafter. It does not build MTP, an adaptive window, packed prefill, true
prefill/decode overlap (G3 constraint 4, ADR 0018), a second speculative
backend selectable per request, or vision.

**The step ABI (ADR 0016).** `ignis_program_decode` keeps its signature. A
`struct ignis_decode_options { size; speculative_window; out_committed_counts; }`
selects the round: `speculative_window == 0` (or a NULL pointer) is today's
round; `speculative_window == k` runs the verify round and fills
`out_token_ids` as `[batch][k+1]` with `out_committed_counts[i]` tokens valid
per lane. The window is fixed for the life of a model load: it is the number
the graphs were captured at, and a call with a different window is rejected,
not padded. `ignis_sampling_params` gains `remaining_tokens` and a stop-id
list; the leaf cuts the accepted prefix at the first stop and clamps the
lane's extent to `min(k, remaining_tokens, remaining_context)`.

**Model load.** `ignis_model_load` gains the speculative backend and window
through its options. Loading DFlash2 binds the 66 `dflash2/*` objects of the
v2 artifact (`docs/maintainer/qwen3.8-27b-artifact.md` in the reference tree
names them; the binder's inventory today leaves them unconsumed by design),
allocates the drafter's per-lane window pool (BF16, 5 layers × 2048 × 8 KV
heads × 128 × K+V = 40 MiB per lane, ×2 with the checkpoint: 640 MiB for
eight lanes) and captures the eight verify graphs. Without the option, none of
that happens and the binder's text scope is unchanged.

**The verify round.** Per lane: column 0 is the anchor (the pending
successor), columns 1..extent the drafts, the tail padded with the anchor;
positions `base + j`. The traversal runs the 64 layers over `batch × (k+1)`
columns with the GDN layers in record mode (ReplaySSM records: conv, key,
value, gate planes at `width = k+1`) and the GQA layers on the hq width-8
small-T verify tile (`gqa_small_t_partial_hq`, one pass for draft 7 + bonus).
Then accept, select the accepted hidden state, cut at the stop, commit: KV
valid frontier moves to the accepted length and the pages past it are
trimmed, the fold rebuilds the GDN slot and the conv taps from the accepted
records (a zero-length commit is a strict no-op), the progress image
advances. Under BF16 the verify tile chunks at 6 and takes two passes; that
route exists for the oracle only (ADR 0022) and its speed is not a cell.

**Accept and RNG.** Vendored verbatim (ADR 0010): `speculative_round.cuh`'s
accept kernel and its large-vocab partial-top-k route, on the already
vendored `sampling_device.cuh`. The RNG is `sampling_uniform(seed, position,
purpose)`, stateless; the three purposes (accept, correction, bonus) are
distinct streams. Presence/frequency penalties see a round-local overlay of
the drafts accepted so far in the same round, which is how the per-sequence
penalty row stays exact across a multi-token commit.

**The drafter.** Vendored verbatim: `dflash2_draft.{cuh,cu}` (two-tap dynamic
conv, selector lattice scores, per-column top-k, selector walk,
predecessors), `cyclic_kv_cache`, the drafter's five-layer forward and the
propose path. Its inputs per round are the target's feature taps for the
columns just verified, which the traversal already produces; its output is
the next round's drafts. Its window is filled at the end of prefill from the
feature taps of the last 2048 prompt tokens, which is the reference's TTFT
penalty and is accepted here: the drafter prefill sits on the critical path
before the first token, as in the reference, because prefill/decode overlap
is out of scope.

**Feature taps.** The target prefill and the verify traversal both export the
hidden states at layers 5, 19, 33, 47 and 61 into chunk-scoped scratch with
no lane dimension (the reference's `prefill_features` / `pending_features`).
They are never persisted and never a section.

**Sections (ADR 0024).** `IGNIS_SEQ_SECTION_DFLASH_WINDOW` and
`IGNIS_SEQ_SECTION_DFLASH_CHECKPOINT`, both `CLONE`; the drafter frontier
joins the progress image. The blob format version bumps; a blob from a load
without the drafter is refused by a load with it and vice versa, by the same
header check G4 built. The snapshot floor grows from 149 MiB to 229 MiB; the
default host budget (`--kv-host-pool-bytes`, 2 GiB) is not changed by this
phase, and raising it is an operator decision the numbers in
`DEFERRED-DECISIONS.md` inform.

**The Rust seam.** `DecodeOutcome::Token(TokenId)` becomes a run:
`DecodeOutcome::Tokens(run)` with `run.len() >= 1`, plus
`Finished(FinishReason)` unchanged, or equivalently a `Vec<TokenId>` per job
followed by an optional finish — the ticket picks the shape that keeps
`Scheduler` and the mock's diff smallest. The server emits one SSE delta per
token of the run; `usage` counts committed tokens. The request log gains
`spec.rounds`, `spec.drafted`, `spec.accepted` per request (the reference's
counters, same names where they exist). Spans stay at decode-round
granularity (CONTEXT.md, `request.id`).

**CLI.** `ignis-server --spec dflash2 --draft-tokens N` (N in 1..7);
`--spec` absent means off. No per-request field: speculation is engine
residency, frozen at load, as in the reference.

**What is not built.** An adaptive window (MTP-only in the reference, and MTP
is deferred). A width-8 int8 verify tile (the reference's `f92234ca` is not
in the pinned tree and int8 KV is not an ignis format). Per-request backend
selection. `--lm-head-draft` (an MTP proposal head).

## Gate G5

The 99% performance gate (ADR 0007) at three context depths, ignis with
DFlash2 against the reference with DFlash2-7, hq-e8-2b on both sides,
measured live/live (ADR 0015) over at least two independent process
launches per engine (ADR 0021), in one GPU-exclusive session, **once, at the
end of the phase**.

| cell | measurement | verdict |
|---|---|---|
| committed tok/s @ 24K, @ 98K, @ 196K | C=1, corpus prompts cut cold at each depth, greedy, 512 committed tokens, first token excluded, decode phase only | pooled ratio ≥ 0.99 against the live reference, per depth |
| greedy equivalence | the canaries with speculation on and off, same seed | identical token sequences (the correctness floor, not a ratio) |
| canary self-consistency | greedy, fixed seed, sane output, speculation on | exit 0 |
| G4 trace replay under speculation | one replay per engine, speculation on both sides | **informational**: ratios reported per class, no threshold |

**Engine configuration.** Both engines at `--max-context 262144`,
`--prefill-chunk 1024`, hq-e8-2b, matched KV capacity (465,984 tokens as G4
run 2), CUDA graphs on, prefix reuse on. Reference: `--spec dflash2
--draft-tokens 7` and otherwise the flags G4's legs used, not the
`ram-dflash2` preset (which caps concurrency at 6 and sets a host tier);
ignis: `--spec dflash2 --draft-tokens 7`. The reference's `SEPT` table is
sizing information, never the reference side of a cell.

**Committed tok/s.** Defined as the reference defines it
(`docs/serving.md`, "decode counts tokens finally committed by decode rounds,
excluding the first token produced by prefill"): `(completion_tokens − 1) /
decode_seconds`, where decode seconds start at the first token and end at the
last. `ignis-bench` already has both timestamps in its `Outcome`; the cell is
the C=1 throughput cell of `g3` at three prompt depths with that counter, not
a new instrument. Pooling across launches is `g4-gate`'s per-cell rule reused.

**Method.** Four launches in the order reference, ignis, reference, ignis,
each stopped and restarted. The equivalence and canary cells run once against
one ignis launch. The trace replay runs once per engine, last. No cell is
re-run to change its number; a miss is filed.

**Why nothing gate-shaped runs per ticket.** The G3 and G4 sessions cost more
than the engine work they checked (`DEFERRED-DECISIONS.md`, G5 item 4).
Inside the phase the check is the op test and, for the substrate, the
equivalence test, which catches a large error in verify, accept, rollback or
fold with one launch and no reference.

## Testing Decisions

- Vendored op tests: the reference's tests for the accept kernel, the
  dynamic conv, the selector lattice and the cyclic KV window, at 27B
  geometry, in the leaf's test binary.
- Verify substrate with a fake drafter (the internal seam): a drafter that
  proposes the spec-off greedy continuation must yield 100% acceptance and
  the spec-off tokens; a drafter that proposes random ids must yield the
  spec-off tokens at ≤ 1 accepted per round. Both greedy and at temperature >
  0 with a fixed seed, where the sampled output must equal the spec-off
  sampled output for the same seed (the RNG is keyed by position, so the
  streams coincide).
- Stop-aware commit: a run whose accepted prefix contains the stop id commits
  through the stop and no further; the sequence's position equals the emitted
  length; a prefix published after it is claimable and yields the same text.
- Graph replay: the verify round's replay matches eager at every batch width
  1..8, and an interleaved prefill chunk between two replays changes nothing
  (the test #102 shipped, at the new window).
- Sections: snapshot with the drafter, release, restore into a fresh handle,
  continue with speculation on, same tokens as never evicted; a blob without
  the drafter sections is refused by a drafter-bearing load and vice versa;
  a device clone for prefix reuse yields the same tokens on the sibling.
- Rust seam: scheduler and server CPU tests with a mock that returns runs of
  length 1..k+1, including a run that ends in EOS and one that ends at
  `max_tokens` mid-run; the request log's counters; the CLI parse.
- Bench: the three-depth cell and its pooling are CPU-tested against a stub
  endpoint; only the gate run itself needs the GPU (ADR 0006).
- Every GPU test runs under the explicit profile (`scripts/gpu-profile.ps1`),
  one process on the card, per `docs/agents/testing.md`.

## Out of Scope

- MTP as a drafter, `--lm-head-draft`, the adaptive window
  (`DEFERRED-DECISIONS.md`, G5 item 1).
- Rebuilding the drafter's state on restore instead of carrying it (G5 item 2:
  re-opened on a measured restore rate, not on argument).
- Packed prefill (decided after #92), prefill/decode overlap (phase 6),
  more than one active prefill (G4, unchanged).
- The G4 `sub` cell (#148): an optimization ticket beside this phase.
- Vision.

## Further Notes

- The reference's `--adaptive-mtp` and `--lm-head-draft` are rejected with
  `--spec dflash2`; the DFlash2 lane has one knob, the window.
- The pinned tree carries the width-8 hq verify tile (commit `9f36497c`'s
  content) but not the width-8 int8 tile (`f92234ca`); neither matters to
  ignis beyond the note above.
- `README.md`'s artifact section now says what the drafter module is for;
  `CONTEXT.md` defines **DFlash2** and marks **MTP** deferred. Terms this
  phase adds to the glossary when they land in code: drafter, draft window,
  verify round, extent, committed token, ReplaySSM record and fold.
