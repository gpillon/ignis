# runtime 06 — hq-e8-2b residual window: the exact keys the reference keeps

GitHub: #257

Under hq-e8-2b, ignis's GQA attention reads **every** key through the codec.
The vendored kernels can keep three kinds of key exact — the first 32 (sink),
the 512 before the current chunk (a recent ring), and the current chunk
itself — but only when the KV view hands them a **residual window**, and ignis
never does. The reference allocates that window for every hq pool. This spec
wires it, and the whole difficulty is not the wiring: it is keeping the ring
true across every state transition a sequence goes through.

## What the kernels do, and when

The window is opt-in per call. `kernel/vendor/src/core/paged_kv_cache.h`
(lines 18-60) documents `residual_k` / `residual_v` / `ring_valid` on
`PagedKVLayerView` and `PagedKVBatchLayerView`: "Empty tensors mean the
feature is off (every cache dtype other than U8, and U8 callers that opt out)."
With the window on:

- **Sink**: keys `< kGqaHqSinkKeys` (32) are read exact, from the side plane,
  with **no validity gate** — the kernels assume they were written before any
  fetch that names them.
- **Recent ring**: a key `k >= 32` lives in ring slot `k & (kGqaHqRecentKeys -
  1)` (512 slots) and is read exact iff that slot's bit is set in
  `ring_valid`. The bit says "this slot holds the row the next fetch will
  name" — it carries **no position**, so keeping it true is the caller's job.
- **Fresh chunk** (prefill only): `has_fresh = cache.residual_k.data !=
  nullptr && new_k.data != nullptr && new_v.data != nullptr`
  (`launcher/gqa_attention_prefill_hq_routes.cuh:70`); the current chunk's
  rows are rotated straight into the scratch, and the ring then serves the 512
  keys *before* the chunk (`gqa_attention_prefill_hq.cuh:160-215`).
  **But it reads the ring after the chunk's own append** (see *As built*):
  `gqa_attention_prompt_launch` enqueues the append before the attention, so
  a key before the chunk whose ring slot the chunk just rewrote is served the
  chunk's row, bit set. Same in the reference.
- Everything else: `hq_decode_row_group`, the codec.
- **Writes are already there.** The hq fill (append) kernel dual-writes every
  appended row that can still be recent — rotated, one BF16 rounding — into
  the side plane and marks its ring bit, whenever `residual_k` is non-null
  (`gqa_attention_prefill_hq.cuh:116-133`, `hq_ring_mark_valid` in
  `hq_codec.cuh:606-614`). Nothing in `kernel/vendor/` has to change.
- Shapes are enforced by `validate_residual` (`ops/wrapper/gqa_attention.cpp:
  76-113`): residual planes BF16 `[256, kv_heads, 32 + 512, slots]`,
  `ring_valid` I32 `[kGqaHqRecentKeys / 32 = 16, slots]`. The comment above
  `hq_ring_slot_valid` that says "four u32 words per slot row" is stale; the
  kernels index `slot * 16`.

## What the two engines do

- **ignis**: `ignis_kv_layer_view` and `ignis_kv_batch_layer_view`
  (`kernel/include/ignis_seq_internal.h:425, 439`) never set the three
  tensors, and nothing in `kernel/src` allocates them. So the prefill route
  runs with `has_fresh == false` and no side row ever fires, and the decode
  route decodes its whole visible history.
- **Reference** (`F:/ai/q38/ninfer`, branch `gpillon/coding` at `a00648cb` —
  the build #173 compares against):
  `src/targets/qwen3_6/impl/state/decoder_state.cpp:68-85` allocates, for
  every hq pool, `residual_k` / `residual_v` BF16 `{head_dim, kv_heads, sink +
  recent, layers * table_rows}` (one tensor, each layer a dim-3 slice) and
  `ring_valid` I32 `{16, table_rows}` shared by all layers ("appends are
  position-driven and therefore layer-uniform"); the views slice them per
  layer and per slot row at `decoder_state.cpp:159-192`.

## Evidence that ignis reads every key decoded

Measured with the consumed-key tap (`kernel/include/ignis_attn_tap.h`,
`crates/core/src/attn_tap.rs`, commit `a774f61` on `point-one-pass`), which
captures the rotated-frame scratch rows the hq prompt route attended over and
compares each with the rotated pre-codec key: on the committed 4096 px pointing
fixture, served render, GQA layer 39, **16,506 rows × 4 KV heads, 0 exact**;
relative L2 min 0.3329, median 0.3696, max 0.7836 — the codec's own
distribution (`docs/findings/2026-09-12-hq-attention-route-agreement.md`).
The three-source rule would have kept 666 of those positions exact.
`crates/server/tests/attn_tap_hq_consumed_gpu.rs` asserts exactly this today.

## Why it matters

- **#173** (open): on 20 token-identical prompts with tools, greedy, thinking
  off, the reference stops at a tool call on 18/20 and ignis on 7/20 at
  hq-e8-2b. Its spec lists "hq target-path numerics (same root as #160/#161)"
  as a candidate class, and the reference there is the `a00648cb` build that
  keeps this window. **Candidate cause, not a proven one.**
- **#160** (closed): ignis hq picks a different greedy first token than the
  reference on G5 depth prompts while ignis BF16 matches it token for token.
  **#161** (open): spec-on vs spec-off divergence at hq.
- The keys this leaves decoded are the ones attention weighs most: during
  decode, the 512 most recent.

What this does **not** fix: the pointing head of
`docs/findings/2026-09-22-the-codec-costs-the-head-its-read.md`. An image is
almost entirely older than the recent window (at 4096 px all but ~512 of
16,384 positions), so the head's read stays decoded with or without it.

## The seam

Nothing under `kernel/vendor/` changes. The work is allocation, views, and
keeping `ring_valid` and the sink rows true across every transition:

1. **Allocation, hq pools only**, in the sequence pool: residual K and V
   planes and `ring_valid`, sized for every block-table row including retained
   slots. It is a **VRAM-plan line** (ADR 0030: reserved at load, never per
   request): `(32 + 512) × 4 × 256 × 2 B` = 1.06 MiB per plane per slot per GQA
   layer, **~34 MiB per slot** across 16 layers for K and V; ~272 MiB at 8
   slots. The KV page budget shrinks by that amount. BF16 pools allocate
   nothing.
2. **Views**: both view builders fill the three tensors, sliced per GQA layer
   and per slot row the way the reference does. The captured decode graphs
   (ADR 0019 slot indirection, ADR 0020) address the batch view by table row;
   the planes must cover every row they can name.
3. **Ring lifecycle** — each of these is a transition where a stale bit, or a
   stale sink row, is read as exact with no error anywhere:
   1. **A slot gets a new sequence** (`ignis_seq_alloc`, `seq.cu:741`): clear
      the row's 16 ring words. Sink rows are safe only because a sequence
      prefilled from position 0 rewrites all 32 before any fetch; every path
      that binds a slot *without* prefilling 0..31 must carry them instead.
   2. **Backward trim / prefix claim** (`ignis_seq_alloc_against_prefix`,
      `ignis_seq_alloc_shared` in `seq_prefix.cu:264, 358`;
      `ignis_seq_alloc_from_checkpoint`, `seq_checkpoint.cu:281`; ADR 0029,
      #183-#191): the side rows must follow the KV they came from, and then
      the ring is **revalidated** to the new window — a slot stays valid only
      if its last writer before the trim lies in `[base - 512, base)`.
      Reference: `revalidate_residual_ring`, `decoder_state.cpp:200-224`,
      called after every trim at `program_impl.h:621-635`.
   3. **Rejected speculative drafts** (the DFlash2 verify round's commit in
      `kernel/src/step.cu`, the licensed-token path around lines 1334-1560):
      the append wrote ring slots for positions that were then rejected,
      clobbering exact rows of older keys still inside the window. Invalidate
      the slots of `[committed, produced)`. Reference:
      `invalidate_residual_ring`, `decoder_state.cpp:226-246`, called at
      `program_impl.h:~960-975`.
   4. **State transfer** (`ignis_seq_snapshot` / `ignis_seq_restore`,
      `seq.cu:837-943`; `ignis_seq_retained_store`, `seq.cu:1169`; checkpoint
      capture/snapshot, `seq_checkpoint.cu:114-142`; prefix snapshot,
      `seq_prefix.cu:385-409`; the KV-RAM arena, #213; ADR 0024): a snapshot
      carries the slot row's side rows and ring words, and a restore
      overwrites the destination row with them. The reference hit exactly
      this bug and documents it at `program_impl.h:1126-1140` ("a restore left
      the destination row holding whatever sequence used it last"); its
      host records carry the side store as their own sections
      (`kv_ram_cache.cpp`, `Section::TextResidualK` onward).
4. **Two tests assert today's behaviour and must flip**:
   `attn_tap_hq_consumed_gpu.rs` and the consumed-hq self-check in
   `attention_head_point_gpu.rs` both fail on any exact row ("the hq residual
   window is on in this build … the three-source rule applies"). After this
   change they must assert the three-source rule instead; their failure on
   the first run is the first confirmation the window is live.

## Acceptance

In this order — each catches a wrong fix before the next costs a GPU run.

1. **Row-level, fresh prefill**, through the consumed-key tap: on the 1024 px
   and 4096 px pointing inputs, every row the three-source rule calls sink,
   ring or fresh is exact (relative L2 < 0.1 against the rotated pre-codec
   key; one BF16 rounding sits at ~0.004), and every row it calls codec is
   **byte-identical** to the capture before this change. The measured codec
   fraction of the image equals the rule's (40.1% on the generated 1024 px
   pointing scenes, 96.3% on the committed 4096 px fixture).
2. **Lifecycle**, one GPU test per transition in *The seam* 3.1-3.4 — a
   reused slot, a claimed prefix, a checkpoint restore, a rejected draft round,
   a KV-RAM restore — each showing the ring bits and sink rows are exactly what
   the rule predicts for the sequence now in the slot, and never a previous
   occupant's. The existing harnesses to extend are `prompt_checkpoint_gpu.rs`,
   `retained_prefix_gpu.rs`, `seq_snapshot_gpu.rs` and `dflash2_round_gpu.rs`.
3. **Route agreement**: `kernel/tests/test_hq_route_agreement.cu` still passes;
   its new median relative L2 is reported beside the 2026-09-12 finding's
   (expected lower: recent keys are now exact).
4. **VRAM plan**: the new reservation is its own named line, and the printed
   plan still matches measured device memory as ADR 0030 requires.
5. **No regression**: `cargo test` green workspace-wide; hq decode throughput
   at one and eight lanes within noise of before.

**Measured after, reported, not a gate:** #173's twenty-prompt comparison
re-run with the window on. If the 7/20 vs 18/20 split moves, comment the
numbers on #173; if it does not, the window is still correct and #173 has
another cause. G5 is not required for this change.

## As built (2026-09-22)

Wired as the seam says; what building it found, and where it departs:

- **The prompt route reads the ring after its chunk's append.** For a chunk
  `[p0, p0 + w)` the vendored `gqa_attention_prompt_launch` runs the append
  (which dual-writes the chunk's last `min(w, 512)` keys into their ring slots
  and sets their bits) before the scratch decode that serves `[p0 - 512, p0)`
  from the ring. So the keys `[p0 - 512, p0 - 512 + min(w, 512))` are served
  the rows of the chunk keys that share their slots — exact rows, of the wrong
  key. The reference (ninfer `a00648cb`) launches the same order. Measured
  through the tap (`attn_tap_hq_consumed_gpu.rs`, GPU run 2026-09-22): on both
  pointing inputs the query chunk is 122 wide at 16,384 / 1,024, and 122 of
  the 512 ring rows come back exact **to the chunk key 512 positions later**
  (rel. L2 0.0019-0.0020 to it), the other 390 exact to their own. At the
  serving chunk of 1,024, every full chunk after the first loses the whole
  ring this way. Kept as the reference has it — `kernel/vendor/` is unchanged
  and #173 compares against a reference that does the same — and written down
  as `ignis_core::hq_ring::PromptSource::Clobbered`. Swapping the two launches
  when the window is on would give the rule this spec first described; that
  is a vendored-file patch outside ADR 0031, the owner's call.
- **AC 1 as measured.** Fresh 122, sink 32, ring 390, clobbered 122, codec the
  rest (96.3% of the image at 4096 px, 40.1% at 1024 px); every exact row within
  0.0020. "Codec rows byte-identical to the capture before this change" holds
  where the key the codec is given is unchanged: at GQA ordinal 0 (L3), with
  only GDN layers above it, 15,840 of 15,840 codec positions (480 of 480 at
  1024 px) have the baseline's key bit for bit and decode to the baseline's
  bytes. At L39 no key is the baseline's — the GQA layers above it now attend
  over exact rows — so the check does not apply there.
- **No revalidation call site.** ignis never trims a live sequence back: a
  prefix is published at exactly its end, a checkpoint captured at exactly its
  opener, a snapshot taken at a chunk boundary, and a claimant or a restore
  resumes at exactly that point. The clone and the blob carry the window as a
  CLONE state section (`IGNIS_SEQ_SECTION_HQ_RESIDUAL`, snapshot format 4), so
  there is nothing to revalidate. The reference's rule is kept host-side
  (`ignis_hq_ring_revalidate_mask`, `kernel/include/ignis_hq_ring.h`) for tests.
- **A failed verify round clears its columns too**, not only a committed one:
  a retry at a smaller extent would otherwise read a failed round's rows as
  older keys'. A guard in `run_verify_round` clears `[p, p + extent + 1)` of
  every lane on any failure after the pass is enqueued.
- **The prefill view narrows the window to the sequence's slot.** The layer's
  prefill path hands A1 the sequence's own block-table row as a one-row table,
  so the window is pre-sliced to that slot; before this, an hq prefill on any
  slot but 0 was refused by the vendored shape check (the lifecycle tests
  found it on slot 3).
- **The route agreement test keeps a codec-only arm.** Its histories fit the
  window, so the production view now agrees with BF16 to one rounding (prefill
  W=200 0.0023, decode 0.0032-0.0033); the same arms over a view with the
  window taken off reproduce the 2026-09-12 numbers exactly (0.227955,
  0.174-0.190) and keep the codec's read path inside attention under test.
- **Lifecycle** (AC 2): `retained_prefix_gpu.rs`, `prompt_checkpoint_gpu.rs`
  (checkpoint claim, KV-RAM restore of the materialized blob, chained turn),
  `seq_snapshot_gpu.rs` (restore into the slot another sequence used) and
  `dflash2_round_gpu.rs` (11 rounds over 4 lanes, 65 rejected columns) run
  under hq as well as BF16: the claimant's window is the publisher's at the
  prefix's end, the whole snapshot after the same tail is the split control's
  byte for byte, the continuation tokens match, and the ring words read out of
  the blob after the prefill, after one-token decode rounds and after every
  verify round are the host rule's (`ignis_core::hq_ring::HqRing`).
- **VRAM plan** (AC 4): `hq_residual_window`, its own line —
  `ignis_seq_pool_plan::hq_residual_bytes`, read back off the pool and held
  equal by `vram_plan_gpu.rs`; 34 MiB per slot, lanes and retained slots alike.
  The retained image's `image_bytes` includes it; `slot_state_bytes` does not.

## Out of scope

- The pointing head's decoded read (above) and its exact-key copy, which is a
  separate change.
- Re-measuring the `point-one-pass` study's hq numbers (set C, C4096): they
  describe ignis *without* the window and say so; re-measuring them is that
  study's follow-up.

## References

- ADR 0022 (two KV formats, BF16 as oracle), ADR 0024 (sequence state
  transfer), ADR 0029 (cross-request reuse), ADR 0030 (device memory reserved
  at load), ADR 0019 / 0020 (decode graphs, batch-wide rounds).
- `docs/specs/runtime/04-reference-feature-floor.md` (where hq-e8-2b came in).
- `docs/findings/2026-09-12-hq-attention-route-agreement.md`,
  `docs/findings/2026-09-22-the-codec-costs-the-head-its-read.md`.
- GitHub #173, #161, #160.
