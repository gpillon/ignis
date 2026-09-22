# runtime 07 — hq-e8-2b: the prefill reads its residual ring before the chunk overwrites it

GitHub: see the issue titled "hq-e8-2b: prefill residual ring is overwritten
before attention" (filed from #257).

With the hq-e8-2b residual window wired (#257, spec runtime/06), the vendored
prompt route serves some keys *before* a prefill chunk the exact rows of keys
*inside* it — later positions, up to ~1,000 ahead of the query reading them.
hq prefill attention is not causal there. The reference (ninfer `a00648cb`)
does the same. This spec makes ignis read the ring as it stood before the
chunk: the first **Ignis-patched** behaviour under ADR 0037.

## The problem

The window serves three kinds of key exact (runtime/06): the 32 sinks, a
512-slot recent ring — key `k >= 32` in slot `k & 511`, exact iff the slot's
bit is set — and, in a prefill, the chunk being attended ("fresh", rotated
straight into the scratch). For a chunk `[p0, p0 + w)` the scratch decode
serves the 512 keys *before* the chunk, `[p0 - 512, p0)`, from the ring
(`gqa_attention_prefill_hq_scratch_kernel`, the `has_fresh` branch: the ring
bound is `fresh_from - kGqaHqRecentKeys`).

That is right only if the ring still holds those keys when the scratch reads
it. It does not: `gqa_attention_prompt_launch` enqueues the chunk's **append**
first and its **attention** second. The append (the hq fill kernel)
dual-writes the chunk's last `min(w, 512)` keys into their ring slots and sets
their bits. A slot is `k & 511`, so chunk key `p0 + i` lands on the slot of key
`p0 + i - 512`, which is inside `[p0 - 512, p0)`. When the scratch decode then
reads that key, the bit is set and the row is the chunk key's.

- For `w <= 512`: keys `[p0 - 512, p0 - 512 + w)` are served the rows of
  `[p0, p0 + w)`.
- For `w >= 512` (every full chunk at the serving width of 1,024): the append
  writes `[p0 + w - 512, p0 + w)`, which covers every slot, so all 512 keys
  before the chunk are served rows of the chunk's last 512 keys.
- A query at `p0 + q` is allowed by the causal mask to see those positions; what
  it gets is the key and value of `p0 + q + d` for `d` up to `w - 1 - q`
  (~1,000 at `w = 1024`). The K rows carry the later position's rotation too.
- The first chunk of a prompt is unaffected (nothing before it), and so is
  every chunk of a single-chunk prompt. Every multi-chunk prompt from its
  second chunk on is affected, and so is every prefill that continues a
  claimed prefix or a checkpoint (its first chunk starts at `p0 > 0`).
- **Decode and verify rounds are unaffected.** Their kernel appends in place
  before it reads, but its ring bound is the tile's last position + 1 minus
  512, so a slot its own columns rewrite belongs to a key already outside the
  window.

## Evidence

Measured through the consumed-key tap on #257's branch
(`attn_tap_hq_consumed_gpu.rs`, `docs/findings/2026-09-22-the-residual-window-was-the-tool-call-gap.md`).
Both pointing inputs have a 122-wide last chunk; of the 512 ring rows, **122
come back exact to the chunk key 512 positions later** (relative L2
0.0019-0.0020 to it) and 390 exact to their own key. The host model of the
rule is `ignis_core::hq_ring`; its `PromptSource::Clobbered` classifies exactly
these rows, and the test holds every captured row to it. The vendored files
involved are byte-identical to the reference at `a00648cb`.

## What correct means

**The prompt route reads the ring as it stood before the chunk it is
attending.** For a chunk `[p0, p0 + w)`, every key `k` in `[p0 - 512, p0)` whose
ring bit is set is served *its own* exact row; the sinks and the chunk's own
keys are exact as today; everything else is the codec. After the call the
ring holds the chunk's rows and bits exactly as today, so the next chunk, a
decode round, a snapshot or a clone sees no difference. The codec planes are
unaffected either way: the codes written are the same bytes.

This is the rule runtime/06 first described ("the 512 before the chunk exact")
and the one the kernel's own comments state; the reference's behaviour is the
departure from it.

## Constraints

- **The fix is a recorded patch to vendored code (ADR 0037).** The contract is
  broken by the order of two launches inside a vendored launcher, and the
  smallest change that restores it is local to that launcher. It carries a
  `reason` naming the bug and its test, becomes a row of `VENDOR.md`'s
  Ignis-patched behaviours, and must keep `scripts/vendor-ninfer.ps1 verify`
  (and so `crates/vendor`'s tests) green.
- **Only the hq route with the window on may change.** The BF16 and INT8 prompt
  routes read the chunk's keys from the cache, so their append must precede
  their attention; the hq route without a window (`has_fresh` false) also
  decodes the chunk's own codes. With the window on, the hq route never reads
  the chunk's codes — the fresh pass stages those rows, and the scratch decode
  skips every position `>= p0` — so its attention does not depend on its
  append having run.
- **Every shape of the hq prompt route**: dense and masked (`valid_columns`),
  and the banded form for histories wider than one scratch band, whose bands
  each re-run the fresh pass.
- **Nothing else moves.** Decode, verify, the ring lifecycle of runtime/06
  (clones, snapshots, invalidation) and the VRAM plan are untouched.

Approaches that look simpler and do not work, for the record:

- clearing the bits of the slots the chunk will write, before the call — the
  append sets them again before the read;
- handing the prompt route a separate copy of the ring words — the append
  marks that copy too, and the *rows* are overwritten regardless of the bits;
- moving the ring bound to the history's last 512 keys minus the chunk — causal,
  but it serves no pre-chunk key exact once `w >= 512`, i.e. at every full
  serving chunk: it trades the leak for most of the window.

## Acceptance

1. **A test that fails without the patch, in the suite the GPU profile runs.**
   The consumed-key tap needs the `attn-tap` feature, which the profile does
   not build, so the guarding test is the kernel's own (CTest): a prompt-route
   call at `p0 >= 512` with the window on, over rows whose later keys would be
   told apart from the earlier ones, fails against the reference's launch order
   and passes with the patch.
2. **Row level, through the tap**, on both pointing inputs: all 512 ring rows
   exact to their own key (0 clobbered), sinks and fresh rows exact as today,
   and the codec rows at GQA ordinal 0 byte-identical to the capture #257's
   build takes. `ignis_core::hq_ring` describes the Ignis-patched rule (the
   ring as the chunks *before* the query chunk left it), and the tap tests hold
   the capture to it.
3. **No regression**: `test_hq_route_agreement.cu`, the runtime/06 lifecycle
   GPU tests and `make ci` green; hq prefill time per chunk within noise of
   before (the patch reorders work, it adds none).
4. **Recorded as Ignis patched**: the manifest patch and its diff, the
   `VENDOR.md` row (what the reference does, what ignis does, the test that
   tells them apart, this issue), and the glossary's **Residual window** entry,
   runtime/06's *As built* and the #257 finding no longer describing the
   clobbered ring as current behaviour.

**Measured after, reported, not a gate:** #173's twenty tool prompts
(`.scratch/diag-160/twenty/` on #257's branch: 17/20 with the window at
reference parity, reference 18/20), re-run with the patch; the numbers go on
#173. Any comparison with the reference from here on names this departure
(ADR 0037).

## Out of scope

- Upstreaming the fix to the reference: an owner decision, separate from this.
- The pointing study's hq re-measurements (its own follow-up).

## References

- ADR 0037 (correctness patches, reference parity vs Ignis patched), ADR 0010,
  ADR 0031, ADR 0022, ADR 0005 / 0007 (correctness first; the reference is not
  the definition of correct).
- `docs/specs/runtime/06-hq-residual-window.md` (§ As built),
  `docs/findings/2026-09-22-the-residual-window-was-the-tool-call-gap.md`.
- GitHub #257, #173.
