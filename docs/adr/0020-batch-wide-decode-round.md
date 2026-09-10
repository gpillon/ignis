# ADR 0020 — the decode round is one batch-wide model traversal

## Status

Accepted (2026-09-10) — GitHub #111. Amends ADR 0019: its staging and
slot-indirection mechanism stands unchanged; its decision to keep the round's
*per-lane sequential structure* is reversed here.

## Context

ADR 0019 captured the decode round as a CUDA graph but deliberately kept the
round's shape: W complete single-sequence forward passes in program order,
sharing only the final `ninfer::ops::sample`. Its own last Consequence named
the follow-up — "a future true batched-decode redesign (fusing the GEMM
projections themselves across lanes, using the same
`_snapshot`/`kv_table_rows` machinery at B=W instead of calling it W times at
B=1)" — and recorded that no ABI or staging change would be needed to get
there.

G3's gate run made that follow-up load-bearing rather than optional. #110's
live/live records decompose a decoding lane's blocked inter-token interval
into a prefill chunk plus a decode round:

| term | ignis | reference | ratio |
|---|---:|---:|---|
| prefill chunk, 1,024 tokens at 32K context | 116.3 ms | 141.2 ms | 0.82 |
| decode round, B=4 | 70.2 ms | 17.3 ms | 4.06 |
| blocked ITL interval, the sum | 186.5 ms | 158.5 ms | 1.18 |

ignis's prefill is 21% faster; the whole ITL p95 gap is the round. At B=4 it
costs 4.81x its own B=1 round while the reference pays 1.07x, which is the
signature of streaming the model's weights once per lane. Requirement 17 of
`.scratch/runtime/specs/03-serving-loop.md` asked for the opposite: "one
decode call spanning every decode-ready lane, so that eight lanes stream the
model's weights once rather than eight times."

The vendored surface already admits the batched form at exactly this
geometry. `gqa_attention` (A1) accepts B=2..8 at W=1..16 with `q` as
`[256,Hq,W,B]`, `positions` as `[W,B]` and `kv_table_rows` as `[B]`;
`gated_delta_net_snapshot` and `causal_conv1d_silu_snapshot` accept the same
B with per-row `initial_state_slots` / `snapshot_base_slots`; every
projection, norm and SwiGLU op takes an arbitrary column count. So the whole
round is expressible with one call per op per layer, and no vendored kernel
changes (ADR 0010 holds).

## Decision

A decode round is one traversal of the model at batch width W, for every
exact width 1..`IGNIS_DECODE_MAX_BATCH`.

- The per-layer graph-safe bodies take a **width**, not a lane index. Their
  activation buffers carry one column per lane; the attention and recurrence
  calls take the lanes on their batch axis with W=1 per row.
- Every per-sequence input the traversal needs — token id, absolute position,
  physical pool slot — is read from the ADR 0019 staging buffers as a
  contiguous `[width]` vector indexed by row. Row `b` addresses lane `b`'s KV
  pages, GDN slot and conv taps through the same device-resident selectors
  ADR 0019 introduced, so isolation is a property of the row index rather
  than of separate calls.
- RoPE positions are read from `sampling_decode_positions` directly rather
  than through `offset_i32_positions` against a device zero. The lanes'
  positions are unrelated to one another, so there is no base to offset from;
  the staged vector *is* the positions vector both `qk_norm_rope` and A1
  want. ADR 0019's `decode_graph_zero` scalar is therefore gone.
- The output head writes `[vocab, width]` straight into
  `sampling_decode_logits`, the layout the batched sampler already reads, so
  the per-lane logits column copy is gone too.
- **One traversal serves both paths.** Graph capture records that traversal;
  a round at a width whose capture failed enqueues the identical call
  sequence directly. The eager fallback is still a fallback — a capture
  failure degrades performance and never refuses service — but it is no
  longer a second, differently-shaped implementation of the round.
- `decode_graph_scratch` is sized once for one token per lane across
  `IGNIS_DECODE_MAX_BATCH` lanes and shared by every width, mirroring how
  `scratch` is sized once for the widest prefill chunk.

`ignis_program_decode`'s signature, the step ABI and every Rust call site are
unchanged.

## Consequences

- The round's dispatch counter (`ignis_program_stats`'s `kernel_count`) is
  now the layer count at every width and on both paths, where it was the
  layer count times the width. That is the leaf-side instrumentation for
  "one traversal per round": at B>1 it must equal the B=1 round's.
- ADR 0019's replay-vs-eager equality test no longer compares two independent
  implementations, since both paths now run the same op sequence. It still
  earns its place — it catches anything the capture bakes in that a direct
  run would have read fresh — but the isolation property it used to get for
  free needs its own test. That test presents the same sequences to the round
  in reverse row order and requires each one's token stream to be unchanged.
- `ignis_program_decode` now advances `seq->gqa_positions` for every GQA
  layer at the end of a successful round. ADR 0019's graph path never did,
  because it read its positions from device staging; a round that fell back
  to eager after a replay then read a stale counter. With one shared path the
  counter has one owner, and it stays truthful for the per-token prefill
  route and the layer bodies' KV-capacity check.
- Only the *decode* round is batched. Prefill remains one sequence per call
  (ADR 0018's chunk-level interleaving is what shares the device between a
  prefilling request and decoding lanes), and the per-token and per-chunk
  layer bodies are untouched.
