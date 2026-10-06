# 07 - MTP speculation for Flash-Next: the checkpoint's own draft head, a verify round on Flash-Next's state

GitHub: #306 (item 12; master #298).

**Status: approved for implementation (owner, 2026-10-06).** Phase A is
measured: **GO** (see "Phase A result (2026-10-06)"). Every *(proposed)* item
below is decided unless the owner decisions here say otherwise; they answer the
open questions at the end.

## Owner decisions (2026-10-06)

- **Head conventions:** the ones Phase A measured best — comb a, norm a, chain
  a (the trunk's four streams fed separately, each normed on its own; every
  later draft fed the head's own pre-mixer 4-stream output); the head's own
  indexer.
- **Container:** a companion container beside the main artifact for now; a
  later phase merges the head into one artifact.
- **MTP experts at 3.0 bits** per weight (mean, allocation as for the trunk),
  calibrated on the trunk's own calibration corpus with the trunk states from
  Phase A's tap; non-experts FP8 as the trunk.
- **Draft width:** by default `k` adapts per round to a row budget over the
  active lanes (decode route ≤ 8 rows; Phase A: k = 2 at 1 lane, 1-2 at 2,
  1 at 3); a server option forces a fixed `k` (e.g. always 3) and another turns
  MTP off. MTP is **on by default** for Flash-Next.
- **KV format:** hq-e8-2b stays Flash-Next's serving default; `--kv-format bf16`
  stays available. The difference (decode tok/s, expert hit rate, prefill ≠
  decode flips) is measured and reported.
- **One ticket** carries phases B-D with numbered acceptance criteria.

Sources:
- the checkpoint's MTP tensors (31, names and shapes from
  `.scratch/flash-next-compression-2026-10-03/real/headers.json` and
  `fn_index.json`) and its config (`fn_config.json`);
- transformers 5.17's `modeling_qwen4_exp.py` (`F:/ai/ngram-venv`): the trunk's
  math, and the proof that it does not implement the head
  (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`);
- ExLlamaV3 1.5.3's `architecture/qwen4_exp_mtp.py` and
  `modules/arch_specific/qwen4_exp_mtp.py` (same venv), whose docstring calls
  its input combine "a semantic guess that must be confirmed by acceptance
  rate";
- ninfer's MTP for the 27B as a **design reference only**
  (`docs/maintainer/qwen3.6-27b-model.md` §7-8, `include/ninfer/ops/mtp_round.h`,
  `impl/runtime/mtp_impl.h`, `mtp_adaptive.h`), and its A/B log
  (`F:/ai/q38/ab_mtp.log`, 2026-08-31);
- ignis's 27B speculation: runtime spec 05, `crates/core/src/speculation.rs`,
  the vendored `speculative_round.h`, `gated_delta_net.h` (replay record) and
  `gdn_replay.h` (fold);
- `docs/findings/2026-10-06-flash-next-decode-round.md` and
  `docs/findings/2026-10-06-flash-next-on-the-5090.md`.

ADRs: 0016 (options structs), 0019 / 0020 (decode graphs, batch-wide round),
0022 (BF16 oracle), 0024 (sequence state transfer), 0029 (reuse), 0030
(reservations at load), 0037 (vendored correctness patches), 0043 (a second
model without a ninfer reference: this code is OURS), 0044 (experts in
trellis).

## Problem Statement

Flash-Next decodes 61-65 tok/s at one lane. The round is latency-bound, not
bandwidth-bound: 13.6 ms of graph replay at 22% of DRAM bandwidth and 33% of
SMs, plus 2.1-2.65 ms of device idle between replays. A round with more rows
costs less than proportionally: three lanes give ~131 tok/s in total, ~22.9 ms
for a 3-row round against ~15.3 ms for a 1-row round.

The checkpoint ships a draft head trained for exactly this: one MTP layer
(2.61 B parameters, 5.2 GB in BF16) that predicts token t+2 from the trunk's
state at t and the token t+1. Three facts stand between it and a faster
decode.

**Nobody documents the head's input.** transformers skips it. ExLlamaV3 guesses
it. The shapes fix most of the math, but not how the 4-stream trunk state
enters the layer, or how the head chains for a second draft. A wrong guess
does not break anything visibly: it just accepts nothing.

**Flash-Next's state is wider than the 27B's.** A verify round must run k+1
columns per lane without committing them, then commit exactly the accepted
prefix. The 27B's substrate (runtime spec 05) does that for GDN and GQA KV.
Flash-Next adds:
- 36 GDN layers in its own state layout;
- QSA KV for 12 layers, plus the indexer's pooled block keys and raw tail;
- the n-gram embedding (host-hashed ids, NVMe rows, a 9-column conv state);
- the hq-e8-2b residual window;
- MoE residency.

The vendored fold refuses Flash-Next by name, and spec 04 answers speculation
with a 400.

**The extra column is not free on an MoE.** k+1 columns route up to 10(k+1)
experts per layer, so the round moves more bytes over PCIe. The marginal cost
of a column decides k and the whole speedup. Today it is only bounded by the
3-lane proxy.

## Solution

Four phases in one ticket, each gated by the previous one's numbers
*(proposed)*:

- **A. Measure acceptance before building anything.** A PyTorch prototype of
  the head on the BF16 MTP weights, fed by the trunk states the engine
  produces, scores every candidate convention in one pass. It returns the
  convention and the per-position acceptance α₁, α₂, α₃. It is a
  pre-registered GO / NO-GO (Acceptance 1). A NO-GO ends the shipping, not the
  study.
- **B. Convert the head** into a companion container pinned to the main
  container's identity. The 71.8 GB container is not touched.
- **C. Build the verify round on Flash-Next's state, with a fake drafter.**
  This proves commit and rollback on every state component and measures the
  marginal column cost c(w) before the head exists.
- **D. Draft with MTP inside the round's graph,** at a fixed k chosen at load.
  An adaptive width is a measured follow-up.

Greedy speculation is lossless by construction. "Spec on equals spec off,
token for token, up to the documented near-tie rule" is the correctness
oracle for the whole of C and D.

## The MTP head

Notation: `S_p` is the trunk's residual stack at position p, `[4][2560]`. It
is the output of layer 47, **before** `hyper_connection_mixer`. Every RMSNorm
in this family is `x · rsqrt(mean(x²) + 1e-6) · (1 + w)` in fp32.

```text
e      = RMSNorm(embed_tokens(t[p+1]), pre_fc_norm_embedding)          [2560]
n_s    = RMSNorm_s(S_p[s], pre_fc_norm_hidden[s])          s = 0..3    [4][2560]   (C-norm)
X_s    = fc_hidden(n_s) + fc_embedding(e)                  s = 0..3    [4][2560]   (C-comb)
# one Qwen4ExpTextDecoderLayer, full_attention, no PLE:
x, X, inj = mtp.layers.0.attn_hyper_connection(X)
X        += QSA(x, position p, the MTP's own KV + indexer)  ⊗ inj
x, X, inj = mtp.layers.0.mlp_hyper_connection(X)
X        += MoE(x; 512 experts, top-10, shared expert + gate) ⊗ inj
S'_p   = X                                                             [4][2560]
logits = lm_head(mtp.hyper_connection_mixer(S'_p))                     draft for t[p+2]
# draft j+1 (C-chain):  S_{p+1} := S'_p,  t[p+2] := draft j,  position p+1
```

**Confirmed by the tensors and the config:**
- The hidden input is the pre-mixer 4-stream stack: `pre_fc_norm_hidden` is
  `[10240]` = 4 × 2560, and the post-mixer state is only 2560 wide.
  `mtp_use_hidden_state_from_layer: None` means the last layer.
- `fc_embedding` and `fc_hidden` are both `[2560, 2560]`, so the 27B's
  `fc(concat(e, h))` here is split into two named halves whose sum is the
  same map. No ordering question is left.
- The token embedding and the output head are the trunk's:
  `mtp_use_dedicated_embeddings: False`, no `mtp.embed_tokens`, no
  `mtp.lm_head`.
- There is no final MTP norm. As in the trunk, the combine-less mixer's
  `hc_norm` is the final norm (`mtp.hyper_connection_mixer` has no
  `block_inject_weight`).
- The layer is a trunk full-attention layer, tensor for tensor:
  - QSA: q `[12288]` = 24 heads × 256 with a sigmoid output gate, k/v
    2 heads × 256, q/k norms;
  - the indexer: `index_qk_proj [640, 2560]`, 4 q heads + 1 k head of 128;
  - the MoE: 512 experts of 640, top-10, a shared expert with its gate;
  - attn and mlp hyper-connections with `block_inject`.
- It has no PLE: no `mtp.layers.0.ple.*` exists. As an HF module that is
  `layer_idx + 1 = 1 ∉ ple_layer_ids = [2]`.
- Rope θ is 10⁷, partial rotary 0.25, as in the trunk (`mtp.rope_theta`).

**Uncertain, decided in phase A:**

| id | candidates | prior |
|---|---|---|
| C-comb | **a** `fc_hidden` per normed stream, embedding broadcast to every stream (ExLlamaV3 `stream_tap=True`); **b** normed streams averaged, then `fc_hidden`, sum broadcast to 4 streams (`stream_tap=False`, trunk-entry style) | a |
| C-norm | **a** grouped RMSNorm, one group per stream; **b** one RMSNorm over all 10240 | a: every other `[10240]` norm in the model (`hc_norm`, the PLE norms) is grouped with group size 2560 |
| C-chain | **a** the next step's stack is the MTP block's own pre-mixer stack `S'_p` (ExLlamaV3); **b** the post-mixer state, repeated ×4 as the trunk does with its embedding | a |
| C-idx | above 2051 tokens the MTP's QSA selects blocks with **a** its own indexer, or **b** the trunk's layer-47 selection (the 2026-09-28 survey says "reuses the QSA indices") | a: the weights exist. Below 2051 tokens attention is dense and the question does not arise. |
| pos | the RoPE position fed to the entry built from (S_p, t[p+1]): **p** (ninfer's `position_begin = chunk_begin`, vLLM's EAGLE convention) or **p+1** | p. RoPE is relative, so a constant offset changes nothing in dense attention. It moves only the indexer's 4-token block boundaries, i.e. only C-idx a above 2051 tokens. Where the entry is *stored* is a different question, settled by prefix sharing (below). |

Phase A decided every row: comb **a**, norm **a**, chain **a**; C-idx and pos
are moot for acceptance (see "Phase A result (2026-10-06)").

The config field `mtp.hybrid: True` is not interpreted by any code read here.
If phase A finds a convention with high acceptance, it is moot.

**The MTP's KV is sequence state.** The entry built from (S_p, t[p+1]) is
stored at index p+1, the token that completes it *(proposed)*. A prefix's MTP
entries are then a function of its own tokens, so they share by refcount
exactly like KV.

A prompt of P tokens gets entries 1..P−1 in prefill, after the trunk, chunk
by chunk: one extra layer of 49, so ~2% of prefill. Entry P needs the first
generated token, so the first decode round writes it (ninfer's
`final_column_uses_generated_token`). Above 2051 tokens the MTP indexer's
blocks follow the storage index; phase A's long windows check the RoPE row
above.

## User Stories

1. As the owner, I want acceptance measured on the BF16 head before any conversion or kernel work, so that a head that does not pay is found for the price of a script.
2. As the owner, I want every candidate input convention scored in one pass, the wrong ones shown near chance, so that the chosen one is a measurement and not a guess.
3. As the owner, I want a pre-registered GO / NO-GO on the projected one-lane speedup, so that the build starts only when the numbers say it pays.
4. As the owner, I want the 71.8 GB container untouched and the head shipped as a companion container pinned to it, so that nobody re-downloads the model to get the head.
5. As the owner, I want a load without `--speculative mtp` to bind nothing of the head, so that today's VRAM plan and graphs are what spec-off runs.
6. As the owner, I want speculation chosen at load with a fixed draft width, so that the serving profile is a flag and the graph count stays small.
7. As the owner, I want greedy spec-on to emit the spec-off tokens, up to the documented near-tie rule, so that one test is the oracle for verify, accept, commit and rollback together.
8. As the owner, I want sampling at temperature > 0 to stay distribution-preserving with the seed-only RNG, so that a request's output depends on its own seed alone, as on the 27B.
9. As the owner, I want rounds, drafted and accepted counts per request in the request log, and acceptance on the Monitor, so that acceptance in agent traffic is a number.
10. As the owner, I want the speedup reported at 1, 2 and 3 lanes, and drafting turned off at a lane count where it loses, so that agents are never slowed by it.
11. As an engine developer, I want the verify round testable with a fake drafter (all-accept, all-reject, random), so that every state component's commit and rollback is proven before the head exists.
12. As an engine developer, I want each state component's rollback to have a kernel-leaf test against sequential width-1 decode, so that a failure names the component.
13. As an engine developer, I want the MTP's KV and the continuation stack to be sections of the sequence image, so that snapshot, claim, KV-RAM and clone carry them or refuse the blob.
14. As an engine developer, I want the round's marginal column cost measured at widths 1..4 on one lane, so that k is chosen from a measurement.

## Implementation Decisions

### Phase A: the acceptance prototype *(proposed)*

- **Trunk states.** A feature-gated test tap, like `attn-tap` and
  `kv-capture`: during a prefill it copies `S_p` (the final pre-mixer stack,
  BF16 `[P][10240]`) to the host. No production build arms it.
  - The corpus is the engine's own greedy generations: N coding and prose
    prompts at ≤ 2K and 8-32K, 512 greedy tokens each, re-prefilled with the
    tap.
  - On a greedy text, the target for draft j at p is the text token t[p+1+j]
    by construction (up to prefill/decode near-ties). This is the quantity
    the verify accepts.
  - GPU cost: minutes.
  - Alternative without engine code: the converter's head-only replay
    (conv.md, ~130 s/layer × 48 over the reference windows). It gives exact
    α₁ against the stored `q_argmax`, but α₂₊ are only approximate on
    non-greedy text. Phase B needs that replay anyway for calibration.
- **The head.** The 31 BF16 tensors are fetched by range with the converter's
  `fetch.py` (5.21 GB from 28 shards).
  - The block is transformers' own `Qwen4ExpTextDecoderLayer(cfg', 0)`, with
    `cfg'` = the text config with `layer_types=['full_attention']` and
    `num_hidden_layers=1`.
  - The mixer is `Qwen4ExpTextGatedResidual(cfg, use_combine=False)`, with
    the trunk's `lm_head`.
  - Only the input combine (C-comb × C-norm) and the chaining (C-chain) are
    written by hand. Everything else is the checkpoint's own math.
- **Scoring.**
  - One causal pass of the MTP layer over a window gives α₁ at every
    position: draft₁ at p is correct iff it equals t[p+2].
  - α₂ and α₃ come from chaining at sampled positions where draft₁ was
    correct. There the chain token equals the text token, so the
    teacher-forced chain is exact.
  - All candidates run on the same windows. Expected:
    - the right convention lands at α₁ ≈ 0.6-0.85, per the external reports
      of ~2.5-3.3 tokens per round;
    - a wrong one lands far below.

    If two candidates land within noise, the cheaper one wins.
  - Long context (> 2051 tokens) decides C-idx with the HF module's own
    indexer.
- **Projection.** The one-lane rate is τ / (R₁ + k·c + k·d). Inputs:
  - τ = 1 + α₁ + α₁α₂ + …;
  - R₁ = 15.3 ms;
  - c = the marginal column cost, 3.8 ms until phase C measures it (the
    3-lane proxy, pessimistic);
  - d ≈ 0.9 ms per draft step: one QSA half ~0.16, one MoE block with
    resident experts ~0.14, the mixer 0.04, the full head GEMV 0.38 (636 MB
    FP8 at ~94% of bandwidth), the combine ~0.02, plus margin.

### Phase B: conversion *(proposed)*

- **A companion container,** e.g. `qwen3_8_flash_next_mtp_trellis_a25-v2.ninfer`.
  - It is pinned to the main container by `weights_id` and the main
    container's sha256, recorded in its conversion JSON. The binder refuses a
    mismatch by name.
  - *As built (layout.md §13.2):* the sidecar records the main container's
    `model_id`, `weights_id`, `Reader::content_hash` and whole-file sha256.
    The packer refuses a companion whose converter calibrated on a main
    container with another sha256; the load compares `model_id`,
    `weights_id` and `content_hash` (the directory hash the leaf already
    computes) and refuses a mismatch or a missing sidecar by name. The load
    does not re-hash 71.8 GB; verify does, offline.
  - Appending to the main container would make it a re-conversion
    (layout.md: any change is). It would also change the identity the owner
    accepted and force a 71.8 GB re-download for a ~0.9 GB head.
- **Formats, as the trunk's (layout.md):**
  - experts: trellis at the trunk's budget, 2.5 b: 786 MB; 3.0 b would be
    944 MB;
  - attention, indexer and HC mix down/up, the shared expert, `fc_hidden`,
    `fc_embedding`: FP8 row-scale;
  - `block_inject`, the router gate, the shared-expert gate and every norm:
    BF16.
  - Non-experts ≈ 91 MB.
- **Calibration.** The experts' Hessians need the MTP layer's MoE input over
  the corpus. The trunk's work files can produce it through the head-only
  replay, which phase A's prototype then extends through the MTP's
  combine + attention.
  - One replay also yields conv.md's proposed `q_lp_at_bf16_ids`.
  - An uncalibrated quantization is the fallback. The head's error costs
    only acceptance, never correctness, so the bar is acceptance (AC2), not
    KLD.
- **Fetch.** Byte ranges via `fetch.py`, with the long network wait conv.md
  asks for.

### Phase C: the verify round on Flash-Next's state

**Round shape** (one graph per lane count 1..3 at the load's k, so three
graphs, as today's three):
1. On the host, as today but for k+1 columns: hash the n-gram ids of
   (anchor, draft₁..draft_k) from the lane's context. The drafts are already
   on the host, because round N's graph ends with round N+1's drafts (below).
   Gather the rows in one batch.
2. Run the trunk over k+1 columns per lane in record mode. Then the mixer
   and `lm_head` on every column.
3. Accept (below), cut at the first stop and at the budget, on the device.
4. Commit the accepted prefix of every state component (table).
5. Phase D only: the MTP alignment pass, then k−1 autoregressive draft steps.
6. Read back the committed tokens, A, and the next drafts.

A lane whose budget or context is short runs at a smaller extent, down to 0
(a plain step), inside the same graph, as runtime spec 05 does.

The graph always runs k+1 columns per lane. The extent is a per-lane mask of
valid columns (the record op's `valid_columns`), not a narrower width: the
vendored record op refuses widths below 2. The n-gram gather and the indexer
honour the same mask.

**State components:**

| component | during verify | commit (A accepted, A+1 committed) | reuse |
|---|---|---|---|
| GDN recurrent, 36 layers | the vendored `gated_delta_net_replay_record`, per layer, at Flash-Next's per-layer state `{128,128,48,slots}`. It admits 16/48 heads and width 2..16, writes k/v/g/β records and leaves the state untouched. | **our fold**: replay the first A+1 records into the slot. The vendored `gdn_replay_fold` refuses Flash-Next twice over: its registration table admits only 48 or 30 layers (`replay.cpp:155`), and it writes the vendored all-layer state view, while Flash-Next keeps its own per-layer `State{conv, recurrent}` (`flash_next/gdn.h`). | the record op as is; the fold OURS (ADR 0043) |
| GDN conv taps `[slots][3][10240]` | our conv reads old taps + the window's columns, and records the window's conv inputs | taps = tail₃(old ‖ accepted columns) | OURS (the conv already is) |
| QSA KV, 12 layers (BF16 or hq-e8-2b) | written at positions f..f+k into pages provisioned before the round | the frontier moves to f+A+1. Rejected entries sit past it and are overwritten. | the 27B's pattern |
| hq-e8-2b residual window | must not overwrite a live entry. An append of k+1 rejected-capable columns would evict exact BF16 rows that spec-off still has. | capacity W + k_max, or the k+1 overwritten slots saved and restored. Decided by the "reject equals never drafted" test. | OURS |
| indexer, per QSA layer | raw keys of the k+1 columns recorded. A block completed inside the window is pooled to a scratch slot that later columns of the window can see. | pages get only blocks whose four tokens are all committed. `tail_keys` = the committed remainder (tail ‖ accepted raw keys) mod 4. A rollback that crosses a block boundary un-pools it. | OURS |
| n-gram conv state, layer 1, 9 columns | inputs recorded | tail₉(old ‖ accepted) | OURS |
| n-gram id context (last two ids) | the window's ids are hashed from context + drafts | context ← the last two committed ids | host side |
| penalty-count row | column j sees the counts with drafts < j | only accepted tokens counted | the 27B's rule |
| MoE residency | `resolve` over the union of the round's rows. The lookahead prefetch runs per row. | nothing to undo: residency is a cache, not state | as is |

**Accept and sampling.** The vendored `speculative_accept_*` and
`speculative_prepare_verify_*`:
- vocab 248320 is the 27B's, so they serve Flash-Next unchanged;
- the greedy branch is the longest matching prefix plus the target's token
  at the divergence;
- the sampling branch is accept with p_target / p_draft and resample from
  the residual, using the stateless RNG keyed by seed, position and purpose.

At temperature > 0 each draft token is *sampled* from the head's processed
distribution, with the seeded RNG under its own purpose key. That same
distribution is what the accept rule divides by. An argmax draft under the
sampling accept would not preserve the target distribution.

**The fake drafter.** Phase C ships with three drafters selectable in tests:
- replay of a recorded spec-off greedy run (all accepted);
- a constant wrong token (all rejected);
- seeded random.

Together they exercise every A in 0..k on every component.

### Phase D: drafting with MTP

- **In the round's graph** (ninfer's shape, our code):
  - after commit, the MTP alignment pass runs over the A+1 committed columns.
    The stacks are the verify's own `S`; the tokens are the committed ones
    shifted by one, the last being the bonus token. It writes the MTP
    entries at indices f+1..f+A+1.
  - The draft from its last column is draft₁.
  - Then k−1 autoregressive steps chain per C-chain. They write MTP KV past
    the frontier, as scratch that the next alignment overwrites.
  - Cost: k MTP passes and k head calls per round.
- **The MTP's KV and indexer** are a 13th QSA section of the paged pool:
  352 B/token, +8.3% on today's 4,224. The KV is in the trunk's format. Under
  hq-e8-2b it has its own residual window, with the same W + k_max rule as
  the trunk's. Snapshot, claim, KV-RAM and clone
  carry it like the trunk's KV (ADR 0024, spec 05). The blob version bumps.
- **The continuation stack** (the trunk's `S` at the last committed position,
  20 KB per lane) is a new image section.
  - A published prefix of length L holds MTP entries 1..L−1, so its pages
    share like KV.
  - Entry L needs t[L], the claimer's own first token. The claimer's prefill
    writes it into its own tail page from the stored stack, then continues
    as usual.
  - The same section bootstraps a restored live sequence: its first round
    runs at extent 0 and drafts from it.
- **MTP experts are resident,** bound at load outside the expert cache. A
  draft step's 10 experts are 15.4 MB, and a miss puts ~124 µs per expert on
  the draft's critical path.
- **The head:** `ProposalHead::Full` only. The artifact has no shortlist head;
  the 27B's shortlist shares the tokenizer and is a possible follow-up, not
  this spec.
- **Width:** fixed k at load (`--draft-tokens`, proposed default from phase
  C's c(w): k = 2 if c ≥ 3 ms, else 3).
  - Above a lane count stated at load, drafting runs at extent 0. Its default
    comes from AC6's measurement.
  - An adaptive width is a follow-up. ninfer's controller is the design
    reference: an online estimate of prefix survival per draft position,
    against a measured cost table per (batch, width, context band).

### Rust seam and serving

- `SpeculativeBackend` gains `Mtp`, with its own `abi_code`.
  `Speculation::new(Mtp, k)` sizes the MTP section, the records and the
  continuation stack, which the leaf reports back for the VRAM cross-check,
  as for DFlash2.
- `DecodeOutcome` already carries a run of committed tokens (runtime spec
  05). Nothing changes at the seam.
- Spec 04's 400 for speculation on Flash-Next becomes "400 unless the load has
  MTP".
- The request log's rounds / drafted / accepted fields and the acceptance
  counters are the 27B's. The Monitor shows them per model.
- Make: `SPEC=mtp DRAFT=<k>` for `MODEL=flash-next`. Once AC6 passes, on by
  default per the measured-better rule. Owner, see the open questions.

### VRAM plan *(proposed)*

| line | bytes |
|---|---:|
| MTP experts, 2.5 b, resident | 786 MB (3.0 b: 944 MB) |
| MTP non-experts (FP8 + BF16) | ~91 MB |
| MTP KV + indexer section | 352 B/token: +171 MB at the 2.05 GB / 486K-token pool, or 1/13 fewer tokens at fixed bytes |
| GDN records (k/v/g/β + conv column ≈ 36 KB per column per layer) | 36 layers × (k+1) × lanes × 36 KB → ~16 MB at k=3, 3 lanes |
| verify scratch (rows = lanes × (k+1) ≤ 12) | tens of MB, measured |
| continuation stack | 20 KB per lane, plus in each image |

Total ≈ 1.1 GB, taken from the expert cache. At the 4G-headroom default the
cache is ~17.6 GB, so it would be ~16.5 GB. The process plus desktop still
fits spec 04's 29 GB. The cache shrinks ~6%; AC7 bounds the cost in hit rate.

### Expected speedup (one lane, before measurement)

τ = 1 + α + α² + … (one α for every position, 0.7, consistent with the
external ~2.5 tokens per round). d = 0.9 ms. R₁ = 15.3 ms. Spec-off: 65 tok/s.

| k | τ | round, c = 3.8 ms | tok/s | round, c = 2.0 ms | tok/s |
|---|---:|---:|---:|---:|---:|
| 1 | 1.70 | 20.0 ms | 85 (1.31×) | 18.2 ms | 93 (1.44×) |
| 2 | 2.19 | 24.7 ms | 89 (1.36×) | 21.1 ms | 104 (1.60×) |
| 3 | 2.53 | 29.4 ms | 86 (1.33×) | 24.0 ms | 106 (1.62×) |

At α = 0.8, k = 3 gives 100-123 tok/s (1.55-1.9×). Three lanes are not
estimated: 9-12 columns widen the expert union and the demand copies, and the
gain is probably small or negative. AC6 measures it, and the lane threshold
protects it.

**A known dependency, not a blocker:** the HC-mix work (`hcmix`) shrinks R₁,
and so the relative gain. At R₁ = 10 ms, k = 2, c = 3 ms: 1.23×.

## Testing Decisions

Tests check what the next layer observes: committed tokens, logits, the state
a later width-1 step reads, the request log, the plan. They never check
record layouts or copy order.

- **CPU, in `cargo test`:**
  - `Speculation` parsing with `Mtp`, and refusal on the 27B
    (DFlash2 stays its backend);
  - section sizes: the MTP section, the continuation stack, the records,
    against this spec's numbers;
  - VRAM plan lines and refusal with the line to shrink named;
  - the companion container refused against a different main container, by
    identity;
  - the host n-gram hashing of a draft window equals the hashing of the same
    text committed token by token;
  - the scheduler and server with a mock compute emitting runs of 1..k+1.
  - Prior art: `crates/core/src/speculation.rs` tests and the DFlash2 plan
    tests.
- **Kernel leaf** (GPU, real geometry): for each component in the table,
  verify w columns, commit A+1, then step once at width 1. The result equals
  A+1 width-1 steps then one more, bit-exact on the state the next step
  reads:
  - GDN state and taps;
  - indexer pages and tail, across a block boundary;
  - the n-gram conv;
  - the hq window: "rejected equals never drafted";
  - the MTP combine and layer against fixtures recorded from the phase A
    prototype.
  - Prior art: the 27B's replay record / fold tests, and
    `test_flash_next_hc.cu`.
- **Whole model** (GPU profile, `--ignored`, fails and never skips when the
  card is busy):
  - greedy spec-on vs spec-off: the same tokens up to the near-tie rule
    (#153, 93f0f68). Run with all three fake drafters and with MTP, at
    < 2051 tokens (dense) and > 8K (sparse), in BF16 and hq-e8-2b KV;
  - sampling: the DFlash2 distribution check
    (`docs/findings/2026-09-24-dflash2-sampled-acceptance.md` method);
  - a claim, a KV-RAM round trip and a clone with MTP on continue exactly
    as with MTP off.
- **Measurement** (once, at acceptance):
  - c(w) at w = 1..4 with the fake drafter, one lane;
  - tok/s and acceptance per position at 1/2/3 lanes × 2K/32K, coding and
    prose;
  - the expert-cache hit rate against spec-off at the same headroom.

## Acceptance

1. **Phase A (GO / NO-GO, pre-registered).**
   - The prototype reports α₁, α₂, α₃ for every candidate convention on the
     same greedy windows (coding and prose, ≤ 2K and 8-32K).
   - The chosen convention's α₁ is above every other's by more than the
     windows' 95% interval, or ties go to the cheaper one.
   - **GO** if the projected one-lane speedup at c = 3.8 ms reaches
     **≥ 1.20× at some k ≤ 3**. With one α for every position, that needs
     α ≥ ~0.56 at k = 1 or ~0.59 at k = 2.
   - Otherwise **NO-GO**: the finding is kept, phases B-D are not built, and
     the follow-up is proposed to the owner.
   - The 1.20× bar sits below AC6's 1.25× on purpose: c = 3.8 ms is the
     pessimistic bound, and AC3 re-projects with the measured c.
2. **Phase B.** The companion container:
   - holds the 31 tensors in the stated formats, bound only with
     `--speculative mtp`;
   - its identity check refuses a different main container;
   - `convert.py verify` decodes its experts bit-identical.
   - The quantized head's α₁ is within 0.03 of the BF16 head's on phase A's
     windows. Otherwise 3.0 b is used and the 0.03 is reported.
3. **Phase C.**
   - The verify round runs at k = 1..3 with each fake drafter. Greedy spec-on
     equals spec-off up to the near-tie rule, at dense and sparse context, in
     both KV formats.
   - Every state component has its leaf test (Testing).
   - c(w) is measured and recorded in the finding.
   - The one-lane projection is redone with the measured c(w). If it falls
     under 1.25×, phase D stops and the owner decides.
4. **Phase D.**
   - MTP drafts in the round's graph.
   - Greedy equality holds as in AC3, and the sampling check passes.
   - A claim, a KV-RAM round trip and a clone continue exactly.
   - `request_done` carries rounds / drafted / accepted. The acceptance
     counters are exported and on the Monitor.
5. **Load and plan.**
   - Without `--speculative mtp` nothing of the head is bound, and the plan
     and graphs are today's.
   - With it, the plan prints the head's lines. Process plus desktop stays
     ≤ 29 GB at three lanes.
   - The leaf's reported VRAM matches the Rust arithmetic.
6. **Speed** (warm cache, 5090, coding and prose):
   - one-lane decode with MTP ≥ 1.25× spec-off at 2K and 32K;
   - two- and three-lane total ≥ spec-off's minus 2%, or the load's lane
     threshold turns drafting off at that count, stated at load.
7. The decode expert-cache hit rate with MTP is within 1 point of spec-off's
   at the same headroom.
8. `cargo test` passes workspace-wide, and
   `cargo check --workspace --features cuda --tests` is clean. The Flash-Next
   GPU tests and the 27B GPU profile are green on a free 5090.

## Phase A result (2026-10-06)

Finding: `docs/findings/2026-10-06-flash-next-mtp-phase-a.md`.

**How it was measured.**
- The trunk states come from a test-only tap of the final pre-mixer stack
  (`residual-tap`, `kernel/include/ignis_fn_residual_tap.h`).
  - The texts are 16 prompts from the converter's reference windows: 12 of
    1,536 tokens (code, Python, prose, English, chat), plus the code and the
    prose document at 8,192 and 24,576 tokens.
  - Each prompt was continued by 512 greedy decode tokens, then re-prefilled
    with the tap armed (`crates/core/examples/flash_next_mtp_phase_a.rs`).
  - The trunk's mixer and BF16 `lm_head` over the tapped stacks reproduce
    the engine's pick at 97.8% of rows, every miss a near-tie ≤ 0.625
    logits.
- The head is the BF16 checkpoint's, fetched by range (7.76 GB with the
  trunk's embed, head and mixer), run on transformers' own layer
  (`tools/flash-next-mtp/`).
- Drafts are scored against the text's own greedy tokens, which makes the
  chained α_j exact. All conventions ran on the same 8,192 generated
  positions.

**Conventions:**

| id | result |
|---|---|
| C-comb | **a**: α₁ 0.828. b (streams averaged) 0.604-0.647, −0.22 ± 0.01 paired |
| C-norm | **a** (grouped): +0.025 ± 0.006 over b (one norm over 10,240) |
| C-chain | **a** (the block's own pre-mixer stack): α₂ / α₃ / α₄ = 0.800 / 0.811 / 0.821; b (post-mixer ×4) 0.748 / 0.727 / 0.732 |
| C-idx | the head's own indexer equals dense attention at 8K and 24K (α₁ −0.001 ± 0.003), so b (the trunk's layer-47 selection, not measured) cannot gain |
| pos | moot: the block offset by one entry gives identical α |

**Acceptance (comb a, norm a, chain a):**
- α₁..α₄ = **0.828 / 0.800 / 0.811 / 0.821**.
- Code: 0.858 / 0.838 / 0.847 / 0.847. Prose: 0.799 / 0.758 / 0.768 / 0.787.
- Long (8K and 24K): 0.825 / 0.780 / 0.778 / 0.781.
- τ = 1.83 / 2.49 / 3.03 / 3.47 at k = 1 / 2 / 3 / 4.

**Verdict (Acceptance 1): GO.** The pre-registered one-lane projection (R₁
15.3 ms, c 3.8 ms, d 0.9 ms) is 1.40× at k = 1, 1.54× at k = 2 and 1.58× at
k = 3, against the 1.20× bar.

**Marginal verify-column cost c.** It was measured with decode rounds whose
lanes hold one text at consecutive positions: the tokens and experts a
verify of w columns runs, in a tight loop on a load of 8 lanes with the
served expert cache.

| lanes | spec-off round | c per column (k = 1 / 2 / 3) | projected speedup (k = 1 / 2 / 3) |
|---|---:|---|---|
| 1 | 11.7 ms | 2.0 / 2.0 / 2.6 ms | 1.47× / 1.67× / 1.59× (k = 4: 1.64×) |
| 2 | 18.5-21.0 ms | 7.8-8.7 / 8.3-9.0 / 6.6-10.4 ms (two rows each) | 1.25× / 1.25-1.28× / 1.16-1.37× |
| 3 | 27.4-40.8 ms | 6.9-14.6 ms (three rows), k = 1 only | 1.17-1.54× |

- The three-lane proxy's 3.8 ms overstated the one-lane column by ~2×.
- At two and three lanes a column is two or three rows of other texts'
  experts. The three-lane cells mostly measure which three texts share the
  expert cache (two passes: 27.4 and 40.8 ms spec-off).
- The decode route caps a round at 8 rows (`IGNIS_MOE_DECODE_MAX_TOKENS`,
  the GDN and QSA lane caps). A per-round row budget within today's kernels
  is therefore k ≤ 7 at one lane (this spec stops at 3), k ≤ 3 at two and
  k = 1 at three.
- The measured spec-off round (10.9-12.1 ms at one lane, 2K context) is
  below the 15.3 ms above: it is a tight loop on HEAD 1166102, after the
  HC-mix work.

**Caveats** (finding §Caveats):
- The quantized head may lose up to 0.03 α₁ (AC2).
- The states are the prefill route's. On the generated text the prefill's
  pick differs from the decode's token at 7.1% of positions, mostly
  near-ties.
- d = 0.9 ms is still the estimate.
- The clone lanes are a proxy for the verify: phase C's c(w) is the number
  of record.

## Phases C-D as built (2026-10-06)

Code: `kernel/src/flash_next/{verify,speculative,mtp}.{h,cu}`, `bind_mtp`
(`bind.cu`), `ignis_artifact::flash_next::mtp`, `ignis_core::flash_next_mtp`.
Where the build departs from the text above:

- **k adapts to a row budget by width, not per round.** A round of w lanes
  verifies `k = min(draft_tokens, budget / w - 1)` drafts per lane
  (`draft_row_budget`, 0 = the decode route's 8; `flash_next_window`); a
  width whose k is 0 runs the one-token round. One pass graph and one commit
  graph per width. `--spec mtp` defaults to 2 draft tokens (phase A's
  projection), `--draft-tokens` forces at most k, `--draft-rows` names the
  budget, `--spec off` turns it off; MTP is on by default with the companion
  present.
- **The round is two graphs with the host between them**: the pass (save,
  record-mode trunk, head, accept) and, after the host's stop cut, the commit
  (fold, then the head's drafting, then the restore). The cut is not on the
  device.
- **GDN rollback** is the vendored replay record in the pass and our fold in
  the commit, whose per-token step is the vendored `recurrent_bf16_body` in
  Fold mode -- the step the decode snapshot runs, so the folded state is bit
  for bit what c one-token rounds leave (leaf test
  `test_flash_next_verify.cu`). The conv taps are rebuilt from the recorded
  conv inputs.
- **The rest is saved, not recorded**: the pass saves each lane's indexer
  tails, n-gram conv columns, hq ring words and the ring rows its positions
  overwrite; the commit rebuilds the tails from the saved tail and the raw
  keys the indexer copies out, the n-gram conv as tail9(saved || inputs) (the
  inputs are the last columns of the post-pass state, since k + 1 <= 9), and
  restores the rejected positions' ring rows and words -- a rejected column
  is a column never drafted, which the whole-model test checks bit for bit
  on the next step's logits.
- **MTP entries are stored at p**, not p + 1: entry (S_p, t[p+1]) is written
  once t[p+1] is known -- a prefill chunk's from the span's next id (the
  span's last from the token just drawn), a one-token round's from the token
  it drew, the verify commit's from the licensed tokens -- so the head's
  frontier is always the sequence's and no stack outlives a round. There is
  no continuation-stack section: a restored or claimed sequence runs one
  round at extent 0, which drafts. A claim of a prefix whose last entry was
  built from the publisher's own next token is the known gap (the claimant's
  first entry past the prefix is right, the prefix's last one is the
  publisher's): an acceptance cost, never a different text.
- **Drafts are the head's argmax at every temperature.** The vendored accept
  treats a draft as a one-hot proposal, which is distribution-preserving for
  an argmax drafter; no draft RNG.
- **The head's chained steps write past the frontier** (up to 2k positions),
  so on an MTP load the pass saves and the commit restores ring rows over 2k
  positions, and the commit restores the indexer tails after the chain.
- **Drafts reach the host** through `ignis_decode_options::out_drafts`
  (`LaneVerifyRun::next_drafts`), because the n-gram rows of a round's
  columns are hashed on the host before it.
- **Observability**: `ignis_speculative_{rounds,drafted_tokens,
  accepted_tokens}_total` and `ignis_speculative_position_{drafted,
  accepted}_total{position}` on /metrics, and a Speculation card on the
  Monitor.

## Out of Scope

- **Adaptive draft width:** a measured follow-up, after AC6.
- **A shortlist proposal head** for the draft steps.
- **MTP on the 27B.** DFlash2 stays its backend (runtime spec 05,
  `DEFERRED-DECISIONS.md` G5 item 1).
- **Tree or multi-branch drafts,** and n-gram / prompt-lookup drafting
  (`ignis-ngram-study`: no signal).
- **Speculation inside prefill,** and prefill/decode overlap.
- **Training or fine-tuning the head.**

## Further Notes

- **Supersedes** spec 04's user story 9 and AC11 for loads with MTP. The 400
  stays for loads without it.
- **The 3-lane proxy overstates c.** Its three rows are three contexts:
  - three n-gram gathers;
  - three KV streams at 32K;
  - three unrelated expert sets.

  Consecutive tokens of one sequence share more experts. Phase C's c(w) is
  the number; the 3.8 ms is only the pessimistic bound phase A uses.
- **The n-gram gather is host time before the round.** At ~12 reads per
  column (0.31 ms per 12 reads over 4 threads), k+1 columns add ~0.3 ms per
  draft column to the device idle between rounds. That is part of c. It
  could later overlap the previous round's tail.
- **Teacher-forced acceptance is the faithful one only on the engine's own
  greedy text.** On held-out text, α₂₊ are approximate. The external numbers
  are on generations:
  - llama.cpp PR #28243 reports ~2.5 tokens per round on a 5090, 2.6-3.3
    elsewhere (`flash-next-research-2026-09-28/SINTESI.md`);
  - ninfer's 27B MTP did 3.0-5.1 tokens per round, 47-78%, adaptive widths
    3-7 (`ab_mtp.log`).
- **ninfer's design, as read (not ported):**
  - one graph per (batch, K) holds the whole round: verify K+1 columns,
    accept, fold, then the MTP. The MTP first runs an **alignment pass**
    over the committed columns from the target's verified hidden states,
    rewriting the MTP KV, then K−1 autoregressive steps on its own hidden.
    So MTP KV past the frontier is always scratch, and the next round's
    drafts are ready when the round ends.
  - Verify writes candidate KV into provisioned but unpublished pages and
    ReplaySSM records. GDN state is never touched until the all-layer fold
    of the accepted prefix.
  - A prompt's MTP KV is built in prefill from hidden at t and the embedding
    of t+1, with the last column waiting for the generated token.
  - A "continuation hidden" per lane, current and turn checkpoint, survives
    across rounds and prefix reuse.
  - The width controller is an online prefix-survival estimate with a
    fast/slow low-pass pair, scored against a per-(batch, band) measured
    round-cost table that graduates from a static prior.

  This spec keeps the round shape, the alignment and the continuation
  state, at Flash-Next's state and in our code. It defers the controller.
  ninfer's bench targets (`target_mtp_round_bench.cpp`, `mtp_pack_bench.cu`)
  were read at header level only.
- Prerequisite to `ready-for-agent`: the owner's answers below.

## Open questions for the owner

1. **Companion container or re-pack?** The proposal is a ~0.9 GB companion
   pinned to the 71.8 GB container's identity, which stays untouched. A
   re-pack is a re-conversion and a re-download.
2. **Expert bits for the head:** 2.5 b (786 MB) like the trunk, or 3.0 b
   (944 MB) to protect acceptance? AC2 picks 3.0 only when 2.5 costs > 0.03
   α₁.
3. **Phase A's trunk states:** the feature-gated engine tap (minutes of GPU,
   exact acceptance on greedy text), or the converter's head-only replay (no
   engine code, ~1.7 h of GPU, which phase B needs anyway for calibration)?
   Proposed: the tap for A, the replay for B.
4. **Calibration of the head's experts:** replay-calibrated, or uncalibrated?
   The head's error costs acceptance only.
5. **The draft width:** fixed k from c(w) (2 or 3). MTP on by default for
   Flash-Next once AC6 passes (the measured-better rule), or opt-in?
6. **The lane threshold:** drafting off from 2 lanes, or from 3, if the
   multi-lane measurement loses?
7. **One ticket with phase A as its gate, or phase A as its own ticket?**
   The few-tickets rule argues for one.
