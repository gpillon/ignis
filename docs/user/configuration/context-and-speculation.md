---
title: "Context, KV and speculation"
description: "Context length, KV format, decode lanes and DFlash2 speculative decoding."
sidebar:
  order: 4
---

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--model-max-context <tokens>` | `IGNIS_MODEL_MAX_CONTEXT` | `40960` | Max per-sequence context (prompt + generation). The KV pool must be able to hold one of them or the load is refused. On the 27B the attention bounds it too: at most 524,288 on `bf16`, 1,048,576 on `hq-e8-2b`; past that the start is refused. Flash-Next has no such bound. A 27B context the profile or the default gave adapts to the free memory ([Adaptation](index.md)); a named one never does. |
| `--model-adapt-context-step <tokens>` | `IGNIS_MODEL_ADAPT_CONTEXT_STEP` | `65536` | The 27B only: when its VRAM plan refuses a context the profile or the default gave, the load lowers it this many tokens at a time, never below this many; a start whose plan does not hold one step is refused. A whole number of KV pages (a nonzero multiple of 64). A Flash-Next start refuses it. |
| `--model-default-max-tokens <n\|0>` | `IGNIS_MODEL_DEFAULT_MAX_TOKENS` | `38912` | The `max_tokens` of a request that sends none (see [Chat completions](../api/chat-completions.md)), its reasoning included, on both models. Never more than what the prompt leaves of `--model-max-context`, so a long prompt is never refused for it. An explicit cap always wins. `0` = none: such a request may generate to the end of the context, as before. (`make` knob `DEFAULT_MAX_TOKENS`.) |
| `--model-prefill-chunk <tokens>` | `IGNIS_MODEL_PREFILL_CHUNK` | `1024`; under the built-in profiles `2048` for the 27B, and for Flash-Next `8192` under `rtx5090` and `rtxpro6000`, `4096` under `rtx-24gb`, `2048` under `rtx-16gb` | The prefill chunk width while a lane decodes, a nonzero multiple of 128. The program's prefill scratch is reserved at load for the wider of it and [the solo width](#the-solo-width). |
| `--model-prefill-chunk-solo <tokens\|auto>` | `IGNIS_MODEL_PREFILL_CHUNK_SOLO` | `auto`; the prefill chunk under `rtx-16gb` and `rtx-24gb` | The prefill chunk width while no lane decodes, [the solo width](#the-solo-width): `auto` is 8192, 4096 or 2048 above the prefill chunk for the 27B on an RTX 30 or RTX 40 card, the widest the VRAM plan holds with nothing else adapted and room for a second request, and the prefill chunk everywhere else; a width is a nonzero multiple of 128, at least the prefill chunk, at most `--model-max-context`. (`make` knob `PREFILL_CHUNK_SOLO`.) |
| `--spec-decode-lanes <n>` | `IGNIS_SPEC_DECODE_LANES` | the model's own: `8` on the 27B, `3` on Flash-Next | `1..=8`, both models: the sequences decoded at once. Scoped like any field both models take (`--qwen38-spec-decode-lanes`, `--qwen38flashnext-spec-decode-lanes`); a general value is both models' on a switch. On the 27B each lane reserves its state (~233 MB with DFlash2 under hq-e8-2b) and its share of the round buffers (~35 MB at a 7-token draft window: sampling, the decode and verify rounds, the drafter's), its decode and verify graphs are captured at widths up to the lanes, and the default host retained slots are two per lane: `2` frees ~1.6 GB of VRAM for the KV pool against `8`. On Flash-Next the lanes share the KV pool ([The KV pool](vram-and-kv-pool.md#the-kv-pool)): min(524,288 tokens, lanes × `--model-max-context`), never below one `--model-max-context` sequence and a page per retained slot. At `make`'s 262,144 tokens that is one context at one lane and 524,288 tokens from two lanes on, so one lane leaves the expert cache about 1 GiB more, and further lanes cost only their state; at 131,072 tokens three lanes get 393,216. `make MODEL=flash-next` runs 3 lanes at up to 262,144 tokens each (the checkpoint's trained positions); the `make` knob is `LANES`, passed in the started model's own section. |
| `--model-decode-share <percent>` | `IGNIS_MODEL_DECODE_SHARE` | `25` on both models | The part of the time decoding lanes keep while a prompt prefills, 0-99: after a chunk that took `t`, the next chunk of the same prompt waits `t * s / (1 - s)` of wall time while they decode, and nothing waits when no lane decodes or after the prompt's last chunk. It trades the prefilling request's TTFT (about x1.33 at 25, x2 at 50, only while lanes decode) for the lanes' rate (15.6-18.3 tok/s on Flash-Next instead of ~1; why 25: [ADR 0018](../../adr/0018-chunk-level-prefill-decode-interleaving.md)). `--model-decode-share 0` restores one decode round per chunk, the pacing before [#92](../../findings/2026-10-07-27b-prefill-chunk-width.md); 50 splits time evenly. A prompt alone is unaffected. |
| `--model-kv-format <fmt>` | `IGNIS_MODEL_KV_FORMAT` | `hq-e8-2b` | `hq-e8-2b` (serving) or `bf16` (retained; the format every correctness oracle runs against — [ADR 0022](../../adr/0022-two-kv-formats-bf16-as-oracle.md)). Decides what a pool byte budget is worth in tokens. |
| `--model-prefill-attention <band>` | `IGNIS_MODEL_PREFILL_ATTENTION` | `fp8` on an RTX 50 or RTX PRO card, `int8` on an RTX 30 or RTX 40 card | The 27B only: the band of the prompt attention's band pass over an `hq-e8-2b` prefill chunk. `8bit` and `fp8` are two spellings of the 8-bit band: Q·Kᵀ in INT8 and P·V in E4M3 on an RTX 50 or RTX PRO card, where it reads `fp8`, and the **integer band** (INT8 Q·Kᵀ, u8×s8 P·V) on an RTX 30 or RTX 40 card, where it reads `int8`. `int8` names the integer band on any card; a default or a profile gives it on an RTX 30 or RTX 40 card alone, and on an RTX 50 or RTX PRO card it runs only when named. `fp16` is the FP16-accumulate fallback, on every card and from every source, a profile's too. Fixed for the life of the load. Flash-Next's attention is QSA, with no such pass: a Flash-Next start refuses an explicit value. A band the build or the card cannot run is the `fp16` band when it comes from a default or a profile, and a refusal naming the option when the operator named it. On an RTX 30 or RTX 40 card (a [build for older cards](../contributing/building.md#other-card-architectures)) the integer band is the default: unset (or a profile's `fp8`) runs it, a named `8bit`, `fp8` or `int8` runs it too, and `GET /v1/config` reports it as `int8` with the card as its source; `--model-prefill-attention fp16` names the FP16 band there instead. An RTX 50 or RTX PRO card keeps `fp8`. `GET /v1/config` and `make config` report the band the load runs, `GET /v1/config/origins` names the card as its source when it decided, which the Playground's configuration view shows as "Decided by the card". A named `fp16` keeps its own source. |
| `--model-prefill-linear <route>` | `IGNIS_MODEL_PREFILL_LINEAR` | `fp16` on an RTX 50 or RTX PRO card, `int8` on an RTX 30 or RTX 40 card | The 27B only: how its trellis prefill multiplies its matrices. `fp16` is the **fused prefill**, the trellis decoded inside the GEMM and the multiply in f16: the default on an RTX 50 or RTX PRO card, and one option away on an RTX 30 or RTX 40 card. `int8` is the **dense prefill**: each matrix decoded once per chunk into one INT8 buffer and multiplied on the integer MMA. It is the default on an RTX 30 or RTX 40 card, where `GET /v1/config` reports it as `int8` with the card as its source, and where `fp16`, named or a profile's, keeps the fused prefill; on an RTX 50 or RTX PRO card the fused prefill is the default, `int8` runs only when named, and a profile's `int8` is `fp16` there. On an RTX 30 or RTX 40 card the 27B's gated delta net prefill runs ignis's own kernels as well, with no option: the dense prefill, the integer band and those kernels are the prefill of those cards, and the load logs the three routes it runs as `ignis.runtime.prefill_routes`. Its one buffer, the largest matrix in INT8 (89 MB on the 27B), is a line of the [VRAM plan](../concepts/vram-plan.md), `trellis_dense`, so the context adapts to it; the INT8 activations take less of the trellis workspace than the fused prefill's. The trellis artifact alone: on the NVFP4 one a named `int8` is refused before the load, naming the option ([Troubleshooting](../troubleshooting.md)). Fixed for the life of the load. Flash-Next has its own prefill and refuses an explicit value. `GET /v1/config` reports the route the load runs under `model.qwen38.prefill_linear`, and `GET /v1/config/origins` its source. |
| `--model-exception-weights <format>` | `IGNIS_MODEL_EXCEPTION_WEIGHTS` | `bf16` | The 27B only: how its nine BF16 exception tensors (the attention projections of the early GQA layers, the GDN output of layer 4) are held on the GPU. `bf16` keeps them as the artifact stores them. `fp8` converts them at load to FP8 row-scale: 0.53 GB less VRAM, cold prefill ~3% faster at 14K-105K, the same 4K KLD, but two decision fixtures read differently ([finding](../../findings/2026-10-21-the-bf16-exception-tensors-in-fp8.md)); they are multiplied on the FP8 tensor cores in prefill and in any call past 8 rows (a verify round of two lanes or more), with BF16 activations in a decode round and a one-lane verify round. Fixed for the life of the load. Flash-Next has none: a Flash-Next start refuses an explicit value. On a card below compute capability 12.0, which only a [build for older cards](../contributing/building.md#other-card-architectures) runs on, `fp8` is refused. |
| `--vram-kv-pool-bytes <bytes\|tokens>` | `IGNIS_VRAM_KV_POOL_BYTES` | the KV pool policy's | The KV pool, on both models: a byte count (`K`/`M`/`G` suffixes) or a token count, `<n>tok`, `<n>Ktok` or `<n>Mtok` with binary multipliers (`512Ktok` is 524,288 tokens: the same context on either model and either format, where a byte count is worth a different one on each). It replaces the policy's size ([The KV pool](vram-and-kv-pool.md#the-kv-pool)). Smaller than one `--model-max-context` sequence plus a page per retained slot, or past the VRAM budget, fails the load by name. (`make` knob `KV_POOL_BYTES`.) |
| `--reuse-kv-host-pool-bytes <bytes>` | `IGNIS_REUSE_KV_HOST_POOL_BYTES` | `2G` | The KV-RAM host tier. Page-locked whole at start and held for the life of the load, so it is RAM the process holds even idle — and the figure Windows reports as shared GPU memory. `0` disables the host tier. (`make` sets `8G`, and Flash-Next's own `2G` as `--qwen38flashnext-reuse-kv-host-pool-bytes`: knob `FN_KV_HOST_POOL_BYTES`.) |
| `--spec-backend <backend>` | `IGNIS_SPEC_BACKEND` | unset | Speculative decoding backend: `dflash2` (Qwen3.8-27B), `mtp` (Qwen3.8-Flash-Next, see [Flash-Next speculation (MTP)](flash-next-mtp.md)), `off`, or unset for none. |
| `--spec-draft-tokens <n>` | `IGNIS_SPEC_DRAFT_TOKENS` | — | 1..7. Required with `--spec-backend dflash2`; with `--spec-backend mtp` the most drafts a lane verifies per round (default 2). |
| `--spec-draft-rows <n>` | `IGNIS_SPEC_DRAFT_ROWS` | `0` (= 8) | Flash-Next MTP only: the rows one verify round may take across all lanes, `0` or 2..32. Each lane drafts `min(draft tokens, rows / lanes - 1)`, so `3` drafts at one lane only, and past 8 five to eight lanes draft too (`16`: 3 drafts up to four lanes, 2 at five, 1 from six to eight). A budget past 8 reserves the wider rounds' VRAM, taken from the expert cache: about 50 MB at 16 rows. Past 8 the round is held numerically equivalent to speculation off, not token for token: greedy output may differ from it where two tokens are nearly tied ([ADR 0056](../../adr/0056-speculation-past-eight-verify-rows-is-numerically-equivalent.md)). Measured slower than 8 on both the RTX 5090 and the RTX PRO 6000 ([Flash-Next speculation (MTP)](flash-next-mtp.md)). (`make` knob `DRAFT_ROWS`.) |
| `--spec-draft-head <head>` | `IGNIS_SPEC_DRAFT_HEAD` | `full` | Needs `--spec-backend`. The head the drafter proposes with: `full` (the target's output head), `shortlist` (the artifact's Q4 head over the 131,072 most frequent tokens, ~357 MB more VRAM: it is loaded beside the full head) or `auto` (`full` on every card, the `rtx-16gb` profile's value; the load logs `draft_head auto -> full`). What a round accepts is still the target's choice, so the text changes only at near ties; measured on coding prompts `shortlist` accepts 7% fewer tokens per round and is slower overall ([finding](../../findings/2026-09-24-upstream-quick-wins-ab.md)). (`make` knob `DRAFT_HEAD`.) |
| `--model-rope-scaling <spec>` | `IGNIS_MODEL_ROPE_SCALING` | `auto` | `auto` decides from the model's `--model-max-context`: no scaling up to the checkpoint's trained 262,144 positions, else YaRN at `max_context / 262,144` (524,288 is 2, 786,432 is 3, 1,048,576 is 4); past 64x (16,777,216 tokens) the start is refused. `none` is the linear table whatever the context; `yarn:F[,t=<c>][,bf=<n>][,bs=<n>]` rescales the envelope by `F` in (1, 64]. An explicit value overrides `auto`. Both models apply it; on Flash-Next one table serves attention, its sparse-attention indexer and the MTP head, with YaRN's attention factor on queries and keys alike, as the model's reference implementations apply it ([ADR 0059](../../adr/0059-flash-next-applies-yarn-on-q-and-k.md)). It scales the text table only: the DFlash2 drafter keeps its own unscaled table at window-local positions, and vision its 2-D one. The load log (`ignis.model.loaded`) names the table run (`rope_scaling`) beside the setting (`rope_scaling_setting`). (`make` knob `ROPE_SCALING`.) |

## The solo width

While a lane decodes, a long prompt is prefilled one chunk at a time and the
lane waits for each chunk between two of its rounds, which is why the prefill
chunk is kept narrow. While no lane decodes, nobody waits, and a wider chunk
spreads each chunk's fixed costs over more tokens: on an RTX 30 or RTX 40 card
the 27B's dense prefill decodes every matrix once per chunk. The **solo width**
(`--model-prefill-chunk-solo`) is the chunk a prompt takes then.

- **The scheduler decides at every chunk boundary.** The solo width while no
  request is decoding or waiting for a lane to decode on; the prefill chunk as
  soon as one is. A long prompt that starts while another request decodes
  widens once that request ends. Two prompts prefilling with nothing decoding
  both take it.
- **Both widths are planned at load.** The prefill scratch is reserved once,
  for the wider, and shows in the VRAM plan's `workspace` line
  ([The VRAM plan](../concepts/vram-plan.md)). Serving never allocates for it
  and never borrows from the KV pool.
- **`auto`, the default,** is resolved at load. On the 27B on an RTX 30 or RTX
  40 card the load plans 8192, 4096 and 2048 in turn, those above the prefill
  chunk, and keeps the first its VRAM plan holds with nothing else adapted (the
  context, the lanes and the drafter's head as the profile asks) and, at more
  than one lane, the KV pool still holding one full sequence and
  `--vram-adapt-spare-tokens` (65,536) beside it: room for a second request
  comes before a wider chunk. When none does, and on an RTX 50 or RTX PRO card,
  and on Flash-Next, `auto` is the prefill chunk: one width, as before.
- **The 16 GB and 24 GB profiles pin it to their chunk**, 1024 under
  `rtx-16gb` and 2048 under `rtx-24gb`: on those cards the KV pool comes first.
  Measured on the 5090 held to those budgets, `auto` would have taken 2048 on a
  16 GB RTX 40 card (33,280 tokens of pool) and 4096 on a 24 GB one (13,312
  tokens). `auto` stays the default for the cards no profile pins: an RTX 30 or
  RTX 40 card of 32 GB, under `rtx5090`, or a larger one, under no profile. On the 5090 a wider chunk bought the 27B at most 4% of cold prefill
  time ([finding](../../findings/2026-10-07-27b-prefill-chunk-width.md)), not
  worth its context; Flash-Next pays an expert stream per chunk, so it wants
  its widest chunk with lanes decoding too
  ([finding](../../findings/2026-10-07-flash-next-prefill-chunk-and-decode-share.md)),
  and the built-in profiles give it 8192 on the RTX 5090 and the RTX PRO 6000
  (4096 and 2048 on the 24 GB and 16 GB classes, placeholders until Flash-Next
  runs on those cards).
- **A width you name** is used as named. Narrower than a prefill chunk you
  also named, the start is refused naming both; narrower than a chunk a
  profile or the default gave (Flash-Next's 8192, for a width named for both
  models), it is that chunk. When the 27B's VRAM plan cannot hold a named
  width, it falls to the prefill chunk as the first [adaptation](index.md),
  before anything else (`adapted: prefill_chunk_solo 8192 -> 2048`).
  Flash-Next adapts nothing: its plan holds the width, or the start is
  refused. Scoped like the prefill chunk: `--qwen38-model-prefill-chunk-solo`,
  `--qwen38flashnext-model-prefill-chunk-solo`.
- **What the load runs** is logged once, `ignis.runtime.prefill_chunk_solo`:
  `prefill_chunk_solo auto -> 8192`, or `prefill_chunk_solo auto -> 2048 (=
  prefill_chunk)` when it is the chunk. `GET /v1/models` reports the width as
  `prefill_chunk_solo`, always a number; `GET /v1/config` keeps `auto`. The
  request log's `ignis.request.admitted` line counts the chunks a prompt ran at
  it, `prefill_chunks_solo`, beside `prefill_chunks_consumed`.
- **What it costs.** One request prefills at a time, so a request that
  arrives during a solo chunk waits for the whole prompt in progress at either
  width. The one that waits longer is a request restored from KV-RAM or
  KV-disk while a prompt prefills: at most one solo chunk, then the chunks
  narrow. The same prompt can end in logits that differ in the last bits
  depending on whether lanes were decoding; the quality gates measure KLD.
