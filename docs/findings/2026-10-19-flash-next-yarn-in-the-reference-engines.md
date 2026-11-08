# HF, vLLM and SGLang run plain YaRN on Flash-Next, its attention factor on q and k, one table for attention, the indexer and the MTP head

- Kind: research
- Status: current
- Observed: 2026-10-19
- Last verified: 2026-10-19
- Scope: kernel / Flash-Next rope sites (`kernel/src/flash_next/qsa.cu`, `indexer.cu`, `rope.cuh`), the shared YaRN table builder (`kernel/src/rope_scaling.h`, `crates/core/src/rope_scaling.rs`)
- Topic: flash-next
- Related: [ADR 0059](../adr/0059-flash-next-applies-yarn-on-q-and-k.md), [ADR 0057](../adr/0057-rope-scaling-is-auto-by-default.md), [spec rope-scaling/02](../specs/rope-scaling/02-flash-next.md), `.scratch/orchestra-2026-10-18/fn-yarn/research.md` (the full read, with every line cited), `.scratch/fn-needle/RESULTS-5090.md` (the ~500K probe), `.scratch/cluster-072/REPORT.md` (the 1M probe)
- Superseded by: none
- Summary: HF, vLLM and SGLang run plain static YaRN on Flash-Next's 64-dim partial rotary, 0.1·ln f + 1 folded into cos/sin on q and k, one table for attention, indexer and MTP head; ignis's ~499K and 1M needle probes separate nothing.

## Question

Flash-Next (`qwen4_exp`) is trained over 262,144 positions and ignis serves it
at 589,824 and 1,048,576. Before ignis applied any YaRN to it: what exactly do
the reference engines do for this model, and does any of them do something
model-specific (an unscaled indexer table, a different attention factor, a
separate MTP table, dual-chunk attention)?

## Evidence

Sources read, at these revisions (no fetched code was run):

| source | revision | read |
|---|---|---|
| HF transformers | `d4804f9d` | `modeling_rope_utils.py`, `models/qwen4_exp/*` |
| vLLM | `e0adacc9` | `vllm/models/qwen4_exp/**`, `model_executor/layers/rotary_embedding/*`, `config/speculative.py` |
| SGLang | `da414add` | `srt/models/qwen4_exp.py`, `qwen4_exp_mtp.py`, `qwen3_5.py`, `srt/layers/rotary_embedding/*`, `srt/layers/attention/qsa/*` |
| Hub `Qwen/Qwen3.8-Flash-Next` | `de4b8e4d` | `config.json`, `README.md` ("Processing Ultra-Long Texts") |
| ninfer upstream | `64d10fbf` | `src/models/qwen4_exp/**` |

What the code says:

- **The table.** All three build HF's YaRN on the 64-dim partial rotary
  (`head_dim` 256 x `partial_rotary_factor` 0.25), θ 1e7, original 262,144,
  `beta_fast` 32, `beta_slow` 1, `truncate` true: ramp bounds **14 / 22**,
  static (one cache built at init). HF `modeling_rope_utils.py:413-483`, vLLM
  `yarn_scaling_rope.py:51-74`, SGLang `mrope.py:477-499`.
- **The attention factor.** `m = 0.1·ln(factor) + 1` (1.0811 at 2.25, 1.1386
  at 4), multiplied into cos and sin, so it scales the rotated 64 dims of
  **q and k** and a score's rotated part by m². The softmax scale stays
  `256^-0.5` in all three. HF `modeling_qwen4_exp.py:118-129`, vLLM
  `yarn_scaling_rope.py:76-85`, SGLang `mrope.py:509-518`.
- **One table everywhere.** The sparse-attention indexer is handed the
  attention's own rotary object, for its queries and for its pooled block keys
  at the block's first position. No engine keeps an unscaled indexer table.
  HF `:694-740`, vLLM `qsa.py:488-495`, SGLang `qwen4_exp.py:1809-1815`.
- **The MTP head.** It builds an ordinary full-attention layer from the text
  config's `rope_parameters`; `mtp.rope_theta` is read by none of them. SGLang
  copies the overridden config into the draft (`model_config.py:860-871`).
  vLLM does not apply a dict `--hf-overrides` to the draft
  (`speculative.py:1097-1098, 1282-1307`).
- **No dual-chunk attention**: vLLM refuses it for this model
  (`nvidia/qsa.py:404-408`). For text tokens the interleaved MRoPE is the 1-D
  table.
- **Precision.** All three compute the angle as an FP32 product and FP32
  cos/sin. HF and vLLM store the table in BF16, SGLang in FP32.
- **Upstream ninfer** refuses any `rope_type` other than `default` for this
  model.
- **The card** recommends static YaRN with `factor` matched to the typical
  context (4 with original 262,144 for 1M, 2 for 524,288).

A probe on ignis after the change (RTX 5090, release candidate 0.7.2,
hq-e8-2b KV, context 589,824, speculation off, temperature 0, one question per
cell, `.scratch/fn-needle/`): at ~499K prompt tokens, `auto` (YaRN 2.25)
retrieved 5/5 single needles (24.6K to 475.2K) and 1/2 three-hop chains;
`none` retrieved 4/5 (it missed the needle at 251.0K) and the same 1/2, the
failed chain answering the other chain's number under both. Cold prefill was
the same under both tables (~720 tok/s, 688-694 s).

A second probe ran at 1,048,576 on the RTX PRO 6000 (release 0.7.2, hq-e8-2b
KV, MTP 3 on, temperature 0, one question per cell, seed 1, ~1,000K prompt
tokens): `auto` (YaRN 4) retrieved 4/5 needles and 0/2 chains, missing the
needle at 948.9K and both chains (last hops at 970.6K and 928.1K); `none`
retrieved 5/5 and 1/2, missing the same 928.1K chain. Every miss answered
another fact's number from the same prompt. Cold prefill was ~3,200 tok/s
(312-313 s) under both tables, and MTP drafted and accepted on the scaled
table with no error.

## Finding

Observed (code): the three reference engines agree on the table, the ramp, the
attention factor and its placement on q and k, and they apply that one table
to the attention, the indexer and (SGLang) the MTP head. Nothing in them is
specific to Flash-Next beyond its partial rotary width and θ, which equal the
27B's, so the 27B's table builder gives Flash-Next's table.

Observed (code): the only disagreement that could matter is vLLM's MTP draft
under the command-line recipe, which does not receive the YaRN override.

Inferred: under that recipe vLLM's MTP head runs the linear table past
262,144, so vLLM is not an oracle for MTP at long context; HF (no MTP) and
SGLang are.

Observed (probe): with YaRN 2.25 Flash-Next answers single-needle retrieval
up to 475K and loses nothing against the unscaled table on this probe. At 1M,
YaRN 4 scored 4/7 against the unscaled table's 6/7, its extra misses at the
deepest positions. Each probe is one question per cell, the two point opposite
ways, and neither separates the tables statistically. The unscaled table did
not degrade past 262,144 on either.

## Implications

- ADR 0059 follows the references: m folded into cos/sin on q and k, the FP32
  angle, the 27B's builder, one table for attention, indexer and MTP.
- The 27B's vendored op puts m² on q only (factor-free cached K): equal scores
  in exact arithmetic, different BF16 rounding and a different cached K.
- vLLM long-context MTP numbers are not a comparison point for ignis past
  262,144.

## Limits and unknowns

- Line citations are to the revisions above; later revisions may move them.
- vLLM's MTP behaviour past 262,144 (an out-of-bounds gather or a mismatched
  table) is inferred from the code and was not run.
- The probes ran with hq-e8-2b only, one seed and one filler (the repo's own
  text). BF16 KV is not covered, and neither probe can tell whether factor 4
  costs retrieval at depth.
- At ~500K and at 1M every question re-prefilled the whole prompt
  (`cached_tokens` 0); why prefix reuse did not engage at those sizes was not
  investigated.

## Follow-ups

- A second seed and a second filler at 1,048,576, if the `auto` default needs
  a retrieval number behind it. Today it rests on the model card and the
  reference engines, not on these probes.
