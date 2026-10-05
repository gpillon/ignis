# Flash-Next artifact: the converter's output contract

GitHub: #299 (spec 01). Owner of this document: the converter (`tools/flash-next-converter/`).
Readers: the Rust packer and reader (`crates/artifact`), the MoE kernels (spec 02),
residency (spec 03), the forward and its gates (spec 04).

This is the byte-level contract between the Python converter, which writes **work
files**, and everything that reads them: the Rust packer that assembles the `.ninfer`
v2 container, the kernels that decode expert projections, and the harnesses that
replay the stored references and routing traces. Spec 01 says *what* is stored; this
document says *how*. Changing anything here means a re-conversion.

All integers and floats are **little-endian**. "fp16" is IEEE binary16, "bf16" is
bfloat16, "fp32" is IEEE binary32. Shapes are row-major, outermost first.

## 1. The output tree

```
F:/ai/models/Qwen3.8-Flash-Next-ignis/
  work/                         converter output, consumed (and deleted) by the packer
    converter.json              run record: everything the sidecar needs from the converter (§9)
    frontend/                   tokenizer and config resources (§8)
    global/                     non-layer tensors (§6)
    ngram/                      n-gram table shards, hash buffers, hot rows (§7)
    layers/L00 .. layers/L47/   one directory per decoder layer (§3-§6)
    state/                      resume bookkeeping, run.json (never read by the packer)
  qwen3_8_flash_next_trellis_a25-v2.ninfer                  the container (the packer's)
  qwen3_8_flash_next_trellis_a25-v2.ninfer.conversion.json  its sidecar (§9)
  references/                   stored references, the G1 fixture, kernel references (§10, §11)
  traces/                       routing traces per domain (§12)
  report.txt                    the plain end-of-run report
  convert.log                   one progress line per layer
```

- The container's identity is `model_id` `qwen3.8-flash-next`, `weights_id`
  `trellis-a25-fp8rows-q4g32-de4b8e4`: it cannot be mistaken for a 27B artifact.
- `references/` and `traces/` are not container objects: they are kept as files beside
  the artifact (about 1.2 GB).
- **A work tree belongs to one configuration.** `work/state/run.json` records the
  revision, the corpus manifest, the canary sequences, `--layers`, `--table-shards`,
  `--budget`, the hot-row sample and the long windows; the converter refuses to continue
  a tree made with any other value. A dry run therefore goes to its own directory.

## 2. Completion protocol

- A directory under `work/` is **complete** only when it contains a file named
  `DONE`. `DONE` is written last, by writing `DONE.tmp`, flushing it and renaming it,
  after every other file of the directory was flushed to disk. Its content is a JSON
  object, `{"files": {"<name>": {"bytes": n, "sha256": "..."}}}`, listing every other file
  of the directory (not its subdirectories).
- A directory without `DONE`, or whose files no longer match it, is garbage: the
  converter checks every `DONE` on resume and redoes such a directory, and the packer
  refuses it.
- `work/layers/LNN/` is complete when the layer is converted. `work/frontend/` and
  `work/global/` are complete before layer 0 starts. `work/ngram/table/` (the 128 shards)
  and then `work/ngram/` (the small files) are complete during layer 0 or 1.
  `work/converter.json` is written last of all, after the end-of-run measurements;
  `"status": "complete"` means a full conversion (`"dry-run"` otherwise).
- The packer reads only `experts.bin`, `experts.idx`, `tensors.json` and the files
  `tensors.json` lists, the `ngram/` files of §7 and `frontend/`. Everything else in a
  layer directory (`layer.json`, `route_*.npy`, `moe_block/`) is the converter's own
  record, read by its end-of-run steps. The packer deletes nothing before
  `converter.json` exists, and a unit's files only after they are durably appended.

## 3. Expert projections: the trellis record

Each routed expert has two **expert projections**:

| projection | code | weight as exllamav3 takes it `(in, out)` | HF tensor |
|---|---|---|---|
| fused gate/up | `gu` (0) | `(2560, 1280)` | `mlp.experts.gate_up_proj[e].T`; output columns `[0, 640)` are **gate**, `[640, 1280)` are **up** (HF `chunk(2, dim=-1)`) |
| down | `dn` (1) | `(640, 2560)` | `mlp.experts.down_proj[e].T` |

Each projection is encoded by `exllamav3==1.5.3`'s `quantize_exl3_batch` with the
**mul1** codebook (`mul1: True`, no `mcg`), `apply_out_scales: True`, at one bit
width **K ∈ {2, 2.5, 3, 4}**, quantized as one matrix (gate/up: one K, one `suh`, one
`svh`). Its record is the quantizer's output tensors, unchanged, laid out as:

| bytes | content | type, shape |
|---|---|---|
| `[0, T)` | `trellis` | int16 `(in/16, out/16, 16·K)`, row-major: tile `(tk, tn)` starts at word `(tk·(out/16) + tn)·16K` |
| `[T, T + 2·in)` | `suh` (input channel scales) | fp16 `(in,)` |
| `[T + 2·in, T + 2·in + 2·out)` | `svh` (output channel scales) | fp16 `(out,)` |
| up to the next multiple of 4096 | zero padding | |

with `T = in · out · K / 8` bytes. Every section starts 16-byte aligned. The record's
size is a pure function of (projection, K): there are **eight K classes**.

| class | K | `k2` | trellis bytes | data bytes | **record bytes** | stored b/w incl. padding |
|---|---|---|---|---|---|---|
| gu-2 | 2 | 4 | 819,200 | 826,880 | **827,392** | 2.0200 |
| gu-2.5 | 2.5 | 5 | 1,024,000 | 1,031,680 | **1,032,192** | 2.5200 |
| gu-3 | 3 | 6 | 1,228,800 | 1,236,480 | **1,236,992** | 3.0200 |
| gu-4 | 4 | 8 | 1,638,400 | 1,646,080 | **1,646,592** | 4.0200 |
| dn-2 | 2 | 4 | 409,600 | 416,000 | **417,792** | 2.0400 |
| dn-2.5 | 2.5 | 5 | 512,000 | 518,400 | **520,192** | 2.5400 |
| dn-3 | 3 | 6 | 614,400 | 620,800 | **622,592** | 3.0400 |
| dn-4 | 4 | 8 | 819,200 | 825,600 | **827,392** | 4.0400 |

- **K is never a float in a file.** It is `k2 = 2·K` (4, 5, 6, 8) in the work files and
  the format code in the container (§4).
- **Half step (K = 2.5).** exllamav3's fractional encoding (`frac_k`): alternating
  2- and 3-bit trellis steps, step mask `0xAAAA` (period 16), packed by
  `ext.pack_trellis_frac`; 40 int16 words per 256-weight tile. Integer K packs with
  `ext.pack_trellis`; 16·K words per tile.
- **The codebook is not stored per record.** exllamav3 emits a constant `mul1` tensor
  (`0x83DCD12D` as int32) only as a flag; every record of this artifact is mul1, so
  the flag is implied by the format code. There is no `mcg` tensor.
- **`suh`/`svh` are real fp16 scales**, not sign vectors: exllamav3's
  `refit_scales` polishes them after quantization. There is no packed `su`/`sv`.
- **Tile content** (the order of a tile's 256 weights inside the trellis) is
  exllamav3's own: see `exllamav3/modules/quant/exl3_lib/quantize.py`,
  `tensor_core_perm` (lines 22-50), and the extension's `reconstruct`; the clean-room
  decoder `kernel/tests/fixtures/flash_next/record.py` (`decode_inner`) pins it.
- **Decode (the oracle).** The weight a record stands for is exactly what this
  produces (it is the converter's `decode`, used for the quantized reference and the
  self-checks):

  ```python
  from exllamav3.ext import exllamav3_ext as ext
  from exllamav3.modules.quant.exl3_lib import quantize as xq
  w = torch.empty((in_, out), dtype=torch.half, device="cuda")
  ext.reconstruct(w, trellis, K, False, True)        # mcg=False, mul1=True; K a float for 2.5
  w = xq.preapply_had_l(w.float(), 128)              # H128/sqrt(128) on each 128-row block
  w *= suh.float()[:, None]
  w = xq.preapply_had_r(w, 128)                      # H128/sqrt(128) on each 128-column block
  w *= svh.float()[None, :]
  W_hf = w.T.to(torch.bfloat16)                      # (out, in), what the quantized reference runs
  ```

  `H128` is `exllamav3.util.hadamard.get_hadamard(128)`. Equivalently, for an input
  row `x` (length `in`): `y = had128(x ∘ suh) · W_rot`, then `y = had128(y) ∘ svh`,
  with `had128` the per-128-block multiplication by `H128/√128` and `W_rot` the
  reconstructed `(in, out)` fp16 matrix. How a kernel fuses this is its own business;
  the bytes are fixed.
- **Bit compatibility.** A record's `trellis`, `suh` and `svh` bytes are exactly the
  tensors `quantize_exl3_batch` returned; `LinearEXL3(trellis=..., suh=..., svh=...,
  mul1=...)` accepts them unchanged.

## 4. Per-layer expert file and index; the container's expert objects

`work/layers/LNN/experts.bin` is the concatenation of the layer's 1024 records in the
order **expert id 0..511, and for each expert `gu` before `dn`**. Every record is a
multiple of 4096 bytes, so every record starts 4096-aligned relative to the file start.

`work/layers/LNN/experts.idx` is the index, 1024 entries of 16 bytes, in the same order:

| offset | type | field |
|---|---|---|
| 0 | u16 | expert id |
| 2 | u8 | projection (0 = `gu`, 1 = `dn`) |
| 3 | u8 | `k2` (4, 5, 6 or 8) |
| 4 | u32 | record bytes (from the class table) |
| 8 | u64 | record offset in `experts.bin` |

Readers check: entries in order, offsets contiguous from 0, sizes equal to the class
table, file size equal to the sum.

**In the container** each expert projection is one tensor object:
- name `layers.{L}.mlp.experts.{E}.gate_up_proj` / `layers.{L}.mlp.experts.{E}.down_proj`;
- format `TRELLIS_MUL1_K2`, `TRELLIS_MUL1_K2P5`, `TRELLIS_MUL1_K3` or `TRELLIS_MUL1_K4`
  (from `k2` 4, 5, 6, 8); layout `trellis-tile16-v1`, 4096-byte alignment;
- shape `[out, in]` (`gu` `[1280, 2560]`, `dn` `[2560, 640]`), stored bytes = the record
  bytes of the class table, padding included;
- within a layer, the `tensors.json` objects come first, then the 1024 records in index
  order, so a layer's experts are one contiguous range. `experts.idx` is not stored: the
  per-layer index is rebuilt from the directory.

## 5. Rate accounting

Acceptance 2's rate is computed as in run 8:

- a projection costs `K + 16·(in + out)/(in·out)` bits per weight: the trellis at
  exactly K, plus the fp16 `suh`/`svh`. The scale overhead is **0.01875** b/w for `gu`
  and **0.03125** for `dn`;
- the layer's rate per projection kind is the mean over its 512 experts; the allocation
  keeps it **≤ `--budget`** (2.50; the fallback re-conversion uses 3.0) separately for
  `gu` and for `dn` (run 8: 2.4992 / 2.5000);
- the **4 KiB padding is not in the rate**. The stored rate including padding (the
  class table's last column) is reported beside it in the sidecar and the report.

## 6. Non-expert tensors

### 6.1 Encodings

Every non-expert tensor is a file whose bytes are **exactly its container payload**,
so the packer copies bytes and the reader's `tensor_encoded_size` validates them:

- **FP8**: format `FP8_E4M3FN_ROW_BF16S`, layout `row-scale-v1` (the container's
  existing registry entries): a u8 E4M3FN code plane `rows × cols`, zero padding to the
  next multiple of 256, then a bf16 scale plane `rows`. Element `(r, c)` decodes to
  `e4m3fn(code[r, c]) · bf16(scale[r])`.
  - The scale of a row is `amax(|W[r, :]|) / 448` rounded **up** to the next bf16
    (so no code saturates or becomes NaN), or 1.0 for an all-zero row; codes are
    `round_to_nearest_even_e4m3fn(W / scale_bf16)`. Every stream that runs "FP8"
    weights in the converter dequantizes the stored bytes: `code · scale` in fp32,
    rounded to bf16, since the reference runs the checkpoint's modules in bf16 (the
    decoded trellis experts are rounded to bf16 the same way). An engine that keeps the
    product in fp32 differs from the reference by that rounding only.
  - **Departure from specs 01/04**, decided by the coordinator (2026-10-05): they say
    "one fp32 scale per output row" *and* "the container's existing row-scale layout",
    whose scale is bf16. The existing layout wins (no new format code). Quality is
    unaffected: the codes are computed against the stored scale, and a bf16 scale moves
    a row's grid by < 0.4 %, against E4M3's 12.5 % relative step.
- **BF16**: format `BF16`, layout `contiguous-le-v1`, the checkpoint's own bf16 values
  copied bit for bit (no conversion).

### 6.2 Which tensor gets which

FP8 (row scale): every 2-D linear weight whose rows and columns are both at least 16,
except the router and the experts:

| tensor (HF suffix) | layers | measured in FP8 by the study |
|---|---|---|
| `linear_attn.in_proj_qkv` [10240,2560], `in_proj_z` [6144,2560], `in_proj_a` [48,2560], `in_proj_b` [48,2560], `out_proj` [2560,6144] | GDN (36) | yes |
| `self_attn.q_proj` [12288,2560], `k_proj` [512,2560], `v_proj` [512,2560], `o_proj` [2560,6144], `self_attn.indexer.index_qk_proj` [640,2560] | attention (12) | yes |
| `mlp.shared_expert.gate_proj` [640,2560], `up_proj` [640,2560], `down_proj` [2560,640] (three tensors, not fused) | all | yes |
| `attn_hyper_connection.input_mix_weight_down` [320,10240], `.input_mix_weight_up` [10240,320], same for `mlp_hyper_connection` | all | **no** |
| `ple.key_proj` [10240,2560], `ple.value_proj` [2560,2560] | 1 | **no** |
| `embed_tokens` [248320,2560], `lm_head` [248320,2560], `hyper_connection_mixer.input_mix_weight_down` / `_up` | global | **no** |

BF16, copied: the router `mlp.gate.weight` [512,2560]; `mlp.shared_expert_gate`
[1,2560]; `*_hyper_connection.block_inject_weight` [4,10240]; every norm
(`*norm*.weight`, `hc_norm`), `linear_attn.conv1d.weight` [10240,1,4], `A_log` [48],
`dt_bias` [48], `ple.conv1d.weight` [10240,1,4].

The FP8 parts the study did not measure are measured in the pass: the FP8-only stream
(BF16 experts, FP8 non-experts, BF16 table) gives their total cost per domain beside the
quantized stream's, the head's FP8 cost is isolated, and an FP8-only KLD above 0.01 on
any domain is flagged in the report. Moving a part to BF16 is a re-conversion.

Not converted: `model.visual.*` (vision is out of this spec), `mtp.*` (out of scope).

### 6.3 Files

`work/layers/LNN/tensors.json` and `work/global/tensors.json`:

```json
{"tensors": [
  {"name": "layers.0.linear_attn.in_proj_qkv.weight",
   "file": "linear_attn.in_proj_qkv.weight.bin",
   "format": "FP8_E4M3FN_ROW_BF16S", "layout": "row-scale-v1",
   "shape": [10240, 2560], "bytes": 26234880, "sha256": "...",
   "fp8_measured_by_study": true}
]}
```

- `name` is the checkpoint name without the `model.language_model.` prefix (globals:
  `embed_tokens.weight`, `lm_head.weight`, `hyper_connection_mixer.*`; the checkpoint
  has no separate final norm, the mixer's `hc_norm` is it). It is the container name,
  verbatim.
- Tensors are **not fused**: one file per checkpoint tensor. Row-concatenating FP8
  row-scale tensors (e.g. GDN `qkv` + `z`) is a byte-level operation on the code and
  scale planes and needs no re-quantization.
- `shape` is the checkpoint shape (`[out, in]` for linears, original rank for the
  rest).

## 7. The n-gram table

### 7.1 Table rows

The checkpoint's table is `layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}`,
each `[2,500,012, 160]` bf16; concatenated in shard order they are the table's
**320,001,536 rows** of 160 values (16 heads × 160 = 2560 after the gather). A row is
**90 bytes**, INT4 in groups of 32:

| bytes | content |
|---|---|
| `[0, 80)` | 160 4-bit codes: byte `j` holds value `2j` in its low nibble and value `2j+1` in its high nibble |
| `[80, 90)` | 5 fp16 scales, group `g` (values `32g .. 32g+31`) at `80 + 2g` |

- Value `i` decodes to `(nibble_i − 8) · scale[i / 32]`.
- Encoding: `scale = fp16_round_nearest(amax_g / 7)`; `q = clamp(round_half_even(x /
  scale), −7, 7)`, computed in fp32 against the **fp16-rounded** scale; `nibble = q + 8`
  (1..15, 0 is never written). A group whose fp16 scale is 0 stores `q = 0`.
- Because quantization is per row, quantizing a gathered row gives exactly the stored
  row. The quantized reference feeds layer 1's PLE with the decoded rows rounded to bf16.

`work/ngram/table/shard_NNN.int4` (NNN = 000..127) holds rows
`[NNN · 2,500,012, (NNN+1) · 2,500,012)`, 225,001,080 bytes each, row `r` at
`(r − NNN·2,500,012) · 90`; `work/ngram/table/DONE` lists the shards. The table is
sharded so the packer can append and delete shard by shard.

**In the container** the table is one tensor `layers.1.ple.ple_embedding.ngram_embedding.weight`,
format `Q4G32_F16S`, layout `row-interleaved-v1`, shape `[320001536, 160]`, 4096-byte
alignment: the shards concatenated, row `r` at `r · 90` from the object's start, no
index. The binder hands its file range to the n-gram reader (host-streamed); nothing in
the directory marks that role.

### 7.2 Hash buffers and per-head sizes

`work/ngram/` also holds, each as raw little-endian **int64**, the checkpoint's own
buffers verbatim:

| file | container object (format `I64`, `contiguous-le-v1`) | shape | value at revision `de4b8e4…` |
|---|---|---|---|
| `layer_multipliers.i64` | `layers.1.ple.ple_embedding.layer_multipliers` | `[3]` | 23703573157769, 20109073645365, 8052911324071 |
| `ngram_heads_vocab_sizes.i64` | `layers.1.ple.ple_embedding.ngram_heads_vocab_sizes` | `[16]` | 20000003 … 20000171 |
| `ngram_heads_offsets.i64` | `layers.1.ple.ple_embedding.ngram_heads_offsets` | `[16]` | 0 … 300001275 |

The multipliers do not fit int32. The hashing is the checkpoint's
`Qwen4ExpTextNGramEmbedding.forward` (transformers 5.17): n-gram orders 2 and 3, 8
heads each, eos-aware right shifts, `ids = (Σ_xor tok_shift_p · mult_p) mod
vocab_size_h + offset_h`.

### 7.3 Hot rows

`work/ngram/hot_rows.u32`: u32 row indices (all below 2^31), ranked by lookup count,
**count descending, ties by row index ascending**, only rows with count ≥ 1, capped at
`floor(2 GiB / 90)` rows. The counts are taken over the union of
- the calibration chunks' valid tokens (16 lookups per token), and
- the research session's n-gram coverage sample (`review/ngram_coverage.py`): up to
  1.2M tokens each of this repository's code and docs (files at `--repo`'s HEAD with
  extensions rs md py ts tsx toml cu h, sorted then shuffled with seed 0),
  Italian Wikipedia (`20231101.it`, first shard) and WikiText-103, documents alternated
  train/test with every 4th held out and only the train documents counted.

The list is expected around 1.2-1.5 GB. `work/ngram/hot_rows.json` records the row
count, the bytes, the shard count (`shards`), `table_rows` and `complete`, and the
lookups and sample tokens behind it. In the container the list is the tensor
`layers.1.ple.ple_embedding.ngram_embedding.hot_rows`, format `I32`, shape `[n]` (the
same bytes). The engine decides how much of it to load.

## 8. Frontend

`work/frontend/` holds, fetched from the pinned revision byte for byte:
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`
(the 27B's `FRONTEND_RESOURCES` names), plus `config.json` (the checkpoint's model
config, for the binder's topology checks). Each file is the container resource
`frontend/<file>`.

## 9. `work/converter.json`: the converter half of the sidecar

The packer merges this object into the sidecar `<artifact>.conversion.json` and adds
what only it knows (`recipe_id`, `artifact.bytes`, `objects.count`, as `Sidecar::load`
requires); each side read-modify-writes only its own keys. Fields:

```jsonc
{
  "schema": "flash-next-converter-v1",
  "status": "complete",                       // "dry-run" for a reduced run
  "verdict": "PASS",                          // PASS | FAIL | DRY-RUN
  "flags": [],                                // e.g. FP8-only KLD above 0.01 on a domain
  "source": {"repo": "Qwen/Qwen3.8-Flash-Next", "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540"},
  "converter": {"commit": "<git sha>", "dirty": false, "seed_rule": "...", "command": "<argv>",
                "run": {...}},                // work/state/run.json
  "versions": {"exllamav3": "1.5.3", "transformers": "5.17.0", "torch": "...", "python": "..."},
  "quantizer": {"codebook": "mul1", "apply_out_scales": true, "K_set": [2, 2.5, 3, 4],
                "hessian": "...", "budget_bits": 2.5, "batch": 32,
                "hessian_fallback": {"<layer>": [expert ids with no calibration token]}},
  "corpus": {"manifest": [{"file", "sha256", "sources"}], "licences": {"<source>": {"what", "licence"}},
             "calibration_chunks": 224, "test_chunks": 66,
             "tokens_per_domain": {"calibration": {...}, "test": {...}},
             "long8192": [{"window": 3, "start": 0, "kind": "code"}, ...],
             "canary": ["rust-hello", ...], "hot_sample_tokens": {...}},
  "k_map": {"layers": [{"layer": 0, "gu": [k2 x 512], "dn": [k2 x 512]}]},
  "k_hist": [{"layer": 0, "gu": {"2": n, "2.5": n, "3": n, "4": n}, "dn": {...}}],
  "rates": {"per_layer": [{"layer": 0, "gu": 2.4992, "dn": 2.5000, "gu_stored": 2.52, "dn_stored": 2.54}],
            "mean": {"gu": ..., "dn": ..., "gu_stored": ..., "dn_stored": ...}},
  "k_classes": [{"class": "gu-2", "k2": 4, "projections": n, "record_bytes": 827392,
                 "selections": n, "traffic_share": 0.0}],   // calibration selections landing on the class
  "expert_traffic": {"layers": [{"layer": 0, "counts": [n x 512]}]},   // selections per expert, BF16 stream
  "moe_error_db": [{"layer": 0, "db": ..., "run6_db": -19.96, "run8_db": -20.97, "per_kind_db": {...}}],
  "acceptance2": {"pass": true, "budget": 2.5},
  "acceptance3": {"mean_db_layers_1_5": ..., "run8_mean_db_layers_1_5": -16.8,
                  "named_layers_worse_than_run6_by_1db": [], "pass": true},
  "kld": {"quantized": {"<domain>": {"kld", "kld_top64", "top1", "ppl", "ppl_bf16", "run6", "limit", "within"}},
          "fp8_only": {"<domain>": {"kld", "kld_top64", "top1", "run6_fp8"}}},
  "kld_long8192": {...}, "kld_head_fp8_only": {"<domain>": ...},
  "mmlu": {"n": 281, "bf16": ..., "quantized": ..., "fp8_only": ...,
           "mcnemar": {"quantized": {"lost": n, "gained": n, "p": ...}, "fp8_only": {...}}},
  "acceptance4": {"pass": true, "kld_pass": true, "mmlu_pass": true, "fallback": "..."},
  "experts_bin": [{"layer": 0, "bytes": n, "sha256": "..."}],     // each layer's experts.bin, as DONE has it
  "decode_sha256": [{"layer": 0, "class": "gu-2.5", "expert": 7, "sha256": "..."}],
  "self_check": {"work_files": {"projections_checked": n, "bit_identical": true, "mismatches": []},
                 "container": "convert.py verify, after packing"},
  "ngram": {"rows": 320001536, "row_bytes": 90, "hot_rows": n, "hot_rows_bytes": n, "shards_converted": 128},
  "references": {...}, "traces": {...}, "time_s": {"total": ..., "per_layer": [...]}
}
```

- `decode_sha256`: per layer, the first expert of every K class present: the sha256 of
  its decoded weight (§3's oracle, bf16 `(out, in)` bytes). `convert.py verify` decodes
  the same projections from the packed container's bytes and must match them; the
  sidecar keeps them after the work files are gone.
- `kld.quantized.*.limit` is run 6's per-domain KLD × 1.1 (code 0.119, prose 0.177, en
  0.114, it 0.056, zh 0.089, math 0.044, py 0.028, de 0.224, ja 0.211); `chat` (0.059)
  and `mmlu` (0.132) are reported without a limit. The FP8-only variant is shown beside
  run 6's FP8-only figures (code 0.058, prose 0.084, en 0.029, it 0.015, zh 0.025, math
  0.015, py 0.009, de 0.024, ja 0.022, chat 0.026, mmlu 0.065).

## 10. Stored references

`references/<set>/` per window set. Sets: `test2048` (the study's 66 held-out chunks,
2048 tokens), `long8192` (8 windows of 8192 tokens: offsets 0, 8192, 16384, 24576 of the
27B KLD study's long window 3, `crates rs` part 1, and of its long window 5, `docs`
prose part 1; neither is in the calibration, which uses long windows 0-2), `canary`
(the G1 sequences, §11).

| file | type, shape | content |
|---|---|---|
| `manifest.json` | | windows in order: `{index, kind, source, length, valid, first_position}`; `P = Σ valid` |
| `tokens.u32` | u32, each window's `length` tokens, concatenated in window order | the fed token ids (EOS padding past `valid`) |
| `bf16_top64_ids.i32` | i32 `(P, 64)` | BF16 stream: the 64 most probable next tokens, descending |
| `bf16_top64_lp.f32` | f32 `(P, 64)` | their log-probabilities (nats) |
| `bf16_lse.f32` | f32 `(P,)` | `logsumexp` of the position's logits |
| `q_argmax.i32` | i32 `(P,)` | quantized stream: argmax |
| `q_top64_ids.i32`, `q_top64_lp.f32`, `q_lse.f32` | as above | quantized stream |

- Position `p` of a window is the distribution after consuming tokens `[0..p]`
  (predicting token `p+1`); positions `0..valid−1` are stored for every window,
  concatenated in window order (`first_position` is the window's first row).
- Logits are the head's bf16 output (`final mixer → lm_head`, bf16 matmul) upcast to
  fp32; log-probabilities and `lse` are computed in fp32. Ties in top-64 and argmax
  resolve to the lower token id.
- **The top-64 scorer** (spec 04 uses the same): with the reference's top-64 ids `I`,
  log-probs `r_i`, tail mass `R = 1 − Σ_{i∈I} e^{r_i}`, and the candidate's full
  log-softmax `c`:
  `KL ≈ Σ_{i∈I} e^{r_i}(r_i − c_i) + R · (log R − log(1 − Σ_{i∈I} e^{c_i}))`,
  with the tail term dropped when `R < 1e-12`. It is a lower bound of the exact KL; the
  converter reports both so the engine's figure is compared like with like.
- **Proposed for the next conversion** (not in the 2026-10-05 run's output):
  `q_lp_at_bf16_ids.f32`, f32 `(P, 64)`, the quantized stream's log-probs at the BF16
  row's top-64 ids (`lq.gather(-1, r_ids)` in `finish.py` `_score_window`, about 35 MB for
  `test2048`). With it the quantized `kld_top64` is reproducible from the set alone;
  without it, only at the positions whose BF16 top-64 ids are all in the quantized top-64,
  since a quantized log-prob outside its own top-64 is not stored: at most 0.01% of the
  positions of any domain in the 2026-10-05 run. The acceptance scorers
  (`crates/bench/src/flash_next/references.rs`) read the file when it is present.

**Kernel references (spec 02).**
- `references/moe_block/LNN/` for layers 2, 24 and 46: 64 tokens (positions
  1024..1087) of the first test chunk through the layer's quantized MoE block —
  `x.bf16` (64, 2560) the block's input, `y.f32` (64, 2560) its output (bf16 upcast),
  `ids.i32` (64, 10) the router's top-10, descending, `weights.f32` (64, 10) their
  weights, `shared.f32` (64, 2560) `sigmoid(shared_expert_gate(x)) · shared_expert(x)`
  alone, and `manifest.json`.
- `references/trellis_checksums.json`: at layers 0, 24 and 47, two projections per K
  class present, `{layer, expert, proj, k2, checksum}` with checksum
  `Σ_i u16(reconstruct_fp16_bits[i]) · (lowbias32(i) | 1) mod 2^64` over the row-major
  `(in, out)` output of `ext.reconstruct(w, trellis, K, False, True)` (hex), as
  `checksum_u16` in `kernel/tests/fixtures/flash_next/record.py`.

## 11. The G1 fixture

`references/g1_flash_next.json` keeps the bench crate's `Fixture` shape verbatim
(`model`, `max_tokens`, `prompts[{id, prompt, text, token_ids}]`) so the existing
reader parses it, and adds per prompt:

- `prompt_token_ids`: the prompt rendered with Flash-Next's `chat_template.jinja`
  from the pinned revision, messages `[{"role": "user", "content": prompt}]`,
  `add_generation_prompt=True`, `enable_thinking=False` (the template then ends with
  `<think>\n\n</think>\n\n`; the 27B fixture's texts are answers only), tokenized with
  the artifact's tokenizer, no special tokens added;
- `token_ids`: the 27B fixture's completion `text` re-tokenized with Flash-Next's
  tokenizer (identical to the 27B's at this revision; recorded anyway);
- `expected_argmax`: `len == len(token_ids)`; entry `i` is the **quantized reference's**
  argmax after `prompt_token_ids + token_ids[0..i)` — `score_canary`'s expected column;
- `bf16_argmax`: the same from the BF16 stream (information only).

Top level adds `"render": {"template": "chat_template.jinja@de4b8e4…",
"enable_thinking": false, "add_generation_prompt": true}` and
`"reference": "quantized"`. `model` is `qwen3.8-flash-next-ignis`.

## 12. Routing traces and traffic

`traces/<domain>/` for every domain of the test chunks (`chat`, `code`, `prose`, `en`,
`it`, `zh`, `math`, `py`, `de`, `ja`, `mmlu`) and `traces/long8192/`:

| file | type, shape | content |
|---|---|---|
| `manifest.json` | | chunks in order, their `valid` lengths, token count `N` (a replay resets at chunk boundaries) |
| `experts.i16` | i16 `(N, 48, 10)` | quantized stream: the router's top-10 expert ids per token and layer, in the router's descending-weight order |
| `weights.f16` | f16 `(N, 48, 10)` | the matching routing weights |
| `lookahead.i16` | i16 `(N, 48, 20)` | layer L's MoE input through router L+1 (BF16 weights), top-20, descending; layer 47's row is −1 |

Token-major, so a replay walks token by token and layer by layer. Only valid tokens are
included. `manifest.json` is exactly:

```json
{"tokens": N, "layers": 48, "set": "test", "chunks": [0, 1, 6, 7], "valid": [2048, 2048, 2048, 2048],
 "stream": "quantized", "order": "token-major (N, layers, k)"}
```

`set` is `"test"` (the 2048-token test chunks) or `"long"` (`traces/long8192/`); `chunks`
are indices within that set, in file order; `valid` has the same length and sums to `N`.
Chunk indices are unique within a set but not across sets (long8192's 0.. collide with
the test chunks'), so a replay keys a chunk by (set, index) and resets its state at each
chunk boundary. Residency takes the first W (default 16) lookahead ids, and its pool sizes
and warm-start order from `converter.json`'s `k_map`, `k_classes[].record_bytes` and
`expert_traffic`.
