# Flash-Next artifact: the converter's output contract

GitHub: #299 (spec 01). Owner of this document: the converter (`tools/flash-next-converter/`).
Readers: the Rust packer and reader (`crates/artifact`), the MoE kernels (spec 02),
residency (spec 03), the forward and its gates (spec 04).

This is the byte-level contract between the Python converter, which writes **work
files**, and everything that reads them: the Rust packer that assembles the `.ninfer`
v2 container, the kernels that decode expert projections, and the harnesses that
replay the stored references and routing traces. Spec 01 says *what* is stored; this
document says *how*.

Every item carries a tag:
- **[decided]** fixed by this document; changing it means a re-conversion;
- **[pack's call]** the container mapping, decided by the packer's owner;
- **[kern's call]** a kernel-side choice that does not change stored bytes;
- **[ASK]** an open question to the coordinator (listed again at the end).

All integers and floats are **little-endian**. "fp16" is IEEE binary16, "bf16" is
bfloat16, "fp32" is IEEE binary32. Shapes are row-major, outermost first.

## 1. The output tree [decided]

```
F:/ai/models/Qwen3.8-Flash-Next-ignis/
  work/                         converter output, consumed (and deleted) by the packer
    converter.json              run record: everything the sidecar needs from the converter (§9)
    frontend/                   tokenizer and config resources (§8)
    global/                     non-layer tensors (§6)
    ngram/                      n-gram table shards, hash buffers, hot rows (§7)
    layers/L00 .. layers/L47/   one directory per decoder layer (§3-§6)
    state/                      resume bookkeeping (never read by the packer)
  references/                   stored references and the G1 fixture (§10, §11); kept
  traces/                       routing traces per domain (§12); kept
  report.txt                    the plain end-of-run report
  convert.log                   one progress line per layer
```

The packer writes the container and its sidecar next to `work/` (names: [pack's
call]). `references/` and `traces/` are not container objects: they are kept as files
beside the artifact (a few hundred MB).

## 2. Completion protocol [decided]

- A directory under `work/` is **complete** only when it contains a file named
  `DONE`. `DONE` is written last, by writing `DONE.tmp`, flushing it and renaming it.
  Its content is a JSON object: `{"files": {"<name>": {"bytes": n, "sha256": "..."}}}`
  listing every other file of the directory.
- A directory without `DONE` is garbage: the converter deletes and redoes it on
  resume, and the packer refuses to read it.
- `work/layers/LNN/` is complete when the layer is converted; `work/global/`,
  `work/ngram/`, `work/frontend/` get their own `DONE`. `work/converter.json` is
  written last of all, after the end-of-run measurements, and its presence (with
  `"status": "complete"`) means the whole work tree is complete.
- The packer may delete a directory's files after appending them to the container.

## 3. Expert projections: the trellis record [decided]

Each routed expert has two **expert projections**:

| projection | code | weight as exllamav3 takes it `(in, out)` | HF tensor |
|---|---|---|---|
| fused gate/up | `gu` (0) | `(2560, 1280)` | `mlp.experts.gate_up_proj[e].T`; output columns `[0, 640)` are **gate**, `[640, 1280)` are **up** (HF `chunk(2, dim=-1)`) |
| down | `dn` (1) | `(640, 2560)` | `mlp.experts.down_proj[e].T` |

Each projection is encoded by `exllamav3==1.5.3`'s `quantize_exl3_batch` with the
**mul1** codebook (`mul1: True`, no `mcg`), `apply_out_scales: True`, at one bit
width **K ∈ {2, 2.5, 3, 4}**. Its record is the quantizer's output tensors, unchanged,
laid out as:

| bytes | content | type, shape |
|---|---|---|
| `[0, T)` | `trellis` | int16 `(in/16, out/16, 16·K)`, row-major: tile `(tk, tn)` starts at word `(tk·(out/16) + tn)·16K` |
| `[T, T + 2·in)` | `suh` (input channel scales) | fp16 `(in,)` |
| `[T + 2·in, T + 2·in + 2·out)` | `svh` (output channel scales) | fp16 `(out,)` |
| up to the next multiple of 4096 | zero padding | |

with `T = in · out · K / 8` bytes. The record's size is a pure function of
(projection, K): there are **eight K classes**.

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

- **K is never a float in a file.** It is stored as `k2 = 2·K` (4, 5, 6, 8).
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
  `tensor_core_perm` (lines 22-50), and the extension's `reconstruct`. This document
  does not paraphrase it.
- **Decode (the oracle).** The weight a record stands for is exactly what this
  produces (it is the converter's `decode`, used for the quantized reference and the
  self-check):

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
  reconstructed `(in, out)` fp16 matrix. [kern's call] how the kernel fuses this; the
  bytes are fixed.
- **Bit compatibility.** A record's `trellis`, `suh` and `svh` bytes are exactly the
  tensors `quantize_exl3_batch` returned (`.contiguous()`, then `.numpy().tobytes()`);
  `LinearEXL3(trellis=..., suh=..., svh=..., mul1=...)` accepts them unchanged.
  [kern's call] a different *tile order* in the container would break this; the doc
  fixes exllamav3's.

## 4. Per-layer expert file and index [decided]

`work/layers/LNN/experts.bin` is the concatenation of the layer's 1024 records in the
order **expert id 0..511, and for each expert `gu` before `dn`**. Every record is a
multiple of 4096 bytes, so every record starts 4096-aligned relative to the file start.
The file is exactly one contiguous range of the container: [pack's call] whether the
container holds it as one object per layer plus the index, or one object per
projection.

`work/layers/LNN/experts.idx` is the index, 1024 entries of 16 bytes, in the same order:

| offset | type | field |
|---|---|---|
| 0 | u16 | expert id |
| 2 | u8 | projection (0 = `gu`, 1 = `dn`) |
| 3 | u8 | `k2` (4, 5, 6 or 8) |
| 4 | u32 | record bytes (from the class table) |
| 8 | u64 | record offset in `experts.bin` |

Readers must check: entries in order, offsets contiguous from 0, sizes equal to the
class table, file size equal to the sum. **The only source of each record's size is
this index**: the container's directory for the experts cannot be final before the
last layer's index exists (or the packer reserves directory space) — [pack's call].

## 5. Rate accounting [decided]

Acceptance 2's rate is computed as in run 8:

- a projection costs `K + 16·(in + out)/(in·out)` bits per weight: the trellis at
  exactly K, plus the fp16 `suh`/`svh`. The scale overhead is **0.01875** b/w for `gu`
  and **0.03125** for `dn`;
- the layer's rate per projection kind is the mean over its 512 experts; the allocation
  keeps it **≤ 2.50** separately for `gu` and for `dn` (run 8: 2.4992 / 2.5000);
- the **4 KiB padding is not in the rate**. The stored rate including padding (the
  class table's last column) is reported beside it in the sidecar and the report.

## 6. Non-expert tensors [decided]

### 6.1 Encodings

Every non-expert tensor is a file whose bytes are **exactly its container payload**,
so the packer copies bytes and the reader's `tensor_encoded_size` validates them:

- **FP8**: format `FP8_E4M3FN_ROW_BF16S`, layout `row-scale-v1` (the container's
  existing registry entries): a u8 E4M3FN code plane `rows × cols`, zero padding to the
  next multiple of 256, then a bf16 scale plane `rows`. Element `(r, c)` decodes to
  `e4m3fn(code[r, c]) · bf16(scale[r])`.
  - The scale of a row is `amax(|W[r, :]|) / 448` rounded **up** to the next bf16
    (so no code saturates), or 1.0 for an all-zero row; codes are
    `round_to_nearest_even_e4m3fn(W / scale_bf16)`. Every stream that runs "FP8"
    weights in the converter dequantizes the stored bytes.
  - **Spec departure, [ASK]:** spec 01 and spec 04 say "fp32 per-row scale" *and*
    "the container's existing row-scale layout"; the existing layout's scale is bf16.
    This document takes the existing layout (no new format code; the materializer and
    the vendored FP8 geometry already read it). The cost is nil for quality: the codes
    are computed against the stored scale, and a bf16 scale moves a row's grid by
    < 0.4 %, against E4M3's 12.5 % relative step. If the coordinator or kern want fp32,
    the converter switches to a new `FP8_E4M3FN_ROW_F32S` (fp32 scale plane) before the
    full run.
- **BF16**: format `BF16`, layout `contiguous-le-v1`, the checkpoint's own bf16 values
  copied bit for bit (no conversion).

### 6.2 Which tensor gets which

FP8 (row scale) — every 2-D linear that is not an expert, not the router and has at
least 16 rows:

| tensor (HF suffix) | layers | measured in FP8 by the study |
|---|---|---|
| `linear_attn.in_proj_qkv` [10240,2560], `in_proj_z` [6144,2560], `in_proj_a` [48,2560], `in_proj_b` [48,2560], `out_proj` [2560,6144] | GDN (36) | yes |
| `self_attn.q_proj` [12288,2560], `k_proj` [512,2560], `v_proj` [512,2560], `o_proj` [2560,6144], `self_attn.indexer.index_qk_proj` [640,2560] | attention (12) | yes |
| `mlp.shared_expert.gate_proj` [640,2560], `up_proj` [640,2560], `down_proj` [2560,640] | all | yes |
| `attn_hyper_connection.input_mix_weight_down` [320,10240], `.input_mix_weight_up` [10240,320], same for `mlp_hyper_connection` | all | **no** |
| `ple.key_proj` [10240,2560], `ple.value_proj` [2560,2560] | 1 | **no** |
| `embed_tokens` [248320,2560], `lm_head` [248320,2560], `hyper_connection_mixer.input_mix_weight_down` / `_up` | global | **no** |

BF16, copied: the router `mlp.gate.weight` [512,2560]; `mlp.shared_expert_gate`
[1,2560]; `*_hyper_connection.block_inject_weight` [4,10240]; every norm
(`*norm*.weight`, `hc_norm`), `linear_attn.conv1d.weight` [10240,1,4], `A_log` [48],
`dt_bias` [48], `ple.conv1d.weight` [10240,1,4].

The FP8 parts the study did not measure are measured in the pass (acceptance 4's
FP8-only stream, plus per-group attributions in the report); any that costs measurably
is named in the report, and moving it to BF16 is a re-conversion.

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
  has no separate final norm, the mixer's `hc_norm` is it). Container names: [pack's call] (the 27B uses
  `text/layers/{i}/...`).
- Tensors are **not fused**: one file per checkpoint tensor. Row-concatenating FP8
  row-scale tensors (e.g. GDN `qkv` + `z`) is a byte-level operation on the code and
  scale planes and needs no re-quantization: [pack's call] / topology's call.
- `shape` is the checkpoint shape (`[out, in]` for linears, original rank for the
  rest).

## 7. The n-gram table [decided]

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
  row; the converter uses that to feed the quantized stream.

`work/ngram/table/shard_NNN.int4` (NNN = 000..127) holds rows
`[NNN · 2,500,012, (NNN+1) · 2,500,012)`, 225,001,080 bytes each, row `r` at
`(r − NNN·2,500,012) · 90`. Concatenated in order, row `r` of the table is at `r · 90`:
the host-streamed object is the concatenation, row stride 90, no index. The table is
sharded so the packer can append and delete shard by shard (a single 28.8 GB file would
need a second copy on F: during packing).

### 7.2 Hash buffers and per-head sizes

`work/ngram/` also holds, each as raw little-endian **int64**:

| file | shape | value at revision `de4b8e4…` |
|---|---|---|
| `layer_multipliers.i64` | `[3]` | 23703573157769, 20109073645365, 8052911324071 |
| `ngram_heads_vocab_sizes.i64` | `[16]` | 20000003 … 20000171 |
| `ngram_heads_offsets.i64` | `[16]` | 0 … 300001275 |

The multipliers do not fit int32, and the container registry has no I64: [pack's call]
(a new `I64` format, or `raw-bytes-v1` resources). The hashing is the checkpoint's
`Qwen4ExpTextNGramEmbedding.forward` (transformers 5.17): n-gram orders 2 and 3, 8
heads each, eos-aware right shifts, `ids = (Σ_xor tok_shift_p · mult_p) mod
vocab_size_h + offset_h`.

### 7.3 Hot rows

`work/ngram/hot_rows.u32`: u32 row indices, ranked by lookup count over the
calibration chunks' valid tokens (16 lookups per token), **count descending, ties by
row index ascending**, only rows with count ≥ 1, capped at `floor(2 GiB / 90)` rows.
`work/ngram/hot_rows.json` records the row count, the bytes it covers and the lookups
it covers. The study's calibration set has ~0.46M tokens, so the list will be far
below the cap (a few million rows, a few hundred MB); the engine decides how much of
it to load. [ASK] whether to extend it with a larger hash-only text sample.

## 8. Frontend [decided]

`work/frontend/` holds, fetched from the pinned revision byte for byte:
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`, `preprocessor_config.json`, `video_preprocessor_config.json`
(the 27B's `FRONTEND_RESOURCES` names), plus `config.json` (the checkpoint's model
config, for the binder's topology checks; [pack's call] whether it becomes an object).

## 9. `work/converter.json`: the converter half of the sidecar [decided]

The packer merges this object into the container's sidecar and adds what only it
knows (`recipe_id`, `artifact.bytes`, `objects.count`, as `Sidecar::load` requires).
Fields:

```jsonc
{
  "schema": "flash-next-converter-v1",
  "status": "complete",                       // "complete" only when every measurement ran
  "source": {"repo": "Qwen/Qwen3.8-Flash-Next",
             "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540"},
  "converter": {"commit": "<git sha of the repo>", "dirty": false,
                "seed": 0, "command": "<argv>"},
  "versions": {"exllamav3": "1.5.3", "transformers": "5.17.0", "torch": "2.13.0+cu130",
               "python": "3.x"},
  "quantizer": {"codebook": "mul1", "apply_out_scales": true, "K_set": [2, 2.5, 3, 4],
                "seed_rule": "L*10000 + proj*1000 + expert",
                "hessian": "g^2-weighted per-expert input metric on calibration tokens; gate/up shrunk 5% toward the layer H, down 5% toward I",
                "budget_bits": 2.5},
  "corpus": {"manifest": [ {"source": "...", "licence": "...", "file": "...",
                            "sha256": "...", "domain": "...", "chunks": n,
                            "tokens": n, "role": "calibration|test|long|canary"} ],
             "tokens_per_domain": {"calibration": {...}, "test": {...}}},
  "k_map": {"layers": [{"layer": 0, "gu": [k2 x 512], "dn": [k2 x 512]}]},
  "k_hist": [{"layer": 0, "gu": {"2": n, "2.5": n, "3": n, "4": n}, "dn": {...}}],
  "rates": {"per_layer": [{"layer": 0, "gu": 2.4992, "dn": 2.5000,
                           "gu_stored": 2.52, "dn_stored": 2.54}],
            "mean": {"gu": ..., "dn": ..., "gu_stored": ..., "dn_stored": ...}},
  "k_classes": [{"class": "gu-2", "projections": n, "record_bytes": 827392,
                 "traffic_share": 0.0}],          // share of calibration (token, expert) selections
  "expert_traffic": {"layers": [{"layer": 0, "counts": [n x 512]}]},
  "moe_error_db": [{"layer": 0, "db": -21.0, "run6_db": -19.96, "run8_db": -20.97,
                    "per_kind_db": {"code": ...}}],
  "acceptance3": {"mean_db_layers_1_5": ..., "run8_mean_db_layers_1_5": -16.8,
                  "named_layers_worse_than_run6_by_1db": [], "pass": true},
  "kld": {"quantized": {"<domain>": {"kld": ..., "kld_top64": ..., "top1": ...,
                                     "run6": ..., "limit": ..., "pass": true}},
          "fp8_only": {"<domain>": {...}}},
  "mmlu": {"n": 281, "bf16": 0.737, "quantized": ..., "fp8_only": ...,
           "mcnemar": {"quantized": {"lost": n, "gained": n, "p": ...}, "fp8_only": {...}}},
  "acceptance4": {"pass": true, "fallback": "3.0-bit mean (45.3 GB pinned)"},
  "self_check": {"projections_checked": n, "bit_identical": true},
  "ngram": {"rows": 320001536, "row_bytes": 90, "hot_rows": n},
  "references": {"dir": "references", "sets": [...]},
  "traces": {"dir": "traces", "domains": [...]},
  "time_s": {"total": ..., "per_layer": [...]}
}
```

`kld.*.limit` is run 6's per-domain KLD × 1.1 (code 0.119, prose 0.177, en 0.114, it
0.056, zh 0.089, math 0.044, py 0.028, de 0.224, ja 0.211; `chat` 0.059 and `mmlu`
0.132 are reported but not in acceptance 4); the FP8-only variant is compared with run
6's FP8-only figures (code 0.058, prose 0.084, en 0.029, it 0.015, zh 0.025, math
0.015, py 0.009, de 0.024, ja 0.022, chat 0.026, mmlu 0.065) for information.

## 10. Stored references [decided]

`references/<set>/` per window set. Sets: `test2048` (the study's 66 held-out chunks,
2048 tokens), `long8192` (8 windows of 8192 tokens, see [ASK] below), `canary` (the
G1 sequences, §11).

| file | type, shape | content |
|---|---|---|
| `manifest.json` | | windows in order: `{index, kind, source, length, valid, first_position}`; `P = Σ valid` |
| `tokens.u32` | u32 `(n_windows, length)` | the fed token ids (EOS padding past `valid`) |
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
  with the tail term dropped when `R < 1e-12`. The converter reports it beside the exact
  full-vocabulary KLD so the two are comparable.
- [ASK] `long8192` selection: proposed 4 windows of 8192 from the 27B KLD study's long
  window 3 (`crates rs`, part 1, 32,768 tokens) and 4 from long window 5 (`docs`
  prose, part 1, first 32,768 of 65,536). Neither is in the calibration or the test
  chunks (those use long windows 0-2 only).

## 11. The G1 fixture [decided]

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
  argmax after `prompt_token_ids + token_ids[0..i)` — `score_teacher_forced`'s
  `oracle` argument;
- `bf16_argmax`: the same from the BF16 stream (information only).

Top level adds `"render": {"template": "chat_template.jinja@de4b8e4…",
"enable_thinking": false, "add_generation_prompt": true}` and
`"reference": "quantized"`. `model` is `qwen3.8-flash-next-ignis`.

## 12. Routing traces and traffic [decided]

`traces/<domain>/` for every domain of the test chunks (`chat`, `code`, `prose`, `en`,
`it`, `zh`, `math`, `py`, `de`, `ja`, `mmlu`) and `traces/long8192/`:

| file | type, shape | content |
|---|---|---|
| `manifest.json` | | chunks in order, their `valid` lengths, token count `N` |
| `experts.i16` | i16 `(N, 48, 10)` | quantized stream: the router's top-10 expert ids per token and layer, in the router's descending-weight order |
| `weights.f16` | f16 `(N, 48, 10)` | the matching routing weights |
| `lookahead.i16` | i16 `(N, 48, 20)` | layer L's MoE input through router L+1 (BF16 weights), top-20, descending; layer 47's row is −1 |

Token-major, so a replay walks token by token and layer by layer. Only valid tokens are
included. `converter.json`'s `expert_traffic` (selections per expert and layer over the
calibration tokens, BF16 stream) and `k_classes[].traffic_share` give residency its
pool sizes and warm-start order. [ASK resid] whether the lookahead width or the
weights are what the CPU model needs.

## 13. Open questions

- **[ASK main]** FP8 scale dtype (§6.1): existing bf16-scale format (this document) or
  a new fp32-scale format (spec text).
- **[ASK main]** `long8192` window selection (§10).
- **[ASK main]** hot-row list size (§7.3): calibration-only, far below 2 GB, or extended.
- **[ASK main]** the corpus token files (study JSONs, ~5 MB): committed into the tool,
  or passed by `--corpus-dir` and pinned by sha256 in the committed manifest.
- **[pack's call]** container names, one object per projection or per layer, I64
  buffers, directory finalization after the last layer.
- **[kern's call]** anything that changes how a record is read, never its bytes.
- **[ASK resid]** trace contents (§12).
