# The 27B NVFP4 artifact: the work tree the packer packs

GitHub: #467 (spec #449). Owner of this document: the packer family `qwen3.8-27b-nvfp4`
(`crates/artifact`: `ignis_artifact::nvfp4_27b` holds the inventory, `ignis_artifact::packer` the
family). Writer of the work tree: the NVFP4 converter (#468). Readers of the sidecar: the loader's
checksum verification (`ignis_artifact::checksum`) and the 16 GB converter (`tools/27b-16gb-converter`,
#465). Why the container is what it is: ADR 0002; why a fine-tune keeps `model_id` `qwen3.8-27b`:
ADR 0074 and spec #449's "Identity".

An NVFP4 container of the 27B, the base's or a fine-tune's, has the served artifact's layout
(`qwen3_8_27b_nvfp4full-v2.ninfer`): the same 1,325 objects, names, formats, layouts and shapes, in
the same order. The converter writes **work files**; the packer checks them against the inventory and
assembles the container and its sidecar. Python never writes the container. Changing anything here
changes every artifact of the family.

All integers and floats are **little-endian**. Shapes are row-major, outermost first.

## 1. The run

```
ignis-artifact-pack --family qwen3.8-27b-nvfp4 --work <dir>/work \
    --out <dir>/<artifact>.ninfer --weights-id <weights id> [--geometry qwen3.8-27b|fixture] [--keep-work]
```

- **The identity** is `model_id` `qwen3.8-27b`, which decides the forward program and the drafter, and
  the `--weights-id` given. Both `--weights-id` and `--out` are required (exit 2 without): the content
  hash covers the identity and the directory, not the payload bytes, so the weights id is what tells
  one build of this layout from another (spec #449: Swift's is `nvfp4full-swift-1.5`), and the file
  name is the build's own (`qwen3_8_27b_nvfp4full-v2-ukisai-swift-1.5.ninfer`).
- **It packs only a whole tree.** Until `converter.json` and every unit's `DONE` exist it creates
  nothing and exits 3 (waiting). Then it checks the whole tree (§7) before it creates anything: a
  refusal (exit 1) leaves no container, no state and no sidecar, and consumes no work file.
- **It writes** `<out>`, `<out>.conversion.json` (§6), and while it runs `<out>.pack-state.json` and
  `<out>.pack.lock`. It deletes each unit's work files once the unit is in the container; `--keep-work`
  keeps them (refused without room for two copies). Interrupted, it resumes where it stopped.
- `--geometry fixture` packs the converter's reduced synthetic tree (§5); the default is the 27B's.

## 2. The tree

```
work/
  converter.json   the converter's record, merged into the sidecar (§6)
  frontend/        DONE and the six resource files
  text/            DONE, tensors.json, one file per tensor
  mtp/             the same
  vision/          the same
  dflash2/         the same
```

- **Completion.** A unit directory is complete when its `DONE` exists, written last:
  `{"files": {"<name>": {"bytes": n, "sha256": "<hex>"}}}` listing every other file of the directory
  (`docs/formats/flash-next-layout.md` §2; `tools/27b-16gb-converter/work.py`'s `mark_done` writes it).
  The packer checks each file's size before it appends a byte and its SHA-256 as it streams in.
- **`frontend/`** holds exactly the six files of §4.1 under their own names, the checkpoint's bytes.
  Any other file in its `DONE` is refused.
- **`tensors.json`** of a tensor unit: `{"tensors": [entry, ...]}`, one entry per object of the unit,
  in any order (the container takes the inventory's):

  ```json
  {"name": "text/layers/0/mlp/gate_up", "file": "layers.0.mlp.gate_up.bin",
   "format": "NVFP4", "layout": "blockscale-k16-m128x4-v1", "shape": [34816, 5120], "bytes": 89128964}
  ```

  - `name` is the container name (§4), starting with the unit and `/`.
  - `file` is the work file in the unit directory, listed in `DONE`. Any name works; the 16 GB
    converter's is the name past the unit with `/` as `.` and `.bin` appended.
  - `format`, `layout` and `shape` must be the inventory's exactly; `shape` is `[]` for the rank-0
    `*_input_scale_divisor` scalars.
  - `bytes` is the file's size, which must be what the layout stores for that shape (§3).
  - Other keys are ignored. Files a unit's `DONE` lists but `tensors.json` does not are the
    converter's own records and stay out of the container.
- **A work file's bytes are its container payload exactly.**

## 3. Encodings

| format | layout | payload of shape `[N, K]` (or the shape given) |
|---|---|---|
| `NVFP4` | `blockscale-k16-m128x4-v1` | `N % 128 == 0`, `K % 64 == 0`. The packed E2M1 code plane (`N·K/2` bytes), zeros to the next multiple of 256, the swizzled E4M3FN scale plane (`N·K/16` bytes, one scale per 16 elements), then the FP32 **weight divisor** (4 bytes, the payload's last). |
| `W8G32_F16S`, `Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S` | `row-split-k128-v1` | Rank 2, any `N`; `K` padded to a multiple of 128. The low plane, the high plane (empty for W8 and Q4) and the FP16 group scales, each plane starting at a multiple of 256. |
| `BF16`, `FP32`, `I32` | `contiguous-le-v1` | The elements in row-major order; rank 0 is one element. |
| resource | `raw-bytes-v1` | The file's bytes. |

- The sizes are the reader's: `ignis_artifact::tensor_encoded_size` (`block_scale_geometry`,
  `row_split_geometry` in `crates/artifact/src/lib.rs`).
- The encoders are ninfer's, which the converter ports: `tools/artifact/layouts.py` (`encode_nvfp4`,
  `encode_row_split`, `encode_direct`), and for NVFP4 parents `tools/convert/qwen3_8_27b/nvfp4_encode.py`
  (`NVFP4_MAXABS_DIVISOR_RNE_V1`: weight divisor `2688 / amax`, E4M3 group scales, round to nearest
  even). The served artifact is the byte-level reference for every one of them.

## 4. The inventory

`ignis_artifact::nvfp4_27b::inventory(geometry)` is the authority; this section states it. The
symbols are the geometry's widths (§5). 1,325 objects: 1,319 tensors and 6 resources.

| unit | objects | formats |
|---|---|---|
| `frontend` | 6 | resources |
| `text` | 908 | 2 W8, 1 Q4, 1 I32, 247 NVFP4, 247 FP32 divisors, 96 FP32, 314 BF16 |
| `mtp` | 12 | 5 W8, 7 BF16 |
| `vision` | 333 | 1 Q6, 54 Q4, 54 Q5, 2 W8, 222 BF16 |
| `dflash2` | 66 | 34 NVFP4, 32 BF16 |

### 4.1 `frontend/`

`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`,
`preprocessor_config.json`, `video_preprocessor_config.json`, in this order, each the resource
`frontend/<file>`.

### 4.2 `text/`

`text/token_embedding` W8 `[V, H]`, then the 64 layers, then `text/final_norm` BF16 `[H]`,
`text/output_head` W8 `[V, H]`, `text/draft_head` Q4 `[R, H]`, `text/draft_head_token_ids` I32 `[R]`.

Layer `i` is a **GQA layer** when `(i + 1) % 4 == 0` (3, 7, ..., 63) and a **GDN layer** otherwise.
Every object is `text/layers/{i}/<suffix>`, in the order listed. Each NVFP4 projection is followed by
its activation divisor, a rank-0 FP32 `<site>/input_scale_divisor`; a projection stored BF16 (the
**positional exceptions**) has none.

GDN layer (15 objects, 14 at layer 4):

| suffix | format | shape |
|---|---|---|
| `input_norm` | BF16 | `[H]` |
| `gdn/a_log`, `gdn/dt_bias` | FP32 | `[Gv]` |
| `gdn/convolution` | BF16 | `[Ck, 2·Gk·Gd + Gv·Gd]` |
| `gdn/a_b_projection` | BF16 | `[2·Gv, H]` |
| `gdn/query_key_value_z` | NVFP4 | `[2·Gk·Gd + 2·Gv·Gd, H]` |
| `gdn/input_projection/input_scale_divisor` | FP32 | `[]` |
| `gdn/norm` | BF16 | `[Gd]` |
| `gdn/output` | NVFP4, **BF16 at layer 4** | `[H, Gv·Gd]` |
| `gdn/output_projection/input_scale_divisor` | FP32 | `[]`, absent at layer 4 |
| `post_attention_norm` | BF16 | `[H]` |
| `mlp/gate_up` | NVFP4 | `[2·I, H]` |
| `mlp/gate_up_projection/input_scale_divisor` | FP32 | `[]` |
| `mlp/down` | NVFP4 | `[H, I]` |
| `mlp/down_projection/input_scale_divisor` | FP32 | `[]` |

GQA layer (12 objects; 11 at layers 11, 15, 19, 23; 10 at layers 3, 7):

| suffix | format | shape |
|---|---|---|
| `input_norm` | BF16 | `[H]` |
| `attention/query_key_gate_value` | NVFP4, **BF16 at layers 3, 7, 11, 15, 19, 23** | `[2·A·D + 2·Kv·D, H]` |
| `attention/input_projection/input_scale_divisor` | FP32 | `[]`, absent where BF16 |
| `attention/query_norm`, `attention/key_norm` | BF16 | `[D]` |
| `attention/output` | NVFP4, **BF16 at layers 3, 7** | `[H, A·D]` |
| `attention/output_projection/input_scale_divisor` | FP32 | `[]`, absent where BF16 |
| `post_attention_norm`, then `mlp/` as in a GDN layer | | |

So 247 NVFP4 parents (48 GDN layers × 3 + 47 + 16 × 2 + 10 + 14), each with its divisor, and 9 BF16
exceptions. The fused planes' rows are the served recipe's (ninfer
`tools/convert/qwen3_8_27b/recipe_nvfp4full.py`), which the packer cannot check:
`query_key_gate_value` is the query rows of `q_proj`, `k_proj`, the gate rows of `q_proj`, then
`v_proj` (`q_proj` interleaves query and gate per head); `query_key_value_z` is `in_proj_qkv` then
`in_proj_z`; `gate_up` is `gate_proj` then `up_proj`; `a_b_projection` is `in_proj_a` then
`in_proj_b`; `convolution` is `conv1d.weight` `[C, 1, Ck]` stored `[Ck, C]`; `a_log` and `dt_bias`
are cast to FP32.

### 4.3 `mtp/`

The checkpoint's MTP layer (ignis binds none of it; the container carries it as the served one does):

| name | format | shape |
|---|---|---|
| `mtp/input_projection` | W8 | `[H, 2·H]` |
| `mtp/embedding_norm`, `mtp/hidden_norm`, `mtp/layer/input_norm` | BF16 | `[H]` |
| `mtp/layer/attention/query_key_gate_value` | W8 | `[2·A·D + 2·Kv·D, H]` |
| `mtp/layer/attention/query_norm`, `mtp/layer/attention/key_norm` | BF16 | `[D]` |
| `mtp/layer/attention/output` | W8 | `[H, A·D]` |
| `mtp/layer/post_attention_norm` | BF16 | `[H]` |
| `mtp/layer/mlp/gate_up` | W8 | `[2·I, H]` |
| `mtp/layer/mlp/down` | W8 | `[H, I]` |
| `mtp/final_norm` | BF16 | `[H]` |

### 4.4 `vision/`

`vision/patch_embedding` Q6 `[vH, vP]`, `vision/patch_embedding_bias` BF16 `[vH]`,
`vision/position_embedding` BF16 `[vN, vH]`; then 27 blocks `vision/layers/{b}/` of twelve:

| suffix | format | shape |
|---|---|---|
| `attention/qkv` | Q4 | `[3·vH, vH]` |
| `attention/qkv_bias` | BF16 | `[3·vH]` |
| `attention/output` | Q5 | `[vH, vH]` |
| `attention/output_bias` | BF16 | `[vH]` |
| `mlp/fc1` | Q4 | `[vI, vH]` |
| `mlp/fc1_bias` | BF16 | `[vI]` |
| `mlp/fc2` | Q5 | `[vH, vI]` |
| `mlp/fc2_bias` | BF16 | `[vH]` |
| `norm1/weight`, `norm1/bias`, `norm2/weight`, `norm2/bias` | BF16 | `[vH]` |

then the 2 × 2 merger: `vision/merger/fc1` W8 `[4·vH, 4·vH]`, `vision/merger/fc1_bias` BF16 `[4·vH]`,
`vision/merger/fc2` W8 `[H, 4·vH]`, `vision/merger/fc2_bias` BF16 `[H]`, `vision/merger/norm/weight`
and `vision/merger/norm/bias` BF16 `[vH]`.

### 4.5 `dflash2/`

The drafter, weight-only: its NVFP4 matrices have **no** divisor objects. `dflash2/feature_projection`
NVFP4 `[H, dT·H]`, `dflash2/context_norm` BF16 `[H]`; then 5 layers `dflash2/layers/{l}/` of twelve:

| suffix | format | shape |
|---|---|---|
| `input_norm` | BF16 | `[H]` |
| `attention/query_key_value` | NVFP4 | `[(dA + 2·dKv)·dD, H]` |
| `attention/query_norm`, `attention/key_norm` | BF16 | `[dD]` |
| `attention/output` | NVFP4 | `[H, dA·dD]` |
| `attention/conv_base` | BF16 | `[2, 2, H]` |
| `attention/conv_proj` | NVFP4 | `[dC, H]` |
| `post_attention_norm` | BF16 | `[H]` |
| `mlp/gate_up` | NVFP4 | `[2·I, H]` |
| `mlp/down` | NVFP4 | `[H, I]` |
| `mlp/conv_base` | BF16 | `[2, 2, H]` |
| `mlp/conv_proj` | NVFP4 | `[dC, H]` |

then `dflash2/final_norm` BF16 `[H]`, `dflash2/selector/hidden` NVFP4 `[dS, H]`,
`dflash2/selector/predecessor` and `dflash2/selector/successor` NVFP4 `[V, dS]`.

## 5. The geometries

Both have every object of §4, with the same names, formats, layouts and order: only the widths differ.
`Geometry::qwen3_8_27b()` is the 27B's (the default); `Geometry::fixture()` (`--geometry fixture`) is the
NVFP4 converter's reduced synthetic checkpoint, which that converter's tests build to these widths. Every
fixture NVFP4 matrix holds whole 128 × 64 blocks, and the fixture's tree is about 5 MB.

| symbol | `Geometry` field | 27B | fixture |
|---|---|---|---|
| `H` | `hidden` | 5,120 | 128 |
| `V` | `vocab` | 248,320 | 256 |
| `I` | `intermediate` | 17,408 | 64 |
| `Gk` | `gdn_key_heads` | 16 | 1 |
| `Gv` | `gdn_value_heads` | 48 | 3 |
| `Gd` | `gdn_head_dim` | 128 | 64 |
| `Ck` | `gdn_conv_kernel` | 4 | 4 |
| `A` | `attention_heads` | 24 | 4 |
| `Kv` | `kv_heads` | 4 | 2 |
| `D` | `head_dim` | 256 | 32 |
| `R` | `draft_head_rows` | 131,072 | 128 |
| `vH` | `vision_hidden` | 1,152 | 32 |
| `vI` | `vision_intermediate` | 4,304 | 64 |
| `vP` | `vision_patch` | 1,536 (3·2·16·16) | 96 (3·2·4·4) |
| `vN` | `vision_positions` | 2,304 (48²) | 16 (4²) |
| `dA` | `dflash2_heads` | 32 | 2 |
| `dKv` | `dflash2_kv_heads` | 8 | 1 |
| `dD` | `dflash2_head_dim` | 128 | 32 |
| `dT` | `dflash2_taps` | 5 | 5 |
| `dC` | `dflash2_conv_rows` | 1,280 | 128 |
| `dS` | `dflash2_selector_rank` | 256 | 128 |

The layer counts (64 text, 27 vision, 5 drafter) and the exception layers are the 27B's at both.

## 6. `converter.json` and the sidecar

`converter.json` is the converter's half of the sidecar. The packer requires these fields and
refuses the tree, naming the field, without them:

```jsonc
{
  "schema": "qwen3.8-27b-nvfp4-converter-v1",    // carried, not checked
  "status": "complete",                            // only "complete" packs, unless --accept-status
  "recipe_id": "qwen3_8_27b_nvfp4full-v2-local",   // the sidecar's recipe id, never the file name's
  "identity": {"model_id": "qwen3.8-27b", "weights_id": "nvfp4full-swift-1.5"},  // must be the run's
  "source": {
    "repo": "ukisai/Swift-1.5-Qwen3.8-27b",        // the BF16 checkpoint's hub repo
    "revision": "b4c84d42903a8646b25857eb2827288d92ed85a4",
    "license": "swift-open-license-1.0"            // the license's identifier, as the catalog names it
  },
  "local_nvfp4": {
    "encoder_profile": "NVFP4_MAXABS_DIVISOR_RNE_V1",
    "parents": {
      "text/layers/0/gdn/query_key_value_z": {"weight_scale_divisor": 185.37930297851562,
                                              "relative_frobenius_error": 0.0949},
      "dflash2/feature_projection": {"weight_scale_divisor": null, "relative_frobenius_error": 0.0949}
    }
  }
}
```

- **`local_nvfp4.parents`** records exactly the 281 NVFP4 objects:
  - each of the 247 `text/` parents with `weight_scale_divisor`, the FP32 weight divisor its work
    file ends with (§3), written as the JSON number of that FP32 value widened to f64 (in Python,
    `json.dump` of `float(numpy.float32(d))` or of `struct.unpack("<f", word)[0]`);
  - each of the 34 `dflash2/` parents with `weight_scale_divisor` `null` or absent (weight-only).
  - Any other per-parent key, such as `relative_frobenius_error`, is carried and not checked.
- Everything else in `converter.json` is carried verbatim: the calibration's provenance, the
  drafter's and the ranking's pins, the converter's commit and versions. It must hold no local path,
  since the sidecar is published beside the artifact.

**The sidecar** `<out>.conversion.json` is `converter.json` plus what only the packer knows:

| field | from | read by |
|---|---|---|
| `recipe_id` | `converter.json` | the loader (`Sidecar::load`) |
| `artifact.bytes` | the packer: the container's size | the loader's checksum verification |
| `objects.count` | the packer: 1,325 | the loader's checksum verification |
| `local_nvfp4.parents.<name>.weight_scale_divisor` | `converter.json`, checked against the work files | the loader's checksum verification: each recorded divisor must equal the container's trailing FP32 word, a `null` one is a presence check |
| `identity`, `source.repo`, `source.revision`, `source.license` | `converter.json` | the 16 GB converter (#465), the hub card |
| `local_nvfp4` (the `dflash2/` records) | `converter.json` | the 16 GB converter, which copies the drafter's records into its own sidecar |

The loader finds the sidecar beside the artifact as `<artifact>.conversion.json` (a served artifact
may carry a `.graft.json` instead) and refuses a load whose verification is not clean.

## 7. Refusals

Each names what failed and leaves nothing written:

- a `converter.json` whose status is not packed, or without `recipe_id`, `source.repo`,
  `source.revision`, `source.license` or `local_nvfp4.parents`, or whose `identity` is not the run's;
- a unit with no `tensors.json`, an object the inventory gives the unit and the tree does not hold
  (`the work tree has no <name>`), one whose format, layout or shape is not the inventory's (both are
  quoted), an object the inventory does not give the unit, one listed twice, and a `frontend/` file
  outside §4.1;
- a work file absent from `DONE`, or whose size is not `DONE`'s, its entry's `bytes` or what its layout
  stores;
- a text NVFP4 parent that `local_nvfp4.parents` does not record or records without a divisor, or
  whose recorded divisor is not the word its file ends with; a drafter parent recorded with a
  divisor; a recorded parent that is not an NVFP4 object of the inventory.

While appending, a file whose SHA-256 is not `DONE`'s stops the run before it is deleted.

The tests are `crates/artifact/tests/nvfp4_27b.rs`, on the fixture's tree; it also checks the
inventory against the served artifact's directory when that file is on the machine.
