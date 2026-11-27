# Image converter

Spec 01 (GitHub #318): the offline tool that
turns `Qwen/Qwen-Image-2.1-Turbo` at revision `fc13ba1986fcb905854dc8de2f52eee395f49b5b` into the
ignis image artifact, a `.ninfer` v2 container of family `Qwen-Image-2.1` (`model_id`
`ignis-image-2.1-turbo`). CPU only. Python writes **work files**; the Rust packer
(`ignis-artifact-pack --family qwen-image`, `crates/artifact`) assembles the container from them,
as for Flash-Next (`docs/formats/flash-next-layout.md` §2 and §6 are the byte contract). The binder
is `crates/artifact/src/qwen_image.rs`.

## Files

| File | What it is |
|---|---|
| `convert.py` | The command: `run` (check, convert, pack) and `fixture` (a synthetic checkpoint and its work tree, for `crates/artifact/tests/qwen_image.rs`). |
| `pin.py`, `pin.json` | The accepted checkpoint: the revision, every file the converter reads with its size and sha256 (a safetensors file's is its LFS oid), every tensor with its dtype and shape. `check` refuses any difference, naming it. |
| `recipe.py` | The tensor map: spec 01's encoding table, each checkpoint tensor's object name, encoding and stored shape. A tensor it does not name is refused. |
| `fp8.py` | FP8 E4M3FN with a bf16 scale per row (`row-scale-v1`): a copy of `tools/flash-next-converter/fp8.py`, plus the round-trip error. |
| `rowsplit.py` | Q4/Q5/Q6 G64 and W8 G32 (`row-split-k128-v1`), the 27B vision tower's formats (this tower uses W8 only): adapted from the reference converter (gpillon/ninfer `a00648c`, Apache-2.0), bit-identical to it. |
| `work.py` | The DONE protocol, copied from `tools/flash-next-converter/layout.py`. |
| `metadata.py` | `frontend/image_model.json` (sampler sigmas, scheduler, image processor, latent normalization, the prompt templates verbatim from diffusers 0.41.0) and the VAE's convolution shapes. |
| `synthetic.py` | The synthetic checkpoint: 2 DiT blocks, 2 text layers, a 3-block vision tower, a small VAE, from diffusers' and transformers' own modules. |
| `fp8_bound.py` | The measurement behind `--fp8-bound`'s default (below). |
| `record.py` | The diffusers recordings (spec 01, AC 5-6): the fp32 oracle of three cases (`run`); the bf16 and simulated W8A8 yardsticks and `thresholds.json`'s bound (`yardstick`, below). |
| `record_encoder_input.py` | Spec 03's CPU fixture (`crates/server/tests/fixtures/image_encoder_input.json`): the pipeline's tokens, `drop_idx`, mrope positions and `pixel_values` on the pinned processor, with a tiny text encoder. |
| `fixtures/edit-condition.png` | The edit case's RGBA condition image, procedural (`record.edit_image(1024)`): shapes on an alpha ramp, alpha 0, partial and 255. |
| `vision_probe.py` | An informational probe: the vision tower's outputs with the artifact's W8 weights and in bf16, against fp32. |
| `ref_layers.py` | Spec 03's per-layer reference: transformers' text model in fp32 on the CPU, layer by layer, on a recorded case (`layer00.npy` .. `layer35.npy`); `IGNIS_IMAGE_ENCODER_LAYERS` names its output for the GPU test's teacher-forced diagnosis. |
| `ref_vision.py` | Spec 03's per-stage reference: transformers' vision tower in fp32 on the edit case, with the checkpoint's weights and with the artifact's W8 weights decoded (through `vision_probe`); `IGNIS_IMAGE_VISION_REF` names its output. |
| `requirements.txt` | The venv `F:/ai/venvs/ignis-image` (torch and torchvision cu130, diffusers 0.41.0, transformers 5.17.0); the recorder uses it too. |

## What is stored

| part | object names | encoding |
|---|---|---|
| DiT block linears, `img_in`, `txt_in`'s two linears, `proj_out` | `transformer/<checkpoint name>` | FP8 E4M3FN, a bf16 scale per output row |
| text encoder decoder-layer linears (36 layers) | `text_encoder/layers.{l}...` | FP8, as above |
| DiT modulation, timestep embedder, `norm_out.linear` (`norm_out`'s modulation), every norm | as above | bf16, the checkpoint's bits |
| token embedding | `text_encoder/embed_tokens.weight` | bf16; the binder keeps it on the host |
| vision tower, merger, 3 DeepStack mergers | the 27B's (`vision/layers/{b}/attention/qkv`, ...; `vision/deepstack/{i}/...`) | W8 G32 for every matrix (the patch embedding as `[1152, 1536]`, qkv/output/fc1/fc2, the mergers), bf16 biases, norms, positions |
| VAE, `quant_conv`, `post_quant_conv` | `vae/<checkpoint name>` | bf16, the checkpoint's shapes |
| frontend | `frontend/<file>` | the tokenizer, chat template, the checkpoint's configs, `image_model.json` |

Not stored: `lm_head` and `model.language_model.norm` (the encoder's output is taken before it).

## The record

`work/converter.json`, merged by the packer into `<artifact>.conversion.json`: every stored
tensor (source name, object, encoding, format, shape, bytes, sha256, max |value|, round-trip
relative error for FP8 and the grouped formats), the tensors not stored, the VAE's convolutions
(kernel, stride, padding as `[left, right, top, bottom]`, channels, the 2x upsampling before the
decoder's upsampler convolutions), the source revision and file hashes, this tool's commit and the
library versions.

## The FP8 bound

The conversion refuses an FP8 tensor whose round-trip relative error,
`||W - decode(encode(W))||_F / ||W||_F` with the decode in fp32 (`e4m3fn(code) * bf16(scale)`), is
above `--fp8-bound`. The default, **0.03**, comes from the 27B's FP8 linears: `fp8_bound.py`
fetched ten of them (`unsloth/Qwen3.8-27B-NVFP4@7d6f8d4`, the source of the 27B artifact's FP8
words, against the BF16 `Qwen/Qwen3.8-27B@1d4bf0f`: q/k/v/o of layer 3, GDN in/out of layer 0, MLP
gate/up/down of layer 60) and measured 2.659-2.663% (max 2.6626e-02); this encoder gives
2.643-2.648% on the same weights. 3% is that maximum rounded up.

## Tests

```
F:/ai/venvs/ignis-image/Scripts/python.exe -m pytest tools/image-converter -q
```

CPU only. A test that needs a file from the owner's layout skips without it; each path has an
environment variable:
- the pinned checkpoint's processor, `IGNIS_IMAGE_SNAPSHOT` (`test_record.py` runs the recorder on
  a tiny checkpoint built around it);
- the reference converter, `IGNIS_NINFER_SOURCE` (`F:/ai/q38/ninfer`);
- the 27B artifact, `IGNIS_ARTIFACT_27B`.

The simulated-W8A8 tests pack a tiny artifact with the release packer and skip until it is built
(`cargo build --release -p ignis-artifact --bin ignis-artifact-pack`). The Rust side:
`cargo test -p ignis-artifact --test qwen_image`.

The commands' default paths (`--out`, `--oracle`, `--artifact`) are the owner's layout, under
`F:/ai/models/ignis-image-2.1-turbo/`.

## The recordings

```
HF_HUB_CACHE=F:/ai/models/hf F:/ai/venvs/ignis-image/Scripts/python.exe tools/image-converter/record.py run \
    --checkpoint F:/ai/models/hf/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba1986fcb905854dc8de2f52eee395f49b5b
```

One GPU session, under the shared GPU lock, writes `F:/ai/models/ignis-image-2.1-turbo/oracle/`:
- a directory per case (`t2i` 1024², `t2i-2k` 2048², `edit` 1024² with the checked-in image): the
  fields of spec 01's list as `.npy`, indexed with shapes and dtypes in `case.json`, the decoded
  `image.png` (`record.FIELDS` describes each);
- a bf16-only `thresholds.json` (schema v1): per case and step, the bf16 pipeline's teacher-forced
  velocity error against fp32 and 5 × it, and the encoder's (`--factor`), which the yardstick below
  supersedes. Written once: the recorder refuses an existing one;
- `manifest.json`, last: the versions, the TF32 flags, the attention check, wall times, peaks.

The fp32 arithmetic is diffusers' own at about half the memory: the weights stay bf16 at rest (the
checkpoint's) and the token embedding, the text layers, `lm_head` and the DiT blocks are upcast to
fp32 in their own forward; everything else is fp32. Each call is split at `encode_prompt`: every
prompt runs first with the encoder on the GPU, then the encoder is dropped and the DiT phase replays
them up to the final latents; then the DiT is dropped and the pipeline's own decode tail runs with
the VAE alone, on the host from `--host-decode-from` (2048) pixels a side, where the fp32 decode
does not fit the GPU budget. The VAE keeps no single-frame causal cache. `test_record.py` shows all
of it bit-identical to `from_pretrained(torch_dtype=float32)` and one call per case. TF32 is off
(asserted) and the DiT's attention is restricted to the memory-efficient SDPA kernel, which raises
rather than fall back to a dense fp32 score matrix. The session needs ~22 GiB of VRAM and, for the
host decode, 40 GB of free host RAM (`--min-free-ram-gb`); it took 494 s.

## The yardsticks and the bound

```
HF_HUB_CACHE=F:/ai/models/hf F:/ai/venvs/ignis-image/Scripts/python.exe tools/image-converter/record.py yardstick \
    --checkpoint F:/ai/models/hf/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba1986fcb905854dc8de2f52eee395f49b5b \
    --replace-thresholds
```

One GPU session against the recorded oracle, which it never records again (the fp32 oracle does not
depend on the artifact). Both yardsticks are teacher-forced (step i's recorded fp32 input latents in),
and each one's DiT reads that pipeline's own encoder output and condition latents:
- **bf16**: diffusers in bf16, as the recorder's own;
- **sim**, simulated W8A8: the pipeline computing in fp32 on the artifact's stored values. Each of
  the 480 FP8 linears becomes a `_SimQuantLinear` of its payload, read from the container
  (`--artifact`) and checked against the sidecar's sha256 and encoding, and fake-quantizes its fp32
  input per token to E4M3 as #319's kernel does (`fake_quant_e4m3`). The W8 vision tower is decoded,
  and every bf16 tensor is checked against its record. Attention stays fp32.

It writes `thresholds.json` (schema v3). For each case it holds each step's velocity error, the
encoder's and the condition latents'. The encoder is measured over its text rows, not the image-pad
rows the DiT discards: each row's relative L2, then their RMS (`row_rel_l2`), because the DiT's
`txt_in` RMS-normalizes each row. With a condition image it is also measured over the text rows
after the vision span (`encoder_after_vision`), the only ones the image reaches. Each of these
quantities has both errors and the bound max(`--sim-factor` 2 × sim, `--bf16-factor` 5 × bf16).
The first row after `drop_idx` (`encoder_row0`) has both errors and no bound. The file also holds
each metric's definition, the factors, the oracle's session, this session, and the artifact's
identity, sha256 and sidecar sha256.

`thresholds.json` is superseded only before any ignis result exists. `--replace-thresholds`
supersedes the recorder's bf16-only file (v1) and the first bound (v2); a later bound takes
`--supersede-bound` as well. The old file is kept as `thresholds.<its session id>.json`. The new one
is written under a temporary name and renamed last, and only if the velocity errors did not move
against the old file's: the bf16's, and the sim's on the same artifact.

`yardstick-report.json` beside it holds:
- both yardsticks' errors, with the encoder's whole-tensor error;
- the encoder error's breakdown, for each yardstick and for edit with DeepStack off: text against
  image-pad rows, each text row's relative L2, and the top rows by reference norm and by error;
- the fp32 encoder replayed against the oracle (the session's wiring check, expected 0);
- the edit encoder with DeepStack off (spec 03 AC 4's premise) under both encoder metrics, beside
  their bounds;
- the velocity errors against the superseded file's;
- the wall times and the peaks.

## The conversion

```
F:/ai/venvs/ignis-image/Scripts/python.exe tools/image-converter/convert.py run \
    --checkpoint F:/ai/models/hf/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba1986fcb905854dc8de2f52eee395f49b5b \
    --out F:/ai/models/ignis-image-2.1-turbo
```

Download the checkpoint first with `HF_HUB_CACHE=F:/ai/models/hf` (32.5 GB; the default cache is
on C:), and build the packer (`cargo build --release -p ignis-artifact --bin ignis-artifact-pack`).
The work tree (~17 GB) is deleted unit by unit as the packer appends it: the disk peak is the
checkpoint, the work tree and the unit being appended (~58 GB with the checkpoint). The output is `ignis_image_2_1_turbo-v2.ninfer` and its
`.conversion.json` in `--out`.
