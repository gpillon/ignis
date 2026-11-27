# 27B 16 GB converter

Spec 01 of the 27B on 16 GB cards (GitHub #338): the
offline tool that turns the public EXL3 quant `turboderp/Qwen3.8-27B-exl3`, branch `SC_3.00bpw_H4`, and
the served artifact `qwen3_8_27b_nvfp4full-v2.ninfer` into the packer's **work tree** for the 16 GB
artifact `qwen3_8_27b_trellis_sc30-v2.ninfer` (identity `qwen3.8-27b` / `trellis-sc30`). The Rust packer
(`ignis-artifact-pack --family qwen3.8-27b-trellis`, `crates/artifact`) assembles the container from
it. Python never writes the container.

- **The body and the output head** are the quant's trellis records, bit for bit: a record is the
  `trellis`, `suh` and `svh` tensors as stored, zero-padded to 4096 (`trellis-tile16-v1`,
  `docs/formats/flash-next-layout.md` §3). Nothing is decoded or re-encoded.
- **Everything else** is the served artifact's object, byte for byte: the token embedding (W8), the
  norms and the GDN small set, the draft head, the MTP layer, the vision tower, the 66 DFlash2 objects,
  the frontend resources. The NVFP4 planes' `*_input_scale_divisor` scalars go with the planes.
- **Names.** A fused NVFP4 plane becomes the quant's split matrices under their exllamav3 names
  (`gdn/in_proj_qkv` + `gdn/in_proj_z`, `attention/q_proj` / `k_proj` / `v_proj`, `mlp/gate_proj` +
  `mlp/up_proj`); a plane the quant holds whole keeps its served name (`gdn/output`,
  `attention/output`, `mlp/down`, `text/output_head`). The table is `plan.py`; the Rust side's is
  `ignis_artifact::text_scope_27b_trellis`.

## Files

| File | What it is |
|---|---|
| `convert.py` | The command: `run` (pins, reconstruction errors, the work tree, `converter.json`) and `verify` (a packed artifact against the plan and its sources). |
| `plan.py` | The object table: the served directory, in its order, with the body's planes and the head replaced. |
| `exl3.py` | The quant's matrices from its safetensors headers (no torch): K, shapes, the `mul1` flag, the record bytes. |
| `container.py` | Read-only access to a `.ninfer` v2 container (the served artifact; the packed one for `verify`). |
| `override.py` | An override directory: EXL3 records that replace the quant's, checked like for like, pinned by the replay's provenance and its partial BF16 reference. |
| `overlay.py` | A BF16 checkpoint with a `replay.py fetch` directory laid over it: which file holds each tensor, the overlay's checked like for like and pinned by `fetch.json`. The KLD reference of the abliterated BF16 reads through it (GitHub #422). |
| `reconstruction.py` | The conversion's GPU step and its only exllamav3 user: each trellis matrix's relative reconstruction error against the BF16 checkpoint. |
| `replay.py` | The abliterated variant's 70 matrices re-quantized by replaying exllamav3's conversion, from the downloaded quant or one converted locally at a named recipe: the plan (no torch), `fetch` of huihui's changed tensors by HTTP range, `run` (the GPU step). |
| `recipe.py` | A named recipe (GitHub #631): `load` reads and checks `recipes/<name>.json`; the text scope's bytes and the bits per weight; a local conversion's per-tensor recipe (`job`) and exllamav3's YAML. |
| `recipes/` | The recipes, one JSON file each, named by its stem: `sc30.json` is the 16 GB artifact's and the default; `tandem25.json` is the public 2.5-bit quant's (`0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-2.5`, a measurement artifact that does not ship); `ig28a.json` is the 12 GB artifact's (GitHub #633), from exllamav3's allocation optimizer, its `about` says how, published as weights id `trellis-ig28` ("The 12 GB artifact" below); `ig28b.json` the second candidate (GitHub #634), from a public GGUF file's allocation (`gguf_recipe.py`), not admitted. |
| `gguf_recipe.py` | A recipe translated from a public GGUF file's per-tensor allocation (GitHub #634): `header` reads the file's header by HTTP ranges (no weights), `recipe` translates its types to integer K under a ceiling. |
| `local_quant.py` | An EXL3 quant of the 27B or a fine-tune converted on this machine by exllamav3 at a named recipe: `run` (the GPU step, its check, the pins and the provenance) and `control` (each matrix's error against the fine-tune's BF16 beside the default artifact's). |
| `work.py` | The DONE protocol the packer checks (copied from `tools/image-converter/work.py`). |
| `test_*.py`, `conftest.py` | Unit tests (CPU only) on a reduced synthetic served artifact and quant. |

## Tests

```
F:/ai/ngram-venv/Scripts/python.exe -m pytest tools/27b-16gb-converter -q
```

The packer's side is `crates/artifact/tests/trellis_27b.rs`; it also binds the real artifact when it is
on this machine.

`test_reconstruction.py` runs the reconstruction-error step on the CPU, with stub torch, exllamav3
and safetensors modules: an override's matrix is read from its record and measured against its
reference BF16 and every other from the quant against the base, one shard of each source is open at
a time, and the CUDA cache is emptied before each matrix.

## The conversion

The reconstruction-error step runs on the GPU (`make gpu-status`, the shared lock
`../.inference-qwen-worktrees/.swarm/gpu-lock.sh`). The quant and the BF16 checkpoint read fastest from
H: (`robocopy /E` of the quant keeps `.cache/huggingface/download`, which pins its revision).

```
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/convert.py run \
    --served F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer \
    --quant H:/models/qwen38-27b-exl3-SC_3.00bpw_H4 --bf16 H:/models/Qwen3.8-27B \
    --work F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/work
cargo build --release -p ignis-artifact --bin ignis-artifact-pack
target/x86_64-pc-windows-msvc/release/ignis-artifact-pack.exe --family qwen3.8-27b-trellis \
    --work F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/work
F:/ai/ngram-venv/Scripts/python.exe tools/27b-16gb-converter/convert.py verify \
    --artifact F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer \
    --served F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer \
    --quant H:/models/qwen38-27b-exl3-SC_3.00bpw_H4 --report <report.json>
```

- **`run`** refuses (exit 2, nothing written) a non-empty `--work`, a quant that is not its recipe's
  ("The recipe" below: by default the 16 GB artifact's, whose `quantization_config.json` says exl3, mul1,
  out scales always, 3.0 bits, head 4), a shard whose
  sha256 is not its LFS oid (the BF16 checkpoint's too), files from two revisions or with no download
  record (a quant converted locally is pinned instead: "A fine-tune" below), a quant matrix with no served plane or a plane
  with no quant matrix, split parts that do not tile their plane, and any matrix whose reconstruction
  error is above `--error-bound` (0.6: a layout error, never a quantization one). `--skip-errors` skips
  the GPU step and writes a `dry-run` tree, which the packer takes only with `--accept-status dry-run`.
- **The packer** consumes the work files as it appends them (`--keep-work` keeps them, given room for
  two copies) and writes `<artifact>.conversion.json`: `converter.json` plus the whole-file invariants.
- **`verify`** prints VERIFY PASS or VERIFY FAILED (exit 3).

## The recipe

`convert.py run` and `local_quant.py run` take a **recipe** by name, `--recipe <name>` (GitHub #631, spec
#630). With none they take `sc30`, the 16 GB artifact's, so the commands above run as they always have.

- **Where.** A recipe is `recipes/<name>.json` beside the tool, committed; its name is the file's stem.
  To write one, copy `sc30.json` under the new name and change what differs.
- **What it states** (schema `qwen3.8-27b-trellis-recipe-v1`):
  - `tensors`: the K of every body matrix, by its exllamav3 key (`model.language_model.layers.<i>.<module>`),
    the 400 of the 27B. The head is not listed: its bits are `quantization_config.head_bits`.
  - `quantization_config`: the fields the quant's `quantization_config.json` must state, compared as they
    are and recorded in `source.quantization_config`. `sc30`'s are exl3, mul1, out scales always, bits 3.0,
    head 4; a downloaded quant states its own (an `out_scales` of `auto`, bits 2.5). A local conversion
    passes exllamav3 its `bits` (`target_bpw`), `codebook`, `out_scales` and `head_bits`.
  - `ceiling_bytes`: the most the text scope's device weights may be. They are counted as the container
    stores them: every trellis record of the body and the head, at the recipe's K, and every other
    `text/` object but the token embedding (host memory) and the draft head and its token ids (not bound
    with the text scope). `sc30` counts 9,824,608,256, which the shipped artifact holds; the load's plan
    line says 9,825,095,804, its arena's alignment beside them. `sc30`'s ceiling is 9,830,000,000.
  - `files`: the pins of every file the recipe's side reads, `{"<file name>": {"bytes": n, "sha256":
    "<hex>"}}`: the downloaded quant's files the converter reads (its shards, index,
    `quantization_config.json` and `config.json`) and the calibration files the local quant reads
    (`quantization_config.json` and `cal_trace.safetensors`). A downloaded quant's pins are its LFS oids for
    the shards and the sha256 of each small file as downloaded; `sc30`'s are the shipped artifact's record's
    `source.files` and the trace's LFS oid. The rest of a conversion's inputs keep their own pins: the BF16
    checkpoint its download records, the served artifact its sha256 in the record, an override its
    `replay.json`, and a quant converted locally its `--quant-pins`, which stand for the recipe's pins of
    the quant's files.
  - `weights_id`: the artifact's, which `--weights-id` overrides (a fine-tune's, the abliterated variant's).
  - `job` (a recipe this machine converts): exllamav3's MTP bits and calibration size, `{"mtp_bits": 3,
    "calibration": {"rows": 250, "cols": 2048}}`. A recipe without it names a downloaded quant;
    `local_quant.py run` refuses it.
  - `about`: free text, for the reader.
- **Refused before the GPU** (exit 2, `REFUSED:` naming the cause): a K outside 2 to 5, a fractional K, a
  head not at 4 bits, a field the schema does not have or lacks; a quant whose configuration is not the
  recipe's, or with a matrix at another K, or one the recipe does not name or names and the quant lacks;
  text-scope device bytes above the ceiling; a file the conversion reads with no pin, of another size, or
  whose sha256 is not its pin's (the quant's files are hashed before the GPU step, the BF16 checkpoint's
  beside it).
- **The record** names it: `recipe` in `converter.json` (and in the local quant's `provenance.recipe.named`)
  is its name, the sha256 of its file, its ceiling, the text scope's bytes and the body's **bits per
  weight**, its average K weighted by each matrix's weights, the head apart as exllamav3's `bits` counts
  it (`sc30`: 2.9998). Nothing of it reaches the container: the 16 GB artifact converted through a recipe
  is the bytes it was.

### A recipe from a GGUF allocation

`gguf_recipe.py` (GitHub #634) makes `ig28b`, the 12 GB class's second candidate, not admitted, from the per-tensor
types of the public dynamic 2-bit GGUF file of the 27B, read from its header and never its weights:

```
F:/ai/ngram-venv/Scripts/python.exe tools/27b-16gb-converter/gguf_recipe.py header     --repo unsloth/Qwen3.8-27B-GGUF --revision 4af51b3542f775b0a8e4b310bfb3f17e01c306c9     --file Qwen3.8-27B-UD-Q2_K_XL.gguf --out <table.json>
F:/ai/ngram-venv/Scripts/python.exe tools/27b-16gb-converter/gguf_recipe.py recipe --table <table.json>     --record F:/ai/q38/ninfer-models/qwen3_8_27b_trellis_sc30-v2.ninfer.conversion.json     --name ig28b --weights-id trellis-ig28b --ceiling 9200000000
```

- **`header`** writes the file's size, LFS oid and revision and every tensor's name, dims and ggml type, reading 1 MiB
  ranges until the header ends.
- **`recipe`** gives each matrix the integer K nearest its type's bits per weight (block scales included) plus one
  shift for the whole body, 2 to 5, the head at 4; the shift is the largest whose text scope (counted as above, from the
  record's shapes and copied objects) is at or under `--ceiling`, and the class that would cross next is promoted by
  its layer's mean bits, then its key, while it fits. The recipe takes the `--base` recipe's configuration (`bits`
  becomes exllamav3's, the body's average K to two decimals), job and calibration pins. It refuses a matrix the header
  lacks or holds at other dims, a ceiling below every matrix at K2, and a recipe name that exists. The rule and its
  choices for `ig28b` are in `docs/findings/2026-11-15-the-12-gb-card-class-online-research.md`.

## The override

`run` and `verify` take an **override**, the directory `replay.py run` writes: EXL3 records in the quant's
layout (index, shards, `quantization_config.json`) that replace the quant's records of the same name, and
`replay.json`, their provenance. It serves a variant made from another BF16 (the uncensored 16 GB artifact,
from the abliterated BF16). Every other object is built exactly as without it, and without `--override`
the work tree and `converter.json` are as before.

```
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/convert.py run ... --override H:/override-386 --override-bf16 H:/abliterated-386 --weights-id trellis-sc30-abliterated
F:/ai/ngram-venv/Scripts/python.exe tools/27b-16gb-converter/convert.py verify ... --override H:/override-386 --weights-id trellis-sc30-abliterated
```

- **The provenance is `replay.json`.** The override's source repository and revision come from it
  (`sources.abliterated`), with the quant it was made against and each record's sha256. The quant is
  named by its revision (`sources.quant.revision`) when it was downloaded, and by its pins file
  (`sources.quant.pins`: name, bytes, sha256) when it was converted locally ("The abliterated replay"
  below; GitHub #703); a record written before #703 names a downloaded quant and is taken as it is. `run`
  refuses (exit 2), naming what is missing, an override with no `replay.json` or one that is not
  `complete` or lacks those fields, one naming its quant by neither a revision nor pins or by both, and a
  record whose sha256 is not the replay's. It refuses an override made against another quant: another
  revision for a downloaded quant; another pins file (name, bytes or sha256) than `--quant-pins` for a local
  one; and one made against a local quant converting a downloaded one, or the reverse. The override's own
  files (shards, index, config, `replay.json`) are pinned by bytes and sha256 computed at conversion time;
  there is no download record to read.
- **A swap is like for like.** `run` and `verify` refuse (exit 2 and VERIFY FAILED) an override record
  whose name is not one of the quant's text matrices, or whose K or shape differs from the quant's
  record, naming the record and what differs.
- **Its error is measured against its own BF16.** `--override-bf16` is the directory `replay.py fetch`
  wrote: partial shards holding only the changed tensors, an index and `fetch.json`. `run` reads each
  override matrix's reference from it, and the quant's matrices are measured against `--bf16` as today.
  It pins the reference by tensor, not by shard: `fetch.json` must be the replay's repository and
  revision, and each override matrix's `.weight` is hashed from its shard and must equal both
  `fetch.json`'s and `replay.json`'s sha256. This check needs no GPU and runs in a dry run too.
- **`--weights-id`** sets the identity the record and `verify` expect (default `trellis-sc30`); pass the
  packer the same `--weights-id` and the output name.
- **`verify`** compares each override record to the override's, bit for bit; it takes only `--override`.
  With `--baseline <default artifact>` it also reads the variant against the artifact it varies: the same
  directory (every object's name, kind, format, layout, shape, offset and size, in order) under another
  identity, so the two have different content hashes; every object's payload the baseline's bit for bit
  except the override's records, which must differ. The report's `baseline` counts the objects that are the
  same and names the ones that differ.

```
F:/ai/ngram-venv/Scripts/python.exe tools/27b-16gb-converter/convert.py verify ... --override H:/override-386 --weights-id trellis-sc30-abliterated --baseline F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer
```

## A fine-tune

`run` also builds the 16 GB artifact of a **fine-tune** of the 27B (GitHub #465, spec #449: Swift 1.5),
from the fine-tune's own NVFP4 artifact and an EXL3 quant converted on this machine at the per-matrix bit
widths of `SC_3.00bpw_H4`, rather than downloaded. Every object outside the body and the head is then the
fine-tune's, byte for byte. Each argument below defaults to the vanilla build: with none of them, the work
tree and `converter.json` are as before.

```
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/convert.py run \
    --served H:/models/<fine-tune NVFP4>/qwen3_8_27b_nvfp4full-v2-ukisai-swift-1.5.ninfer \
    --served-weights-id nvfp4full-swift-1.5 \
    --quant H:/models/<local quant> --quant-pins H:/models/<local quant pins>.json \
    --quant-repo <the local quant's name> --quant-branch <its recipe> \
    --bf16 H:/models/<fine-tune BF16> --bf16-repo ukisai/Swift-1.5-Qwen3.8-27b \
    --license swift-open-license-1.0 --weights-id trellis-sc30-swift-1.5 --work H:/<work>
F:/ai/ngram-venv/Scripts/python.exe tools/27b-16gb-converter/convert.py verify \
    --artifact H:/<out>/qwen3_8_27b_trellis_sc30-v2-ukisai-swift-1.5.ninfer \
    --served H:/models/<fine-tune NVFP4>/qwen3_8_27b_nvfp4full-v2-ukisai-swift-1.5.ninfer \
    --quant H:/models/<local quant> --weights-id trellis-sc30-swift-1.5 --report <report.json>
```

- **The served artifact.** `--served-weights-id` is its weights id (default `nvfp4full`); its model id is
  `qwen3.8-27b` always, and `run` refuses (exit 2) any other identity. Its sidecar is the one the loader
  reads: its `.graft.json` when it has one, else the `.conversion.json` the `qwen3.8-27b-nvfp4` packer
  family writes; `run` refuses an artifact with neither. Of a conversion record `run` reads only
  `recipe_id` and `local_nvfp4` (whose `parents` it keeps for the copied NVFP4 objects, the DFlash2
  module's); of a graft record also its provenance of the DFlash2 module (`grafted_from`, `source`,
  `objects`).
- **A locally converted quant** has no download record, so its revision cannot be read: `--quant-pins` pins
  it instead by its files' bytes and sha256, the pins its conversion recorded, as a JSON file:
  `{"files": {"<file name>": {"bytes": <n>, "sha256": "<hex>"}}}` (other keys are ignored, so the file can
  carry the conversion's provenance too). Every file the converter reads (the shards, the index,
  `quantization_config.json`, `config.json`) needs its pin. `run` refuses (exit 2), naming the file, one
  with no pin, of another size, or whose sha256 is not its pin's, before the GPU step. A local quant takes
  `--quant-repo` and `--quant-branch` too, so the record never names the downloaded quant's. It is checked
  against its recipe (`--recipe`, `sc30` by default) as the download is; its pins stand for the recipe's
  pins of the quant's files.
- **The labels** are the record's: `--license` (default `apache-2.0`), `--quant-repo` and `--quant-branch`
  (default `turboderp/Qwen3.8-27B-exl3`, `SC_3.00bpw_H4`) in `source`, `--bf16-repo` (default
  `Qwen/Qwen3.8-27B`) in `reference`. The BF16 checkpoint is pinned by its download record as before.
- **`--weights-id`** is the fine-tune's own 16 GB weights id, so its content hash is not the vanilla one's;
  pass the packer the same `--weights-id` and the output name.
- **`verify`** reads neither sidecar nor pins: `--served` is the fine-tune's NVFP4 artifact, whose objects
  the packed one's copies must be, and `--quant` the local quant.

## The fine-tune's local quant

`local_quant.py` makes that local quant (GitHub #471): exllamav3 1.5.3's own conversion of the fine-tune's
BF16 at the bit widths of `SC_3.00bpw_H4` (the recipe `sc30`), with that quant's own calibration trace. It
converts the base 27B's BF16 at a recipe of our own the same way, `--recipe <name>` (GitHub #631). Of the
quant only two files are fetched, at the revision the default 16 GB artifact's record names (its `source`):

```
F:/ai/ngram-venv/Scripts/hf.exe download turboderp/Qwen3.8-27B-exl3 quantization_config.json cal_trace.safetensors \
    --revision 86b95530f9d365d90dd24d01a997a8de0cdf6c55 --local-dir H:/swift-471/sc30
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/local_quant.py run \
    --bf16 H:/models/Swift-1.5-Qwen3.8-27b --bf16-repo ukisai/Swift-1.5-Qwen3.8-27b \
    --recipe-quant H:/swift-471/sc30 \
    --record F:/ai/q38/ninfer-models/qwen3_8_27b_trellis_sc30-v2.ninfer.conversion.json \
    --out H:/models/Swift-1.5-Qwen3.8-27b-exl3-sc30 --work H:/swift-471/work [--check]
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/local_quant.py control \
    --quant H:/models/Swift-1.5-Qwen3.8-27b-exl3-sc30 \
    --bf16 H:/models/Swift-1.5-Qwen3.8-27b --bf16-repo ukisai/Swift-1.5-Qwen3.8-27b \
    --record F:/ai/q38/ninfer-models/qwen3_8_27b_trellis_sc30-v2.ninfer.conversion.json \
    --out <main checkout>/.scratch/471/control.json
```

- **The recipe** is `--recipe` ("The recipe" above; `recipe.job`, no torch): every body matrix at its K, at
  the shape the record gives it; the head at `head_bits`, 4; the recipe's bits, codebook and output scales,
  and its `job`'s MTP bits and calibration size. `sc30` is `SC_3.00bpw_H4`'s: K2 34, K3 297, K4 66, K5 4
  over 401 matrices, the head included, bits 3.0, `mul1`, out scales always, MTP bits 3, calibration 250 x
  2,048. exllamav3's YAML (`<out>.recipe.yaml`) lists the 400 body matrices only: it takes the head from
  `head_bits` and would ignore an entry for it.
- **`run`** refuses (exit 2), before the GPU and writing nothing: a recipe the load cannot bind or with no
  `job`, or not of the record's matrices; the two files at another revision than the
  record's quant, a configuration whose sha256 is not the record's, a trace that is not its LFS oid, either
  without the recipe's pin or not its bytes and sha256, a BF16
  checkpoint without its download records or without a recipe matrix at its shape, text-scope device bytes
  at the recipe's K above its ceiling, a non-empty `--out` or
  `--work`, and a disk with less free space than the output, exllamav3's work files and three copies of
  its fp32 calibration state (54 GiB for the 27B; `--no-checkpoints` leaves the copies out, 25 GiB, and
  gives up `--resume`). `--check` stops there.
- **The conversion** is `exllamav3.conversion.convert_model`'s parser, `prepare` and `main`, run in this
  process, at `sc30`: `-hb 4 -mb 3 -vb 16 -cb mul1 --out_scales always -cd <trace> -cr 250 -cc 2048 --recipe`. The
  vision tower stays unquantized as in the quant (1.5.3 would quantize it at 6 bits). `--resume` resumes an
  interrupted job from `--work` (exllamav3's `-r`). The time and the peaks are logged and recorded: the
  CUDA allocator's (allocated, reserved) and the process's working set and commit.
- **The check.** Every matrix of the recipe at its K and shape (the head at 4), nothing else quantized but
  the MTP layer at its bits, the configuration `convert.py run` takes; a quant that is not the recipe's is
  refused naming the matrix, and gets no pins.
- **`<out>.pins.json`**, beside the quant: `files`, every file of the quant by bytes and sha256 (the
  `--quant-pins` above), and `provenance`: the BF16's repository, revision, index and shards (LFS oid),
  the quant's repository, branch and revision with its configuration's sha256 and the exllamav3 that made
  it, the calibration trace's sha256 and size, the recipe (`named`, the named recipe as the converter's
  record has it; its YAML's sha256, its formats, the YAML itself),
  exllamav3's version and arguments (paths by role), the run's seconds and peaks, the converter's commit
  and the versions. No local path.
- **`control`** (GPU) measures each matrix's relative reconstruction error against the fine-tune's BF16
  (`reconstruction.rel_errors`) beside the default 16 GB artifact's error on the same matrix against the
  base BF16 (its record's `trellis.matrices`), under the record's object names, and refuses a record that
  does not hold the quant's matrices at their K.

## The 12 GB artifact

The 12 GB class's artifact (spec #630) is the recipe `ig28a`, the candidate the gate admitted (GitHub #635),
published as weights id `trellis-ig28`: `qwen3_8_27b_trellis_ig28-v2.ninfer` at
`gpillon/Qwen3.8-27B-trellis-2.8bpw-ignis`, revision `15333a0829f055e7ee25d6d5ef84fd8d521518c6`, 12,754,663,684
bytes, sha256 `fab06bea49ce114da99ca69588f00b10fb5081c564f9ce60ecbef8b3dbf6d6a5`. The second candidate, `ig28b`, was
not admitted (its KLD is in `docs/findings/2026-11-15-the-12-gb-card-class-online-research.md`). The published file
comes from a local quant of the base 27B's BF16 at `ig28a`, then the conversion and the packer above:

```
F:/ai/ngram-venv/Scripts/hf.exe download turboderp/Qwen3.8-27B-exl3 quantization_config.json cal_trace.safetensors \
    --revision 86b95530f9d365d90dd24d01a997a8de0cdf6c55 --local-dir H:/630/sc30
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/local_quant.py run --recipe ig28a \
    --bf16 H:/models/Qwen3.8-27B --bf16-repo Qwen/Qwen3.8-27B --recipe-quant H:/630/sc30 \
    --record F:/ai/q38/ninfer-models/qwen3_8_27b_trellis_sc30-v2.ninfer.conversion.json \
    --out H:/models/Qwen3.8-27B-exl3-ig28a --work H:/630/quant-work --no-checkpoints
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/convert.py run --recipe ig28a --weights-id trellis-ig28 \
    --served F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer \
    --quant H:/models/Qwen3.8-27B-exl3-ig28a --quant-pins H:/models/Qwen3.8-27B-exl3-ig28a.pins.json \
    --quant-repo Qwen/Qwen3.8-27B --quant-branch exl3-ig28-local --bf16 H:/models/Qwen3.8-27B --work H:/630/work
target/x86_64-pc-windows-msvc/release/ignis-artifact-pack.exe --family qwen3.8-27b-trellis --work H:/630/work \
    --out F:/ai/models/Qwen3.8-27B-trellis-2.8bpw-ignis/qwen3_8_27b_trellis_ig28-v2.ninfer --weights-id trellis-ig28
F:/ai/ngram-venv/Scripts/python.exe tools/27b-16gb-converter/convert.py verify \
    --artifact F:/ai/models/Qwen3.8-27B-trellis-2.8bpw-ignis/qwen3_8_27b_trellis_ig28-v2.ninfer \
    --served F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer \
    --quant H:/models/Qwen3.8-27B-exl3-ig28a --weights-id trellis-ig28 --report <report.json>
```

- **The inputs** are public: the 27B's BF16 (its download records pin the revision the record names,
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`), the NVFP4 artifact the 16 GB conversion reads, and the 16 GB
  artifact's calibration trace and configuration at that quant's revision, which `ig28a.json` pins by sha256.
  The recipe file stays as committed: the record pins its sha256, `0f5e8460...`.
- **`--weights-id trellis-ig28`** names the artifact; it overrides the recipe's own `weights_id`, `trellis-ig28a`,
  the candidate's, and the packer and `verify` take the same.
- **The labels** `--quant-repo Qwen/Qwen3.8-27B --quant-branch exl3-ig28-local` are the published record's
  `source`: the quant was made from that repository's BF16, here, at `ig28a`.
- **What was shown**: the same quant packed twice, once under other labels, gives the same file, the sha256 above;
  the labels are the record's, not the container's. The local quant is exllamav3's own conversion on this machine
  (2,602 s on the RTX 5090, 19.6 GiB of VRAM allocated, 42.1 GiB of host working set); that a second quant gives the
  same bytes is not shown.

## The sidecar

`converter.json` (merged into `<artifact>.conversion.json`) carries no local path and no timestamp:

- `recipe`: the named recipe, its file's sha256, its ceiling, the text scope's device bytes and the body's
  bits per weight ("The recipe" above). The packer's own `recipe_id` beside it is the artifact's file stem.
- `source`: the quant's repository, branch, revision and every file the converter reads, by bytes and
  sha256; its `quantization_config` (the recipe's fields); its `license`. A local quant has no revision (`null`) and `pins`, its
  pins file by name, bytes and sha256.
- `served`: the served artifact's file name, bytes, sha256 and identity; its graft sidecar's sha256
  (`graft_sidecar`), or its conversion sidecar's sha256 and recipe id (`conversion_sidecar`).
- `reference`: the BF16 checkpoint the errors are measured against: repository, revision, its index
  and shards by bytes and sha256 (none in a dry run).
- `graft` (only with a graft sidecar): its provenance of the DFlash2 module (`grafted_from`, the drafter's
  config); `local_nvfp4`: the served sidecar's per-object NVFP4 records of the copied objects, which
  `ignis_artifact::checksum::verify` reads.
- `trellis`: per matrix its source key, `k2` (2·K), format, shape, bytes, sha256 and relative
  reconstruction error; the K histogram, the bound, the maximum and the mean.
- `override` (only with an override): its repository and revision (the replay's), the quant revision it was made
  against (`null` for a local quant, which `quant_pins` names by its pins file's name, bytes and sha256, as
  `source.pins` does) and every file read, by bytes and sha256; `matrices`, the object names it replaced (each also marked
  `override: true` in `trellis.matrices`); its `reference` BF16, the repository, revision, the `fetch.json` and
  index files, the shards (LFS oid) and each override matrix's tensor by sha256 (the errors in
  `trellis.matrices` are measured against it; none in a dry run).
- `copied`: per object its bytes and sha256; `dropped`: the served objects left out; `quant_unused`:
  the quant's matrices not taken (its MTP layer).
- `converter`: the commit the converter ran from and whether its files differed from it (`dirty`);
  `versions`: Python, torch, exllamav3, safetensors.

## The abliterated replay

`replay.py` makes the override records of the uncensored 16 GB artifact (spec #386, GitHub #421): the
70 matrices the huihui-ai abliteration changed (layers 17-51: `mlp.down_proj`, and
`linear_attn.out_proj` or `self_attn.o_proj` by layer type), re-quantized at the K the quant gives each
(all 3, layer 23's `o_proj` 4) by replaying exllamav3's own conversion of them.

```
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/replay.py fetch \
    --quant Y:/models/qwen38-27b-exl3-SC_3.00bpw_H4 --out H:/abliterated-386
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/replay.py run \
    --quant Y:/models/qwen38-27b-exl3-SC_3.00bpw_H4 --bf16 Y:/models/Qwen3.8-27B \
    --abliterated H:/abliterated-386 --out H:/override-386
```

- **`fetch`** takes the 70 BF16 tensors of `huihui-ai/Huihui-Qwen3.8-27B-abliterated` at revision
  739e3c5b (the 32 GB artifact's) by HTTP range, never whole shards (8.44 GB of 9 shards): each
  shard's tensors in a file of the shard's name, an index, and `fetch.json` (each shard's LFS oid,
  each tensor's source range and sha256). A tensor whose bytes are not its header's length, or whose
  dtype or shape is not the plan's, is refused. The abliterated BF16 is the base with that directory
  laid over it: exllamav3's tensor collection takes the later file of a key, so the base is never
  copied. A harness that reads it with safetensors resolves each tensor's file through `overlay.py`
  (`Reference(base, overlay).files`), which refuses an overlay tensor the base does not have, or has
  with another dtype or shape, and one whose bytes are not `fetch.json`'s sha256.
- **`run`** (GPU, exllamav3 1.5.3; `make gpu-status` first) replays `convert_model.py` for those
  matrices: the quant's `cal_trace.safetensors` rows (250 x 2,048); the state through the token
  embedding and layers 0..L-1 as quantized (the quant's records, the re-quantized ones in their
  place); the Hessian captured on layer L of the abliterated BF16, run whole; each matrix quantized
  alone by the converter's call (`make_quant_args`, `get_H_data`, `Linear.convert_exl3`) with the
  `mul1` codebook, output scales and the seed L+1, the layer's module index. `--check` checks the
  sources and stops before the GPU; `--last 17` is a smoke run of one layer.
- **The output** is the quant's layout (index, one shard per layer, `quantization_config.json` at
  exllamav3 1.5.3), which `exl3.Quant` reads, holding exactly the plan's records at the quant's K and
  shape, and `replay.json`, written last: the control (each matrix's relative reconstruction error
  against the abliterated BF16, beside the quant's own against the base BF16 and against the
  abliterated BF16), each matrix's seed, K and proxy error, the sources' revisions, the calibration
  trace's sha256, the versions and the command, its paths by role. turboderp's quant was made with
  exllamav3 1.4.2, so its bytes are not reproduced bit for bit.
- **`--check`** prints the plan's K by module and the sources it pinned, then `CHECK PASS`; it reads
  no GPU and writes nothing.

### From a quant converted locally

The uncensored 12 GB artifact (spec #701, GitHub #703) replays the same 70 matrices on the 12 GB
artifact's body, the local quant at the recipe `ig28a` ("The 12 GB artifact" above), each at the K
`ig28a` gives it: `linear_attn.out_proj` K3 25 and K4 1, `mlp.down_proj` K2 1 and K3 34,
`self_attn.o_proj` K3 5 and K4 4. The abliterated tensors are the same 70 (`fetch` above names them
from either quant), here in `H:/abliterated-701`:

```
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/replay.py run --check --recipe ig28a \
    --quant Y:/models/Qwen3.8-27B-exl3-ig28a --quant-pins Y:/models/Qwen3.8-27B-exl3-ig28a.pins.json \
    --trace Y:/models/qwen38-27b-exl3-SC_3.00bpw_H4 --bf16 Y:/models/Qwen3.8-27B \
    --abliterated H:/abliterated-701 --out H:/override-701
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/replay.py run --recipe ig28a \
    --quant Y:/models/Qwen3.8-27B-exl3-ig28a --quant-pins Y:/models/Qwen3.8-27B-exl3-ig28a.pins.json \
    --trace Y:/models/qwen38-27b-exl3-SC_3.00bpw_H4 --bf16 Y:/models/Qwen3.8-27B \
    --abliterated H:/abliterated-701 --out H:/override-701
```

- **`--recipe`** names the recipe the quant is, as `convert.py run` takes it (`sc30` by default, the
  16 GB route's): the quant's `quantization_config.json` must state the recipe's fields and every matrix
  must be at the recipe's K, or `run` refuses (exit 2), naming the field or the matrix. A downloaded quant
  is replayed at `sc30` only, since the record names it as `SC_3.00bpw_H4`.
- **`--quant-pins`** is the pins file `local_quant.py run` wrote beside the quant, which stands for the
  download record: every file the replay reads (the shards, the index, `quantization_config.json`,
  `config.json`) must have its pin, of its bytes and sha256.
- **`--trace`** is the directory holding the calibration trace the recipe pins (`ig28a` pins
  `SC_3.00bpw_H4`'s, so the downloaded quant's directory serves): `cal_trace.safetensors` must be the
  recipe's pin, bytes and sha256. A local quant takes both arguments, a downloaded one neither.
- **Refused** (exit 2, before the GPU, writing nothing), naming the file: a quant file with no pin, of
  another size or whose sha256 is not its pin's; a trace the recipe does not pin, missing, of another
  size or whose sha256 is not the pin's. The quant's files are hashed last, after the cheaper checks.
- **The record** names the quant by its recipe (name and file sha256) and its pins file (name, bytes,
  sha256), with `revision` `null`, where a downloaded quant is named by its repository, branch and
  revision; the calibration trace by the recipe that pins it, its bytes and sha256. Every other field is
  as from a downloaded quant, and the command names `--quant-pins` and `--trace` by role. The conversion
  takes the override with `--quant-pins` on that same quant ("The override" above).
- With a downloaded quant nothing changes: the record is the one the replay wrote before #703, field for
  field, and the conversion's `override` section with it.
