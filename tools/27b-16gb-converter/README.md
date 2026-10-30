# 27B 16 GB converter

Spec 01 of the 27B on 16 GB cards (`docs/specs/27b-16gb/01-the-16gb-artifact.md`, GitHub #338): the
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
| `override.py` | An override directory: EXL3 records that replace the quant's, checked like for like. |
| `reconstruction.py` | The GPU step and the only exllamav3 user: each trellis matrix's relative reconstruction error against the BF16 checkpoint. |
| `work.py` | The DONE protocol the packer checks (copied from `tools/image-converter/work.py`). |
| `test_*.py`, `conftest.py` | Unit tests (CPU only) on a reduced synthetic served artifact and quant. |

## Tests

```
F:/ai/ngram-venv/Scripts/python.exe -m pytest tools/27b-16gb-converter -q
```

The packer's side is `crates/artifact/tests/trellis_27b.rs`; it also binds the real artifact when it is
on this machine.

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

- **`run`** refuses (exit 2, nothing written) a non-empty `--work`, another quant recipe (its
  `quantization_config.json` must say exl3, mul1, out scales always, 3.0 bits, head 4), a shard whose
  sha256 is not its LFS oid (the BF16 checkpoint's too), files from two revisions or with no download
  record, a quant matrix with no served plane or a plane
  with no quant matrix, split parts that do not tile their plane, and any matrix whose reconstruction
  error is above `--error-bound` (0.6: a layout error, never a quantization one). `--skip-errors` skips
  the GPU step and writes a `dry-run` tree, which the packer takes only with `--accept-status dry-run`.
- **The packer** consumes the work files as it appends them (`--keep-work` keeps them, given room for
  two copies) and writes `<artifact>.conversion.json`: `converter.json` plus the whole-file invariants.
- **`verify`** prints VERIFY PASS or VERIFY FAILED (exit 3).

## The override

`run` and `verify` take an **override**, a directory of EXL3 records in the quant's layout (index, shards,
`quantization_config.json`) that replaces the quant's records of the same name, for a variant made from
another BF16 (the uncensored 16 GB artifact, from the abliterated BF16). Every other object is built
exactly as without it, and without `--override` the work tree and `converter.json` are as before.

```
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-16gb-converter/convert.py run ...     --override H:/models/<override> --override-repo <owner/override-repo>     --override-bf16 H:/models/<variant-bf16> --override-bf16-repo <owner/variant-bf16>     --weights-id trellis-sc30-abliterated
F:/ai/ngram-venv/Scripts/python.exe tools/27b-16gb-converter/convert.py verify ...     --override H:/models/<override> --weights-id trellis-sc30-abliterated
```

- **A swap is like for like.** `run` and `verify` refuse (exit 2 and VERIFY FAILED) an override record
  whose name is not one of the quant's text matrices, or whose K or shape differs from the quant's
  record, naming the record and what differs.
- **Its error is measured against its own BF16.** The reconstruction step reads each override matrix
  from the override and its reference from `--override-bf16`; the quant's matrices are measured against
  `--bf16` as today. The same pins apply: revision, sha256 of every file read, shards against their LFS oid.
- **`--weights-id`** sets the identity the record and `verify` expect (default `trellis-sc30`); pass the
  packer the same `--weights-id` and the output name.
- **`verify`** compares each override record to the override's, bit for bit; it takes only `--override`
  (the source repositories are the conversion record's).

## The sidecar

`converter.json` (merged into `<artifact>.conversion.json`) carries no local path and no timestamp:

- `source`: the quant's repository, branch, revision and every file the converter reads, by bytes and
  sha256; its `quantization_config`.
- `served`: the served artifact's file name, bytes, sha256 and identity; its graft sidecar's sha256.
- `reference`: the BF16 checkpoint the errors are measured against: repository, revision, its index
  and shards by bytes and sha256 (none in a dry run).
- `graft`: the served graft sidecar's provenance of the DFlash2 module (`grafted_from`, the drafter's
  config); `local_nvfp4`: its per-object NVFP4 records of the copied objects, which
  `ignis_artifact::checksum::verify` reads.
- `trellis`: per matrix its source key, `k2` (2·K), format, shape, bytes, sha256 and relative
  reconstruction error; the K histogram, the bound, the maximum and the mean.
- `override` (only with an override): its repository, revision and every file read, by bytes and sha256;
  `matrices`, the object names it replaced (each also marked `override: true` in `trellis.matrices`); its
  `reference` BF16, the repository, revision and files the override matrices' errors are measured against
  (none in a dry run).
- `copied`: per object its bytes and sha256; `dropped`: the served objects left out; `quant_unused`:
  the quant's matrices not taken (its MTP layer).
- `converter`: the commit the converter ran from and whether its files differed from it (`dirty`);
  `versions`: Python, torch, exllamav3, safetensors.
