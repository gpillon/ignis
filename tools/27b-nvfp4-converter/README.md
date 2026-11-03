# 27B NVFP4 converter

Spec #449, GitHub #468: the offline tool that turns a BF16 checkpoint of `Qwen/Qwen3.8-27B`, the base's or a
**fine-tune**'s, into the work tree the `qwen3.8-27b-nvfp4` packer family packs
(`docs/formats/27b-nvfp4-work-tree.md`), under a weights id of its own. The Rust packer
(`ignis-artifact-pack --family qwen3.8-27b-nvfp4`, `crates/artifact`) assembles the container and its
`.conversion.json`; Python never writes the container.

It is a Python port of ninfer's nvfp4full recipe (`F:\ai\q38\ninfer\tools\convert\qwen3_8_27b\`: the
inventory, the recipe, the NVFP4 encoder, the calibration and its corpus, `verify`), the one behind the served
`qwen3_8_27b_nvfp4full-v2.ninfer`, with one change: **all 247 text NVFP4 matrices are encoded here from the
BF16 checkpoint, and all 247 activation divisors are calibrated here.** The served artifact copies 112 of
each (the MLP of layers 0-55) from a third-party NVFP4 quant of the base, and no such quant exists for a
fine-tune. Everything else is the served recipe's:

- the inventory: the 1,325 objects, their names, formats, layouts and order, the positional BF16 exceptions
  (`attention/query_key_gate_value` at layers 3, 7, 11, 15, 19, 23, `attention/output` at 3 and 7,
  `gdn/output` at 4) included;
- the encoder `NVFP4_MAXABS_DIVISOR_RNE_V1` (weight divisor `binary32(2688 / max|W|)`, one E4M3FN scale per
  16 elements, E2M1 rounded to nearest even) and the grouped W8/Q4/Q5/Q6 one, whose bytes are ninfer's
  (`test_encode.py` compares them when ninfer's checkout is on the machine);
- the fixed calibration corpus: ten documents, 2,690 tokens of the 27B's tokenizer, one prefill;
- the W8 token embedding and output head, the Q4 draft head of 131,072 rows picked by ninfer's frequency
  ranking (its total row, carried in `data/`), the MTP layer, the vision tower in W8, Q4, Q5, Q6 and BF16,
  the norms and the GDN small set as the checkpoint holds them, the six frontend resources byte for byte;
- the 66 DFlash2 objects, encoded weight-only from the base's drafter (`incoai/Qwen3.8-27B-DFlash2` at
  `dedf8df6`), trained for the base and used unchanged for every fine-tune.

## Files

| File | What it is |
|---|---|
| `convert.py` | The command: `calibrate` (the GPU step), `run` (the work tree and `converter.json`), `verify` (a packed artifact against its sources) and `compare` (#469). |
| `compare.py` | `compare`: an artifact built from the base checkpoint against the served one, in #469's three tiers; `rebuild`, the machine-local test's tier 1 from the sources. |
| `inventory.py` | The container's objects from a geometry: the 27B's, or the tests' reduced one (the packer's `--geometry fixture`). |
| `recipe.py` | Each object's route and its parts in the checkpoint or the drafter; the local-only recipe's counts; the served artifact's recipe, which `compare` reads. |
| `encode.py` | The encoders: NVFP4 (`blockscale-k16-m128x4-v1`), grouped (`row-split-k128-v1`), direct (`contiguous-le-v1`). |
| `calibrate.py` | The corpus, the 247 sites and the streamed prefill that measures them. |
| `checkpoint.py` | The sources' checks: the 27B's tensors and configuration, the drafter's, the download-record pins, the ranking and the draft head's rows. |
| `shards.py` | Safetensors reads that keep at most one shard open; headers read without a handle. |
| `container.py` | Read-only access to a `.ninfer` v2 container (copied from `tools/27b-16gb-converter`). |
| `work.py` | The DONE protocol the packer checks (copied from `tools/image-converter`). |
| `data/ranking.totals.i64` | The draft head's ranking, `--ranking`'s default: row 0 of ninfer's `ranking.train.counts.i64`, the totals. `data/README.md` gives its provenance and licence. |
| `test_*.py`, `conftest.py` | The tests (CPU only) on a reduced synthetic checkpoint, and the machine-local checks of the real sources. |

## The Python environment

`F:/ai/ngram-venv` (Python 3.12) holds everything it needs: torch 2.13 (CUDA 13.0), transformers 5.17
(`Qwen3_5Model`), accelerate 1.15 (`init_empty_weights`), safetensors 0.8, tokenizers 0.23, numpy 2.5, psutil
(the peaks) and pytest 9.1. Another environment needs the same packages; transformers must have `Qwen3_5Model`.

The calibration runs the GDN's chunked delta rule through transformers' reference torch function by default.
transformers would otherwise take flash-linear-attention's kernel when that package is installed (it is in this
venv): it runs on CUDA only, and its numerics differ from the reference's. `--gdn-kernel installed` asks for
whatever transformers picks; the calibration record names the one used.

## Tests

```
F:/ai/ngram-venv/Scripts/python.exe -m pytest tools/27b-nvfp4-converter -q
```

- On a reduced synthetic checkpoint at the fixture geometry (`conftest.make_sources`: every tensor of the 27B's
  checkpoint at narrow widths, three shards cut through layers, a word-level tokenizer of 256 ids, download
  records), the base's drafter at those widths and a ranking, on the CPU: the calibration equals the whole
  model's forward; the work tree holds the inventory under the weights id, every NVFP4 parent encoded from its
  BF16 rows beside its calibrated divisor; without `--ranking`, the draft head's rows from the committed totals;
  every refusal below, by name, with nothing written; at most one safetensors handle open, in calibration,
  conversion, `verify` and `compare`; the encoders give the same bytes a block of rows at a time as at once.
  The committed totals' size and sha256 are their pin, and at the 27B's geometry a ranking matching neither pin
  is refused.
- The packed artifact (the packer's own checks of the tree, then the sidecar's fields the loader's checksum
  verification reads), `verify`, passing and failing by name, and `compare` need the packer binary:
  `IGNIS_ARTIFACT_PACK`, else this checkout's `target/**/ignis-artifact-pack.exe`. They skip, saying so,
  without one (`cargo build -p ignis-artifact --bin ignis-artifact-pack`). `compare` runs on the packed
  artifact against a copy of it under the served recipe's `.graft.json`: tier 1 and tier 2 byte-equal and
  tier 3 exactly the 112 MLP parents of layers 0-55; a tier-1 difference named and failing; a tier-2 parent
  perturbed by hand (three codes, a scale, its weight and activation divisors doubled) measured right; its
  refusals.
- Machine-local, skipped where absent: the 27B inventory is the served artifact's directory, object for object;
  the base and Swift checkpoints on H: pass the cheap checks; the corpus is 2,690 tokens of the 27B's
  tokenizer; ninfer's ranking is pinned and the committed totals are its first row, and both give the base
  checkpoint's draft head the same 131,072 ids; the encoders' bytes are ninfer's; the served recipe's
  third-party parents are ninfer's `SOURCE_NVFP4_WEIGHT_SPECS`; and **#469's real-artifact test**: the 765
  tier-1 objects the recipe makes of the base checkpoint on the CPU (`H:/models/Qwen3.8-27B`, every object but
  the drafter's, the draft head's ids from the committed totals) are the served artifact's byte for byte, and so
  are the 66 DFlash2 objects of the drafter (`H:/models/Qwen3.8-27B-DFlash2`), a test that also skips until
  the drafter's file is there and hashes to its LFS oid; one of the 135 parents ninfer encoded itself re-encodes
  byte-equal. Measured here, each test alone
  (`pytest -s` prints the cost): the 765 take 25-30 s and peak at 6.6 GiB working set and 10.7 GiB commit; the
  66 take 94 s, the drafter's hash included, and peak at 8.1 GiB working set and 11.4 GiB commit. Both are the
  whole process's peaks, and the open file's mapping counts in them (a shard up to 4 GB, the drafter 3.8 GB).

No test touches the GPU: `conftest.py` hides CUDA from the process and fails the session if CUDA was
initialized.

## Inputs and outputs

- **The BF16 checkpoint** (`--checkpoint`), downloaded with the Hugging Face CLI so its download records exist
  (`.cache/huggingface/download`): every file read (the index, `config.json`, the six frontend files, the 18
  shards) at one revision, the one `--source-revision` names; each shard's SHA-256 its LFS oid, each small
  file its git blob id. Its tensors must be the 27B's 1,199 (names, shapes, BF16) and its configuration's
  members the 27B's.
- **The drafter** (`--dflash2`): `config.json` and `model.safetensors` of `incoai/Qwen3.8-27B-DFlash2` at
  `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, with their download records; the file's SHA-256 its LFS oid
  (`67fc76d6...`), its configuration and its 81 tensors the base's drafter's.
- **The ranking** (`--ranking`, optional): by default `data/ranking.totals.i64`, the total row of ninfer's
  `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64` (1,986,560 bytes, sha256 `1e8fb35c...`, pinned;
  `data/README.md`); `--ranking` also takes ninfer's whole file (11,919,360 bytes, sha256 `c692dc76...`, pinned),
  whose row 0 gives the same ids. The total row picks the draft head's rows over the tokenizer's 248,077 ids, the
  special ones forced in. `converter.json` names the file used.
- **The calibration** (`--calibration`): the record `calibrate` writes from the same checkpoint.
- **The labels**: `--weights-id` (the build's own; `nvfp4full`, the served artifact's, is refused),
  `--source-repo`, `--source-revision`, `--license` (the identifier the catalog names, ADR 0078). They go into
  `converter.json`.
- **Out**: the work tree (`--work`, absent or empty): `frontend/`, `text/`, `mtp/`, `vision/`, `dflash2/`, each
  with its `DONE` written last, and `converter.json` written after them. The packer adds the whole-file
  invariants (`artifact.bytes`, `objects.count`) in the sidecar.

## The run on the 5090

The GPU steps are `calibrate`, `run` and `verify`; each needs the card to itself (`make gpu-status` first, and
the server on :8000 has several users: ask before stopping it). Read H:'s free space right before each step that
writes (`python -c "import shutil;print(shutil.disk_usage('H:/').free/2**30)"`). Nothing is read from or written
to Y:. Below, the base checkpoint (#469); for Swift (#470) the checkpoint is `H:/models/Swift-1.5-Qwen3.8-27b`,
`--source-repo ukisai/Swift-1.5-Qwen3.8-27b --source-revision b4c84d42903a8646b25857eb2827288d92ed85a4
--license swift-open-license-1.0 --weights-id nvfp4full-swift-1.5` and the output
`qwen3_8_27b_nvfp4full-v2-ukisai-swift-1.5.ninfer`.

```
# the drafter, once, onto H: (3,848,817,896 bytes), its file list checked first
F:/ai/ngram-venv/Scripts/hf.exe download incoai/Qwen3.8-27B-DFlash2 config.json model.safetensors \
    --revision dedf8df68adfb1afeaf7b7480c0a0243108177b4 --local-dir H:/models/Qwen3.8-27B-DFlash2

# 1. calibration: the 247 divisors, a JSON of some tens of KB
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-nvfp4-converter/convert.py calibrate \
    --checkpoint H:/models/Qwen3.8-27B --out H:/nvfp4-469/calibration.json

# 2. conversion: the work tree, 19,406,582,553 bytes (18.1 GiB)
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-nvfp4-converter/convert.py run \
    --checkpoint H:/models/Qwen3.8-27B --source-repo Qwen/Qwen3.8-27B \
    --source-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --license apache-2.0 \
    --dflash2 H:/models/Qwen3.8-27B-DFlash2 \
    --calibration H:/nvfp4-469/calibration.json --weights-id nvfp4full-local-469 --work H:/nvfp4-469/work

# 3. packing (CPU): the container, 19,406,942,468 bytes, and its .conversion.json
cargo build --release -p ignis-artifact --bin ignis-artifact-pack
target/x86_64-pc-windows-msvc/release/ignis-artifact-pack.exe --family qwen3.8-27b-nvfp4 \
    --work H:/nvfp4-469/work --out H:/nvfp4-469/qwen3_8_27b_nvfp4full-v2-local-469.ninfer \
    --weights-id nvfp4full-local-469

# 4. verify: VERIFY PASS or VERIFY FAILED (exit 3)
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-nvfp4-converter/convert.py verify \
    --artifact H:/nvfp4-469/qwen3_8_27b_nvfp4full-v2-local-469.ninfer \
    --checkpoint H:/models/Qwen3.8-27B --dflash2 H:/models/Qwen3.8-27B-DFlash2 \
    --calibration H:/nvfp4-469/calibration.json --weights-id nvfp4full-local-469 --report H:/nvfp4-469/verify.json

# 5. #469 only, CPU: the scratch artifact against the served one, in three tiers (`compare` below)
F:/ai/ngram-venv/Scripts/python.exe -u tools/27b-nvfp4-converter/convert.py compare \
    --served F:/ai/opencode/inference/models/qwen3_8_27b_nvfp4full-v2.ninfer \
    --artifact H:/nvfp4-469/qwen3_8_27b_nvfp4full-v2-local-469.ninfer \
    --checkpoint H:/models/Qwen3.8-27B --report H:/nvfp4-469/compare.json
```

- **Disk.** The packer deletes each unit's work files once the unit is in the container (`--keep-work` keeps
  them, and needs room for two copies), so packing peaks near one copy, ~18.1 GiB, on top of the work tree's
  disk; the drafter is 3.6 GiB more.
- **Time and peaks.** Every command logs its seconds, the CUDA allocator's peaks and the host's peak working set
  and commit (`calibration: ...`, `conversion: ...`, `verify: ...`), and the time it took to hash the sources
  against their records (`pins: ...`: 55.6 GB of shards, each command). The calibration record keeps its own.
  The estimates, not measured: calibration minutes (one prefill of 2,690 tokens with one layer on the card,
  bound by reading the shards); conversion and `verify` tens of minutes each (bound by reading the shards and
  writing or reading the tree). Every matrix is encoded 8,192 rows at a time (`encode.ROWS`), so the W8 output
  head (248,320 x 5,120) never stands on the card in FP32 at once; the VRAM peak is a block's temporaries, a GB
  or two (estimated, not measured): the card must be free all the same.
- **An interrupted run** leaves a partial tree without `converter.json`, which the packer does not take; delete
  it and run again (`run` refuses a non-empty `--work`).

## `run`

Refused (exit 2), naming what failed and writing nothing:

- a `--work` that is not empty; `--weights-id nvfp4full`;
- a checkpoint whose tensors are not the 27B's (a missing or extra tensor, another shape, not BF16) or whose
  configuration's members are not the 27B's;
- a file with no download record, files from two revisions or from another revision than `--source-revision`, a
  shard whose record holds no LFS oid or whose SHA-256 is not its oid, a small file that is not its git blob;
- a drafter at another revision, or whose configuration or tensors are not the base's drafter's;
- a ranking whose sha256 is neither pin (the committed totals', ninfer's whole file's), a tokenizer whose id
  count is not the 27B's;
- a calibration measured on another checkpoint (another revision or another shard's LFS oid), over another
  corpus, or that does not cover the 247 sites with finite positive binary32 divisors;
- a recipe other than the local-only one: its route counts (6 resources, 247 local NVFP4 parents, 247
  calibrated divisors, 9 BF16 exceptions, 401 objects as the checkpoint holds them, 349 W8, Q4, Q5, Q6 and BF16
  objects of the embedding, the heads, the MTP layer and the vision tower, 66 drafter objects; no source
  parents), or a text NVFP4 parent not encoded here;
- a disk with less free space than the work tree.

The source files are hashed last, after every other check.

**One shard at a time.** Every safetensors handle the tool opens, in calibration, conversion and `verify`, goes
through `shards.ShardReader`, which holds one and closes it before it opens the next (on Windows each open
handle costs its file's size in commit: GitHub #422's `os error 1455`); the checkpoint's reader is closed before
the drafter's opens. ninfer's calibration kept a handle on every shard it read.

## `converter.json`

The converter's half of the sidecar (`docs/formats/27b-nvfp4-work-tree.md` §6), with no local path:

- `schema`, `status` (`complete`), `recipe_id` (`qwen3_8_27b_nvfp4full-v2-local`: the sidecar's, never the
  file name's), `identity` (`qwen3.8-27b` and the `--weights-id`), `geometry`, `device`;
- `source`: `repo`, `revision`, `license` and `files`, every checkpoint file read by bytes and sha256;
- `drafter`: its repository, revision, files and configuration; `ranking`: the file used (the committed
  `ranking.totals.i64` or ninfer's `ranking.train.counts.i64`), its bytes, sha256 and origin,
  the tokenizer's id count, the forced special ids and the rows;
- `calibration`: the record's sha256 and its provenance (corpus, encoder profile, GDN rule, device, versions);
- `recipe`: the route counts; `local_nvfp4`: the encoder profile and the 281 NVFP4 parents, each text parent
  with its `weight_scale_divisor` (its work file's last FP32 word) and each drafter parent with none, each with
  its `relative_frobenius_error`, and their maximum and mean;
- `converter` (the commit and whether the tool's files differed from it) and `versions`.

## The calibration record

`calibrate` writes `schema`, `encoder_profile`, `divisor_formula`, `corpus` (documents, tokens, sha256),
`checkpoint` (its revision and shards by bytes and sha256: what `run` and `verify` hold it to), `gdn_kernel`,
`device`, `seconds`, `peaks`, `versions`, `converter`, and `measured_sites`: per divisor object its module, its
`amax` and its `input_scale_divisor`, `binary32(2688 / amax)`. It refuses an existing `--out`. The model is
built with empty weights and each decoder layer's parameters are read in before its forward and dropped after
it, the token embedding and the final norm resident.

## `verify`

It reads a packed artifact against its sources: the identity (`qwen3.8-27b` and `--weights-id`), the directory
against the inventory object for object (name, kind, format, layout, shape, bytes, order), then every object's
payload against the one its sources give: each of the 247 local parents re-encoded byte-equal from the BF16, each
divisor the calibration's word, the drafter's parents re-encoded, every other object its source's bytes. It
prints the failures by object name and VERIFY PASS or VERIFY FAILED (exit 3); `--report` writes them with the
counts checked. Its sources are checked as `run`'s are (exit 2).

## `compare` (#469)

The proof of the tool before it builds a fine-tune: after the run above on the base checkpoint (the drafter,
calibration, conversion, packing, `verify` PASS), `compare` sets the scratch artifact beside the served
`qwen3_8_27b_nvfp4full-v2.ninfer`, object by object. It runs on the CPU (`--device cuda` when the card is free),
reads both artifacts and the base checkpoint's BF16 rows, and writes nothing but its report.

- **Which object is which.** Each artifact's sidecar names its recipe: the served `.graft.json`
  `qwen3_8_27b_nvfp4full-v2`, the scratch `.conversion.json` `qwen3_8_27b_nvfp4full-v2-local`. The routes are
  `recipe.py`'s: `recipes` for the local one, `served_recipes` for the served one, which is the same but for
  the MLP parents of layers 0-55 and their divisors, copied from the third-party quant. The served sidecar
  records only the drafter's 34 parents, so that split is ninfer's `SOURCE_NVFP4_MLP_LAYERS`, stated in
  `recipe.py` and checked against ninfer's inventory by a machine-local test.
- **Tier 1, binding** (831 objects): every object neither calibrated nor taken from the quant (the 6 frontend
  resources, the 401 norms and GDN small set, the 9 BF16 exceptions, the 349 objects of the embedding, the heads,
  the MTP layer and the vision tower, the 66 DFlash2 objects) must be byte-equal. A difference names the object,
  its route and its differing bytes, and the comparison ends TIER 1 FAILED (exit 3): the work stops and the owner
  hears of it.
- **Tier 2** (the 135 parents both recipes encode locally, and their divisors): byte-equal, or per parent the
  differing words (E2M1 codes of `N·K`, E4M3 scales of `N·K/16`, the FP32 weight divisor, and any padding byte),
  the weight divisor on both sides with its relative difference `(scratch - served) / served`, and the relative
  reconstruction error `|W - W_hat| / |W|` against the BF16 rows on both sides; per divisor, both values and
  their relative difference. A difference is a finding for the owner, not noise.
- **Tier 3** (the 112 parents the served artifact took from the quant, and their divisors): each parent's
  reconstruction error against the BF16 on both sides, the weight divisors and the activation divisors with
  their relative differences, and the means and maxima of both sides' errors.
- **Out.** `--report` (JSON: both artifacts' identities, sizes and recipes, the checkpoint's revision, the
  three tiers with every entry, the verdict, seconds and peaks) and beside it the same name with `.txt`: the
  summary it prints, which ends TIER 1 PASS or TIER 1 FAILED.
- **Refused** (exit 2, nothing written): an artifact with no sidecar, or whose sidecar names a recipe `compare`
  does not know, records another size or object count, or records as encoded locally a parent its recipe takes
  from the quant; a directory that is not the inventory's; recipes that put an object in no tier; a checkpoint
  that is not the one the scratch artifact's record names (its revision and every shard's LFS oid, read from the
  download records: `run` and `verify` hashed the shards, `compare` does not).
- **Cost** (estimated from one parent, not measured whole): it reads every payload of both artifacts once
  (19.4 GB each: tier 1 6.0 GB, tier 2 5.0 GB, tier 3 8.4 GB) and the 112 tier-3 parents' BF16 rows (about
  30 GB), and decodes both sides of each tier-3 parent, about 2 s per `gate_up` side and 1 s per `down` side
  on this box's CPU: some ten minutes in all. At most one shard is open at a time, and the log names every
  measured parent as it goes.
