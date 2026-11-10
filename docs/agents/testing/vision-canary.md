# Testing against the vision canary

## The vision canary fixture (GitHub #178)

`crates/server/tests/fixtures/vision_canary/` is ADR 0014's floor applied to
multimodal prompts: four generated images (a rendered number, a coloured
square, three circles, a line of screenshot text), one unambiguous question
each, and the reference's greedy answers.
`crates/server/tests/vision_canary_gpu.rs` scores ignis against them by
teacher-forced agreement over the first 32 answer positions, each prompt
prefilled twice — whole, and in 48-token spans so the placeholder run crosses
span boundaries — and **each way clears the 95% floor on its own**, pooled
over the four canaries the way ADR 0014 pools its own suite.

The questions ask for a sentence, not a word, so the score has resolution:
the four answers are ~50 scored positions per prefill mode, and one near-tie
flip is two percent. At one token per answer it was a seventh of the suite,
which is noise rather than a floor. Measured 2026-09-16: whole 48/50 = 96.0%,
spanning 49/50 = 98.0%, and decode continuing the prefill path's own greedy
chain 46/46.

Re-record it (needs the GPU and the reference, ADR 0006 — stop
`ignis-server` first; the images are regenerated deterministically, so a
re-record with the same PIL/font is a no-op on them):

```powershell
F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe `
  F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --host 127.0.0.1 --port 8080 --vision --greedy --no-thinking `
  --max-context 8192 --max-concurrency 1 --kv-capacity auto
python tools/vision-canary/record.py crates/server/tests/fixtures/vision_canary
# then stop ninfer-serve and commit the fixture
```

`crates/runtime/tests/cuda_leaf_vision_gpu.rs` is the same load's serving-shape
check: an image prompt chunked at 64 tokens while three text lanes decode, each
lane still answering its own question, and 100 image requests leaving no media
embedding live and the leaf's footprint where it was. Its second test evicts an
image request to KV-RAM mid-decode and checks the restored one continues with
the never-evicted tokens (GitHub #194).

## Flash-Next's image checks (GitHub #560)

ADR 0087 admits Flash-Next's images without the 335 GiB BF16 checkpoint on
three checks, whose method and rules are fixed in
[the finding](../../findings/2026-11-10-flash-next-images-checked-without-the-bf16-checkpoint.md)
before any run:

- **The encoded tower against the BF16 tower**, with the 27B's as the bar:
  `crates/server/tests/vision_tower_dump_gpu.rs` encodes a set of images
  through each family's processor and media encode, as served, and dumps the
  patches and the embedding rows; `tools/flash-next-vision/tower_check.py`
  runs the checkpoint's own tower (transformers, float32, the CPU) on the
  same patches, from the first shard's BF16 tensors only, and compares.
- **Known answers**: `tools/vision-known-answers/` (a manifest of the four
  canaries and public photos with their licenses and answers, `fetch.py`,
  `score.py`), through the HTTP API of one running server at a time.
- **The decode gate**: `tools/flash-next-vision/decode_gate.py`, Flash-Next
  with vision on and no image in flight against vision off, C=1 and C=3, four
  legs in ABBA order, once at the end of the phase.

Commands, Git Bash from the repository root of a `make build CUDA=1` of the
branch. `PY` is a Python with transformers 5.17 or later (its `qwen4_exp` is
Flash-Next's tower), torch, numpy and Pillow; on the development machine
`F:/ai/ngram-venv`. The scripts that read downloaded images run under `-I`.

```bash
PY=F:/ai/ngram-venv/Scripts/python.exe; W=H:/fn-vision-560

# 0. The inputs (CPU and network, once): the BF16 towers' caches (~0.9 GB each,
#    by range), the public photos (each in its own directory, held to its SHA-1)
#    and the tower check's share of them.
$PY tools/flash-next-vision/convert.py fetch --source H:/fn-vision-556/source
$PY tools/flash-next-vision/tower_check.py fetch-27b --source $W/27b-visual
$PY -I tools/vision-known-answers/fetch.py --root $W/known --tower $W/tower-images

# 1. The served tower (GPU, the card exclusively; Flash-Next's load needs ~40 GB
#    of free RAM, one load at a time; ~10 minutes). Writes $W/tower-dump/<family>/.
IGNIS_TOWER_IMAGES=$W/tower-images IGNIS_TOWER_OUT=$W/tower-dump \
  cargo test -p ignis-server --features cuda --test vision_tower_dump_gpu -- --ignored --nocapture --test-threads=1

# 2. The BF16 towers on the same patches and the record (CPU, ~4 GB of RAM, a few
#    minutes per family: run it with no Flash-Next load on the host).
$PY tools/flash-next-vision/tower_check.py reference --family qwen3.8-flash-next \
  --source H:/fn-vision-556/source --dump $W/tower-dump
$PY tools/flash-next-vision/tower_check.py reference --family qwen3.8-27b --source $W/27b-visual --dump $W/tower-dump
$PY tools/flash-next-vision/tower_check.py compare --dump $W/tower-dump --out $W/tower-check.json --csv $W/tower-check.csv

# 3. Known answers (GPU, one server at a time; a few minutes each after the start).
make start MODEL=flash-next VISION=1 IDLE_UNLOAD=0 LOOP_GUARD=off
$PY -I tools/vision-known-answers/score.py run --url http://127.0.0.1:8000 --images $W/known \
  --label flash-next --out $W/known-runs
make stop
make start VISION=1 IDLE_UNLOAD=0 LOOP_GUARD=off
$PY -I tools/vision-known-answers/score.py run --url http://127.0.0.1:8000 --images $W/known \
  --label qwen27b --out $W/known-runs
make stop
$PY -I tools/vision-known-answers/score.py compare --flash-next $W/known-runs/flash-next.jsonl \
  --qwen27b $W/known-runs/qwen27b.jsonl --out $W/known-runs/compare.json

# 4. The decode gate (GPU, ~30 minutes): four legs, vision the only difference,
#    the context pinned so no leg adapts it and the VRAM budget so no leg plans
#    another expert cache when the card's free VRAM moves between legs (a run
#    without it planned four). HOST_MARGIN_BYTES=0 joins KNOBS, in every leg
#    alike, only when a start is refused for host memory.
G=$W/gate; KNOBS="MODEL=flash-next IDLE_UNLOAD=0 LOOP_GUARD=off MAX_CONTEXT=589824 VRAM_BUDGET=27G"
for leg in off-1:0 on-1:1 on-2:1 off-2:0; do
  label=${leg%:*}; vision=${leg#*:}
  make gpu-status
  make start $KNOBS VISION=$vision || break
  $PY -I tools/flash-next-vision/decode_gate.py run --url http://127.0.0.1:8000 --label $label \
    --vision $([ $vision = 1 ] && echo on || echo off) --out $G --server-log .scratch/serve/ignis-server.log
  cp .scratch/serve/ignis-server.log $G/$label.server.log
  make stop
done
$PY -I tools/flash-next-vision/decode_gate.py verdict --legs $G/off-1.json $G/on-1.json $G/on-2.json $G/off-2.json \
  --out $G/verdict.json
```

`compare` prints the bar's line and `score.py` an `ALARM` line for a wrong
canary or a clear gap to the 27B: both go to the owner, who decides whether a
KLD against BF16 on image prompts runs. `verdict` exits non-zero on a failed
gate and names the pair, the expert cache or the prompts that failed it. The
tools' own tests are CPU-only: `$PY -m pytest tools/flash-next-vision` and
`$PY -m pytest tools/vision-known-answers`.

When the tower's bar is above the 27B's, `tower_check.py diagnose` says where
the excess comes from (GitHub #560), on the CPU from the same dump: the
reference tower with its matrices taken from the BF16 checkpoint, from the
container's stored words decoded as the kernels decode them, or re-encoded in
another format the kernels serve, kind by kind and block by block (the
variant grammar is in the script's docstring). The `stored` variant against
the served rows says whether the encode computes what the stored words say,
the recipe check whether the stored words are the recipe's encoding of the
BF16 tensors, and the stages where the error grows. One forward per variant
per item, plus the BF16 one: about three minutes a pass per family over the
32 items at 6 threads, and up to ~10 GB of RAM at the largest item (its
stages held beside the model and the decoded words; each further source
another ~1.8 GB), so one run at a time and none beside a Flash-Next load.

```bash
D=$W/tower-diagnosis
$PY tools/flash-next-vision/tower_check.py diagnose --family qwen3.8-27b --source $W/27b-visual \
  --container F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer --dump $W/tower-dump \
  --out $D/qwen27b-stored.json --variant stored
FN="diagnose --family qwen3.8-flash-next --source H:/fn-vision-556/source --dump $W/tower-dump \
  --container F:/ai/models/Qwen3.8-Flash-Next-ignis/qwen3_8_flash_next_vision-v2.ninfer --against $D/qwen27b-stored.json"
$PY tools/flash-next-vision/tower_check.py $FN --out $D/flash-next-stored.json --variant stored
# Every matrix W8 (the merger's stored words already are):
$PY tools/flash-next-vision/tower_check.py $FN --out $D/flash-next-w8.json --variant stored+patch_embedding=W8+block=W8
```

Without `--variant` it runs `bf16`, `stored` and each kind's stored words
alone (`bf16+qkv=stored`, ...); `--against` adds each variant's paired bar
against the other family's stored tower.

On that diagnosis the owner had Flash-Next's companion re-encoded with every
matrix W8 (GitHub #556, 2026-11-10); the grouped one is kept in
`H:/fn-vision-560/companion-q4q5/`. The recipe check reads each family's own
recipe: Flash-Next's every matrix W8, the 27B's grouped (`tower_check.py`'s
`TOWERS`). The re-encoded companion is checked by the same commands: step 1
again (the dump of the new companion; the 27B's rows do not change), step 2
and `compare`, then the stored variant alone, whose recipe check holds all
333 objects equal and whose served rows sit at the kernels' rounding from the
stored-weight rows:

```bash
$PY tools/flash-next-vision/tower_check.py $FN --out $D/flash-next-w8-stored.json --variant stored
```
