# Flash-Next's images checked without the BF16 checkpoint: the encoded tower against its BF16 tower, known answers beside the 27B's, and the decode gate

- Kind: experiment
- Status: current
- Observed: 2026-11-10
- Last verified: 2026-11-10
- Scope: Flash-Next's vision companion (`qwen3_8_flash_next_vision-v2.ninfer`, every matrix W8) and the 27B's tower,
  each encoded by the media encode under the `host` residency and answered through the server's HTTP API, on the RTX
  5090 (PCIe Gen 3 x16 host, Windows/WDDM); MTP-3 on image prompts; Flash-Next's decode with vision on and off
- Topic: vision
- Related: [ADR 0087](../adr/0087-flash-nexts-vision-encoder-ships-in-a-companion-container-and-encodes-in-memory-its-prefill-already-reserves.md),
  [ADR 0075](../adr/0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md),
  [ADR 0056](../adr/0056-speculation-past-eight-verify-rows-is-numerically-equivalent.md),
  [ADR 0088](../adr/0088-a-vendored-registry-that-lacks-a-case-ignis-needs-may-be-extended-or-rewritten-when-small.md),
  [the vision residency's acceptance](2026-10-30-vision-residency-acceptance.md),
  [the NVFP4 converter reproduces the served 27B](2026-11-03-nvfp4-converter-reproduces-the-served-27b.md),
  https://github.com/gpillon/ignis/issues/560, https://github.com/gpillon/ignis/issues/555,
  https://github.com/gpillon/ignis/issues/565, https://github.com/gpillon/ignis/issues/567
- Superseded by: none
- Summary: With every matrix W8, Flash-Next's tower is 0.059 in relative L2 from its BF16 tower, the 27B's 0.183, on 32 images (the grouped Q4/Q5 encoding's 0.305 was the alarm); known answers 66/66 beside the 27B's 66/66; RTX 5090.

## Question

ADR 0087 admits Flash-Next's images without the 335 GiB BF16 checkpoint on three checks beside the positions' fixture
(#558): is the encoded tower as close to its BF16 tower as the 27B's is to its own, measured the same way; how does
Flash-Next answer images with known answers beside the 27B; and does a vision load, with no image in flight, decode at
least 99% as fast as a text-only one at one lane and at three (#555's phase gate)? An alarm from the first two goes to
the owner, who decides whether a KLD against BF16 on image prompts runs. Beside them: what a vision load plans, and
what MTP does on image prompts (#564, #565).

## Evidence

The measures, the bars and the alarm rules of checks 1 to 3 were fixed on 2026-11-09, before any run; the runs
landed on 2026-11-10. The image MTP test's rule (section 4) and the decode gate's VRAM budget were set on 2026-11-10,
after the runs that called for them, and say so where they appear.

**Setup.**
- Trees: the tower check, the GPU suite and Flash-Next's known answers on `555-integration` at 3fae18c5, the W8
  companion pinned (#560's comment of 2026-11-10); the 27B's known answers on the release build made before the
  re-encode, at ef632ec3, the 27B's tower and recipe the same in both; the bind CTest's item bounds at f749cea0.
- The vision companion `qwen3_8_flash_next_vision-v2.ninfer`, every matrix W8 (482,499,840 B, sha256 `0110bb79…`,
  weights id `vision-w8-bf16-de4b8e4`, #556), pinned to the main container `12e3b265…`; the 27B
  `qwen3_8_27b_nvfp4full-v2.ninfer`.
- The first run of check 1, on the companion first shipped in the 27B tower's grouped formats (284,229,120 B, sha256
  `d97353d4…`, weights id `vision-w8q4q5q6-bf16-de4b8e4`, kept in `H:/fn-vision-560/companion-q4q5/`), raised the
  alarm: pooled relative L2 0.3046 against the 27B's 0.1834, the bar +0.123 [+0.112, +0.134]
  (`H:/fn-vision-560/tower-check-q4q5.json`). `tower_check.py diagnose` (`H:/fn-vision-560/tower-diagnosis/`, the
  table below) found the encode right and the excess the recipe's. The owner had the companion re-encoded with every
  matrix W8 (2026-11-10); checks 1 and 2 below are the re-encoded companion's.
- The BF16 towers, each the first shard's 333 `model.visual.*` tensors fetched by HTTP range, no other byte of either
  checkpoint read:
  - Flash-Next: `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d`, `model-00001-of-00131.safetensors`, 897,862,112 bytes from
    offset 142,293,832, sha256 `665127be…` (`tools/flash-next-vision/convert.py fetch`);
  - the 27B: `Qwen/Qwen3.8-27B` at `1d4bf0f2`, `model-00001-of-00018.safetensors`, 921,460,192 bytes from offset
    3,045,270,360, sha256 `c623c5ec…` (`tools/flash-next-vision/tower_check.py fetch-27b`). The served 27B's tower is
    the converter's from this revision byte for byte ([finding](2026-11-03-nvfp4-converter-reproduces-the-served-27b.md)).
- The commands are in [docs/agents/testing/vision-canary.md](../agents/testing/vision-canary.md#flash-nexts-image-checks-github-560).
  Raw output in `H:/fn-vision-560/`: the dumps (`tower-dump/`, the grouped run's `tower-dump-q4q5/`), the records
  (`tower-check.json` and `.csv`, `tower-check-q4q5.*`, `tower-diagnosis/`, `known-runs/`), the decode gate's legs with
  their server logs (`gate*/`). The GPU tests' output: `.scratch/555/gpu/<run>.log` of the coordinator's checkout.

**The plan figures** (the plan a load logs, ADR 0087's memory decision): `flash_next_vision_large_image_gpu` at
3fae18c5 (`.scratch/555/gpu/f_large.log`), the RTX 5090 profile's three lanes, an 8,192-token chunk, a 589,824-token
context, the two loads one after the other at the same budget.

| | text-only | vision on, `host` |
|---|---:|---:|
| VRAM budget bytes | 29,768,867,840 | 29,768,867,840 |
| expert cache bytes (`ignis.runtime.flash_next_plan`) | 17,311,517,264 | 17,311,517,264 |
| program bytes (the prefill scratch among them) | 2,522,868,208 | 2,522,868,208 |
| prefill staging bytes (the expert staging ring) | 1,574,895,616 | 1,574,895,616 |
| KV pool bytes | 2,491,554,816 | 2,491,554,816 |
| media embedding bytes in VRAM | 0 | 0 |
| item bound (`vision_item_max_tokens`) | 0 | 16,384 |
| tower in host memory (`flash_next_host_plan` `vision_tower_bytes`) | 0 | 482,705,408 |
| embedding pool in host memory, envelope 32,768 (`vision_pool_bytes`) | 0 | 167,772,160 |

The tower's sizes (#556's W8 build): 482,705,408 bytes are its host spans (each object's span padded to 4 KiB, what
the host plan charges), 481,445,216 its payload and 481,451,264 what a `device` residency places; the grouped
companion's were 284,454,912, 283,172,768 and 283,180,544. The 27B's are 297,000,960 bytes of host spans and
295,711,648 in VRAM. The pool is 2,560 columns of BF16 per token of the envelope: 83,886,080 bytes at a 16,384-token
context (the canary's load, `f_canary.log`), 167,772,160 at the default envelope of 32,768.

The item bound is the largest item whose encode fits the prefill scratch and the staging ring, at most 16,384. At an
8,192-token chunk the prefill scratch alone holds it: `ignis_kernel_flash_next_bind_test` (CTest, 2026-11-10, at
f749cea0) prints 16,384 merged tokens in a 1,016,074,496-byte prefill scratch under either residency; at a 1,024-token
chunk the scratch alone holds 6,543 (`device`) or 5,320 (`host`), and 16,384 with the RTX 5090's 1,574,895,616-byte
ring. The 8,192 the same test printed on 2026-11-09 at c00f2e49 was the test's own envelope, not the scratch's
(8b42bb5a). The server logs 16,384 at both chunks (at 1,024: program 1,422,341,872 bytes, ring 1,574,895,616), and a
4,096 x 4,096 image of 16,384 merged tokens (16,412 prompt tokens) answers "47" at both, its encode 3.65-3.69 s.

**1. The encoded tower against the BF16 tower** (spec #555's seam 2).
- Images: the vision canary's four; sixteen synthetic ones drawn by the dump test (rings, checkers, stripes, gradients
  with blocks, 32 x 32 to 1,440 x 1,440, square to 1:17); twelve of the known-answer photos at a longer side of 384,
  640, 896 or 1,280 px (`fetch.py --tower`: every fifth photo of the manifest's order). 32 items, the largest 2,025
  merged tokens, every one within the 4,096 the CPU reference reaches.
- Served: `crates/server/tests/vision_tower_dump_gpu.rs`, each family's own vision processor on the image's bytes, then
  the leaf's media encode under `host` (one load per family, one at a time), the embedding rows the pool holds read
  back.
- Reference: `tools/flash-next-vision/tower_check.py reference`, the checkpoint's own tower as transformers 5.17
  implements it (`Qwen4ExpVisionModel` for Flash-Next's `qwen4_exp`, `Qwen3_5VisionModel` for the 27B's `qwen3_5`),
  its weights the cached BF16 tensors held in float32, computing in float32 on the CPU, on the very patch rows the
  encode read.
- The measure, fixed before the run: per image, the relative L2 of the served rows against the reference rows
  (`rel_l2`); per row, its relative L2 (median, p90, max); the cosine per row centred on the family's reference set
  mean (a raw cosine is ~0.995 for any two rows of one tower, so it is reported only beside the centred one). Two
  controls by the same measure: the served rows against the reference rows half the item away, and against the next
  image's. An error near its control is a layout or binding fault, not a quantisation error.
- The bar, fixed before the run: on the images both families encoded, the mean over images of Flash-Next's `rel_l2`
  less the 27B's, with its paired bootstrap 95% interval (images resampled, 10,000 draws, seed 560). Flash-Next's tower
  is at most the 27B's when that mean is not above zero; an excess goes to the owner, the interval saying whether it
  is beyond noise.

The record's pooled and median figures and its `bar` as written; the other rows are the medians, means and ranges of
the per-image figures in each family's `items`.

| | Flash-Next, W8 | Flash-Next, grouped (the alarm) | the 27B |
|---|---:|---:|---:|
| images | 32 | 32 | 32 |
| pooled relative L2 | 0.0586 | 0.3046 | 0.1834 |
| median of the images' relative L2 | 0.0512 | 0.3089 | 0.1844 |
| largest image's relative L2 | 0.1428 | 0.4399 | 0.3351 |
| median row relative L2, median over images | 0.0449 | 0.3059 | 0.1250 |
| centred cosine, mean over images | 0.9968 | 0.9216 | 0.9415 |
| raw cosine, mean over images | 0.9980 | 0.9418 | 0.9904 |
| control within the item, relative L2, median (range) | 1.12 (0.84-1.27) | 1.14 (0.86-1.32) | 0.79 (0.53-0.92) |
| control across images, relative L2, median (range) | 1.22 (1.05-1.53) | 1.22 (1.07-1.53) | 0.82 (0.74-1.42) |

The bar: Flash-Next less the 27B, mean −0.130, 95% interval [−0.149, −0.114], Flash-Next lower on 32 of 32 images;
pooled ratio 0.32; at most the 27B (`H:/fn-vision-560/tower-check.json` and `.csv`). The grouped encoding's: mean
+0.123 [+0.112, +0.134], lower on 0 of 32, pooled ratio 1.66 (`tower-check-q4q5.json`).

**Where the grouped encoding's error came from** (`tower_check.py diagnose`, `H:/fn-vision-560/tower-diagnosis/`:
`qwen27b-stored.json`, `flash-next-stored.json`, `flash-next-w8.json` for the simulation, `flash-next-w8-stored.json`
for the re-encoded companion). The reference tower is run again with its matrices decoded from the container's stored
words as the kernels decode them; the stages are the tower's after each block and each merger step, pooled over the
32 items.

| | the 27B | Flash-Next, grouped | Flash-Next, W8 |
|---|---:|---:|---:|
| recipe check: stored words equal to the recipe's encoding | 333 of 333 | 333 of 333 | 333 of 333 |
| stored-weight rows against BF16, relative L2 | 0.181 | 0.302 | 0.029 |
| the same, centred | 0.286 | 0.343 | 0.033 |
| share of that error's energy one vector every row shares | 4.3% | 31.1% | 37.4% |
| served rows against the stored-weight rows | 0.033 | 0.051 | 0.053 |
| stage error after block 26 / merger fc1 | 0.236 / 0.250 | 0.197 / 0.362 | 0.019 / 0.035 |

- The weights quantize alike in both towers: relative Frobenius error per kind, the 27B's then Flash-Next's, patch
  embedding Q6 0.0247 / 0.0247, qkv Q4 0.117 / 0.118, attention output Q5 0.051 / 0.051, fc1 Q4 0.114 / 0.115, fc2
  Q5 0.057 / 0.057, the merger W8 0.005 / 0.005; with every matrix W8, every kind 0.005-0.006.
- On the centred measure the grouped excess was smaller but above zero: +0.050 [+0.031, +0.067], Flash-Next lower on
  6 of 32 (`flash-next-stored.json`); the pre-registered bar stays the raw one.
- The W8 simulation before the re-encode (`flash-next-w8.json`): 0.029 against the 27B's stored 0.181, mean −0.157
  [−0.175, −0.141], lower on 32 of 32; a tower of 481,445,216 bytes in place of 283,172,768, the same staging slot
  (22,608,384 bytes). The re-encoded companion's stored-weight rows give the same 0.029 (`flash-next-w8-stored.json`).
- Published practice (`.scratch/555/research-vision-tower-quant.md`, read 2026-11-10): no release of this tower
  quantized below 8 bits was found; the official quantized releases keep the visual weights in 16-bit, the official
  GGUF projector for this shape is F16 or Q8_0 (its fc2 F16, 4,304 columns not a multiple of 32), and no source
  reports a tower-output relative L2, so there is no external bar for the figure.

**2. Known answers** (spec #555's seam 1).
- The set: `tools/vision-known-answers/manifest.json` (sha256 `40eab647…` in both runs), the vision canary's four
  images and 62 photos from Wikimedia Commons, each under public domain, CC0, CC BY or CC BY-SA (its page, author,
  license, the original's SHA-1 per image), each with one question and its answer, checked by eye on the photo before
  the run: 27 objects or animals, 10 colours, 10 counts (one to six items), 10 texts (a word or number on a sign or the
  road), 5 scenes. Photos with a brand, a flag or a person as the subject were left out, and so were colours between
  two names and counts with overlapping items. The images are not in the repository; `fetch.py` downloads them, holds
  each to its SHA-1 and scores an upright RGB copy no longer than 1,280 px, so both families encode the same patches.
- Asked through `POST /v1/chat/completions` of a server started with `make start` (Flash-Next with `VISION=1`, the 27B
  as its profile serves it), one at a time, the question then the image, greedy, thinking off, `max_tokens` 64. The
  27B's run started at 01:43, Flash-Next's at 03:55 (`H:/fn-vision-560/known-runs/`).
- The matching rule, fixed before the run: a reply is right when, for every needle group of the image, one of its
  spellings appears in the lowercased reply as a whole word; the canaries' needles are their GPU test's (47; red; 3
  or three; config.yaml and not found).
- The alarm rules, fixed before the run: a canary Flash-Next answers wrongly; Flash-Next clearly below the 27B, meaning
  the paired bootstrap 95% interval of the accuracy difference (Flash-Next less the 27B) lies wholly below zero.

| | Flash-Next | the 27B |
|---|---:|---:|
| right, all images | 66 of 66 | 66 of 66 |
| canaries | 4 of 4 | 4 of 4 |
| objects / colours / counts / texts / scenes | 27 / 10 / 10 / 10 / 5 | 27 / 10 / 10 / 10 / 5 |

Paired: Flash-Next only 0, the 27B only 0; difference 0.0, 95% interval [0.0, 0.0]; McNemar p 1.0. Alarms: none
(`known-runs/compare.json`). The grouped companion was not run on the set; it answered the canary 8 of 8 (#560).

**3. The decode gate** (spec #555's phase gate, once at the end).
- Four legs, ABBA (vision off, on, on, off), each a `make start MODEL=flash-next IDLE_UNLOAD=0 LOOP_GUARD=off
  MAX_CONTEXT=589824 HOST_MARGIN_BYTES=0 VRAM_BUDGET=27G` with `VISION=0` or `VISION=1` and nothing else different,
  the RTX 5090 profile's three lanes, MTP off; no image sent in any leg.
- `tools/flash-next-vision/decode_gate.py run`: nine coding prompts, greedy, thinking off, `ignore_eos` with 512
  tokens each; C=1 the nine one after another (each request's tokens after the first over its streaming time from
  the first content chunk to the last, the mean of the nine); C=3 the nine through three clients at once, two rounds
  (completion tokens over the round's wall time, the mean of the two).
- The rule, fixed before the run: at C=1 and at C=3 alike, each adjacent pair's on/off ratio (on-1 over off-1, on-2
  over off-2) at least 99%; the expert cache equal in every leg; every prompt's text the same in every leg.
- Two earlier attempts are not the gate: the first stopped at its first vision-on start, 625 MiB short of the 4 GiB
  host margin (`H:/fn-vision-560/gate-attempt1-refused/`), and the second, every leg at `HOST_MARGIN_BYTES=0`,
  planned four different expert caches because the card's free VRAM moved between legs
  (`H:/fn-vision-560/gate-attempt2-budget-drift/`), so the run below pins `VRAM_BUDGET=27G` in every leg.

- The result (`H:/fn-vision-560/gate/verdict.json`; the release build of f749cea0, 2026-11-10): C=1, tokens per
  second 92.5 (off-1), 92.4 (on-1), 92.5 (on-2), 92.5 (off-2), the pairs 99.87% and 100.02%, pooled 99.95%; C=3,
  100.9, 100.9, 101.0, 101.0, the pairs 100.04% and 99.99%, pooled 100.01%. The expert cache 16,531,515,984 bytes
  in every leg; every prompt's text the same in every leg. **The gate passes.**

**4. MTP on image prompts** (#564, #565).
- `flash_next_vision_mtp_gpu` on the W8 companion (`.scratch/555/gpu/f_mtp.log`): greedy MTP-3 on against off on the
  four canary images, one lane, BF16 KV, 64 tokens a prompt. 3 of 8 image runs split from MTP off, each at a prefill
  route gap of one BF16 ulp (0.125), within ADR 0056's near tie (a pick gap within 16 ulps, teacher-forced), which the
  test holds a split to since the owner's call of 2026-11-10, and within the one-ulp rule it held before. Acceptance:
  66.0% on image prompts (177 of 268 drafts, 2.79 tokens a round) against 54.2% on text prompts (154 of 284, 2.51).
  The same test on the grouped companion (`s_mtp2.log`): 4 of 8 runs split at 1-6 ulps, acceptance 66.4% (2.80).
- `flash_next_vision_speculative_gpu`, on the grouped companion (`s_specimg.log`): after an image, rounds whose drafts
  are all rejected, all accepted or the head's own commit the same text and leave the same state as rounds with no
  drafts, bit for bit, and a head's load's plain rounds equal a headless load's. The routes' drift, median BF16 ulps
  on image prompts against text controls of the same length: plain round against prefill 7.50 against 3.75 (2.00x),
  verify round against prefill 8.94 against 3.50 (2.55x), verify against plain, teacher-forced, 5.36 against 2.50
  (2.14x).
- #565's arms against the same text controls, in the same three ratios (`s_arms.log`, `s_armg.log`): the text's own
  rows placed where the image's sit, through the image's path, 1.87x / 2.00x / 1.80x; those rows at text positions
  2.07x / 2.03x / 2.19x, with the text's n-gram rows 2.33x / 2.49x / 1.55x, and through the embedding lookup bit for
  bit what the pool's scatter writes; the text controls with only the vision start and end tokens inserted, no image,
  2.00x / 2.18x / 1.80x. The reference implementation's multimodal forward, read against ours on the CPU, hashes the
  placeholder ids into the n-gram rows as ours does, puts the vision rows into every stream and treats no modality
  apart (#565).

## Finding

Observed:
- With every matrix W8, Flash-Next's served tower is 0.0586 in pooled relative L2 from its BF16 tower, the 27B's
  0.183, lower on 32 of 32 images: at most the 27B on the pre-registered bar, mean −0.130 [−0.149, −0.114].
- In the 27B tower's grouped formats the same tower was 0.305, above the bar (+0.123 [+0.112, +0.134]), with its
  controls at 0.86-1.53: precision, not a layout fault.
- In both encodings the encode computes what the stored words say (333 of 333 objects the recipe's; the served rows
  0.05 from the stored-weight rows, the 27B's 0.03). The grouped excess was the recipe's: the weights quantize alike in
  both towers, their errors track through the 27 blocks, and at Flash-Next's merger they grow (0.362 after its fc1
  against the 27B's 0.250), 31% of the output error one vector every row shares, against 4%.
- Known answers: 66 of 66 for Flash-Next and for the 27B, every canary right, no alarm.
- At the RTX 5090's settings a vision load under `host` plans every VRAM line the text-only load plans, the expert
  cache among them, and holds its tower (482,705,408 bytes) and its pool (167,772,160 at the default envelope) in
  host memory. The item bound is the full 16,384 at an 8,192- and at a 1,024-token chunk, the second with the ring.
- On image prompts greedy MTP on equals off up to one-ulp near ties, and MTP accepts more drafts than on text (66.0%
  against 54.2%); a verify round leaves the state a plain round leaves, bit for bit. The routes drift about twice as
  far on image prompts as on text, and as far on text carrying only the vision start and end tokens.

Inferred:
- The re-encoded tower's remaining error is mostly the encode's arithmetic, not its stored formats: 0.053 served
  against stored-weight rows beside 0.029 stored-weight against BF16 (the two are not additive; together about the
  0.059 measured).
- A retrained tower does not inherit its sibling's quantization recipe: the same formats cost Flash-Next 1.7 times the
  27B's error, at a merger the sibling does not share. The tower check against the BF16 tower from shard 1 is what saw
  it; the known answers did not separate the encodings (the grouped one answered the canary 8 of 8, both families
  answer the set at its ceiling).
- The drift on image prompts is a property of the model's inputs, the two special tokens, not of the image path: no
  piece of the path, replaced by the text's, brings it back to the text's, and the two tokens alone carry it. The
  re-encode does not touch it, since the arm that carries it has no image.

## Implications

- Neither quality check raises an alarm on the W8 companion: Flash-Next's companion ships with every matrix W8, and
  the KLD against BF16 on image prompts does not run. The 27B's tower and its recipe stay as they are. ADR 0087's
  acceptance rests on these and on the decode gate's line above.
- A new tower, or a tower's new recipe, gets the same check against its BF16 tower before it ships, on the same
  pre-registered bar; a sibling's recipe is evidence for nothing.
- The image MTP test holds a split to ADR 0056's near tie, as the text path's wide verify rounds are held; speculation
  that never changes an answer is #567's.

## Limits and unknowns

- One host, one card, Windows only.
- The tower is compared on items of at most 2,025 merged tokens (the CPU reference reaches 4,096); the item bound's
  16,384 is not compared against BF16. The encode's arithmetic at that size is checked against itself: `host` against
  `device` byte for byte (#557's `ignis_kernel_media_encode_test`, the 27B's residency acceptance).
- The reference computes in float32 from the BF16 weights. The split of the error into the stored formats' part and
  the encode's arithmetic's is the diagnosis's two measures, which are not additive.
- The raw relative L2 bar is sensitive to a tower's shared component: the 27B's rows carry a larger one in the
  denominator, so its raw error reads lower than its centred one (0.181 against 0.286). The bar is the raw one, fixed
  before the run; the W8 companion's stored-weight rows are below the 27B's on the centred one too (−0.267
  [−0.297, −0.240], `flash-next-w8-stored.json`), its served rows measured raw only.
- The known-answer set is small and its questions short: it shows a gross loss, not a few points of accuracy, and at
  66 of 66 for both it bounds nothing finer.
- The MTP figures are four canary images at 64 tokens; the drift study ran on the grouped companion, before the
  re-encode.
- The decode gate has no image in flight: an image's encode inside a running decode is not gated (a 16,384-token
  encode took 3.65-3.69 s).
- The 27B's tower re-encoded with every matrix W8 was not measured.

## Follow-ups

- #565: the drift experiment and the image MTP test's rule, closed when they reach `main`.
- #567 (parked): routes that agree bit for bit, so greedy speculation never changes an answer.
