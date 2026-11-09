# Flash-Next's images checked without the BF16 checkpoint: the encoded tower against its BF16 tower, known answers beside the 27B's, and the decode gate

- Kind: experiment
- Status: current
- Observed: 2026-11-09
- Last verified: 2026-11-09
- Scope: Flash-Next's vision companion (`qwen3_8_flash_next_vision-v2.ninfer`) and the 27B's tower, each encoded by the
  media encode under the `host` residency and answered through the server's HTTP API, on the RTX 5090 (PCIe Gen 3
  x16 host, Windows/WDDM); Flash-Next's decode with vision on and off
- Topic: vision
- Related: [ADR 0087](../adr/0087-flash-nexts-vision-encoder-ships-in-a-companion-container-and-encodes-in-memory-its-prefill-already-reserves.md),
  [ADR 0075](../adr/0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md),
  [the vision residency's acceptance](2026-10-30-vision-residency-acceptance.md),
  [the NVFP4 converter reproduces the served 27B](2026-11-03-nvfp4-converter-reproduces-the-served-27b.md),
  https://github.com/gpillon/ignis/issues/560, https://github.com/gpillon/ignis/issues/555
- Superseded by: none
- Summary: DRAFT, figures pending: Flash-Next's encoded tower [TBD] its BF16 tower against the 27B's [TBD]; known answers [TBD] of [TBD] beside the 27B's [TBD]; decode with vision on [TBD] of off at C=1, [TBD] at C=3, RTX 5090.

> **Draft (2026-11-09).** The tools, the method and the rules below were written before any run; every `[TBD]` is a
> figure the runs fill. The file's date is the day the method was fixed: if the runs land on another day, the file
> is renamed to their date (and `Observed` with it) before the first figure is committed.

## Question

ADR 0087 admits Flash-Next's images without the 335 GiB BF16 checkpoint on three checks beside the positions' fixture
(#558): is the encoded tower as close to its BF16 tower as the 27B's is to its own, measured the same way; how does
Flash-Next answer images with known answers beside the 27B; and does a vision load, with no image in flight, decode at
least 99% as fast as a text-only one at one lane and at three (#555's phase gate)? An alarm from the first two goes to
the owner, who decides whether a KLD against BF16 on image prompts runs.

## Evidence

**Setup.**
- Build: `555-integration` at [TBD], `make build CUDA=1` in [TBD]; the vision companion
  `qwen3_8_flash_next_vision-v2.ninfer` (284,229,120 B, sha256 `d97353d4…`, weights id `vision-w8q4q5q6-bf16-de4b8e4`),
  pinned to the main container `12e3b265…`; the 27B `qwen3_8_27b_nvfp4full-v2.ninfer`.
- The BF16 towers, each the first shard's 333 `model.visual.*` tensors fetched by HTTP range, no other byte of either
  checkpoint read:
  - Flash-Next: `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d`, `model-00001-of-00131.safetensors`, 897,862,112 bytes from
    offset 142,293,832, sha256 `665127be…` (`tools/flash-next-vision/convert.py fetch`);
  - the 27B: `Qwen/Qwen3.8-27B` at `1d4bf0f2`, `model-00001-of-00018.safetensors`, 921,460,192 bytes from offset
    3,045,270,360, sha256 `c623c5ec…` (`tools/flash-next-vision/tower_check.py fetch-27b`). The served 27B's tower is
    the converter's from this revision byte for byte ([finding](2026-11-03-nvfp4-converter-reproduces-the-served-27b.md)).
- The commands are in [docs/agents/testing/vision-canary.md](../agents/testing/vision-canary.md#flash-nexts-image-checks-github-560).
  Raw output: `H:/fn-vision-560/` ([TBD]: the dumps, the references, the runs' JSON, the legs' server logs).

**The plan figures** (the plan a load logs, ADR 0087's memory decision):

| | text-only | vision on, `host` |
|---|---:|---:|
| expert cache bytes (`ignis.runtime.flash_next_plan`) | [TBD] | [TBD], equal |
| prefill staging bytes (the expert staging ring) | [TBD] | [TBD] |
| item bound (`vision_item_max_tokens`) | -- | 16,384 [TBD: as logged] |
| tower in host memory (`flash_next_host_plan` `vision_tower_bytes`) | -- | 284,454,912 |
| embedding pool in host memory at 16,384 context (`vision_pool_bytes`) | -- | 83,886,080 |

The tower's two sizes: 284,454,912 bytes are its host spans (each object's span padded to 4 KiB, what the host plan
charges); 283,180,544 bytes are its payload, what a `device` residency places. The 27B's are 297,000,960 and 295,711,648.
The pool is 2,560 columns of BF16 per token of the envelope: 83,886,080 bytes at 16,384 tokens, 167,772,160 at the
default 32,768. The item bound is the largest item whose encode fits the prefill scratch and the staging ring, at most
16,384: before the ring was lent to the encode (#558), the prefill scratch alone held 8,192 (1,016,074,496 bytes, the
CTest run of 2026-11-09 at c00f2e49); with the ring, #558's GPU test expects 16,384 at the 8,192 and the 1,024 chunk
([TBD]: the figure the runs' plans log).

**1. The encoded tower against the BF16 tower** (spec #555's seam 2).
- Images: the vision canary's four; sixteen synthetic ones drawn by the dump test (rings, checkers, stripes, gradients
  with blocks, 32 x 32 to 1,440 x 1,440, square to 1:17); twelve of the known-answer photos at a longer side of 384,
  640, 896 or 1,280 px (`fetch.py --tower`: every fifth photo of the manifest's order). 32 items, every one at most
  4,096 merged tokens, so the reference runs on the CPU.
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

| | Flash-Next | the 27B |
|---|---:|---:|
| images | [TBD] | [TBD] |
| pooled relative L2 | [TBD] | [TBD] |
| median of the images' relative L2 | [TBD] | [TBD] |
| median row relative L2, median over images | [TBD] | [TBD] |
| centred cosine, mean over images | [TBD] | [TBD] |
| control within the item, relative L2, median | [TBD] | [TBD] |
| control across images, relative L2, median | [TBD] | [TBD] |

The bar: Flash-Next less the 27B, mean [TBD], 95% interval [[TBD], [TBD]], Flash-Next lower on [TBD] of [TBD] images;
pooled ratio [TBD]. Per image: `H:/fn-vision-560/tower-check.json` and `.csv` ([TBD]).

**2. Known answers** (spec #555's seam 1).
- The set: `tools/vision-known-answers/manifest.json`, the vision canary's four images and 62 photos from Wikimedia
  Commons, each under public domain, CC0, CC BY or CC BY-SA (its page, author, license, the original's SHA-1 per
  image), each with one question and its answer, checked by eye on the photo before the run: 27 objects or animals,
  10 colours, 10 counts (one to six items), 10 texts (a word or number on a sign or the road), 5 scenes. Photos with a
  brand, a flag or a person as the subject were left out, and so were colours between two names and counts with
  overlapping items. The images are not in the repository; `fetch.py` downloads them, holds each to its SHA-1 and
  scores an upright RGB copy no longer than 1,280 px, so both families encode the same patches.
- Asked through `POST /v1/chat/completions` of a server started with `make start` (Flash-Next with `VISION=1`, the 27B
  as its profile serves it), one at a time, the question then the image, greedy, thinking off, `max_tokens` 64.
- The matching rule, fixed before the run: a reply is right when, for every needle group of the image, one of its
  spellings appears in the lowercased reply as a whole word; the canaries' needles are their GPU test's (47; red; 3
  or three; config.yaml and not found).
- The alarm rules, fixed before the run: a canary Flash-Next answers wrongly; Flash-Next clearly below the 27B, meaning
  the paired bootstrap 95% interval of the accuracy difference (Flash-Next less the 27B) lies wholly below zero.

| | Flash-Next | the 27B |
|---|---:|---:|
| right, all images | [TBD] | [TBD] |
| canaries | [TBD] of 4 | [TBD] of 4 |
| objects / colours / counts / texts / scenes | [TBD] | [TBD] |

Paired: Flash-Next only [TBD], the 27B only [TBD]; difference [TBD], 95% interval [[TBD], [TBD]]; McNemar p [TBD].
Alarms: [TBD].

**3. The decode gate** (spec #555's phase gate, once at the end).
- Four legs, ABBA (vision off, on, on, off), each a `make start MODEL=flash-next IDLE_UNLOAD=0 LOOP_GUARD=off
  MAX_CONTEXT=589824` with `VISION=0` or `VISION=1` and nothing else different ([TBD]: `HOST_MARGIN_BYTES`), the RTX
  5090 profile's three lanes, MTP off; no image sent in any leg.
- `tools/flash-next-vision/decode_gate.py run`: nine coding prompts, greedy, thinking off, `ignore_eos` with 512
  tokens each; C=1 the nine one after another (each request's tokens after the first over its streaming time from
  the first content chunk to the last, the mean of the nine); C=3 the nine through three clients at once, two rounds
  (completion tokens over the round's wall time, the mean of the two).
- The rule, fixed before the run: at C=1 and at C=3 alike, each adjacent pair's on/off ratio (on-1 over off-1, on-2
  over off-2) at least 99%; the expert cache equal in every leg; every prompt's text the same in every leg.

| leg | C=1 tok/s | C=3 tok/s | expert cache bytes |
|---|---:|---:|---:|
| off-1 | [TBD] | [TBD] | [TBD] |
| on-1 | [TBD] | [TBD] | [TBD] |
| on-2 | [TBD] | [TBD] | [TBD] |
| off-2 | [TBD] | [TBD] | [TBD] |

Pairs: C=1 [TBD], [TBD]; C=3 [TBD], [TBD]. Pooled: C=1 [TBD], C=3 [TBD]. Texts identical: [TBD]. Verdict: [TBD].

## Finding

Observed:
- [TBD]

Inferred:
- [TBD]

## Implications

- [TBD: ADR 0087 accepted on these figures, or the alarm the owner decides on.]

## Limits and unknowns

- One host, one card, Windows only.
- The tower is compared on items of at most 4,096 merged tokens, the CPU reference's reach; the item bound's 16,384 is
  not. The encode's arithmetic at that size is checked elsewhere against itself, not against BF16: `host` against
  `device` byte for byte (#557's `ignis_kernel_media_encode_test`, the 27B's residency acceptance).
- The reference computes in float32 from the BF16 weights; it does not split the error into the stored formats' part
  and the encode's arithmetic's.
- The known-answer set is small and its questions short: it shows a gross loss, not a few points of accuracy.

## Follow-ups

- [TBD]
