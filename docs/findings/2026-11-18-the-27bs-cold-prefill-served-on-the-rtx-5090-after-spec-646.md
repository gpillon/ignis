# The 27B's cold prefill served on the RTX 5090 after spec #646: a cold 16K prompt 18.2% sooner, a cold 64K prompt 13.7%, the first request after a load 29.3%, decode unchanged

- Kind: experiment
- Status: current
- Observed: 2026-11-18
- Last verified: 2026-11-18
- Scope: prefill / the 27B NVFP4 artifact on the RTX 5090 (Windows, PCIe Gen 3), main at 5926d31f against the integration build of spec #646 at ba23cefa, one build each, the parts' measurement switches as controls
- Topic: prefill
- Related: https://github.com/gpillon/ignis/issues/654,
  https://github.com/gpillon/ignis/issues/646,
  [ADR 0100](../adr/0100-the-27b-cuts-a-reusable-prompts-tail-once-at-the-generation-opener-on-every-card.md),
  [ADR 0101](../adr/0101-on-windows-the-vram-plan-reserves-the-w4a4-tma-products-descriptor-slots-as-a-line-of-their-own.md),
  [the prompt-reuse tax](2026-09-18-prompt-reuse-tax-on-short-ttft.md),
  [the GDN prefill on Blackwell](2026-11-08-blackwell-gdn-prefill-on-ignis-kernels.md),
  [the 27B's prefill chunk at 2,048](2026-10-21-the-27b-prefill-chunk-at-2048.md)
- Superseded by: none
- Summary: RTX 5090, 27B NVFP4, served: cold 16K 1.515 to 1.239 s (-18.2%), cold 64K 6.798 to 5.865 s (-13.7%), first request after a load 1.755 to 1.240 s, decode 389.8 vs 389.6 tok/s; text moves only through the single tail cut.

## Question

Spec #646 took the 27B's cold prefill apart on the RTX 5090 and built five changes: the W4A4 products' activations laid out K-tile by K-tile (#652), SwiGLU in the gate and up product's epilogue, quantizing straight to the down product's input (#651, #653), the TMA descriptors kept on the device on Windows (#650), one tokenization and no grammar built for a request without tools (#647, #648), and one tail cut at the generation opener on every card (#649). The prototypes were measured in two halves, never all together and never served from one build. #654 asks what a user sees with every part in, against main:

- time to first token of a cold 16K and a cold 64K prompt, and of the first request after a load;
- decode tokens per second on a greedy 8K-prompt cell;
- whether the greedy text equals main's with the single tail cut off by its switch, and is stable across launches with it on;
- what the reuse of a conversation does where the tail cut changed.

## Evidence

**Builds.** Main at 5926d31f and the integration build at ba23cefa, each `make build UI=0` (release, CUDA), one build each. The new path is the default of the integration build; each part's measurement switch (`IGNIS_SECOND_TAIL_CUT=1`, `IGNIS_TMA_DESCRIPTOR_CACHE=0`, `IGNIS_PREFILL_SWIGLU=split`, `IGNIS_W4A4_ACTIVATION_LAYOUT=token`) gives main's path from it.

**Conditions.** RTX 5090 under Windows, PCIe Gen 3 x16; across the legs the power draw peaked at 577-587 W and the SM clock's median was 2,400-2,490 MHz. The machine was quiet: no builds, no other GPU process. Flags on both arms: the 27B NVFP4 artifact, `--model-kv-format hq-e8-2b`, `--model-max-context 262144` (the context pinned: no adaptation on either arm), `--model-rope-scaling none`, `--model-exception-weights fp8` (named by the flag on both arms; the user configuration file in force on both held the same value, with a VRAM headroom of 1,500M and a host margin of 1G), `--reuse-kv-host-pool-bytes 8G`, DFlash2 with 7 drafts, `--server-system-message-policy merge`, `--server-developer-message-policy inplace`. The load: a prefill chunk of 2,048, a solo width of 2,048, eight decode lanes.

**Prompts.** ignis's own generated word-soup system block followed by a coding request, sized by the server's own tokenizer through `/v1/tokenize`. Cold means a fresh nine-digit seed in every request, so nothing in it is reusable. "16K" is 15,898-16,415 tokens and "64K" is 63,590-65,636. One output token; time to first token measured from the client on the stream. The decode cell is an 8K prompt at temperature 0 run to its stop.

**Launches and order.** One server start per leg. Eight legs in the order M N N M M N N M (M main, N the integration build), then two control legs and two reuse legs. Each leg: the first request after the load (a 16K prompt, reported apart), then 16K and 64K alternated four times, then the decode cell. Per arm: 4 first requests, 12 cold 16K, 16 cold 64K, 4 decode runs. A first attempt whose prompts were miscalibrated (about 21K and 85K) was voided and its legs discarded.

### The cells

| cell | main, median (range) | integration, median (range) | change |
|---|---|---|---|
| first request after a load, 16K | 1.755 s (1.716-1.866) | 1.240 s (1.212-1.304) | -29.3% |
| cold 16K | 1.515 s (1.395-1.614) | 1.239 s (1.166-1.283) | -18.2% |
| cold 64K | 6.798 s (6.400-7.015) | 5.865 s (5.804-6.085) | -13.7% |
| ms per 1K prompt tokens, 16K | 93.0 (87.8-98.5) | 75.6 (73.4-78.2) | -18.7% |
| ms per 1K prompt tokens, 64K | 103.7 (100.6-107.0) | 89.4 (88.5-92.8) | -13.7% |
| decode, 8K prompt, greedy | 389.8 tok/s (367.4-393.3) | 389.6 tok/s (381.6-393.7) | -0.1%, inside the spread |
| TTFT of the 8K decode cell's prompt | 0.869-0.889 s | 0.725-0.763 s | |

The integration build's first request after a load (1.240 s) equals its later ones (1.239 s): no one-time cost is left. On main it is 0.24 s above them. The ranges of the two arms do not overlap on any prefill cell.

### The text

Greedy, the 8K cell, four launches per arm.

- Main: the same text on 4 of 4 launches (3,205 tokens, hash prefix `ac1f9fd4356e`).
- Integration, defaults: the same text on 4 of 4 launches (3,435 tokens, `b92644094e51`), different from main's.
- Control X, the integration build with every switch on main's path (`IGNIS_SECOND_TAIL_CUT=1 IGNIS_TMA_DESCRIPTOR_CACHE=0 IGNIS_PREFILL_SWIGLU=split IGNIS_W4A4_ACTIVATION_LAYOUT=token`): `ac1f9fd4356e`, main's text, and the routes line reads split and token.
- Control Y, the integration defaults with `IGNIS_SECOND_TAIL_CUT=1` alone: `ac1f9fd4356e`, main's text, the routes line reading quant and ktile.

So on this cell the single tail cut is what moves the text: it moves the 27B's split points on Blackwell, and a different split of the prefill can change a token where two are nearly tied (ADR 0007, ADR 0029). The layout, the fused SwiGLU and its quantizing epilogue, the descriptor cache and the host changes are byte for byte main's here. The 8K cell's split in Control Y has no chunk exactly 1,024 tokens wide, where the single SwiGLU rounding would differ from the split route's; a 1,024-wide chunk was not run served and is held by the kernel's byte-identity tests, not by this measurement.

### Reuse (#649)

One 8K conversation, requests in sequence, the integration build against main (tokens cached over tokens in the prompt):

| request | integration | main |
|---|---|---|
| identical retry | 8,172 / 8,176 | 8,148 / 8,152 |
| next turn | 8,172 / 8,263 | 8,148 / 8,239 |
| first turn, its last words before the opener changed | 8,064 / 8,178 | 8,128 / 8,154 |

A retry and the next turn resume from the previous prompt's checkpoint on both builds. The edited request is where the single cut shows: main resumes it at the opener's page floor (127 pages), the integration build at the reuse point before it (126 pages, 8,064 tokens), and the integration build still answers sooner (0.131 s against 0.166 s). The 8,064 is the reuse point, not a measure of the system block's length.

### The load

- The routes line of the integration load names `prefill_swiglu=quant` and `w4a4_activation_layout=ktile`; `ignis.model.loaded` carries `second_tail_cut=false`. Under the controls the line reads `split` and `token`.
- The VRAM plan gained one line, `tma_descriptors`, 632,320 bytes (ADR 0101), and the KV pool went from 14,919 to 14,918 pages of 64 tokens (954,816 to 954,752 tokens). The context was pinned at 262,144 on both arms, so nothing adapted; the CUDA context and the free memory at the start were the same (452,161,536 and 30,515,085,312 bytes).

## Finding

With every part of spec #646 in, served on the RTX 5090: a cold 16K prompt reaches its first token 18.2% sooner (1.515 to 1.239 s), a cold 64K prompt 13.7% sooner (6.798 to 5.865 s), and the first request after a load 29.3% sooner, no longer slower than its successors. Decode is main's (389.8 against 389.6 tokens per second, inside the 367-394 spread). The gain is larger than the prototypes' (-8.7% at 16K, -5.3% at 64K, the first request 1.686 to 1.283 s, the layout alone another -45 ms), which measured the parts in two halves on an earlier main; this is the first measurement of all of them together, and it does not separate them.

Per-part shares are not re-separated served here. The per-part figures are the prototypes' and the spec's (#646), measured in their two halves: the layout's 5.5-11.9% less time per product call in a served prefill and its -45 ms at 16K, the rest's -8.7% at 16K and -5.3% at 64K. The controls separate only what moves the text, not what each part saves in time.

What the model writes: main's text under every switch except the single tail cut, which on the 8K cell gave a different, launch-stable text. A request edited only in its last words before the generation opener resumes from the reuse point before the opener's page floor, one page lower here, and is still sooner.

Not confirmed here: the RTX PRO 6000 (188 SMs), the other card that runs the 27B's NVFP4 artifact on its W4A4 route, still has to confirm the K-tile-major layout and the fused SwiGLU as defaults: the cold 16K and 64K cells, the first request after a load, decode, and the text with the single cut off. The layout's gain comes from the L2 behaviour of the products on the RTX 5090's 170 SMs and is not assumed for another card. The descriptor cache (Windows) and the host changes are not specific to a card's SM count.

Limits: one machine, one artifact, one output token on the prefill cells, at most sixteen requests per cell, FP8 exception weights on both arms; the 46K and 105K figures the guide quotes were not re-measured.
