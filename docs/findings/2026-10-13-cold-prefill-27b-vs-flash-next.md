# Cold prefill on the same text: the 27B at 5.2-9.7K tok/s, Flash-Next at 3.1-3.4K, held by the experts it streams over PCIe Gen 3

- Kind: experiment
- Status: corrected
- Observed: 2026-10-13
- Last verified: 2026-10-13
- Scope: serving / cold prefill rate at the make defaults, the 27B and Flash-Next on the same prompts; Flash-Next's expert bytes per prompt
- Topic: flash-next
- Related: [upstream quick wins](2026-09-24-upstream-quick-wins-ab.md) (the README's 27B cells), [Flash-Next prefill chunk decomposition](2026-10-07-flash-next-prefill-chunk-decomposition.md), [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md)
- Superseded by: none
- Corrected by: [The 27B's prefill attention without prompt planes](2026-10-21-prefill-attention-without-prompt-planes.md), [The 27B's cold prefill at 396K tokens](2026-10-31-cold-prefill-at-396k-tokens.md)
- Summary: On the same cold text Flash-Next prefills a flat 3.1-3.4K tok/s, held by 2.2-2.4 MB of experts per token over PCIe Gen 3; the 27B's 9.6K/7.6K/5.2K tok/s at 14K/46K/105K rose after #335 (7,853 at 105K on 2026-10-31).

> **Corrected 2026-10-21.** The 27B's cold prefill rose after #335, +30% at 46K and +53% at 105K ([prefill attention without prompt planes](2026-10-21-prefill-attention-without-prompt-planes.md)), 7,853 tok/s at 105K on 2026-10-31 ([cold prefill at 396K](2026-10-31-cold-prefill-at-396k-tokens.md)). The Flash-Next side stands.

## Question

The README gives the 27B's cold prefill: 6,600 tok/s at 46K and 4,600 at 105K. A reader asked what Flash-Next does. How fast does each model prefill the README's cells today, on the same text?

## Evidence

**Setup.**
- Build: release, branch `host-margin-config` at 32e30ed (main plus three host-plan and make commits, none on the prefill path).
- Each server ran alone on the card, one load per model, with exactly the flags `make config` prints, plus `IDLE_UNLOAD=0`:
  - Flash-Next (`MODEL=flash-next`): hq-e8-2b, 262,144 context, 3 lanes, `--prefill-chunk 8192`, prompt reuse on, no speculation;
  - 27B: hq-e8-2b, 524,288 context (YaRN x2), `--prefill-chunk 1024`, DFlash2 with 7 draft tokens.
- Link: PCIe Gen 3 x16; the host is the limit (i9-10900K on Z490), not the card.
- Client (`client.py`):
  - the README's definition: `prompt_tokens / TTFT`, TTFT to the first content token;
  - streamed, `max_tokens` 4, thinking off, the request names no model;
  - a warm-up of 8,224 tokens, then 14K, 46K and 105K, twice;
  - text: slices of this repo's docs, Rust and CUDA sources, each opened by its own nonce, sized with `/v1/tokenize`;
  - the same `slices.json` for both models, so both saw the same text. Their `prompt_tokens` agree on every request, and `cached_tokens` was 0 on every one.
- Nothing else reached either server during the cells: Flash-Next's log admits one outside request before the run, plus the client's seven.

**TTFT and rate** (two reps):

| prompt tokens | 27B TTFT | 27B tok/s | Flash-Next TTFT | Flash-Next tok/s | Flash-Next expert bytes over the link |
|---:|---:|---:|---:|---:|---:|
| 8,224 (warm-up) | 0.86 s | 9,513 | 2.89 s | 2,841 | 21.2 GB |
| 14,282-14,284 | 1.48 / 1.48 s | 9,625 / 9,679 | 4.63 / 4.52 s | 3,082 / 3,160 | 34.5 / 35.1 GB |
| 45,598-45,612 | 5.96 / 5.99 s | 7,659 / 7,610 | 13.30 / 13.44 s | 3,429 / 3,393 | 105.7 / 106.8 GB |
| 105,272-105,280 | 20.35 / 20.43 s | 5,173 / 5,152 | 32.44 / 31.00 s | 3,245 / 3,396 | 232.6 / 232.9 GB |

- The bytes are the per-request delta of `ignis_expert_bytes_moved_total{phase="prefill"}`. The 27B moves none: its weights are resident.
- `ignis_expert_residency_stall_seconds_total{phase="prefill"}` grew 0.18-1.46 s per request. That counts only the demand copies on the compute stream, not the prefetch the compute waits on.

Raw data: `.scratch/fn-prefill-cells-2026-10-13/` in the main checkout (untracked): `client.py`, `slices.json`, `fn.jsonl`, `27b.jsonl`, both servers' logs and `make config` output.

## Finding

Observed:

1. **Flash-Next prefills at 3.1-3.4K tok/s from 14K to 105K tokens**, flat with length. The 27B goes from 9.6K at 14K to 5.2K at 105K.
2. **The 27B is 3.1x Flash-Next at 14K, 2.2x at 46K and 1.5-1.6x at 105K.**
3. **Flash-Next moves 2.2-2.4 MB of experts per prompt token**: 21 GB over the 3 traversals of the 8.2K prompt, 106 GB over the 7 of a 46K one, 233 GB over the 14 of a 105K one (the log's `prefill_chunks_consumed`).
4. **The 27B's cells are above the README's**: 7.6K tok/s at 46K (6.6K there), 5.2K at 105K (4.6K there). The README's ran at 262,144 context on 2026-09-24's main.

Inferred:

- **On this host Flash-Next's prefill is held by the link.** At the copy engine's measured 11.7-12.3 GB/s, 233 GB keeps the link busy ~19-20 s of a 31-32 s TTFT, most of it under compute. [The decomposition](2026-10-07-flash-next-prefill-chunk-decomposition.md) measured an 8192-token chunk's compute (1.72-1.79 s) and its expert stream (1.72-1.87 s) as equal, with 19-23% of the chunk waiting on copies alone.
- **Its rate is flat because every chunk re-streams the experts.** Each 8192-token chunk touches ~90% of the pool, so the cost is per chunk, and the rate is the chunk's.
- **A Gen 4 or Gen 5 host shrinks the copy side ~2x or ~4x, and the chunk becomes compute-bound, which is about where it already is.** Copies and compute are equal per chunk, so the gain is bounded, not 2-4x. Here a 46K prompt runs ~2.4 s per 8192 tokens.
  - With the copies hidden and the n-gram gather (~0.46 s per chunk, card idle) left as is: 8192 / (1.75 + 0.46) s ≈ 3.7K tok/s.
  - With the gather pipelined too: 8192 / 1.75 s ≈ 4.7K tok/s.
  - Not measured: no faster host was available.

## Implications

- To answer "how fast does Flash-Next prefill": ~3.1-3.4K tok/s on this host, against 5-10K for the 27B. That is a PCIe Gen 3 number: the expert stream ties the compute chunk by chunk, so the link sets the pace.
- A measurement on a Gen 4 or Gen 5 host would test the inferred 3.7-4.7K range and say how much of the gap is the link. A card that holds every expert in VRAM (48 GB or more) removes the stream altogether, which is a different reference point.
- The README's 27B prefill sentence is now conservative (6.6K / 4.6K against 7.6K / 5.2K here).

## Limits and unknowns

- One load per model, two reps per cell, one text family (repository sources).
- The 27B ran at the make default of 524,288 context; the README's cells used 262,144.
- TTFT includes HTTP, tokenization, the n-gram gather and the scheduler; it is the rate a client sees, not the card's.
- The Gen 4/5 numbers are inferred from this host's measured bytes and compute, not measured.
