# The 27B's cold prefill at 396K tokens: 98.7 s, 4,012 tok/s on main, two thirds of it attention

- Kind: experiment
- Status: current
- Observed: 2026-10-31
- Last verified: 2026-10-31
- Scope: serving / the 27B's cold prefill rate at 396,032 prompt tokens and 524,288 context on the RTX 5090 (PCIe Gen 3 host), `ignis-bench ttft` with a 105,472-token calibration cell
- Related: GitHub [#332](https://github.com/gpillon/ignis/issues/332) (point 1), [#342](https://github.com/gpillon/ignis/issues/342), [the prefill attention without prompt planes](2026-10-21-prefill-attention-without-prompt-planes.md), [the 2,048 chunk](2026-10-21-the-27b-prefill-chunk-at-2048.md), [cold prefill 27B vs Flash-Next](2026-10-13-cold-prefill-27b-vs-flash-next.md)
- Superseded by: none

## Question

#332 reports a cold prefill of about 396K tokens with the 27B at 512K context in about 221 s (1.8K tok/s), on a
build from before 0.7.3. The repo's own cells stop at 105K. How fast does main prefill that length, and is the
reported rate what the build of the time would give?

## Evidence

**Setup.**
- Build: main `cf7e7eeb`, `make build UI=0` (release, CUDA).
- Server: `make start MAX_CONTEXT=524288 IDLE_UNLOAD=0 LOOP_GUARD=off UI=0`, every other make default:
  - hq-e8-2b KV, 524,288 context (YaRN x2), DFlash2 with 7 draft tokens, prompt reuse on;
  - the `rtx5090` profile (`auto`), so a 2,048-token prefill chunk;
  - a KV pool of 1,023,040 tokens.
- Card: alone under the GPU check, 2.2-2.3 GiB held by the desktop before the start.
- Client: `ignis-bench ttft --cells 105472,396032 --samples 2 --corpus bench_corpus.ids`.
  - Prompts are cut from the 65,536-id bank, tiled about six times at 396K.
  - Requests are streamed, thinking off, `max_tokens` 8.
  - TTFT is taken to the first content token; rate = prompt tokens / TTFT.
  - One warm-up per cell, then the samples, each prompt of the run its own.
  - The bench proved every sample cold (`all cold`, computed prefill = prompt length).
- Telemetry (`nvidia-smi`, 1 s):
  - VRAM peak 31,183 of 32,607 MiB, so the card was not oversubscribed;
  - while busy, 571 W mean (the 575 W limit) and SM clock 2,518 MHz mean;
  - host CPU load 21-93%, median 40%, from two other sessions' CPU work.

**TTFT** (one launch):

| cell | warm-up | samples | median | tok/s (median) | server's admission line | prefill traversals |
|---:|---:|---:|---:|---:|---:|---:|
| 105,472 | 13.70 s | 13.49 / 13.37 s | 13.43 s | 7,853 | 13.06-13.37 s | 54 |
| 396,032 | 99.26 s | 98.84 / 98.57 s | 98.71 s | **4,012** | 97.2-98.0 s | 196 |

The README's 105K cell, 7,865 tok/s from #335's driver at the 2,048 chunk, agrees with this one within 0.2%.

Raw data: `.scratch/measure-332-373/` (untracked):
- `ttft-332.json`, the record;
- `server-332.log`, the server log;
- `gpu-332.csv` and `cpu-332.log`, the telemetry.

## Finding

Observed:

1. **Main prefills 396,032 cold tokens in 98.7 s, 4,012 tok/s**, the three runs within 0.7% of each other.
2. **That is 2.24x the rate #332 reports** (1,792 tok/s for 396K in 221 s).
3. The rate falls with length, from 7,853 tok/s at 105K to 4,012 at 396K: 127.3 µs per token, then 249.2.

Inferred (a two-point fit, linear in the keys already seen):

- **Per token, the cost is a fixed ~83 µs plus ~0.84 ns per key already seen.** The 0.84 ns agrees with #335's
  kernel bench (0.86-0.92 ns per key and token over 16 layers); the 83 µs agrees with #342's fixed part (~80 µs).
- **At 396K the attention is two thirds of the prefill** (~166 of 249 µs per token); at 105K it is a third.
- **The reported 221 s is about what the build of the time gives.**
  - The same fit on the 2026-10-13 cells (main `32e30ed`, chunk 1024, the vendored prompt-plane attention) gives
    ~83 µs plus 2.10 ns per key.
  - At 396,032 tokens that is ~499 µs per token: ~198 s, ~2,000 tok/s.
  - The report is 12% slower than that. Its measurement (the whole request, not TTFT; see Limits) can account for
    some of the 12%.

## Implications

- To answer "how fast is a cold prefill at ~400K": about 100 s and ~4,000 tok/s on this card at the make defaults.
  The default 2,400 s request deadline (ADR 0058) covers it many times over.
- At long context the prefill lever is the attention, not #342's fixed part: past ~200K tokens the attention is
  most of the cost. At the 105K cells the README quotes, the fixed part is still two thirds.
- A host's PCIe generation does not enter the 27B's rate: its weights are resident and the card ran at its power
  limit. A card held to a lower power limit, or a desktop that pushes VRAM to the ceiling, would read slower.

## Limits and unknowns

- One launch, two samples and a warm-up per cell. The fit uses two lengths only, and nothing between 105K and 396K
  was measured.
- The user's build was not rebuilt and re-run, so the ~198 s figure for it is an extrapolation.
- The user's method is known only from the report. Their 30 s → 900 s timeout fix points to non-streamed requests
  timed whole, so their 221 s likely includes the answer's generation and the client's upload. Their repeat of the
  probe took 1.6 s in all, which bounds that part to a couple of seconds.
- Not measured: the 917,504-token `auto` context, the 1,024 chunk on this build, and a prompt of natural text
  rather than a tiled bank. Dense attention's cost does not depend on content.
