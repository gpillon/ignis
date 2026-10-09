# At a 1 GiB VRAM headroom a browser drawing a page makes WDDM page the model out; at 1.5 GiB neither model moves, and Flash-Next's cache is 2.5 GiB larger than at 4G

- Kind: experiment
- Status: current
- Observed: 2026-10-09
- Last verified: 2026-10-09
- Scope: server / the derived VRAM budget's default headroom (`--vram-headroom-bytes`, `DEFAULT_VRAM_HEADROOM_BYTES`); make / Flash-Next's `VRAM_HEADROOM`; serving / one-lane decode on the 27B and Flash-Next under Windows (WDDM) with desktop GPU clients
- Related: [ADR 0030](../adr/0030-device-memory-reserved-at-load.md), spec [vram-budget/01](../specs/vram-budget/01-vram-budget.md), spec [flash-next/04](../specs/flash-next/04-the-flash-next-forward-and-serving.md) (story 11, AC 10)
- Superseded by: none

## Question

The owner set Flash-Next's make headroom from 4G to the server's 1G (main
d039ddc), to give the expert cache the difference. Served from the
Playground, Flash-Next then decoded at 4.6 tok/s, and the 27B at ~40 tok/s.

1. Why, with ~1.5 GB of VRAM still free?
2. Which headroom keeps both models at speed with the owner's desktop open?

## Evidence

**Setup.**
- Host: RTX 5090 (32,607 MiB) driving the Windows 11 desktop, with ~40 GPU
  client processes (dwm, Chrome/Edge, Slack, Discord, Telegram, WhatsApp,
  Claude desktop, iCUE...). PCIe Gen 3 x16.
- Binary: release build of `8b67773` (the `experts8-base` worktree); the
  27B's and Flash-Next's `make config` flags, with only
  `--vram-headroom-bytes` changed.
- Flash-Next ran with `--kv-host-pool-bytes 1G`, for the host's free RAM.
- One request, repeated: a 2K-token Python module to explain, greedy,
  thinking off, 512 tokens. On the 27B the DFlash2 rounds and the acceptance
  are identical across runs (68 rounds, 444/471), so ms per round compares
  equal work.
- "UI open" means the Playground open in Chrome, served by Vite on :5174
  against the server.
- Scripts: `ctxbench.py` (session scratchpad); records are in each server
  log's `ignis.request.done` events, in the main checkout's `.scratch/`
  (`owner-test-*.server.log`, `owner-test-probe.jsonl`).

**27B: one server, one binary, the browser toggled without a restart (1 GiB
headroom).**

| state | VRAM used | ms per round | tok/s | ITL max |
|---|---:|---:|---:|---:|
| UI open | 31,078-31,084 MiB | 54-76 | 100-134 | 128-207 ms |
| browser closed | 30,991 MiB | 18.9-20.0 | 377-399 | 45-53 ms |
| UI reopened | 31,078 MiB | 68.7-75.5 | 100-110 | 128-160 ms |

**27B by headroom, UI open.**

| headroom | VRAM used | ms per round | tok/s | ITL max | KV pool |
|---|---:|---:|---:|---:|---:|
| 1 GiB | 31,08 GB | 69-76 | 100-110 | 128-160 ms | 8.70 GB |
| 1.5 GiB | 30,60 GB | 16.4-20.3 | 371-460 | 18 ms | (larger than 4G's) |
| 4 GiB | 28,09 GB | 18.1-21.6 | 348-417 | 19-20 ms | 5.44 GB (590K tokens) |

**Flash-Next by headroom, UI open.**

| headroom | expert cache | tok/s (code prompt) | ITL mean / max | tok/s (fusion A-B-A's 3 prose texts) |
|---|---:|---:|---:|---|
| 1 GiB | 18.3 GiB | **4.6** | - | - |
| 1.5 GiB | 17.8 GiB | 90.0 | 10.95 / 24-26 ms | 105.0 / 92.5 / 100.1 (mean 99.2) |
| 4 GiB | 15.3 GiB | 79.6 | 12.4 / 32-33 ms | 98.1 / 86.9 / 98.3 (mean 94.4) |

- The same 4 GiB binary on a quiet host overnight (the step-8 A-B-A's base
  legs): 112.3 / 93.5 / 104.9.

**A false lead, recorded so it is not repeated.** Before the browser toggle,
main's 02:31 release binary measured 70-102 tok/s on the 27B, and a
2026-09-28 binary 361-451. That read as a build regression. It was not: the
`experts8-base` binary of the same code gave 264-388 a minute later, and then
134 with the UI open. The three binaries were measured at different moments
of the desktop's state, not interleaved.

## Finding

Observed:

1. **At a 1 GiB headroom the slowdown follows the desktop, not the server.**
   - The same request on the same server goes from ~19 to ~70 ms a 27B decode
     round when the browser opens, back when it closes, and again when it
     reopens.
   - At full GPU utilisation the power is low (Flash-Next ~152 W against
     ~300 W in a healthy decode): the SMs wait on memory.
   - This is the 2026-09-22 collapse (memory: 200 -> 25 tok/s at the VRAM
     ceiling), reproduced on demand.
2. **At 1.5 GiB neither model moved with the UI open.**
   - The 27B ran at 371-460 tok/s with an 18 ms ITL max.
   - Flash-Next's ITL max was 24-26 ms.
3. **Flash-Next at 1.5 GiB is +13% over 4 GiB on the code prompt and +5% on the
   prose texts.** It has 2.5 GiB more expert cache: ~0.58 ms a token per GB
   on the code prompt, above the decode-headroom study's 0.25-0.4 estimate.
4. **The open UI costs ~9% even at 4 GiB** (prose texts: 94.4 against 103.6 on a
   quiet host), with no paging sign (ITL max unchanged): the browser takes GPU
   time slices.

Inferred, not measured:

- **The mechanism is WDDM residency.** A desktop client's allocation, with the
  card near full, makes the video memory manager evict part of the model's
  allocations to system memory, and the GPU reads them over PCIe. An ETW
  (`wpr` GPU profile) trace was not taken. The per-process "Local Usage"
  counter stayed at 29.04 GiB through the slowdown, so it counts committed
  bytes, not residency.
- **Flash-Next collapses further than the 27B** (4.6 against ~100 tok/s), and
  the cause is not separated. Flash-Next has 38.4 GiB of pinned host memory
  mapped for the GPU (non-local), against the 27B's 11.6, and issues expert
  copies every round.

## Implications

- **The server's default headroom is 1.5 GiB since 2026-10-09**
  (`DEFAULT_VRAM_HEADROOM_BYTES`). Make passes none for either model, so both
  take it.
- `VRAM_HEADROOM=4G` is the safe choice on a desktop with heavy GPU clients:
  games, 4K video, many browser tabs.
- **Speed measurements on this host need the desktop's state held fixed, and
  A and B interleaved.** Sequential runs at different moments can differ 3-4x
  with no code change.

## Limits and unknowns

- One host, one desktop session; the threshold between 1 and 1.5 GiB depends
  on what the desktop holds. A heavier client can still cross 1.5 GiB.
- One prompt for the headroom comparison on each model, plus the three prose
  texts on Flash-Next. No three-lane run.
- No ETW trace, so the mechanism is inferred.

## Follow-ups

- An ETW trace of a slow round at 1 GiB, to confirm the paging and to
  separate the role of Flash-Next's pinned host memory.
- Watch for the collapse in serving: a decode round several times its usual
  time at full utilisation is the signature, and the Monitor could flag it.
