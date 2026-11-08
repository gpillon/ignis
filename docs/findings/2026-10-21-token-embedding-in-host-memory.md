# The 27B's token embedding in mapped host memory: 1.35 GB of VRAM for ~146K more KV tokens, at ~1 µs per prompt token

- Kind: experiment
- Status: current
- Observed: 2026-10-21
- Last verified: 2026-10-21
- Scope: kernel / the vendored W8 gather reading `text/token_embedding` from mapped pinned host memory; serving / the 27B's VRAM plan, host plan, decode and cold prefill on the RTX 5090 (PCIe Gen 3 host)
- Topic: artifacts-and-gates
- Related: [ADR 0062](../adr/0062-the-token-embedding-lives-in-host-memory.md), [spec 27b-16gb/03](../specs/27b-16gb/03-vram-recovered-on-every-card.md) (ACs 6, 7), GitHub #335, [cold prefill 27B vs Flash-Next](2026-10-13-cold-prefill-27b-vs-flash-next.md) (the prefill driver and prompts), [HQ verify decode cost](2026-09-24-hq-verify-decode-cost.md) (the decode driver), [VRAM headroom and WDDM paging](2026-10-09-vram-headroom-wddm-paging.md)
- Superseded by: none
- Summary: Moving the 27B's token embedding to mapped pinned host memory frees its 1,350,860,800 bytes of VRAM for ~146K more hq KV tokens (+18.5%), at ~1 µs per prompt token over PCIe Gen 3, with decode and cold prefill within main's spread.

## Question

ADR 0062 moves the 27B's 1.35 GB token embedding (W8G32 `[248320, 5120]`, 5,440 bytes a row) from VRAM to a
`cudaHostAllocMapped | cudaHostAllocPortable` allocation that the gather reads over the bus. Does the VRAM come back,
and do decode (1 and 8 lanes) and cold prefill (8K, 46K, 105K) stay within noise of today's main? How does the mapped
allocation look to WDDM, and does a host under load change the answer?

## Evidence

**Setup.**
- Builds: main `7eba161c` and branch `335-host-embedding` at `f3fee218` (main plus the host-mapped placement), both
  `make build CUDA=1 UI=0`, release; the same artifact (`qwen3_8_27b_nvfp4full-v2.ninfer`).
- Each launch ran the flags `make config IDLE_UNLOAD=0 LOOP_GUARD=off UI=0` prints: hq-e8-2b, 524,288 context
  (YaRN x2), `--model-prefill-chunk 1024`, DFlash2 with 7 draft tokens, an 8 GiB KV-RAM arena, the `rtx5090` profile.
- Order: branch-1, main-1, branch-2, main-2 (branch first, so a cold first launch counts against it), then one launch
  of each build with the host under load. One session, the GPU lock held throughout, nothing else on the card.
- Decode: `ab.py` (`--phases c1,c8`), the 2026-09-24 coding prompts (sources frozen at `7eba161c`), greedy, thinking
  off, 512 tokens: eight prompts one at a time (c1), then the eight together, twice (c8).
- Cold prefill: `client.py` with the 2026-10-13 `slices.json`: 8,224 / 14,282 / 45,612 / 105,272 tokens, then
  14,284 / 45,598 / 105,280, each opened by its own nonce; `cached_tokens` was 0 on every request. The 8,224 cell
  follows the decode phase, so it is warm.
- VRAM: the plan's event (`ignis.runtime.vram_plan`) and the per-process WDDM counters `Dedicated Usage` and
  `Shared Usage`.
- Raw data: `.scratch/335-embedding-ab/` in the main checkout (untracked): `leg.sh`, `run_all.sh`, `analyze.py`,
  each leg's `server.log`, `decode.txt`, `prefill.jsonl`, `vram.txt`, `results/<leg>.json`, `gather-bench.txt`.

**The gather alone** (`kernel/tests/bench_host_mapped_embedding.cu`, the whole table, ids spread over the vocabulary,
median of 200 launches):

| tokens | from VRAM | from mapped host memory | bytes read over the bus |
|---:|---:|---:|---:|
| 1 (a decode step) | 10.0 µs | 11.4 µs | 5,440 |
| 8 (eight lanes) | 9.4 µs | 15.2 µs | 43,520 |
| 64 (a verify round, 8 lanes x 8) | 11.2 µs | 89.7 µs | 348,160 |
| 1,024 (a prefill chunk) | 27.8 µs | 996.6 µs | 5,570,560 |

A chunk's 5.6 MB arrive at ~5.6 GB/s, about half the link's copy-engine rate: the vendored kernel reads a byte per
thread.

**VRAM** (per launch; the two launches of a build agree to the byte on every line but the pool, which takes what NVML
reports free at start):

| line | main | branch | difference |
|---|---:|---:|---:|
| `weights_bytes` | 18,585,584,640 | 17,234,723,840 | −1,350,860,800 (the table) |
| KV pool | 791,488-792,960 tokens | 933,952-940,800 tokens | +~146K tokens (+18.5%) |
| WDDM dedicated usage | 30.48-30.49 GB | 30.44-30.51 GB | the budget, spent either way |
| WDDM shared usage | 12,389,974,016 | 13,742,637,056 | +1,352,663,040 (the mapped allocation) |

The branch's host plan (`ignis.runtime.qwen27b_host_plan`): 39.8 GB available, 13.66 GB planned (the table's
1,350,864,896-byte allocation, 16 retained host slots' 3,720,515,584, the 8 GiB arena), 26.15 GB left.

**Decode** (tok/s; c1 the mean of the eight prompts, c8 the aggregate of each round):

| leg | c1 | c8 round 0 | c8 round 1 |
|---|---:|---:|---:|
| branch-1 | 383.7 | 1,072.7 | 1,096.3 |
| main-1 | 383.3 | 1,099.6 | 1,101.2 |
| branch-2 | 383.2 | 1,096.5 | 1,099.1 |
| main-2 | 384.3 | 1,104.1 | 1,102.6 |
| **branch / main** | **−0.1%** | | **−1.0% over all rounds, −0.4% without the session's first** |

**Cold prefill** (tok/s, the mean of the four samples per build: two launches, and two reps per launch but at 8K):

| prompt | main | branch | difference | TTFT difference | the gather's share (bench) |
|---:|---:|---:|---:|---:|---:|
| 8,224 | 9,381 | 9,204 | −1.9% | +20 ms (0.875 → 0.895 s) | +8 ms |
| 14,282 | 9,611 | 9,542 | −0.7% | +10 ms | +14 ms |
| 45,612 | 7,507 | 7,475 | −0.4% | +27 ms | +44 ms |
| 105,272 | 5,096 | 5,088 | −0.15% | +30 ms | +100 ms |

Main's own spread between its launches: 1.0% at 8K, 1.8% at 46K, 1.0% at 105K.

**Greedy text:** c1 8/8 identical across all six launches; c8 identical in every round but one lane of main-2's
round 0 (7/8), the multi-lane nondeterminism main shows against itself (2026-09-24 finding).

**Host under load** (`pressure.py`: 15 GiB of the 37 GiB available held and touched page by page, four busy cores,
for the whole launch; one launch per build):

| leg | available at load | c1 | c8 | 8K | 46K | 105K |
|---|---:|---:|---:|---:|---:|---:|
| branch-load | 23.8 GB | 382.0 | 1,089.9 / 1,099.7 | 8,045 | 7,494 / 7,262 | 4,929 / 4,960 |
| main-load | ~24 GB (no host plan logs it) | 370.8 | 1,064.7 / 1,072.9 | 9,039 | 7,372 / 7,284 | 4,963 / 4,949 |

The branch planned and pinned its 13.66 GB with 23.8 GB available and served; both builds lose 0-4% to the load,
neither more than the other beyond one sample (the branch's 8K, 1.02 s).

**Correctness on the card** (`crates/core/tests/host_embedding_gpu.rs`, GPU profile): a host-mapped load and the
all-on-device load give the same 89,891,840 span logits of a 362-token, three-chunk prefill bit for bit, and the same
16 greedy tokens eagerly and replayed from the decode graphs; the leaf finds exactly the table's bytes in host memory.
The leaf's CTest (132 tests, `test_host_mapped_embedding` among them) passed, and `cargo test --workspace` 3,333 tests
(0 failed, 7 ignored).

## Finding

Observed:

1. **The VRAM plan's weights line drops by exactly the table, 1,350,860,800 bytes, and the KV pool takes it:** ~146K
   more hq-e8-2b tokens at the make defaults (+18.5%).
2. **WDDM counts the mapped table as shared memory, not dedicated:** the process's shared usage grows by the
   allocation, its dedicated usage stays the budget.
3. **Decode is unchanged at one lane (−0.1%) and ~0.4% slower at eight**, at the edge of the round-to-round spread.
   The bench accounts for it: a verify round of 64 tokens gathers ~80 µs longer.
4. **Cold prefill pays ~1 µs per prompt token**, ~1 ms per 1,024-token chunk: −0.15% to −0.7% from 14K to 105K,
   inside main's own launch-to-launch spread, and −1.9% (20 ms) on the one 8K sample per launch.
5. **A loaded host changes nothing specific to the branch:** the start plans and pins under RAM pressure, and both
   builds slow alike.

Inferred:

- ADR 0062's acceptance (decode and cold prefill within noise of today's) holds at one lane and from 14K up. Two
  cells sit at or past the edge of the noise: eight lanes (−0.4% warm, −1.0% over every round, against a 0.4%
  round-to-round spread) and the 8K prompt (−1.9%, 20 ms, against main's 1.0% launch spread). That is the gather's
  ~1 µs per token over a Gen 3 link, measured rather than noise; whether it is acceptable is the owner's call.
- The gather reads at about half the link's rate because the vendored kernel loads a byte per thread; an adapted
  gather with 16-byte loads would roughly halve the prefill cost. At 0.5-1% of prefill that is not worth an op now.
- A Gen 4 or Gen 5 host shrinks the cost further; nothing here was measured on one.

## Implications

- AC 7 of spec 27b-16gb/03 passes for the embedding with that measured cost, ~1-2% on an 8K prompt and 0.4-1.0% at
  eight lanes: ship the host placement on every card class, as ADR 0062 says, unless the owner rejects the cost.
- The ~146K tokens are AC 8's to spend: the `rtx5090` profile's re-measure starts from a pool 1.35 GB larger.
- The 27B now has a host plan: a start whose RAM cannot hold the table, the host retained slots and the KV-RAM arena
  beside the 4 GiB margin is refused, naming the 27B's own flags. Under the make defaults that is ~18 GB available,
  which the 64 GB box has with room to spare (26 GB left at load).

## Limits and unknowns

- One session, two launches per build and one under load; one text family per cell.
- The 8K cell has one sample per launch, after the decode phase.
- The load leg holds RAM and CPU; no second GPU workload (a game, a browser's WebGL) shared the card, beyond the
  owner's ordinary desktop.
- PCIe Gen 3 x16 host only (i9-10900K, Z490).
- The drafter's own embedding reads go through the same table; they are inside the decode numbers, not measured apart.

## Follow-ups

- AC 8: re-measure the `rtx5090` profile with the larger pool.
- Only if a profile ever needs the 0.5-1% of prefill back: an adapted gather with vector loads (ADR 0051).
