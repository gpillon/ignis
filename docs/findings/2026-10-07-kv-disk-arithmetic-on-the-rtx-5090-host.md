# The arithmetic behind KV-disk on the RTX 5090's host: a blob's size, the volume's rates and the restore floors' break-even

- Kind: experiment
- Status: current
- Observed: 2026-10-07
- Last verified: 2026-10-07
- Scope: runtime / KV-disk tier (its defaults, its restore floors, its file layout), the KV pool's default
- Topic: kv-and-state
- Related: GitHub #309, [ADR 0045](../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md), [Flash-Next on the 5090](2026-10-06-flash-next-on-the-5090.md), [the agent turn tail](2026-10-07-flash-next-agent-turn-tail.md), [KV-disk contention on the volume](2026-10-08-kv-disk-contention-on-the-volume.md)
- Superseded by: none
- Summary: On the RTX 5090's host a whole-context Flash-Next blob is ~1.25 GB and reads back in ~0.45 s at the NVMe's ~2.9 GB/s, 7x under the 3.2 s chunk an 8,192-token floor skips; the 27B breaks even near 16K tokens at a 524K blob.

## Question

What sizes and rates set KV-disk's defaults, its restore floors and the shape
of its files on the development host? The figures were carried in ADR 0045's
text until 2026-11-27; this finding keeps them.

## Evidence

The host of the RTX 5090, on 2026-10-07 and 2026-10-08. The disk figures are
the F: disk bench of 2026-09-29 (`.scratch/disk-bench/RESULTS.md`, untracked,
local to the owner's clone); the rest is arithmetic over the plan's lines and
over the findings linked above.

- **The host is RAM-poor and NVMe-rich.** Flash-Next pins 37.8 GB of experts.
  With ~48 GB available the KV-RAM arena fits at 0.5-2 GiB; on 2026-10-07 the
  host plan refused 2 GiB by 0.66 GB.
- **A blob's size.** A whole-context (262,144-token) Flash-Next lane's blob is
  ~1.25 GB: 4,224 B/token paged (KV 3,456, indexer keys 768) plus a 130 MB
  state image. A 27B blob is ~230 MB of image plus 9,216 B/token under
  hq-e8-2b (65,536 B/token under BF16).
- **The volume.** F: is an NVMe behind the chipset at Gen3 x4, ~96% full.
  Unbuffered, it reads ~2.9-3.0 GB/s and writes 1.2-1.5 GB/s, with
  garbage-collection stalls up to ~150 ms.
- **Prefill, the cost a restore saves.** A full 8,192-token Flash-Next chunk
  takes ~3.2 s, ~1.7 s of it expert copies, and even a traversal of a few
  tokens re-streams 0.13-0.66 s of experts. The 27B prefills ~8-9.6K tokens/s,
  ~0.11 ms/token.
- **Hashing.** SHA-256 runs ~0.5 GB/s on a CPU without SHA extensions, slower
  than the read it would guard.
- **Preallocation.** NTFS zero-fills up to any write past a file's valid data
  length, and writing a whole 16 GiB budget at start takes ~15 s at 1.2 GB/s.

## Finding

Arithmetic from the evidence, not measured as such:

- **Flash-Next's restore floor, 8,192 tokens, is one prefill chunk.** Skipping
  it skips ~3.2 s. At ~2.9 GB/s even a whole-context blob reads in ~0.45 s, so
  the floor buys a margin of about 7x.
- **The 27B's floor, 16,384 tokens, is its break-even at make's default
  context.** A blob reads in ~0.08 s + 3.2 µs/token and prefill costs
  ~0.11 ms/token, so a restore pays past ~750 + 0.03 x (blob tokens): ~8.6K
  tokens at a 262K blob and ~16K at a 524K one.
- **What a budget holds.** 16 GiB holds about 13 whole-context Flash-Next
  lanes or about 65 conversations of 30K tokens; 4 GiB about three
  whole-context lanes.
- **A spill's time.** Writing a whole-context Flash-Next blob takes ~1 s at
  1.2-1.5 GB/s, the time its victim keeps its pages on the device.
- **Three agents on the default pool.** At the 38,912-token default cap, three
  requests that send no `max_tokens` reserve their whole bounds together on
  the 524,288-token pool for prompts up to ~135K tokens each; with no default
  cap two whole-context reservations fill it.

## Implications

- The floors are per family because prefill speed is: the slower prefill
  takes the lower floor (`restore_floor_tokens` in
  `crates/runtime/src/kv_disk.rs`).
- A whole-file hash, a preallocated arena file and a blob staged whole in
  pageable RAM each cost more here than what they would buy, which is why
  ADR 0045 rejected them.

## Limits and unknowns

- The disk bench's raw output is not in the repository, and its figures are
  one volume's, nearly full. [KV-disk contention on the volume](2026-10-08-kv-disk-contention-on-the-volume.md)
  later measured the tier's own writes at 1.5-1.6 GiB/s there.
- The break-even figures are computed, never measured as a restore against a
  prefill of the same prompt.
- Another host's volume, link and RAM move every figure.

## Follow-ups

None.
