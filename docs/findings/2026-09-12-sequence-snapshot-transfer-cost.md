# Sequence snapshot transfer cost is PCIe-link-bound at ~10 GB/s on this host

- Kind: experiment
- Status: current
- Observed: 2026-09-12
- Last verified: 2026-09-12
- Scope: kernel / sequence state transfer, KV-RAM host tier
- Related: [GitHub #124](https://github.com/gpillon/ignis/issues/124),
  [GitHub #125](https://github.com/gpillon/ignis/issues/125),
  [ADR 0024](../adr/0024-sequence-state-transfer.md),
  [Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md)
- Superseded by: none

## Question

ADR 0024 justifies the KV-RAM host tier with an asymmetry it states as an
estimate: a full-context snapshot is "about 528 MB and near 21 ms per
direction over pinned PCIe", against a re-prefill of "~4.6 s". The ticket
that implements the transfer asks for that cost to be measured rather than
assumed, for a short and a full-context sequence.

Two things were open. What does a snapshot actually cost per direction, and
is the tier's justification still sound if it costs more than the estimate?

## Evidence

`kernel/tests/test_seq_snapshot.cpp` prints both figures on every run. It
builds a sequence pool at the real Qwen 3.8-27B geometry (16 GQA layers of 4
KV heads x 256, 48 GDN layers of 48 value heads x 128x128, vocab 248,320)
under hq-e8-2b KV, gives one sequence a frontier, and times
`ignis_seq_snapshot` / `ignis_seq_restore` over a `cudaMallocHost` region —
the transport the host tier will use. One untimed pass first, then the mean
of three.

Three consecutive runs of 2026-09-12, RTX 5090, no other process holding the
card. Means of the three (each itself the mean of three reps):

| sequence | blob | snapshot | restore | effective |
| --- | ---: | ---: | ---: | ---: |
| 128 tokens ("short") | 148.89 MiB | 13.7 ms | 12.6 ms | 11.4 / 12.4 GB/s |
| 40,960 tokens ("full context") | 507.76 MiB | 44.6 ms | 41.6 ms | 11.9 / 12.8 GB/s |

Spread across the three runs is under 5%. One earlier reading on the same
machine came in at 52.1 / 50.0 ms for the full-context cell (10.2 GB/s), so
treat ~10 GB/s as the slow end of what this host does rather than as a
different result.

The blob sizes match the spec's own model of what a sequence is made of. The
short sequence is the floor: 144 MiB of GDN recurrent state, 2.81 MiB of conv
taps, 0.95 MiB of penalty counts, and 1.125 MiB for the two KV pages its 128
tokens occupy. The full-context blob adds 360 MiB of KV (640 pages at 589,824
bytes) and nothing else.

The link this machine runs the card on:

```
$ nvidia-smi --query-gpu=pcie.link.gen.gpucurrent,pcie.link.gen.gpumax,pcie.link.gen.hostmax,pcie.link.width.current --format=csv
pcie.link.gen.gpucurrent, pcie.link.gen.gpumax, pcie.link.gen.hostmax, pcie.link.width.current
3, 5, 3, 16
```

The GPU negotiates Gen 5; the **host** caps at Gen 3. Gen 3 x16 is 15.75 GB/s
of theoretical payload bandwidth.

## Finding

**Observed.** Snapshot and restore both run at 11–13 GB/s regardless of blob
size, with restore consistently a few percent faster than snapshot. A
full-context snapshot is ~45 ms per direction, not ~21 ms.

**Observed.** The card is on a Gen 3 x16 link because the host board caps it
there, not because the GPU cannot do better.

**Inference.** The transfer is link-bound, not pattern-bound. 12 GB/s is 76%
of the Gen 3 x16 theoretical maximum, the ordinary efficiency of a pinned
copy, and the rate is flat from 149 MiB to 508 MiB — the signature of a
saturated link rather than of per-copy overhead. The KV section is moved as
one copy per (plane, page), 40,960 of them at full context, and even that does
not move the rate, which is further evidence the link is the limit.

**Inference.** ADR 0024's 21 ms is an estimate at PCIe 5.0 rates. It is right
about the mechanism and optimistic by roughly 2x about this machine.

**The tier's justification survives.** Against the measured prefill rate of
0.0974 ms/token at a 1,024-token chunk width
([Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md)), 40,960
tokens is at least ~4.0 s of re-prefill, and more in practice because
attention cost grows with the prefix. A 42 ms restore is therefore around
100x cheaper than the work it replaces. That is the low edge of ADR 0024's
"two orders of magnitude" rather than the ~220x the 21 ms figure implied, and
the decision does not turn on the difference.

## Implications

- **Eviction budgeting (#125) should use ~90 ms, not ~42 ms**, for a
  full-context round trip on this host, and should tolerate ~100 ms at the
  slow end. An eviction that frees a lane pays one direction now and the
  other when the request resumes.
- **The floor dominates a short sequence.** A 128-token sequence costs 149
  MiB and 14 ms because the 144 MiB GDN slot is paid whatever the prompt
  length. This is why the host tier is bounded by a byte budget rather than a
  lane count: 3.4 short sequences cost what one full-context sequence costs,
  and a lane count would price them identically.
- **The KV format changes only the KV part.** A BF16 full-context snapshot
  would carry 360 MiB x 7.11 of KV instead, a 2.6 GiB blob, so roughly 240 ms
  per direction. hq-e8-2b is what makes a full-context sequence transferable
  at all on this link.
- **A faster host would change the number, not the conclusion.** On a Gen 5
  x16 board the same copies would land near the ADR's 21 ms.

## Limits and unknowns

- One machine, three runs of three reps per cell. The rate is flat across a
  3.4x size range and stable within 5% across those runs, but one earlier
  reading was 17% slower, and nothing here establishes variance across
  reboots or driver versions.
- The measurement uses a sequence whose device state was written directly
  rather than produced by a forward pass. Transfer cost does not depend on
  the content of the bytes, but this does not exercise any interaction
  between an in-flight forward pass and a concurrent transfer — which is
  precisely what #125's eviction path will do, on its own stream.
- The re-prefill comparison extrapolates an 8,192-token measurement to 40,960
  tokens. It is a floor, not a measurement, at that length.
- Nothing was measured for the device-to-device clone that prefix reuse will
  use (#126); that path never crosses this link and is expected to be two
  orders of magnitude faster again.

## Follow-ups

- #125 (KV-RAM tier) records snapshot and restore wall time per request, which
  will replace this synthetic measurement with the production one.
- #126 (device prefix reuse) measures the device-to-device clone, the other
  half of ADR 0024's cost asymmetry.
