# Findings

This collection preserves durable, reusable, evidence-backed results that can
inform future work in this repository. To add, update, or supersede a finding,
follow the [authoring convention](../agents/findings.md).

## Classification

- `discovery`: durable observations from exploring the codebase, architecture,
  runtime, or workflow.
- `research`: conclusions drawn from identifiable external sources.
- `experiment`: results from reproducible tests, benchmarks, or measurements.

## Status

- `current`: the finding remains usable.
- `superseded`: the finding is retained as historical context and links to the
  material that replaces it.

## Index

| Finding | Kind | Scope | Observed | Status | Superseded by | Summary |
|---|---|---|---|---|---|---|
| [hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md) | discovery | kernel / paged KV cache, scheduler capacity | 2026-09-11 | current | none | A sequence-token costs 9,216 bytes under hq-e8-2b against 65,536 under BF16 (7.11x), so 8 lanes at 40,960 context need ~3.02 GB instead of 20 GiB. |
| [Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md) | experiment | kernel / prefill chunking, GPU profiling | 2026-09-11 | current | none | A 1,024-token prefill chunk is 90% per-layer compute and 0.5% the forced synchronization, so #92's "8 semaphores" premise does not hold; packing helps only below ~1,024 tokens/traversal by filling GEMM shape and amortizing launch overhead, not by removing syncs. |
| [Sequence snapshot transfer cost](2026-09-12-sequence-snapshot-transfer-cost.md) | experiment | kernel / sequence state transfer, KV-RAM host tier | 2026-09-12 | current | none | Snapshot and restore both run at ~12 GB/s because this host caps the 5090 at PCIe Gen 3 x16, so a full-context round trip costs ~90 ms rather than ADR 0024's estimated ~42 ms — still around 100x cheaper than the re-prefill it replaces. |
| [hq attention route agreement](2026-09-12-hq-attention-route-agreement.md) | experiment | kernel / GQA attention routes, hq-e8-2b KV cache | 2026-09-12 | current | none | hq attention agrees with ignis's own BF16 route to ~0.23 median relative L2 (cosine > 0.95) on identical keys and values, and its handful of rows past 1.0 are softmax flips between near-tied keys — so the enforced bound is the median plus a capped fraction, never a per-row maximum. |
| [Device prefix clone cost](2026-09-12-device-prefix-clone-cost.md) | experiment | kernel / device prefix reuse, sequence state transfer | 2026-09-12 | current | none | The device-to-device clone behind prefix reuse costs 0.33 ms for 148 MiB of mutable state, 3.6x ADR 0024's estimate but 73x cheaper than a PCIe round trip and ~6,000x cheaper than re-prefilling the head; issuing it as one strided copy per section instead of one per layer is worth 4.6x. |
| [hq prompt workspace under-report](2026-09-12-hq-prompt-workspace-under-report.md) | discovery | kernel / vendored GQA workspace query, hq-e8-2b prefill | 2026-09-12 | current | none | The vendored capacity query under-reports an hq-e8-2b prompt call by one split-partial set at widths 9..16, so a short prompt overruns its arena; ignis corrects it at the caller rather than patching the vendored source. |
