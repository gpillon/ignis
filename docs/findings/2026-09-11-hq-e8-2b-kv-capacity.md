# hq-e8-2b stores a sequence-token in 9,216 bytes, 7.11x denser than BF16 KV

- Kind: discovery
- Status: current
- Observed: 2026-09-11
- Last verified: 2026-09-11
- Scope: kernel / paged KV cache, scheduler capacity
- Related: [ADR 0022](../adr/0022-two-kv-formats-bf16-as-oracle.md), [spec 04](../../.scratch/runtime/specs/04-reference-feature-floor.md), [#65](https://github.com/gpillon/ignis/issues/65), [#112](https://github.com/gpillon/ignis/issues/112)
- Superseded by: none

## Question

Phase 4 adopts hq-e8-2b so that N=8 lanes are reachable at long context. How
much KV does a sequence-token actually cost in each format at this model's
geometry, and does the standard target profile fit?

## Evidence

From the vendored codec, `kernel/vendor/src/ops/kernel/hq_codec.cuh`:

```
inline constexpr int kHqRowBudgetBytes = 64;  // 512-bit code budget = 2 bits/dim
inline constexpr int kHqMetaBytes      = 8;
inline constexpr int kHqHeadDim        = 256;
```

Its header states the property that makes this arithmetic valid at all: "every
(token, kv_head) row occupies exactly kHqRowBudgetBytes of code plane and
kHqMetaBytes of metadata plane, so capacity math, page addressing, and CUDA
Graph address stability are unchanged from the fixed-width formats."

Model geometry from `CONTEXT.md`: 16 GQA layers, 4 KV heads of 256, K and V
both stored.

BF16 per sequence-token: `16 × 4 × 256 × 2 roles × 2 bytes = 65,536`. This
matches the figure already recorded in `.scratch/DEFERRED-DECISIONS.md` item 6
and in the G3 verdict.

hq-e8-2b per sequence-token: `16 × 4 × 2 roles × (64 + 8) = 9,216`.

## Finding

**Observed.** A sequence-token costs 65,536 bytes under BF16 KV and 9,216 bytes
under hq-e8-2b at this geometry — a factor of **7.11**, not the 8x the "2 bits
per dimension" headline suggests, because the 8-byte metadata row costs 12.5%
on top of the code budget.

**Inferred.** Eight lanes at a 40,960-token context need 327,680 resident
tokens, which is **20.0 GiB** under BF16 and **3.02 GB** under hq. Against
~19 GB of weights on a 32 GB card, the first does not fit and the second
leaves room. This is the arithmetic behind the claim that `N-lane concurrency`
is a short-context promise until hq lands.

**Inferred.** Because bytes per token are fixed in both formats, token capacity
is a pure function of the byte budget and the format. A pool configured in
bytes therefore needs no format-specific configuration, only a derived capacity
it reports.

## Implications

- The KV pool is sized by a byte budget and reports a derived token capacity
  (ADR 0022); no token count needs to be compiled in or configured per format.
- The standard target profile requirement at G4 — at least 8 × 40,960 resident
  tokens under hq — is satisfiable with roughly 3 GB of the card's free space.
- A full-context sequence's KV is about 378 MB under hq, which dominates the
  144 MiB GDN slot in any snapshot taken at long context and is inverted at
  short context.

## Limits and unknowns

- This is arithmetic over declared constants, not a measurement. No hq kernel
  has ever run in this tree: P1-15 vendored the hq routes compiled but untested.
- It says nothing about hq's accuracy, its encode cost per appended token, or
  its effect on attention throughput. Those need the GPU.
- The exact-key side store named in the review is not in the vendored tree, so
  its bytes are not in these numbers.
- Escalation (re-encode at alpha/2, then alpha/4) changes the encoded values,
  never the row budget, so it cannot change these figures.

## Follow-ups

- P4-04 reports the derived token capacity at load, which is where these
  numbers stop being arithmetic.
- P4-12 (the G4 gate run) retires the KV-format inequality recorded beside the
  G2 and G3 verdicts.
