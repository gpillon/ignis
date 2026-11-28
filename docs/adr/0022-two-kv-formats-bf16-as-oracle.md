---
status: accepted
---

# Two KV formats, with BF16 retained as the correctness oracle

Both KV formats ship, and the format is a model-load option, fixed for the life of a load. hq-e8-2b is the serving
default: the capacity argument is decisive at the 27B's geometry, a sequence-token costing about a seventh of BF16's
bytes ([finding](../findings/2026-09-11-hq-e8-2b-kv-capacity.md)), and without it `N-lane concurrency` in `GLOSSARY.md`
is a short-context promise. The reference served in hq-e8-2b the whole time, so while ignis carried only BF16 every
comparison of the two recorded the format difference as a known inequality rather than measuring through it (the
measuring rule in the opening comment of `crates/bench/src/lib.rs`, `g2::Verdict::notes`). BF16 is retained, and is
the format every correctness oracle runs against; the GPU profile keeps its correctness checks on it.

The codec is not the difficulty: it is vendored, and its format keeps the paged-KV contract's fixed-bytes-per-token
property. The difficulty is that every correctness oracle the project owns was calibrated on BF16: the teacher-forced
canary floor (ADR 0082), the f64 layer references and the chunked-versus-per-token self-oracle. Replacing the format
outright would have re-derived all of those tolerances under a lossy format in the same change that introduced the
lossy format, and #96 is the project's own record of what a tolerance nobody re-derived can hide: roughly 80 times of
unused slack, carried over from an unrelated oracle.

- **hq earns its own acceptance in-house**: the op-level codec error is measured on real KV rows, and the hq
  attention route is checked against the BF16 route on identical keys and values. The tolerance is derived from the
  measured codec error and never copied from another oracle.
- **The KV pool is sized by a byte budget.** A resident pool is what the budget leaves, and an offloaded one is ADR
  0045's token default. Either way, the pages are derived from the format in force, and the capacity is reported at
  load.

## Considered Options

- **Replacing BF16 with hq.** Rejected for the tolerances above, and because keeping BF16 means the engine can always
  answer "is this hq's fault?" by reloading. That diagnostic is the main reason retention beats replacement.
- **An external oracle for hq**: recording the reference's hq-mode output and checking ignis against it. That is token
  agreement, which ADR 0007 exists to refuse, and it would make a second engine's quantization choices the definition
  of ignis being correct.

## Consequences

- Two attention route families are live, and both are in the GPU profile. The cost is real, and is accepted in
  exchange for keeping a non-lossy reference path inside the engine.
- A format change is a new load (a restart or a reloading PATCH), never a switch within a load; one decode-graph
  capture set per load, and the round's widths are unchanged (ADRs 0019 and 0020).
- Anything that depends on the exact bytes of the KV cache (an exact-key side store, the host tier's snapshot layout,
  a drafter's KV carry) must be written against the section table (ADR 0024), not against one format.
