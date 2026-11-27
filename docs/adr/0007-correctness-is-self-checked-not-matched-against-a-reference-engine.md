---
status: accepted
---

# Correctness is self-checked, not matched against a reference engine

ignis is our engine, not a recreation of ninfer, so matching ninfer's output is not what makes it correct.
Correctness is self-checked: the engine must produce sane output for the same model, greedy at a fixed seed, and the
canary check (`ignis-bench canary`, sane and deterministic output) is that check. The engine is not required to agree
with a reference's tokens, at 99% or byte for byte, and parity is not the word for acceptance.

## Considered Options

- **A gate of at least 99% token agreement with ninfer's baseline, a divergence report its artifact**, the acceptance
  this ADR replaced: rejected with the framing it came from, that acceptance is "match ninfer's output".
- **Byte-identical greedy output on the canary suite**: rejected. It is not guaranteed across builds, where CUDA
  toolkit drift, kernel porting subtleties and a changed GEMM accumulation order can flip an argmax near a tie, so a
  gate on it would be brittle and would take "every floating-point operation is bit-identical" for "the architecture
  works".

## Consequences

- Speed is no part of this check. The 99% performance gate against ninfer, the first rung of a ladder, which
  comments still cite under this number, is retired: the gate is ADR 0082's and none of it is a ratio against ninfer,
  `ignis-bench`'s report and gate checks stay as instruments, and the image model never had a performance gate
  (ADR 0050).
