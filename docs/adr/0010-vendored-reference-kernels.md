---
status: accepted
---

# Kernel policy: vendor the reference ops verbatim, under a manifest

An op the program takes from the reference is a verbatim vendored copy of the reference's op: the kernel, its
launcher, its public wrapper and header, its storage-layout decoders and its own op test, never a scalar
re-implementation. This is what "port the proven CUDA" means in ADR 0005: vendored verbatim under the manifest. The
hand-written stand-ins it replaced, under headers that claimed a 1:1 port, were 10 to 100 times slower than the
reference's ops and wrong as well, since the storage-layout decoders went with the kernels and the scale plane and
the weight divisor were the first things lost; a reviewer who trusted the header would not have opened the
reference's file.

- **Vendored files keep the reference's namespaces, include paths and tensor conventions** (feature-major: the
  token axis second), so a `diff` against the source stays trivial and a reference update is a re-run of the script.
- **Provenance is a manifest, not a header comment.** A copy-and-verify script maintains the reference's pinned
  commit, the list of vendored files with their content hashes, and any local patch recorded as a diff. A file whose
  hash matches neither the source nor its recorded patch fails verification. The `NOTICE` attribution is kept.
- **A port claim must be diffable.** Anything that is not a verbatim vendored file, including anything edited beyond
  a recorded patch, is labelled our own implementation and carries no provenance claim.
- **The program layer is ours** (ADR 0009) and is not vendored: the per-layer op sequence, the sequence-state
  wiring, the prefill and decode loops and the step ABI.
- **Each vendored op brings its reference test** into a kernel-leaf test executable (CTest), run at real 27B
  geometry against the reference's own fp64 references and tolerances. An op ticket closes only with that test green
  on the GPU.
- **Our own kernels come one op family at a time, each justified by a measurement** and gated again as ADR 0005
  asks, by ADR 0082's gate. A family already at roofline in the reference is not a candidate. ADR 0031 says what the
  measurement has to show, per op, and what a patch or a replacement owes.

A vendored file is not hand-edited. The reasons that admit a recorded patch are other ADRs': a measured bottleneck
(ADR 0031), a demonstrated correctness bug (ADR 0037), whose patch makes the behaviour it changes Ignis patched
where every other is reference parity, portability (ADR 0077) and a registry that lacks a case (ADR 0088). What a
kernel may be beside a vendored op, and that a vendored op may come from a third party, is ADR 0051's.

## Considered Options

- **Writing our own tensor-core kernels first**: months per op family to reach the reference's numbers, and nothing
  to win where the reference is already at roofline.
- **Keeping the rewrites because "the reference's dispatch layer is C++ state the ABI forbids"**: a misreading of
  ADR 0001. Dispatch stays inside the leaf, and only the step crosses the boundary (ADR 0009).

## Consequences

- An op vendored and compiled ahead of the phase that uses it does not drift, because it is never hand-edited.
- A reference update is a mechanical re-run of the copy-and-verify script and of the vendored op tests, not a
  re-port.
- The vendored ops are Apache-2.0: attribution stays in `kernel/NOTICE`, and the manifest records exactly which
  files it covers.
