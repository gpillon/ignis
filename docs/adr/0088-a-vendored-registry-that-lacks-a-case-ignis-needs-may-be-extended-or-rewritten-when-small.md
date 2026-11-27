---
status: accepted
---

# A vendored registry that lacks a case ignis needs may be extended, or rewritten when it is small

Extending a vendored registry that lacks a case ignis needs is a standing reason for a recorded patch, the fourth
beside a measured bottleneck (ADR 0031), a demonstrated correctness bug (ADR 0037) and portability (ADR 0077). A
registry, a table that maps a shape to the launch that serves it such as the W8 linear's `select_w8_a16_launch`,
fails another way than those three name: a shape the reference never served throws, though the kernels it would
route to run it. The image model's vision tower and Flash-Next's 2,560-wide merger (ADR 0087) each needed such an
arm, and a standing reason replaces one ticket's sanctioned edit each time. The patch's `reason` names the shapes
added, why the routes fit them (tiles, alignment) and the test cases that throw without it; the routes follow the
registry's neighbouring arms unless a measurement says otherwise. Where the registry is small, rewriting it as
ignis's own code outside the vendored tree is the alternative, as ADR 0031 allows a replacement.

## Consequences

- `kernel/vendor/VENDOR.md` lists this reason beside the other three, with the patches that stand on it.
- An extension adds cases and moves none: every shape the reference routed keeps its route, so it adds no row to
  `VENDOR.md`'s table of Ignis-patched behaviours.
