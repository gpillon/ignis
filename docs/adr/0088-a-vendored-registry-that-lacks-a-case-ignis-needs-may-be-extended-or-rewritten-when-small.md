---
status: accepted
---

# A vendored registry that lacks a case ignis needs may be extended, or rewritten when it is small

ADR 0010 vendors the reference's kernels verbatim, and ADRs 0031, 0037 and
0077 name the reasons a vendored file may carry a recorded patch: a measured
bottleneck, a demonstrated correctness bug, portability. A registry, a table
that maps a shape to the launch that serves it such as the W8 linear's
`select_w8_a16_launch`, fails another way: a shape the reference never served
throws, though the kernels it would route to run it. The image model's vision
tower (GitHub #320) and Flash-Next's 2,560-wide merger (GitHub #557, ADR 0087)
each needed such an arm, admitted until now as one ticket's sanctioned edit.
The owner decided on 2026-11-09 that extending a vendored registry that lacks
a case ignis needs is a standing reason for a recorded patch. Its `reason`
names the shapes added, why the routes fit them (tiles, alignment), and the
test cases that throw without it; the routes follow the registry's
neighbouring arms unless a measurement says otherwise. Where the registry is
small, rewriting it as ignis's own code outside the vendored tree is the
alternative, as ADR 0031 allows a replacement.

## Consequences

- The W8 dispatch's arms for the image model's tower and for Flash-Next's
  merger stand on this ADR; `kernel/vendor/VENDOR.md` lists it beside the
  other three reasons.
- An extension adds cases and moves none: every shape the reference routed
  keeps its route, so it adds no row to VENDOR.md's table of ignis-patched
  behaviours.
