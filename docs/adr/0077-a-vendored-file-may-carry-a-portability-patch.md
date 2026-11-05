---
status: accepted
---

# A vendored file may carry a portability patch

Running ignis on Ampere (sm_86) and Ada (sm_89) cards (#456) needs the leaf
compiled for more than `120a`. On sm_89, 32 vendored files fail to compile
because they reach programmatic dependent launch, an sm_90 instruction, through
one header, `src/core/pdl.cuh` (finding
[RTX 20, 30 and 40 cards](../findings/2026-11-02-rtx-20-30-40-what-blocks-the-leaf.md)).
ADR 0031 admits a recorded patch for a measured bottleneck and ADR 0037 for a
demonstrated correctness bug. A file that is right and fast on the card it was
written for, but does not build or launch on another, is neither. This extends
ADR 0037 with a third reason: a vendored file may carry a recorded patch that
makes it build and run on an architecture the build targets, on these terms:

- **Nothing changes where the reference runs.** On `120a` the patched file
  compiles to the same device code, instruction for instruction (the default
  build's `cuobjdump -sass` compared before and after), and any host change is
  a branch the 5090 always takes the way the reference does. So the behaviour
  stays **reference parity** wherever the reference runs, and the patch adds no
  Ignis-patched behaviour.
- **It is the smallest change**: an architecture guard (`#if __CUDA_ARCH__ >=
  900`, a no-op below) or a host check, never a kernel rewritten. A kernel with
  no counterpart on the older architecture, or one its compiler refuses as
  written (more static shared memory than an sm_8x block may declare), is not
  patched: it compiles for the arch-specific Blackwell targets alone, so a
  launch anywhere else fails with CUDA's "no kernel image is available", never
  as a stand-in.
- **The `reason` names the CUDA architecture or the card size, this ADR and
  what it fixes**: a compile, or a launch that fails at run time there.

Our own copy of `pdl.cuh` under `kernel/src`, ADR 0031's option (b), was
rejected. The 32 files include it by its vendored path, so a copy would need
either 32 patches to their includes or a header of the same path earlier in the
include order, a silent substitution that the manifest cannot see.

The first patch is `src/core/pdl.cuh` (#456). Its device calls compile to
nothing below sm_90. `launch_dependent` drops the programmatic-serialization
launch attribute on a device below compute capability 9.0, which it reads once
per process, because whether such a device ignores the attribute or rejects it
is untested. The proposal answers #456's first owner decision ("a new
exemption, or our own copy under `kernel/src`?"); the owner decides at merge.

The second is the vendored GDN gating test, `tests/ops/test_gdn_gating_proj.cpp`
(#478). What it fixes is a card size, not a CUDA architecture: the test's
cooperative schedules are sized for the RTX 5090's 170 SMs, and the planner
resolves the 27B's fused norm/control route with no fallback, so on a card
with fewer than 120 SMs, RTX 50 included, the test aborted at its first check.
These terms therefore also admit a host check on the card's SM count in a
test, which holds no device code: the test asks the planner's own residency
query and skips, naming it on its output, each schedule the query refuses. On
the 5090 every schedule is resident, so the test runs and prints what it did
before. The `reason` names the card size and the planner query it reuses.

The third is the GDN gating planner,
`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp`, with the query
it asks from `bf16_gdn_gating_proj_kernels.{h,cu}` (#478). The planner admits
a cooperative split-K schedule only for a grid the card holds resident, its
SMs times the CTAs one SM holds, and took the second factor from constants
true of the `120a` build. An sm_86 build of those kernels uses more registers
(72 in place of 62), so the 27B's split4 and split2 hold one CTA per SM, not
two, and the 35B's split8, split4 and split2 hold three, not four. On an RTX
3090 (82 SMs) the planner admitted grids the card cannot hold, a 27B prefill
chunk of 1025 to 1664 tokens among them, and the launch failed with
`cudaErrorCooperativeLaunchTooLarge`. The planner now takes each kernel's
CTAs per SM from CUDA's occupancy query on the kernel it launches, once per
device. That is a host change the 5090 takes the way the reference does: on
compute capability 12.0 the query gives the former constants, which the
gating test checks (it fails if one differs, and on the 5090 if it skips any
schedule), and the `120a` build's device code is unchanged (`cuobjdump -sass`
of the kernels' object, before and after). The test's patch now also prints
each kernel's CTAs per SM.
