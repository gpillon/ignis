---
status: accepted
---

# A vendored file may carry a portability patch

A vendored file may carry a recorded patch that makes it build and run on an architecture the build targets, or on
a card smaller than the RTX 5090: the third reason for one, beside ADR 0031's measured bottleneck and ADR 0037's
demonstrated correctness bug. Running ignis on Ampere (sm_86) and Ada (sm_89) cards needs the leaf compiled for more
than `120a`, and a file that is right and fast on the card it was written for but does not build or launch on
another is neither a bottleneck nor a bug
([finding](../findings/2026-11-02-rtx-20-30-40-what-blocks-the-leaf.md)). The patch is admitted on these terms:

- **Nothing changes where the reference runs.** On `120a` the patched file compiles to the same device code,
  instruction for instruction (the default build's `cuobjdump -sass` compared before and after), and any host change
  is a branch the RTX 5090 always takes the way the reference does. So the behaviour stays **reference parity**
  wherever the reference runs, and the patch adds no Ignis-patched behaviour.
- **It is the smallest change**: an architecture guard (a no-op below the architecture it guards) or a host check,
  never a kernel rewritten.
- **A kernel with no counterpart on the older architecture, or one its compiler refuses as written, is not
  patched.** It compiles for the arch-specific Blackwell targets alone, so a launch anywhere else fails with CUDA's
  "no kernel image is available", never as a stand-in. An option's value the card or the build cannot run falls
  back when it comes from a default or a profile, and is refused, naming the option, when the operator names it;
  the routes of ADR 0080 for 8.x cards, the dense prefill and the integer band of the 8-bit prompt
  attention, are checked the same way, so a build that does not carry a route refuses it by name instead of running
  another.
- **The `reason` names the CUDA architecture or the card size, this ADR and what it fixes**: a compile, or a launch
  that fails at run time there.
- **In a test, which holds no device code, the terms also admit** a host check on the card's SM count, and a
  criterion derived for the route the card runs, applied only where that route runs in place of the reference's: a
  check that fails only because the card runs another route is neither a compile nor a launch. On the RTX 5090 every
  check stays the reference's.
- **Where the reference's own choice cannot be held by a smaller card, the terms admit a host change that picks
  what the card holds** (ADR 0091: the GDN gating planner takes the largest smaller split of the model's family the
  card holds resident, and its capacity query covers every width), and a test-only host setter that no engine path
  and no wrapper calls, under which the RTX 5090 runs a smaller card's routes. They admit both because with no cap
  the RTX 5090 takes the reference's branch everywhere, which the test checks.

Which files carry a portability patch, and what each fixes, is in `kernel/vendor/VENDOR.md`.

## Considered Options

- **Our own copy of `pdl.cuh` under `kernel/src`**, ADR 0031's option (b), for the header through which 32 vendored
  files reach programmatic dependent launch: they include it by its vendored path, so a copy would need 32 patches to
  their includes, or a header of the same path earlier in the include order, a silent substitution that the manifest
  cannot see.
