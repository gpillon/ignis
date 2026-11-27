---
status: accepted
---

# Step-ABI calls take extensible options structs

A step-ABI call that needs per-call modulation takes a pointer to an extensible options struct, whose first field is
`uint32_t size`, the `sizeof` of the struct the caller compiled against; `NULL` means the production defaults, and
the leaf rejects a size it does not recognize. The ABI then grows by fields on these structs, not by parameters and
not by entry points. The span, position and batch parameters of ADR 0009 cover the shape of the work, a chunked
prefill is still "this token span from this position", but not how a call is modulated: the first need, a prefill
route and a compute policy chosen per call, had no place in the signature, and answered one parameter at a time
every later phase would have been another ABI break.

- **An addition is a field append plus a size bump**, and old callers keep compiling. One exception: a field that
  fits the struct's tail padding leaves `size` as it was, so the size check accepts the layout before it and the
  one after it alike. The layout pin guards it instead: a `static_assert` on `sizeof` and on each appended field's
  `offsetof` in the header, mirrored by an `offset_of!` test on the Rust mirror, so a drift on either side fails
  the build or `cargo test`.
- **No compatibility wrapper is kept.** The only consumer of this ABI is the Rust binding in this repository, kept
  1:1 with the header and compiled from the same tree, so the leaf and the binding change together. If the ABI ever
  gains a consumer outside the repository, stability is declared then, in its own ADR, with a versioning story,
  which also says whether an append into padding needs a version field of its own: it is not simulated before.
- **A new step-ABI call that plausibly needs per-call modulation takes an options pointer from its first version**,
  even if the struct starts with nothing but `size`.
- **The prefill call's options select the route and the compute policy**: chunked by default and per-token for
  tests, the engine's default policy or `A16Only` forced. The per-token prefill route is a supported, documented,
  test-only route: it is the chunked path's self-oracle (the same prompt through both routes must agree), and the
  policy override lets a test compare routes on identical inputs.
- **The rule is about steps, not about lifetime calls or queries.** A step is a forward pass, whose route, compute
  policy or sampling would otherwise multiply parameters and `_ex` variants. A call that creates or destroys an
  object a step runs against (the sequence pool, a sequence) has nothing to modulate per call, and a query has no
  knob: these are entry points and take no options struct. So are the state-transfer and prefix calls (ADR 0024: a
  snapshot's size and its format version are queries, publish, claim and release are lifetime operations on an
  object the leaf owns, and the allocation that claims a shared prefix is a second constructor, not a flag on the
  first, such a sequence being built differently, not stepped differently), and the decode graphs' capture, a
  one-time startup action with no per-round counterpart (ADR 0019). A snapshot control, that is a policy, a stream
  or a partial extent, still goes in an options struct when a phase needs one, and is not one more entry point.

## Considered Options

- **A parallel `_ex` entry point, the old symbol kept as a defaulting wrapper**: rejected. There is no external
  caller to preserve the old symbol for, so the wrapper is dead code from the day it is written, and the pattern
  repeats at every phase that modulates a call.
- **Change the signature a parameter at a time**: honest for an ABI with one consumer, but it makes every phase an
  ABI break.
- **A model-load option in place of a per-call one**: rejected. The self-oracle test would load the 19 GB model
  twice to compare two routes, and a test-only switch would sit in production configuration.
