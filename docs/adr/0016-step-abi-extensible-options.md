# ADR 0016 — Step-ABI calls take extensible options structs, and G2 spends the one extension

## Status

Accepted (2026-09-07, grilling session for GitHub #63). **Amends ADR 0009** —
specifically its claim that chunked prefill would be "a leaf change, not an
ABI change". The step-level, device-resident, opaque-handled design of ADR
0009 is unchanged.

Sources: `.scratch/runtime/specs/02-real-prefill.md` (GitHub #63).

## Context

ADR 0009 put the span and start-position parameters on `ignis_prefill` from
day one precisely so that G2's chunked prefill and G4's tail prefill would
need no ABI change. That reasoning held for the *shape of the work*: a
chunked prefill is still "this token span, starting at this position", and
the leaf can cut it into chunks internally.

It did not hold for *how the call is modulated*. G2 keeps the G1 per-token
prefill path as a test-only route so the chunked path has a self-oracle
(same prompt, both routes, must agree), and it wants tests able to force the
`A16Only` compute policy to compare routes on identical inputs. Selecting a
route per call is not expressible in the current signature, which has no
options or flags parameter.

So G2 must either change the signature or work around it. Three options:

- **(a) A parallel `_ex` entry point**, with the old symbol kept as a
  defaulting wrapper. Preserves the existing ABI literally. But the only
  consumer of this ABI is the Rust binding in this same repository, compiled
  from this same tree — there is no external caller to preserve it for, so
  the wrapper is dead code from the day it is written. And the pattern
  repeats: G3's sampling parameters and G4's snapshot controls would each add
  another `_ex`.
- **(b) Change the signature, and say so.** Honest, and correct for a
  single-consumer ABI. But done naively (add a parameter now, add another at
  G3, another at G4) it makes every phase an ABI break.
- **(c) Make it a model-load option instead.** No signature change, but the
  self-oracle test would have to load the 19 GB model twice to compare two
  routes, and a test-only switch would sit in production configuration.

The real decision is not "break or not" — it is *what shape the ABI takes so
that this is the last break of its kind.*

## Decision

- **Step-ABI calls that need per-call modulation take a pointer to an
  extensible options struct**, whose first field is `uint32_t size` carrying
  `sizeof` the struct the caller compiled against. `NULL` means "production
  defaults". The leaf rejects a size it does not recognize.
- **G2 spends this extension on `ignis_program_prefill`**, adding
  `struct ignis_prefill_options` with the prefill route (chunked by default,
  per-token for tests) and a compute-policy override (engine default, or
  force `A16Only`).
- **Later phases add fields, not parameters and not entry points.** G3's
  sampling parameters and G4's snapshot controls extend the relevant options
  struct; each such addition is a field append plus a size bump, and old
  callers keep compiling.
- **No compatibility wrapper is kept.** The Rust binding
  (`crates/core/src/step.rs`, kept 1:1 with the header) is updated in the
  same change. If this ABI ever gains a consumer outside this repository,
  stability is declared then, in its own ADR, with a versioning story — not
  pre-emptively simulated now.
- **ADR 0009 is amended, not superseded.** Its span+position and batch
  parameters did their job: no chunking-driven parameter was needed. Its
  blanket "not an ABI change" wording is corrected to "not an ABI *shape*
  change; per-call options were added under ADR 0016."

## Consequences

- G2 is an ABI break, recorded as such. The leaf and the Rust binding are
  updated together, which is already how every change in this repository
  ships.
- The default call site does not get more complicated: passing `NULL` keeps
  today's ergonomics for the production path, and only tests construct an
  options struct.
- The per-token prefill route becomes a supported, documented, test-only
  route rather than dead code left behind by G2. Deleting it later is a G3
  decision.
- A future reader who wonders why a C struct carries its own `sizeof` finds
  the reason here: it is the mechanism that lets G3 and G4 extend prefill and
  decode without another break.
- The rule generalizes: any new step-ABI call that plausibly needs per-call
  modulation should take an options pointer from its first version, even if
  the struct starts with nothing but `size`.
