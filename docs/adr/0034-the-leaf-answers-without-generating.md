# ADR 0034 — the leaf answers without generating

## Status

Accepted (2026-09-20, owner). **Amends the *Per-lane sampling* invariant** in
`CONTEXT.md`, which said the leaf "returns token ids and never ships logits to
the host": that claim is now the decode path's, and this ADR names the two
paths that sit beside it.

Sources: owner grill 2026-09-19/20 (Q1, Q4, Q7 of the design tree);
`docs/findings/2026-09-19-typed-option-logit-readout.md` and
`docs/findings/2026-09-19-constrained-digit-readout-points.md`, both measured on
branch `jev-classify`; TypeSafe's Jev API reference (`docs.typesafe.ai/api`) for
the interface pattern; SemIf (github.com/TheoLeeCJ/SemIf, MIT) for the
technique and the fixture.

## Context

Most of what an agent asks a model is not a request for prose. *Route this.
Retry that. Does the evidence support the claim? Which of these 200 queues owns
this ticket? Where is the button?* A chat model answers each of them by
generating a sentence that software immediately parses back into a branch — and
pays a decode round per token to do it, plus a parser that has to survive every
shape the model might choose. On this engine the parse is not hypothetical: in
three runs against three near-identical screenshots the model returned a bare
JSON array twice and a markdown-fenced object once.

The answer is already in the model before a single token is generated. A prompt
that names each option by a label puts the whole decision in one position's
next-token distribution, and reading the logits of those labels *is* the answer.
Measured on the served 27B: the model's own unrestricted winner is a declared
label on 144 of 144 authored decisions, the declared labels hold a median 99.8%
of the distribution, and the result is 0.934 balanced accuracy at 35 ms with
**zero tokens generated**. It holds over an image (4/4 on the vision canaries,
mass ≥ 0.999) and it does not decay with width (mass p50 ≥ 0.996 out to 256
declared options, winner in the declared set at 100% of rows at every width).

Nothing below the kernel leaf has to change to do this. `cuda_leaf.rs` already
calls `step::prefill_program_sampled(..., None)`, where that `None` is an
`out_logits` buffer GitHub #72 added for a diagnostic. The work is entirely
above it.

But the glossary said the leaf never ships logits to the host, and that
sentence is load-bearing: it is why "what a request generates depends on its own
seed alone, never on which lanes happened to share its round". A feature that
quietly contradicted it would leave a future reader unable to trust either the
sentence or the code.

The second half of the problem is coordinates. A readout reads one position, so
it yields one symbol — never a pair of numbers. Reading a point needs a run of
positions, each restricted to the digits. Measured the same way: one prefill
plus six digit-restricted steps lands inside the target button on all three
synthetic 4096-pixel screenshots, worst error 2.1% of the side, and reproduces
the box the model gives unprompted to within (0,9) on a 0–999 scale. The
restriction reads the model rather than overruling it — the digit mass is 1.000
at every position after the first.

Doing that against today's leaf means a host round-trip per digit: 607 KB of
logits across PCIe and a synchronization, six times, on a lane that blocks its
round meanwhile. That is what the experiment did, because an experiment is alone
on the card. A service is not.

## Decision

The leaf grows two paths beside decode, and the sampling invariant is restated
rather than excepted.

- **A readout receives logits and draws no sample.** `StepLeaf::prefill` takes
  an optional logits buffer; the `Compute` seam carries the **answer tokens**
  in and the readout out. The gather happens inside `RuntimeCompute::prefill_step`
  — the full-vocabulary buffer (151,936 × f32 ≈ 607 KB per decision) never
  crosses the seam, only the answer logits, the full-vocabulary log-sum-exp and
  the unrestricted argmax.
- **A constrained decode restricts sampling, per lane, in the leaf.** The
  sampling ABI takes a set of permitted token ids; the leaf samples only among
  them and still returns a token id. No logits cross for this path, so it
  composes with temperature, with seeds and with speculation exactly as decode
  does.
- **The invariant is the decode path's.** *Per-lane sampling* now reads: the
  decode path never ships logits to the host; a readout is the one path that
  does, and it draws no sample — no seed, no RNG, no penalty history. Lane
  independence is therefore *stronger* stated this way than it was stated
  loosely, because a readout has nothing with which to disturb another lane.
- **The answer alphabet is computed from the loaded tokenizer, never compiled
  in.** A label is admitted only if it encodes to exactly one token that decodes
  back to itself. 114 of the 676 uppercase bigrams fail that in the 27B's
  tokenizer, and a two-token label would have its first token's logit read — a
  token that belongs to some other label. This is a correctness rule, not an
  optimization.

## Considered options

**Host-side constraint for the digits** — read the logits each round, pick on
the host, force the token back. It needs no kernel change and it is exactly what
the experiment did. Rejected: six host synchronizations per number on a lane
that blocks its round, 607 KB each way, and it would contradict the invariant
three days after we sharpened it. A thing that is fine alone on a card and wrong
under load is not a design.

**Generate and parse, with no new paths at all** — keep `/v1/chat/completions`
and ask for JSON. It works today and needs nothing. Rejected for the decision
endpoint, though it remains the right answer for anything that wants prose: it
costs a decode round per token where a readout costs none, it cannot return a
calibrated distribution over the options (only the one the model picked), and
the parser has to survive every output shape — which, measured, is more than one.

**A separate classification model.** Rejected: the point is that the served
model already knows the answer. Loading a second set of weights to ask it a
smaller question spends VRAM that the first model's KV wants, and gives up the
one property that makes this worth doing — the decision is made by the same
model that would have written the sentence.

## Consequences

- The `Compute` trait is no longer purely "give me tokens". It is the only
  GPU-coupled seam in the engine and it now carries a second kind of answer, so
  every CPU-only implementation of it — the mock above all — must produce a
  deterministic readout, or the scheduler's tests stop covering the path
  (ADR 0006: the engine is testable without a GPU).
- A decision holds no residency and takes no decode lane, so **eviction
  priority** has nothing of its to take and the admission classes mean something
  different for it. That is why a decision request defaults to `Agent` where
  every other route defaults to `Interactive` (`CONTEXT.md`, *Lane tag*).
- The sampling ABI gains a per-lane token set, which every future sampling
  change has to carry. That is the cost of not having the host in the loop.
- `ignis_decoded_tokens_total` does not move for a readout, which is correct and
  makes decisions invisible to the existing panels. The decision counters and
  the **answer mass** histogram exist because of it: a silent collapse of answer
  mass in production means well-formed noise, and it is the only failure of this
  endpoint that nothing else would show.
- The endpoint can be reached by an unmodified Jev client, which is the reason
  to copy their wire shape rather than invent one. Their vocabulary (`noul`,
  `criteria`, `instructions`) therefore enters our glossary as imported, not
  ours.
