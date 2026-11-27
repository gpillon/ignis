---
status: accepted
---

# A grammar checks each sampled token and repairs a miss, and the model is told the schema

Structured output holds an answer to a grammar by letting the model sample freely and checking each sampled token
on the host before the model consumes it. Only a **miss** sends a permitted set down, as a mask over the whole
vocabulary, to **repair** that one position from the scores the round left on the device. The constraint depends on
what the model has written and its allowed set can be most of the vocabulary, far past the short permitted set of
ADR 0034. The check is cheap here because a plain round's draw already waits host-side as the pending token, and
after a verify pass the host already chooses how many licensed tokens to commit before the recurrent state is
folded. The server also describes the schema to the model by default, at the end of the prompt, because a model
that has not seen the schema misses on nearly every key.

- **Same result as masking every step.** Identical under greedy to masking every step of the route that drew each
  token; under temperature, the same distribution except for the top-k cut the vendored sampler applies
  (llama.cpp's check-first default accepts the same). With DFlash2 on,
  a token is drawn from a verify column, whose numerics can part from a plain round's at a near-tie, as
  they do for an unconstrained answer: a constrained answer equals the speculation-off answer up to that near-tie
  rule, never further.
- **Speculation stays on for a constrained request.** The host cuts a verified run at its first miss and repairs
  that position. No sibling's round loses its drafts or its graph because one request is constrained.
- **Two forms of permitted set.** The short set stays for schedules and forced literals; the mask is the second
  form. It goes down as the short set's lists do (ADR 0034), and no logits come up: the leaf replaces a lane's
  pending token, samples one position again from the scores the round kept, restricted by the mask, and makes a
  verify commit whose cut the host decides with the grammar's check.
- **The schema description goes after the last message.** It does not touch the part of the conversation that is
  reused across turns. A request field with an `ignis_` name turns it off for a client whose prompt already
  describes the schema.
- **The grammar is ignis's own** (`crates/grammar`), CPU-only, with no grammar engine vendored. Under `strict: true`
  it enforces OpenAI's strict subset exactly and refuses, naming the keyword or the limit, what it cannot
  guarantee; without `strict` it enforces what it can and names the rest as inert.
- **The same grammar holds a held tool's calls** (ADR 0094), in the dialect the model writes them in.

## Considered Options

- **A full mask every step**, as the other engines we read do, which they overlap with the forward pass.
  The result is the same, but it needs CPU work every
  step, a round that hands control back between its forward pass and its draw, and, under speculation, one mask per
  draft position, which forces a DFlash2 pass to read the drafts back to the host before verify.
- **Writing the grammar-forced text ourselves** (jump-forward). Deferred. One engine shipped it and removed it, and
  text tokenized unlike the model's own choice hurts the model's next predictions (DOMINO, arXiv 2403.06988).
  Describing the schema removed the key misses it would have saved.
- **Leaving the prompt as the client wrote it**, as other local engines do. Measured worse: about a hundred times
  the misses, an estimated several times the DFlash2 decode speed lost, keys forced in odd token splits, and one
  recursive schema that ran away to `max_tokens` (the
  [finding](../findings/2026-11-12-structured-output-grammar-measured-on-the-27b.md), which also has what a check,
  a repair and a schema's compilation cost).
- **Vendoring a grammar engine.** The inference path is ignis's own Rust; a JSON-schema automaton is a few
  thousand lines we can hold to a per-token reference and an independent validator.

## Consequences

- **The grammar produces a subset of the valid documents and refuses some schemas by name**: keys only in schema
  order, literals in canonical spelling, integers as digits, bounded numbers in plain decimal; under `strict`, no
  non-regular pattern and no keyword outside OpenAI's strict subset.
- **A constrained answer can loop**, and only `max_tokens` stops it (#584): the loop guard watches only the
  reasoning.
- **Flash-Next comes later (#581).** Its MTP verify round overwrites the scores inside the round.
- **The metrics (ADR 0017) count completed requests held to a grammar, their misses and their repairs**, a held
  tool's calls a kind of their own (ADR 0094), and time
  each repair where it happens, the one place such a request reads the clock: each miss is reported with its repair
  time as a fact of its own, so a round with no grammar adds nothing.
