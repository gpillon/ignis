---
status: proposed
amends: [ADR-0034]
---

# A grammar checks each sampled token and repairs a miss, and the model is told the schema

Structured output (#287) needs a constraint that depends on what the model has written, and whose allowed set at a
step can be most of the 248,320-token vocabulary, where ADR 0034's permitted set holds at most 32 ids. Every engine
we read (2026-11-12) builds a full-vocabulary mask on the CPU at every step, overlaps that with the forward pass,
and under speculation builds one mask per draft position, which forces it to read the drafts back to the host
before the verify pass. ignis instead lets the model sample freely and checks each sampled token against the
grammar on the host before the model consumes it. Two things make that cheap here: a plain round's draw waits
host-side as the pending token, and after a verify pass the host already chooses how many licensed tokens to
commit before the recurrent state is folded. Only a **miss** sends a permitted set down, as a mask over the whole
vocabulary, to **repair** that one position from the scores the round left on the device. The server also
describes the schema to the model by default, at the end of the prompt, because a model that has not seen the
schema misses on nearly every key.

- **Same result as masking every step.** Identical under greedy; under temperature, the same distribution except
  for the top-k cut the vendored sampler applies (llama.cpp's check-first default accepts the same).
- **Speculation stays on for a constrained request.** The host cuts a verified run at its first miss and repairs
  that position. No sibling's round loses its drafts or its graph because one request is constrained.
- **Two forms of permitted set.** The short set (at most 32) stays for schedules and forced literals; the mask is
  the second form. It goes down as ADR 0034's lists do, and no logits come up.
- **The schema description goes after the last message.** It does not touch the part of the conversation that is
  reused across turns. A request field with an `ignis_` name turns it off for a client whose prompt already
  describes the schema.
- **The grammar is ignis's own** (`crates/grammar`), CPU-only, with no grammar engine vendored. Under `strict: true`
  it enforces OpenAI's strict subset exactly and refuses, naming the keyword or the limit, what it cannot
  guarantee; without `strict` it enforces what it can and names the rest as inert.

## Considered options

- **A full mask every step**, as everyone else does. The result is the same, but it needs CPU work every step, a
  round that hands control back between its forward pass and its draw, and a DFlash2 pass split to read the drafts
  back before verify.
- **Writing the grammar-forced text ourselves** (jump-forward). Deferred. One engine shipped it and removed it, and
  text tokenized unlike the model's own choice hurts the model's next predictions (DOMINO, arXiv 2403.06988).
  Describing the schema removed the key misses it would have saved.
- **Leaving the prompt as the client wrote it**, as other local engines do. Measured worse: 4.2 misses per 100
  tokens against 0.04, an estimated 8–18% of DFlash2 decode speed lost against 1–2%, keys forced in odd token
  splits, and one recursive schema that ran away to `max_tokens`.
- **Vendoring a grammar engine.** The inference path is ignis's own Rust; a JSON-schema automaton is a few
  thousand lines we can hold to a per-token reference and an independent validator.

## Consequences

- **Costs measured** ([finding](../findings/2026-11-12-structured-output-grammar-measured-on-the-27b.md)), on the
  27B's vocabulary:
  - a check takes 0.2 µs (1.0 µs at p99);
  - a repair's permitted set takes 34 µs on a state's first visit (0.71 ms at p99) and about 1 µs from the cache;
  - an SDK-sized schema compiles in about 2 ms, one at OpenAI's limits in about 27 ms, once per distinct schema;
  - in the spike's check-first runs, a repair took 26 µs typically and 107 µs at most.
- **The leaf gains three things:**
  - replacing a lane's pending token;
  - sampling one position again from the scores the round kept, restricted by a mask;
  - a verify commit whose cut the host decides with the grammar's check.
- **The grammar produces a subset of the valid documents and refuses some schemas by name.** It writes keys only in
  schema order, literals in canonical spelling, integers as digits and bounded numbers in plain decimal. It refuses
  non-regular patterns and keywords outside OpenAI's strict subset under `strict`. The finding lists both.
- **A constrained answer can loop**, and today only `max_tokens` stops it (#584). The loop guard watches only the
  reasoning.
- **Flash-Next comes later (#581).** Its MTP verify round overwrites the scores inside the round.
