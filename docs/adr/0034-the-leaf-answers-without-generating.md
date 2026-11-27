---
status: accepted
---

# The leaf answers without generating

Most of what an agent asks a model is a decision, not a request for prose: route this, retry that, which queue owns
this ticket, where is the button. A chat model answers by generating a sentence that software parses back into a
branch, paying a decode round per token and a parser that has to survive every shape the model might choose, while
the answer is already in the model before a token is generated: a prompt that names each option by a label puts the
decision in one position's next-token distribution
([finding](../findings/2026-09-19-typed-option-logit-readout.md)). So the leaf grows two paths beside decode, and
the sampling invariant is restated for them rather than excepted.

- **A readout receives logits and draws no sample.** The answer tokens go in across the `Compute` seam and the
  readout comes out. The gather happens below the seam: the full-vocabulary buffer never crosses it, only the answer
  logits, the full-vocabulary log-sum-exp and the unrestricted argmax.
- **A constrained decode restricts sampling, per lane, in the leaf.** A readout reads one position, so it yields one
  symbol and never a number or a point: those need a run of positions, each restricted to its alphabet
  ([finding](../findings/2026-09-19-constrained-digit-readout-points.md)). The sampling ABI takes a set of permitted
  token ids; the leaf samples only among them and still returns a token id. No logits cross for this path, so it
  composes with temperature, with seeds and with speculation exactly as decode does. A permitted set goes down in
  two forms, the short list of a schedule or a forced literal and a mask over the whole vocabulary, with which a
  grammar repairs one position from the scores the round left on the device (ADR 0089); in both, no logits come up.
- **The invariant is the decode path's.** *Per-lane sampling* reads: the decode path never ships logits to the host;
  a readout is the one path that does, and it draws no sample: no seed, no RNG, no penalty history. Lane
  independence, that what a request generates depends on its own seed alone and never on which lanes shared its
  round, is stronger stated this way, because a readout has nothing with which to disturb another lane.
- **The answer alphabet is computed from the loaded tokenizer, never compiled in.** A label is admitted only if it
  encodes to exactly one token that decodes back to itself: a two-token label would have its first token's logit
  read, a token that belongs to some other label. This is a correctness rule, not an optimization.

## Considered Options

- **Host-side constraint for the digits**: read the logits each round, pick on the host, force the token back. It
  needs no kernel change. Rejected: a host synchronization and a full-vocabulary buffer across the bus per digit, on
  a lane that blocks its round meanwhile, and it would contradict the invariant. It is fine alone on a card and
  wrong under load.
- **Generate and parse, with no new paths**: it remains the right answer for anything that wants prose. Rejected for
  a decision: it costs a decode round per token where a readout costs none, it cannot return a calibrated
  distribution over the options, only the one the model picked, and the parser has to survive every output shape,
  which, measured, is more than one.
- **A separate classification model**: the served model already knows the answer. A second set of weights spends
  VRAM that the first model's KV wants, and gives up the property that makes this worth doing, that the decision is
  made by the same model that would have written the sentence.

## Consequences

- The `Compute` trait is no longer purely "give me tokens". It is the only GPU-coupled seam in the engine and it
  carries a second kind of answer, so every CPU-only implementation of it, the mock above all, must produce a
  deterministic readout, or the scheduler's tests stop covering the path (ADR 0006).
- A decision holds no residency and takes no decode lane, so eviction priority has nothing of its to take and the
  admission classes mean something different for it. That is why a decision request defaults to `Agent` where every
  other route defaults to `Interactive`.
- The sampling ABI carries a per-lane token set, which every future sampling change has to carry too. That is the
  cost of not having the host in the loop.
- `ignis_decoded_tokens_total` does not move for a readout, which is correct and makes decisions invisible to the
  existing panels. A counter of decisions and a histogram of the **answer mass** exist because of it: a silent
  collapse of answer mass in production means well-formed noise, and it is the only failure of this endpoint that
  nothing else would show. Both are counted per question, and both are absent from the exposition until a decision
  has been served (ADR 0017 carries them, and the reason for the absence). `confidence` is deliberately not
  exported: it is per caller and per domain, and an aggregate over callers with different thresholds means nothing.
- The endpoint can be reached by an unmodified client of the decision API whose wire shape it copies, which is the
  reason to copy that shape rather than invent one. That API's vocabulary (`noul`, `criteria`, `instructions`)
  enters the glossary as imported, not ours.
