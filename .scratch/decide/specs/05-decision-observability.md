# 05 - decision observability

GitHub: #241

A readout generates nothing, so `ignis_decoded_tokens_total` does not move and a
decision is invisible to every existing panel. Worse, the failure this endpoint
can actually have is silent: if **answer mass** collapses, the answers are
well-formed noise with a plausible argmax.

- A counter of decisions, labelled by primitive type.
- A histogram of answer mass. The measured baseline is p50 >= 0.996 across
  widths from 8 to 256 options, and >= 0.999 with an image as evidence.
- `confidence` is **not** exported. It is per-caller and per-domain; an aggregate
  histogram over callers with different thresholds means nothing.
- The request log records a decision the way it records a request, with the
  question count and the primitive types.

## Acceptance

1. A served decision increments the counter for its type and observes its mass.
2. `ignis_decoded_tokens_total` is unchanged by a readout-only request.
3. The metrics appear on the metrics listener and are absent before any decision
   is served (no zero-valued clutter for a server that never sees one).

## References

- ADR 0017 (the Prometheus contract), ADR 0034.
- Findings: the measured mass baselines in
  `2026-09-19-typed-option-logit-readout.md`.
