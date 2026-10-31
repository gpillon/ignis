# 03 - a readout-only decision over an image reports no media cost at all

The `media.*` attributes — `encode_seconds`, `preprocess_seconds`,
`vision_tokens`, `bytes`, `cache_hits`, `cache_misses` — ride
`ignis.request.admitted`. That event fires when a request is **admitted onto
a decode lane**, and a decision made of readouts never takes one: it ends
where its prefill ends (GitHub #238).

So an image put to `noul`, `choice` or `score` encodes its picture, spends
seconds doing it, and reports **nothing**. The same image put to `point` or
`box` reports everything, because a constrained decode does take a lane.

## How it was found

Measuring the encode against image size
(`docs/findings/2026-09-20-number-width-and-decide-e2e.md`, slice 02's
table): the first sweep used a `noul` question and came back with an empty
log. Re-running it with a `point` produced every row. The cost was always
being paid; only the reporting depended on the primitive.

## Why it matters beyond this session

An operator's only view of what an image costs is this event. A fan-out of
twenty `choice` questions over one screenshot is the most expensive thing
this endpoint can be asked to do — twenty encodes, twenty prefills of 16K
tokens — and on today's log it is invisible. `ignis.decide.done` carries the
request-level summary but no media at all.

## Shape

Two candidates, and the choice is the point of the ticket:

- Emit the media attributes on a completion event every request reaches,
  rather than on admission. A decision reaches `ignis.request.done`.
- Or add them to `ignis.decide.done`, summed over the fan-out, which is
  where a reader looking for "what did this decide request cost" already
  goes.

The first is narrower and fixes every caller; the second answers the
question an operator actually asks. They are not exclusive.

## Acceptance

1. A `noul` over an image reports its `media.encode_seconds` and
   `vision_tokens` somewhere a log reader will find them.
2. The numbers match what the same image reports through a `point` question
   today — this is a reporting gap, not a measurement one.
3. A text-only request reports no media fields, as now.
4. Nothing new is computed on the model thread to satisfy this: the stats
   already exist, they are attached to the wrong event.

## References

- Finding: `docs/findings/2026-09-20-number-width-and-decide-e2e.md`.
- `crates/server/src/telemetry.rs` (`note_media`, `on_admitted`),
  `crates/server/src/media.rs::MediaStats`.
- GitHub #238 (why a decision never takes a lane), #241 (`ignis.decide.done`).
- ADR 0011 (the request log), ADR 0017 (what belongs on the metrics
  listener instead).
