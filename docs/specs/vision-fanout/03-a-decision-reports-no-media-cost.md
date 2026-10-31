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

## As built (2026-10-31, branch `245-decision-media-cost`)

Where the implementation chose what the sections above leave open.

- **The first shape: `ignis.request.done`.** Its line carries the same
  eight `media.*` attributes as `ignis.request.admitted`, from the same
  per-request state on the telemetry consumer (`Telemetry::on_done` builds
  the `MediaSummary` `on_admitted` builds), and none on a request without
  media. `admitted` keeps its fields; a request that takes a lane now reports
  the same numbers on both lines. Nothing moved onto the model thread: the
  acquisition stats arrive with the submission and the encode micros with the
  `PrefillChunk` events, as before, and only the line that prints them
  changed.
- **Not the second shape.** `ignis.decide.done` carries no media sum, for two
  reasons. Every reader of these fields reads them per question: the
  finding this spec came from, slice 02's sweep, the acceptance of slices 01
  and 04 ("the second question's `media.encode_seconds` is 0"), and the guide;
  nothing in the Playground, the bench or the docs reads a decide-level
  sum. And `ignis.decide.done` is emitted by the handler, not the consumer:
  the encode micros reach the handler only on each question's event stream,
  which the three collectors `ask` calls (`collect_readout`,
  `collect_attention`, `collect_draws`) drop today, so a sum there means
  tallying them in each collector and carrying the tally back through `ask`,
  `alone`, `in_waves`, `led` and the shortlist. Summing only the
  acquisition half the handler already holds would print a decide line whose
  biggest cost, the encode, is missing. It stays open as its own ticket if an
  operator asks for the per-call figure: the request log still cannot join a
  decide line to its questions (`log_decision`'s doc).
- **Which decisions this covers.** Every decision answered by its prefill:
  `noul`, `choice` and `score`, and since GitHub #260 a `point` or `box` read
  from the heads, which on a calibrated load is a `point`'s default. A
  measurement that wants a lane-taking question over an image for
  comparison asks a `point` with `"method": "chain"`, or a `box`.
- **The CPU seam** is `crates/server/tests/media_request_log.rs`: a `noul`
  over a PNG through `/v1/decide` on the mock, which logs no `admitted` and
  reports its media on `done`, and the same image through a chain `point`,
  whose `admitted` and `done` carry the same numbers as the `noul`'s `done`
  (`preprocess_seconds`, each request's own wall clock, compared for
  presence). Its `EncodingCompute` now keeps the mock's outcomes, so a
  readout still reaches its caller, and reports encode time only on a chunk
  that covers an image placeholder, as the leaf does.
