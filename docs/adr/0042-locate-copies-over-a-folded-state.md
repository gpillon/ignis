# ADR 0042 — a locate narrows by attention heads and decides by a labelled choice, by kind of text

## Status

Proposed (2026-09-28, revised twice the same day — spec
`docs/specs/decide/22-locate-by-copy-over-a-folded-state.md`, GitHub #278).
Accepted when spec 22's acceptance holds, with its numbers written here.
**Extends ADR 0041**, whose head vote becomes one of two methods, unchanged.
**Partial acceptance, 2026-09-28**
(`docs/findings/2026-09-28-locate-shortlist-acceptance-partial.md`): every
rule judged on the runs made holds but rule 9 (`auto` misread one Apache
window; the fold's rule is corrected since); the rest of the runs are a
verification still to make, and this ADR stays Proposed until then.
The file keeps the name of its first version, whose decision (a constrained
copy by default) this revision replaces before anything was built; the copy
is a future study (#279).

Sources: `docs/findings/2026-09-27-locate-at-length.md`,
`docs/findings/2026-09-27-the-instance-is-read-while-copying.md`,
`docs/findings/2026-09-28-locating-a-line-in-real-logs.md` (specs 20, 21),
`docs/findings/2026-09-28-zero-decode-locate-exploration.md`,
`docs/findings/2026-09-28-zero-decode-locate-very-long-logs-and-prose.md`
(spec 23) and
`docs/findings/2026-09-28-zero-decode-locate-paragraphs-not-found-records.md`.

## Context

ADR 0041 serves `locate` as a vote of 32 heads at the copy scaffold, one
prefill, measured to 4,554 keys. The texts a `locate` is for are logs,
documents and record arrays of hundreds of thousands to millions of tokens.
On real logs the vote finds the right *kind* of line and loses the instance
among its near-duplicates (20 of 58 on R2). The research after it found:

- the vote's **reading** is the weak link at length: a line is marked at its
  end (last token, separator, next line's first token), and 32 heads chosen
  for that on short sets find 72-74% of real-log lines in one pass;
- one pass still loses the instance among many near-duplicates, and a
  labelled `choice` over the heads' first 16 candidates resolves them
  (55-57 of R2's 58, no token written);
- folding the log into templates first removes the long prefill: the heads
  keep 5 templates and 16 rows, a `choice` decides at each level — 21 of 23
  on a fresh cluster capture of 100K to 1M tokens, registered (spec 23), in a
  median 2.6 s;
- prose does not fold, is read best by the heads' sum over a sentence, holds
  a gold sentence among the first 16 up to 1M tokens read window by window,
  and often answers with several sentences; the `choice`'s probabilities give
  them;
- an array of JSON records is read best by the end heads over the array
  itself, then a `choice` over 16 (58 of 60 to 3,500 records, 29 of 30 at
  10,000 in windows; the fold: 52 of 60), and is told by its shape;
- a "none" option in the last `choice`, read with a yes/no in the same
  request, says "not found" when that `choice` sees checkable candidates —
  a log's rows as their original lines (every found answer kept, 16 of 20
  authored and 41 of 58 target-removed absent questions flagged), prose in
  its paragraphs, whole records — and fails over a fold's values-only rows
  (37 of 100 found answers kept).

A constrained copy (this ADR's first version) resolves the instance too — the
free-generation route it would constrain read 55 of R2's 58 over the fold —
but needs a decode lane per question and leaf changes (the draw reported in
the call, a vocabulary-wide permitted set) that the heads-and-choice route
does not.

## Decision

- **`locate` has a `kind`, a `method` and a `compression`, all enums.**
  `kind` is `auto` (default), `log`, `prose` or `records`; `method` is
  `shortlist` (default) or `vote`; `compression` is `template_fold` or
  `none`, its default following the kind (`log` → `template_fold`, `prose`
  and `records` → `none`). A new way to read, compress or tell the kind is a
  new value with its own measurement; an unknown value, and a combination
  measured bad or not applying (`vote` with a fold, a fold of prose, a kind
  that contradicts a records array), is refused by name.
- **`shortlist`: the heads narrow, a labelled `choice` decides; nothing is
  generated.** The heads' rows at the scaffold, less a content-free twin's,
  standardized and summed, rank the candidates; the first few (5 templates,
  16 rows, sentences or records) go to the endpoint's own labelled `choice`.
  Logs and records are read by the **end heads** at each segment's closing
  keys; prose by the **sum heads** (the vote's) over each sentence's keys.
- **The last `choice` sees what makes its answer checkable**: a fold's rows
  as their original lines, prose candidates in their paragraphs, records
  whole.
- **`found`**: the same request asks the last `choice` again with a "none"
  option (and, for logs, a yes/no); `found` below 0.5 names no segment and
  keeps the ranking. Only the measured routes carry it.
- **Long texts are read in windows** of at most `LOCATE_WINDOW_KEYS` (200,000
  keys on the served 27B), cut at segment boundaries (paragraph breaks when
  there are any), each with its twin, scores standardized per window and
  merged. The context stops being the limit of a `locate`.
- **`auto`**: a JSON array of objects is `records`; otherwise the fold's own
  statistic — a target whose first 2,000 segments fall at least half in
  shared templates is a log.
- **The answer names the resolved kind, method and compression**, and gains
  `pointers` — every final candidate the `choice` gave at least 0.05 — and
  `found`.
- **The vote is kept** unchanged under `vote` + `none`, bounded by
  `LOCATE_MAX_KEYS`. The calibration table gains the end heads, the sum heads
  and the window per artifact; both methods need it.
- **`template_fold`** is `tools/locate-sets/compress.py`'s fold, ported as a
  pure host function and held to golden cases, reversible through its map,
  computed once per target per request. The readings, windows, `auto`, the
  renders and the `found` rules are likewise pure host functions held to the
  Python that measured them.

## Considered options

**The constrained copy as default** (this ADR's first version). Equal on the
measured sets; costs a decode lane per question and leaf and ABI changes. A
future study as a later `method` value (#279).

**Booleans** (`fold: true`, `copy: true`). Rejected: combinations by accident
of spelling, and no room for a third value.

**The vote with a better reading only** (the end heads' end reading, one
pass). 72-74% on real logs — better than 34.5-56%, not enough among many
near-duplicates (6 of 16 with six or more siblings).

**The labelled `choice` alone** over the fold or the whole log. It degrades
with the number of candidates (80-88% over 64 near-duplicates or 256 raw
lines; 16 of R4's 23 over the fold): it is a short-list reader.

**Generate the line.** 86.2% of R2 on the whole log, at its whole prefill,
and free text matched back to a line.

**One head reading over the whole text, no windows.** Bounded by the context
(262K tokens) and slower per key past ~200K; windows read a 1M-token text
with the same heads.

**Records as logs** (the fold over their spaced JSON, which `auto` did
before). 52 of 60 against 58: the fold's template key splits records whose
values differ in word count and caps the selecting values out of level 1.
Kept as an opt-in (`records` + `template_fold`) for its latency — no long
prefill — without `found`.

**"Not found" from the "none" option over a fold's values**, or from a
threshold on the `choice`'s confidence. The first kept 37 of 100 found
answers; the second separated absent questions worse (AUC 0.60-0.88) than the
"none" option shown checkable candidates (0.88-1.0).

**Checking the pick alone.** Nearly as good for logs (AUC 0.89-0.96), worse
for prose (0.85: a HotpotQA sentence often answers only with another
paragraph), and one more request.

**Paragraphs as prose's pointer.** A better first pointer at ~1M tokens
(35 of 36 against 32), measured with an instruction the caller does not send
and with no better pointer set: a later value once measured with the
caller's instruction.

## Consequences

- A caller who sent a `locate` with no fields got the vote and now gets the
  shortlist of the kind `auto` resolves; the answer says so, and the vote
  remains one field away.
- A `locate` can answer "not found" (`found` below 0.5, no segment); a caller
  that wants the nearest candidate anyway reads the ranking.
- A `locate` generates nothing and holds no decode lane; a folded question is
  a few short prefills in sequence (level 1, its `choice`, level 2, the last
  request), and prose and records pay each window's prefill once per state.
- A load with no `locate` calibration refuses both methods.
- The rows' room reserved at load grows from 32 x 4,554 keys to 32 x 200,000
  (25.6 MB on the served 27B).
- Folding removes lines' order and every time from level 1: a question that
  needs context across lines, or names a line by its time alone, is better
  asked with `none`.
- `found` is uncalibrated, carried only by the measured routes; prose at ~1M
  tokens stays the weak point; both are research before they change.
- `ignis_locates_total{kind, method, compression, found}` joins the metrics
  contract (ADR 0017).
