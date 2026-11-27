---
status: accepted
---

# A locate narrows by attention heads and decides by a labelled choice, by kind of text

The texts a `locate` is for are logs, documents and record arrays of hundreds of thousands to millions of tokens,
and the head vote of ADR 0041 is one pass over the text, measured on short spans: on real logs it finds the right kind of
line
and loses the instance among its near-duplicates
([finding](../findings/2026-09-28-locating-a-line-in-real-logs.md)). Read at each line's end the heads keep the
right line among their first few candidates even at great length, and a labelled `choice` resolves near-duplicates
among a handful though it degrades among hundreds
([finding](../findings/2026-09-28-zero-decode-locate-exploration.md)). So by default the heads narrow and a `choice`
decides, with a reading per kind of text, and the vote stays as the other method.

- **`locate` has a `kind`, a `method` and a `compression`, all enums.** `kind` is `auto` (the default), `log`,
  `prose` or `records`; `method` is `shortlist` (the default) or `vote`; `compression` is `template_fold` or `none`,
  its default following the kind: a log is folded, prose and records are not. A new way to read, compress or tell
  the kind is a new value with its own measurement; an unknown value, and a combination measured bad or not applying
  (the vote with a fold, a fold of prose, a kind that contradicts a records array), is refused by name.
- **`shortlist`: the heads narrow, a labelled `choice` decides, and nothing is generated.** The heads' rows at the
  copy scaffold, less a content-free twin's, standardized and summed, rank the candidates, and the first few
  (templates, rows, sentences or records) go to the endpoint's own labelled `choice`. Logs and records are read by
  the **end heads** at each segment's closing keys, since at length a line is marked at its end; prose by the **sum
  heads**, the vote's, over each sentence's keys.
- **The last `choice` sees what makes its answer checkable**: a fold's rows as their original lines, prose
  candidates in their paragraphs, records whole.
- **`found`**: the same request asks the last `choice` again with a "none" option, and for logs a yes/no; a `found`
  under its threshold names no segment and keeps the ranking. Only the measured routes carry it: over a fold's
  values-only rows the "none" option fails.
- **A long text is read in windows**, cut at segment boundaries (paragraph breaks when there are any), each with its
  twin, the scores standardized per window and merged. The context stops being the limit of a `locate`.
- **`auto`**: a JSON array of objects is `records`; otherwise the fold's own statistic tells a log, a target whose
  first segments fall largely in shared templates, from prose.
- **The answer names the resolved kind, method and compression**, and carries `found` and `pointers`, every final
  candidate the `choice` gave a share worth naming: prose often answers with several sentences, and the `choice`'s
  probabilities give them.
- **The vote is kept unchanged** under `vote` with no compression, bounded by its measured ceiling. The calibration
  table gains the end heads, the sum heads and the window per artifact, and both methods need it: a load with no
  `locate` calibration refuses both.
- **The fold, the readings, the windows, `auto`, the renders and the `found` rules are pure host functions** held to
  the Python that measured them by golden cases. The fold is reversible through its map and computed once per target
  per request.

## Considered Options

- **A constrained copy of the line as the default**: equal on the measured sets, and it costs a decode lane per
  question and leaf and ABI changes (the draw reported in the call, a permitted set as wide as the vocabulary) that
  the heads and the `choice` do not. A future study, as a later `method` value (#279).
- **Booleans** (`fold: true`, `copy: true`): combinations by accident of spelling, and no room for a third value.
- **The vote with the end reading only**, one pass: better than the vote, and not enough among many near-duplicates.
- **The labelled `choice` alone**, over the fold or the whole log: it degrades with the number of candidates. It is
  a reader of short lists.
- **Generate the line**: it pays the whole log's prefill, and free text has to be matched back to a line.
- **One head reading over the whole text, with no windows**: bounded by the context and slower per key at length;
  windows read a text of a million tokens with the same heads.
- **Records as logs**, the fold over their one-line JSON, which `auto` did before: worse, because the fold's
  template key splits records whose values differ in word count and caps the selecting values out of level 1
  ([finding](../findings/2026-09-28-zero-decode-locate-paragraphs-not-found-records.md)). Kept as an opt-in,
  `records` with `template_fold`, for its latency, with no long prefill and without `found`.
- **"Not found" from the "none" option over a fold's values, or from a threshold on the `choice`'s confidence**: the
  first loses most found answers, and the second separates absent questions worse than the "none" option shown
  checkable candidates.
- **Checking the pick alone**: nearly as good for logs, worse for prose, where a sentence often answers only with
  another paragraph, and one more request.
- **Paragraphs as prose's pointer**: a better first pointer at the longest lengths, but measured with an instruction
  the caller does not send and with no better pointer set. A later value, once measured with the caller's
  instruction.

## Consequences

- A `locate` sent with no fields gets the shortlist of the kind `auto` resolves, and the answer says so; the vote is
  one field away.
- A `locate` can answer "not found": no segment, and a caller that wants the nearest candidate anyway reads the
  ranking.
- A `locate` generates nothing and holds no decode lane. A folded question is a few short prefills in sequence, and
  prose and records pay each window's prefill once per state.
- The rows' room reserved at load (ADR 0041) is sized for a window, not for the vote's ceiling.
- Folding removes lines' order and every time from the fold's first level: a question that needs context across
  lines, or names a line by its time alone, is better asked with `none`.
- `found` is uncalibrated, and prose at about a million tokens stays the weak point; both are research before they
  change.
- `ignis_locates_total`, by kind, method, compression and found, joins the metrics contract (ADR 0017).
- The acceptance on registered sets is partial
  ([finding](../findings/2026-09-28-locate-shortlist-acceptance-partial.md)): every rule judged on the runs made
  holds but the one on `auto`, which misread one window before the fold's rule was corrected, and the runs not made
  are a verification still to make.
