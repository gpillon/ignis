# 22 - locate over very long texts: heads narrow, a labelled choice decides

> **Registered 2026-09-28; revised twice on 2026-09-28, before sets R3, P3
> and J3 exist.** The first version made a constrained **copy** the default;
> the research that followed (spec 23 and its findings) measured a route that
> writes no token at all at least as well, and the owner chose it. The second
> revision replaced the default, added `kind`, and re-registered the
> acceptance. This third revision takes the last round of research
> ([paragraphs at 1M, not found, JSON records](../../findings/2026-09-28-zero-decode-locate-paragraphs-not-found-records.md)):
> a **`records`** kind, a **`found`** answer ("not found"), the log route's
> last `choice` over the rows' original lines, and the acceptance re-registered
> with absent questions and a records set. The acceptance rules (§ The
> acceptance run) and the sets' seeds are fixed here and are not edited once
> R3's, P3's and J3's manifest hashes are recorded in this banner. A failed
> rule is reported as failed and the defaults do not ship until the owner
> decides; this spec does not pre-decide a fallback.
>
> **Sets registered 2026-09-28, before any route was asked on them** (the
> builders and the judge: commits da56f5c and e9e317e — `r3set.py`,
> `accept22.py`, `r3_judge.py`; `prosehay.py`, `zd_prose_nf.py` and
> `zd_records.py` unchanged). Manifest sha256:
>
> - **R3** `19ee6376a81567ec4c9b4955e762800d9fb3cc46118d1b1cc3f6a71897edc9a2`
>   (`r3set.py build`: windows `bd66600822c15069f0672cfef47294fdb2f5b42ff828d41aa97138597f39e7a9`,
>   questions `126fbc57f3880c30af15d52191a7aa7e8fc1716eb165f8264c97e419d991bc1b`):
>   a capture of the owner's cluster on 2026-09-28, 12:45-12:46Z,
>   `--since=6h --timestamps`, 147 running pods, read only (287,001 lines
>   merged); the full LogHub logs, Zenodo record 8196385. 42 windows (10
>   cluster, 32 LogHub), 127 targets drawn; **120 present** questions (39
>   cluster, 81 LogHub), 120 target-removed and 42 authored absent; 16
>   windows at 100K tokens hold a present question. **Shortfalls the
>   sources forced:** three windows drew no target (HPC 100K, Zookeeper 100K
>   and 200K: every line repeats another but for numbers) and 7 drawn
>   targets were dropped because no question can single them out (each
>   shares every content word with another line: c0050k-0:381,
>   h-hadoop-100k:158, h-mac-100k:315, h-mac-200k:121,
>   h-proxifier-100k:1613, h-proxifier-200k:598, h-windows-100k:220).
>   **Two departures from § Set R3, decided before any run:** a LogHub
>   system's stream is the first 8 MB of its log with every line of its 2k
>   sample removed — so "sharing no line with that system's 2k sample" holds
>   by removal, and a LogHub window is the log less those lines (0 to 6,006
>   per system with the blank ones, most about 2,000), not always a
>   contiguous stretch; and past
>   100K tokens a window's targets are drawn from 400 sampled eligible
>   lines, `r2set.py`'s sibling count being quadratic in the window
>   (`z23_r4.py` sampled 120 for R4).
> - **P3** `34eea67ae01f7607834eed451c97927c85f3c2d8f0e90d4265d7d8f90ad1d721`
>   (108 present questions over 18 windows; E1-E3 excluded with A-F, P and
>   P2), and its absent set
>   `e249c30a553b7420b02bf07fc2ddfeecd2a159c0ec76319cceeea27b92136cff`
>   (72 gold-removed, 12 from another 200K window).
> - **J3** `a8f7d66f3beda649c7e00cf5c82c7abef712c8d3dfece56428b1e2186d956481`
>   (9 arrays, 90 present and 18 absent questions).
>
> **What the implementation decided before the runs** (and the acceptance
> therefore judges): a window is `LOCATE_WINDOW_KEYS` less a margin of an
> eighth, at most 256 keys, and less still on a load whose context would not
> hold it (P3's 200K windows, ~200,100 tokens, read as two); one segment
> longer than a window is refused (`locate_segment_too_long`, § Refusals); a
> fold's level-1 text past the context is refused and a level-2 text past a
> window is windowed; a step rendered from an earlier step's answer (a
> fold's level 2, every `choice`) that faults answers its question with an
> error rather than a 422, since it runs after a prefill.

GitHub: #278. ADR: [0042](../../adr/0042-locate-copies-over-a-folded-state.md) (Proposed, revised).
Later: a constrained copy (`method: "copy"`) is a future study, #279.

`locate` (spec 18, ADR 0041) names the **segment** of a `state` an
instruction asks for, today by one prefill's **head vote**, measured to 4,554
keys. The texts a `locate` is for are much longer — logs of hundreds of
thousands of lines, documents and record arrays past the context — and there
the vote finds the right *kind* of line and loses the instance. This spec
ships what the research on very long texts found (specs 20, 21 and 23 and
their findings): **attention heads narrow the text to a few candidates and a
labelled `choice` decides among them**, with no token generated, and the same
`choice` says when **nothing** answers. Logs are first **folded** into
templates and their values; prose and record arrays are read by the heads
**window by window**; the kind of text is told by the state's shape and by
the fold. The caller can choose everything by name, as three enums on the
question: `kind` (`auto`, `log`, `prose`, `records`), `method` (`shortlist`,
`vote`) and `compression` (`template_fold`, `none`).

One ticket, whose acceptance is numbered at the end.

## Problem Statement

A caller with a very long text wants the place that answers an instruction:
in a log, the one line — the failed write, the refused login, the pod that
restarted; in prose, the one or several passages that answer a question; in
an array of records, the record. The texts are 16K tokens to past a million.
And sometimes the text does not hold the answer at all.

- **The served vote refuses past 4,554 keys** (`locate_too_long`), and
  lifting the ceiling does not help on real logs: 28 of 50 on set R, **20 of
  58 (34.5%)** on R2, falling with length
  ([locate at length](../../findings/2026-09-27-locate-at-length.md),
  [real logs](../../findings/2026-09-28-locating-a-line-in-real-logs.md)).
- **Its misses are near-duplicates, and its reading, not its heads, is the
  weak link.** The vote sums each head's attention over a line's keys. At
  length the line is marked at its **end** (its last token, the separator,
  the next line's first token): read there, by 32 heads chosen for it on
  short sets, one pass finds 72-74% of real-log lines against the vote's
  34.5-56%; a line's first key is where no head reads it
  ([zero-decode locate on real logs](../../findings/2026-09-28-zero-decode-locate-exploration.md)).
- **One pass still loses the instance among many near-duplicates** (6 of 16
  targets with six or more siblings); a second, short pass over the heads'
  first candidates resolves them.
- **Generating the line works and costs too much**: 86.2% of R2 on the whole
  log, but the first question pays the whole log's prefill (47-139 s) and
  the answer is free text matched back to a line.
- **Prose is not logs.** It does not fold (the fold groups 4-8% of its
  lines), its answers often span several sentences in several paragraphs,
  and it is read best by the heads' sum over a sentence, not its end.
- **Records are neither.** An array of JSON objects folds badly (the fold's
  template key splits records whose values differ in word count) and has few
  near-duplicates per question; the end heads read it directly.
- **Every `locate` answers something**, even when the text holds nothing
  that answers; and nothing past the context is answered, and the caller
  cannot choose a reading or a compression, nor get more than one segment
  back.

## Solution

Three fields on a `locate` question, all enums, all optional:

- **`kind`**: `"auto"` (default), `"log"`, `"prose"` or `"records"` — which
  reading the text gets. `auto` says `records` when the state is an array of
  JSON objects (§ `auto`), else folds the target's first 2,000 segments with
  content and says `log` when at least half of them fall in templates of two
  or more, `prose` otherwise (logs measured 0.96-0.98, prose 0.04-0.07 on
  every fresh window, spec 23's HA; the shape rule 400 of 400 record arrays).
- **`method`**: `"shortlist"` (default) or `"vote"`.
- **`compression`**: `"template_fold"` or `"none"`; its default follows the
  kind — `template_fold` for a log, `none` for prose and records.

| | `kind: log` | `kind: prose` | `kind: records` |
|---|---|---|---|
| **`shortlist` + `template_fold`** | (default) fold; the **end heads** read the templates and keep 5; a labelled `choice` picks one; its rows (values): the end heads keep 16 (all when ≤ 16); a last labelled `choice` over **those rows' original lines** picks the line; `found` | refused: prose does not fold | opt-in: the `log` route over the records as spaced-JSON lines, its last `choice` over the original lines as for a log (measured 52/60 with the older last step over the rows' values, against 58/60; no long prefill); no `found` |
| **`shortlist` + `none`** | the end heads read the whole target in windows and keep 16 lines; a labelled `choice`; no `found` | (default) the **sum heads** read the target in windows and keep 16 sentences; a labelled `choice` over them *in their paragraphs*; pointers at p ≥ 0.05; `found` | (default) the **end heads** read the array in windows and keep 16 records; a labelled `choice` over them as spaced-JSON lines; `found` |
| **`vote` + `none`** | today's `locate`, byte for byte, whatever the kind; no `found` | the same | the same |
| **`vote` + `template_fold`** | refused (read 28 of R2's 58) | refused | refused |

- **`template_fold`** is `tools/locate-sets/compress.py`'s fold, as the first
  version of this spec described it: templates Drain-style, **level 1** one
  line per template with its slots' distinct values and `(xN)`, **level 2**
  the chosen template's lines as their values, values first and time last; a
  map back to the original segments. A 100K-token log folds to a median
  ~10K-token level 1; a 1M-token log to 25-34K.
- **The heads' reading** is a lift, as the vote's: each head's attention row
  at the scaffold, less the content-free twin's, **standardized over the
  candidates** and summed over the heads. The **end reading** (logs, records)
  scores a segment by its last key, its separator and the next segment's
  first key; the **sum reading** (prose) by all its keys. Candidates are kept
  in document order.
- **The labelled `choice`** is the endpoint's own `choice` over the
  candidates, each prefixed with its answer label. It generates nothing: it
  reads the labels' logits, as every `choice` does. What it is shown is what
  makes its answer checkable: a log's last `choice` sees the original lines
  (a level-2 row is its values only), prose sees each candidate inside its
  paragraph, a record is shown whole.
- **`found`**: in the same request as the last `choice`, a second `choice`
  over the same candidates plus one option "no line (sentence) of the
  evidence answers the criterion", and for logs a yes/no "Is there a line in
  the evidence that answers this question: …". `found` is a number; below
  0.5 the answer names no segment.
- **Windows.** A text the heads read that is longer than the calibrated
  window (`LOCATE_WINDOW_KEYS`, 200,000 keys on the served 27B) is cut into
  windows at segment boundaries — at an empty segment when one is within the
  window (prose's paragraph breaks), else at the last segment that fits (a
  record, a line) — each read as its own prefill with its own twin; scores are
  standardized within a window and merged. The text's raw length is then
  bounded by the request, not the context.

Measured (the numbers are spec 23's and the findings'; the third round's are
exploratory):

| logs, top-1 | R (50) | R2 (58) | ~1M-token logs (41) | **R4, fresh (23)** |
|---|---|---|---|---|
| `shortlist` + `template_fold`, last `choice` over the rows' original lines | 46 | 55 | — | 21 |
| the same, last `choice` over the rows' values (spec 23's route) | 45 | 55 | 36 | **21 (91.3%, registered)** |
| labelled `choice` alone over the fold | 42 | 53 | 33 | 16 |
| the heads alone at both levels | 30 | 29 | 17 | — |
| served vote | 28 | 20 | — | — |
| median wall time | 0.8 s | 1.3 s | 5.0 s | 2.6 s |

| prose (HotpotQA gold sentences in a haystack) | ≤200K | 1M |
|---|---|---|
| a gold sentence in the sum heads' first 16 | 100% | 100% |
| the `choice`'s pick is a gold sentence (fresh P2) | 91.7% | 58% (7/12; 10/12 by paragraph) |
| paragraph pointer F1 (fresh P2, all) | 0.77 | |

| records (spec 18's kinds, 5 targets per array) | 1,000-3,500 records (60) | 10,000 records, ~550K tokens (30) |
|---|---|---|
| `records` + `none`: end heads' first 16 → `choice` | **58** | **29** |
| the end heads alone, top-1 | 52 | 27 |
| `records` + `template_fold` (the `log` route) | 52 | — |
| wall time | first question: each window's prefill (8 s at 55K, ~45 s per 200K); then 1.0-3.3 s per window + 0.3 s | |

| `found` (below 0.5 = not found) | present answers kept | absent flagged |
|---|---|---|
| logs R+R2 (authored absent 20; target removed 58) | 101/101 | 16/20; 41/58 |
| logs R4 (target removed 23) | 21/21 | 20/23 |
| prose P2 ≤200K (gold paragraphs removed 72; another window's question 12) | 66/67 | 45/72; 12/12 |
| records (a value no record holds 12 + 6) | 57/58, 28/29 | 12/12, 6/6 |
| logs, the "none" option over the rows' **values** instead (not shipped) | 37/100 | — |

Every answer is an index into the **original** state, with the segment's
value as the caller sent it.

## User Stories

1. As an agent reading a 100K-token service log, I want to ask which line is
   the failed write and get its index in about a second, so that finding a
   line does not cost the log's whole prefill.
2. As that agent, I want the answer to be the instance I described — this
   session's socket error, not another's — so that I act on the right line
   among its near-duplicates.
3. As an agent with a million-token log, I want a `locate` to answer at all,
   so that the context is not the limit.
4. As an agent reading a long document, I want every passage that answers my
   question — one or several — so that a two-part answer is not cut to one.
5. As an agent holding an API's answer of thousands of JSON records, I want
   the record my instruction describes, so that I do not scan the array.
6. As an agent, I want a `locate` to tell me when the text holds no answer,
   rather than name its nearest miss, so that I do not act on a wrong line.
7. As a caller, I want `found` as a number with the documented threshold, and
   the candidates still ranked when it is below, so that I can apply my own
   threshold or take the best guess knowingly.
8. As a caller, I want the default to be the route that measured best for my
   kind of text, and the kind told for me when I do not say it, so that I
   do not have to know the research to get the good answer.
9. As a caller, I want to name the kind, the method and the compression, so
   that I can trade accuracy, latency and context for my case — a folded
   record array answers without the long prefill, and less often right.
10. As a caller, I want all three to be enums that name what they do, so that
    a new reading or compression later is a new value, not a contradicting
    flag.
11. As a caller, I want an unknown value refused with the accepted values
    named, so that a typo never runs the default silently.
12. As a caller, I want a combination that was measured bad or does not apply
    (`vote` with a fold, a fold of prose, `records` on a state that is not an
    array of objects) refused by name, so that I never get an unmeasured
    answer.
13. As a caller, I want `kind` and `compression` refused on every other
    question type, so that a field I wrote is never ignored.
14. As a caller, I want the answer to name the kind, method and compression
    that produced it, so that a default I did not write is visible.
15. As a caller, I want `segment` to index the state I sent, whatever was
    folded or windowed, so that I map the answer back without the server.
16. As a caller, I want `pointers` — every candidate the `choice` gave at
    least 5% — so that I can take one answer or several.
17. As a caller, I want to know which line I get when several are the same
    text (identical, or identical but for their time), so that a repeated
    event gives a predictable answer: the first.
18. As a caller, I want `confidence` and `found` documented as what they are —
    the `choice`'s probabilities, never calibrated probabilities — so that I
    set thresholds knowingly.
19. As a caller asking several `locate`s over one state, I want the fold
    computed once and every prefill a later question can share claimed, so
    that the second question costs a question.
20. As a caller mixing `locate` with `noul` or `choice` over one state, I
    want every answer back in one request.
21. As a caller whose question needs context across lines, or names a line by
    its time alone, I want the documentation to say that folding removes
    that and `compression: "none"` keeps it.
22. As a Playground user, I want the Decide tab to offer the kind, method and
    compression beside a `locate` and show every pointer and `found`, so that
    I can compare them on my own evidence.
23. As an operator, I want to count `locate`s by kind, method, compression and
    whether they found an answer, and to see per question how many windows,
    templates, rows and candidates it read, so that a slow or wrong answer is
    attributable.
24. As a maintainer, I want the fold, the readings, the windowing, the merge,
    the candidate renders, `auto` and the `found` rule to be pure host
    functions held to golden cases the Python reference writes, so that the
    served route is the measured route.
25. As a maintainer, I want the vote and every non-`locate` request unchanged
    bit for bit.
26. As the owner, I want the defaults judged once on fresh real logs, fresh
    prose and fresh record arrays, present and absent questions, with rules
    written here before the sets exist, and the vote and short states
    guarded, so that the new default is not tuned to its own test.

## Implementation Decisions

### The wire

- **`kind`** on a `locate`: `"auto"` (default), `"log"`, `"prose"`,
  `"records"`. Unknown: `kind_unknown`, naming the four. On any other type:
  `kind_unsupported`.
- **`method`** on a `locate`: `"shortlist"` (default) or `"vote"`. The field
  exists for `point` and `box` (`head`, `chain`); spec 18's refusal of it on
  a `locate` is lifted. An unknown value on a `locate` is `method_unknown`
  naming `shortlist` and `vote`; `shortlist` and `vote` on a `point` or `box`
  are `method_unknown` naming theirs; on every other type `method` stays
  `method_unsupported`.
- **`compression`** on a `locate`: `"template_fold"` or `"none"`, default by
  the resolved kind (`log` → `template_fold`, `prose` and `records` →
  `none`). Unknown: `compression_unknown`; on any other type:
  `compression_unsupported`.
- **Refusals, all before any prefill** (a 422, nothing reaches the engine):

  | code | when |
  |---|---|
  | `compression_unsupported` | `template_fold` with `kind: "prose"` (prose does not fold) or with `method: "vote"` (measured 28 of R2's 58) |
  | `kind_mismatch` | `kind: "records"` on a state that is not a records array (§ `auto`), or `kind: "log"` / `"prose"` on one (a records array is read as records; the fold of it is `records` + `template_fold`) |
  | `locate_uncalibrated` | the load has no `locate` calibration — for either method: both read heads |
  | `locate_too_long` | `vote` only, as today: the target past `LOCATE_MAX_KEYS` |
  | `locate_segment_too_long` | one segment longer than a window: nothing smaller can be cut from it |
  | `context_exceeded` | a level-1 text, a level-2 text or a `choice` prompt past the context (a fold's level-2 text past `LOCATE_WINDOW_KEYS` is windowed, not refused) |
  | `locate_too_few_segments` | fewer than two segments with content (unchanged) |
  | `locate_unsupported` | the loaded template cannot say where its tokens sit (unchanged) |

- **The answer** keeps `Answer::Locate`'s shape and gains five fields:

  ```json
  {"type": "locate", "kind": "log", "method": "shortlist", "compression": "template_fold",
   "found": 0.97,
   "segment": 1482, "value": "…the line as sent…", "confidence": 0.91,
   "ranking": [{"segment": 1482, "share": 0.91}, {"segment": 1480, "share": 0.05}],
   "pointers": [{"segment": 1482, "value": "…", "share": 0.91},
                {"segment": 1480, "value": "…", "share": 0.05}]}
  ```

  - `kind` is the resolved kind (`auto` never appears).
  - `segment`, `value` and `confidence` are the pick: the last `choice`'s
    most probable candidate. Under a fold, `confidence` is the product of the
    two levels' pick probabilities.
  - `ranking` is the last `choice`'s candidates by probability (at most
    five), each share multiplied by the earlier level's pick probability, so
    the first entry is the pick and its share is `confidence`.
  - `pointers` is every candidate of the last `choice` whose share is at
    least **0.05**, best first — always at least the pick. The threshold is
    the one chosen on P's dev split (spec 23 froze it).
  - **`found`** (§ Not found) is present for the three measured routes —
    `log` + `template_fold`, `prose` + `none`, `records` + `none` — and
    absent otherwise. When it is **below 0.5**, `segment`, `value` and
    `confidence` are `null` and `pointers` is empty; `ranking` still lists the
    candidates.
  - **Several segments with one text** (a fold's row of lines identical but
    for their time): the first. `vote` + `none` is unchanged.
  - `vote` + `none`: today's answer plus `kind` (resolved, unused),
    `method`, `compression`, and `pointers` holding the winner alone.
- **`usage`**: `input_tokens` counts every prefill (windows, twins, levels,
  `choice`s); `output_tokens` is 0: nothing is generated.
- **Thinking** is refused as for every decision.

### `auto`

A pure host function, computed once per target per request:

1. **A records array**: the target is a JSON array of at least two elements,
   every element an object, and not an array `Evidence::read` reads as
   content parts (every element an object with a string `type`) — `records`.
   It needs no statistic: all 400 record arrays of sets A-F and every J array,
   no log or prose state.
2. Otherwise fold (`template_fold`, below) the target's first 2,000 segments
   with content, and say `log` when the segments in templates of two or more
   are at least half of them, `prose` otherwise. An array that is not a
   records array (strings, numbers, mixed) folds as its elements' text.

### `template_fold`

Unchanged from the first version of this spec: `tools/locate-sets/compress.py`'s
`fold` with `values=True`, `summarize`, `level2` and `_common_affixes` at the
settings R2 was judged with (`SIM` 0.5, the `TIME`, `VARIABLE`, `LABEL` and
`TOKEN` patterns, a slot's values within a 600-character budget, else the
first 6 cut to 24 characters and `|+N`, `(xN)`, level-2 rows values-first with
their first time last), ported as a **pure host function in `ignis_core`**
with Python's arithmetic (code points, Unicode classes, first-seen cluster
order). It folds a string's lines as sent and an array's elements (a string
as its text, anything else as its **spaced JSON** — Python's
`json.dumps(element, ensure_ascii=False)`: `", "` and `": "` separators,
keys in the order sent); empty segments are left out. The map (template, row)
→ original segments; the answer is the first. Kept as measured, and
documented: a bracket-opened line is read as a source label — unless the
bracket holds a time, which is then the line's timestamp (corrected
2026-09-28 after R3's first run: Apache's `[Sun Dec 04 04:47:44 2005]` and
Proxifier's `[10.30 16:49:06]` put every line in a template of its own); level
1 drops every time; folding removes lines' order and neighbours.

### The heads and their readings

- **The calibration** (`ignis_core::locate::LocateCalibration`) gains, per
  artifact: the **end heads** (the `log` and `records` reading), the **sum
  heads** (the `prose` reading) and **`LOCATE_WINDOW_KEYS`**. On the served
  NVFP4 27B:
  - end heads, in rank order (spec 23's `endheads.json`, chosen on sets A+B
    by single-head top-1 at a line's last key, next first key or separator):
    L47.h20, L47.h4, L51.h4, L47.h17, L47.h1, L51.h12, L51.h23, L47.h5,
    L47.h3, L47.h15, L39.h15, L47.h9, L47.h13, L51.h16, L39.h0, L39.h12,
    L43.h22, L43.h20, L47.h2, L39.h23, L55.h13, L43.h9, L43.h18, L51.h17,
    L55.h23, L43.h7, L51.h2, L55.h20, L47.h23, L35.h18, L43.h8, L43.h14;
  - sum heads: the vote's 32, as calibrated (ADR 0041);
  - `LOCATE_WINDOW_KEYS` 200,000 (spec 23 read prose in windows of at most
    210,000 tokens; the records were read in windows of 200,000 tokens).
  `max_keys` (4,554) stays the vote's ceiling alone.
- **A reading is one prefill per window and its content-free twin** (the
  vote's render: layout L1, the kind text, the instruction, the forced
  `{"quote":"`; the twin's instruction `N/A`), the readout returning the 32
  heads' rows over the window's span (ADR 0041's rows readout). A records
  window is the sub-array itself, as a JSON state — its segments are its
  records, as a `locate` segments any JSON array. The rows' room is reserved
  at load for 32 heads x `LOCATE_WINDOW_KEYS` (25.6 MB of f32 on the served
  27B; ADR 0030), and stated in the VRAM plan.
- **The host reading**, pure and held to golden cases from the Python
  reference (`tools/locate-sets/zd_offline.py`'s `zsum`, `zd_cache.py`'s
  `key_features`, `zd_windows.py`'s `sub_windows`, `zd_records.py`'s `cut`,
  `zd_prose.py`'s merge): each head's softmax over the window's span; per
  segment the question's mass less the twin's at the segment's keys (sum), or
  at its last key, the keys between it and the next segment and the next
  segment's first key (end); each head's lifts standardized over the window's
  segments and summed over heads; windows' scores standardized and
  concatenated; empty segments and, for prose, lines that open with `# `
  (titles) never candidates.
- **The shortlist** is the first K segments by that score, shown in document
  order: K = 5 templates at a fold's level 1, K = 16 rows at level 2 (all
  when ≤ 16), K = 16 lines, sentences or records without a fold.

### The labelled `choice`

- The endpoint's `choice`, as served, over the candidates labelled with its
  answer alphabet (spec 18's labelled route): a derived state holding the
  candidates as `label: text` lines, the instruction as sent, one option per
  label. No new primitive.
- **What each route shows** (each held to golden cases from its reference):
  - a fold's level 1: the kept templates' level-1 lines
    (`zd_logpipe.py`); skipped when the fold has one template;
  - **a fold's last `choice`: the kept rows' original lines** — each row's
    first original segment, in document order (`zd_notfound.py`'s
    `--raw-final`); asked even over one row, for `found`; the heads still
    rank the rows by their values;
  - prose: each candidate inside its paragraph (the segments between the
    empty segments around it), in document order, the candidates labelled
    and the paragraph's other lines indented and unlabelled (`zd_prose.py`'s
    `render`);
  - records: each candidate as its spaced JSON on one line, in array order
    (`zd_records.py`'s `ask`);
  - a log without a fold: the kept lines as sent, in document order.
- The `choice`'s probabilities are the candidates' shares.

### Not found

Computed with the last `choice`, in **the same request** (the candidates'
render is one shared prefix), for the three measured routes:

- **`plain`**: the labelled `choice` above — the pick, the ranking, the
  pointers.
- **`none`**: the same `choice` with one more option, named `none`,
  described `"No line of the evidence answers the criterion"` (logs,
  records) or `"No sentence of the evidence answers the criterion"` (prose);
  `p_none` is its probability.
- **logs only, `found?`**: a `noul` whose instruction is `"Is there a line in
  the evidence that answers this question: "` followed by the `locate`'s
  instruction as sent; `p_yes` is its probability of true.

`found` = `(1 - p_none + p_yes) / 2` for `log` + `template_fold`, `1 - p_none`
for `prose` + `none` and `records` + `none`; not found below **0.5**. Neither
rule is calibrated: they are the rules the third round measured (the log one
chosen after seeing R4, the acceptance below judges it fresh). A route
without a measured rule answers without `found`: `log` + `none`, `records` +
`template_fold` (its "none" was measured only over the rows' values, 35 of
60 present kept, never over its original lines), and the vote.

### Fan-out, reuse and cost

- A request mixing kinds, methods and other primitives over one state
  answers every question. The fold is computed once per target; a fold's
  level-1 prefill (and its twin) is shared by every question over that
  target; a window's prefill is shared by every question over it — the
  scheduler submits a window's questions together, windows in order, so a
  window's prefix is retained while its questions run.
- Measured cost: a folded `log` question a median 0.8-2.6 s in spec 23's
  runs (5 s at ~1M tokens), and 0.7-1.4 s with the last request over the
  original lines (p90 up to 4.2 s, 14 s at ~1M; R, R2, R4), with no long
  prefill; a `prose` or `records` question pays each
  window's prefill once (45-68 s per 200K-token window) and then 1-4 s of head
  reading per window and ~0.3-0.5 s for the last request (`plain`, `none` and
  `found?` share its prefix). A folded record array (`records` +
  `template_fold`): median 1.3-1.7 s, no long prefill. The documentation
  states each.

### Observability and documentation

- **Metrics** (ADR 0017): `ignis_decisions_total{type="locate"}` still counts
  one per question. A new counter
  `ignis_locates_total{kind="log|prose|records",method="shortlist|vote",compression="template_fold|none",found="true|false|unmeasured"}`,
  absent until the first `locate`. No answer mass. ADR 0017 is amended.
- **The request log**: per `locate` its resolved kind, method, compression,
  windows read, `found` with `p_none` (and `p_yes`), and for a fold the
  level-1 templates, the chosen template's rows and the candidates of each
  `choice`.
- **The load event** `ignis.decide.locate` names the calibration's heads and
  window.
- **OpenAPI** (ADR 0036): the three fields, the answer's new fields, the
  refusals, the defaults.
- **`docs/user/README.md`**, "Finding a line or an item": the kinds and
  `auto`, the defaults, the combinations with what each costs and measured,
  `pointers`, `found` (what it is, its threshold, which routes carry it, the
  ranking kept below it), the first-of-identical rule, what `confidence`
  means, windows, the fold's limits, prose's and records' first-read cost.
- **`CONTEXT.md`**: *Shortlist*, *End reading*, *Sum reading*, *End heads*,
  *Template fold*, *Level 1*, *Level 2*, *Window*, *Pointer*, *Records
  array*, *Found*; *Locate* amended.
- **ADR 0042** accepted with the acceptance's numbers; ADR 0041's status
  names it.
- **The Playground's Decide tab** (#277): a `locate` card gets three
  selectors (kind with its four values, method, compression) in the shape of
  the point/box selector — the default choice sends no field; the answer
  panel names the resolved kind, method and compression, shows `found` (and
  "not found" below 0.5, with the ranking still listed) and lists every
  pointer with its share. The dev mock answers each combination, found and
  not found.

## The acceptance run (registered before sets R3, P3 and J3 exist)

### Set R3 (logs)

Seed **20261050**: a fresh capture of the owner's cluster (`kubectl logs
--since=6h --timestamps`, read only, after every earlier capture, merged by
`prodset.py timeline`) cut into 2 windows per tier of 16K / 50K / 100K /
200K tokens and 2 of ~1M, 4 targets per window; the **full** LogHub logs
(not the 2k samples), one window of 100K tokens per system and, where the log
holds it, one of 200K, sharing no line with that system's 2k sample, 3
targets per window. Targets by r2set.py's sibling rule, drawn round-robin over
the bins 0, 1-5, 6-50, > 50; questions written after reading the windows and
before any answer, checked by `r2set.py build`'s rules; a target no question
can single out is dropped, not replaced. At least 80 present questions and
12 windows at 100K tokens; a shortfall the sources force is recorded before
any run. **Absent questions**:
- **authored**: one per window, written as R2's were — a line the window
  could plausibly hold and does not, its distinctive words together in no
  line of the window (checked by the same rules);
- **target removed**: every present question asked again over its window
  with its target line removed (`zd_notfound.py --deleted`'s construction;
  the sets' rule makes the question unanswerable there, its near-duplicates
  stay).
Never committed; its manifest sha256 enters this banner.

### Set P3 (prose)

`tools/locate-sets/prosehay.py --seed 20261110`, excluding every HotpotQA
question of sets A-F, P and P2; tiers 16K / 64K / 128K / 200K (four windows
each) and 1M (two windows); six questions per window. **Absent questions**:
`tools/locate-sets/zd_prose_nf.py build --seed 20261160` over P3 — every
question of the 16K, 64K and 128K windows over its window without its gold
paragraphs (its HotpotQA distractor paragraphs stay), and 3 questions per 200K
window from another 200K window, none of their gold sentences in its text.
Both manifests' sha256 enter this banner.

### Set J3 (records)

`tools/locate-sets/zd_records.py build --seed 20261170 --lengths 1000 3500
10000`: spec 18's three record kinds, one array per kind and length (nine
arrays, ~55K / ~195K / ~550K tokens), five targets per array with a lexical
and a paraphrase question each, one absent value per array with a lexical and
a paraphrase question — 90 present and 18 absent questions. Its manifest
sha256 enters this banner. The records are synthetic (spec 18's generator):
the set tests the reading and the route, not real-world record questions.

The builders and the judge (`r3_judge.py`) are committed, and every hash
written here, **before any route is asked on any of the three sets**.

### The runs

On the served artifact under `make start`'s defaults, one `locate` per
request through `/v1/decide`:

1. **R3, no fields** (the defaults: `auto`, `shortlist`, `template_fold` for
   the logs), first, on a freshly started server — present, authored absent
   and target-removed questions.
2. **R3, `choice` alone over the fold** (`folded_locate.py --route choice
   --values`, the reported comparison), present questions.
3. **P3, no fields** (`auto`, `shortlist`, `none` for prose), present and
   absent questions.
4. **J3, no fields** (`auto`, `shortlist`, `none` for records), present and
   absent questions.
5. **Set F, `method: "vote"`, `compression: "none"`**, named explicitly.
6. **Set F, no fields.**

### Scoring

A present question is **right** when the answer names its target (for prose,
a gold sentence) **and** `found` ≥ 0.5 where the route carries `found`: a
present question answered "not found" is a miss, as the caller sees it. An
absent question is **flagged** when `found` < 0.5.

### The rules

1. **Logs.** The defaults are right on **at least 85%** of R3's present
   questions.
2. **Logs, the heads' part.** The defaults' top-1 on R3 (the pick, whatever
   `found` says) is **at least** the labelled `choice` alone over the same
   fold (run 2).
3. **Latency.** Over R3's 100K-token windows the median wall time of each
   window's first question under the defaults — the client's request time,
   the fold included — is **at most 3.0 s**.
4. **Prose.** On P3's windows up to 200K tokens the defaults are right on
   **at least 85%** of the present questions, and over all of P3's present
   questions the paragraph-level pointer F1 averages **at least 0.75** (a
   present question answered "not found" has no pointers: F1 0). (The 1M
   group is reported, not asserted: spec 23 found it the weak point.)
5. **Records.** The defaults are right on **at least 90%** of J3's present
   questions (explored, counting `found`: 57/60 and 28/30; the pick alone
   58/60 and 29/30), and on at least 85% of its 10,000-record arrays'
   (explored: 28/30).
6. **Not found, logs.** Of R3's present questions whose pick is its target,
   **at least 95%** keep `found` ≥ 0.5 (explored: 101/101, 21/21); **at
   least 60%** of the target-removed questions (explored: 41/58, 20/23) and
   **at least 65%** of the authored absent ones (explored: 16/20) are
   flagged.
7. **Not found, prose.** Of P3's present questions up to 200K whose pick is
   a gold sentence, **at least 95%** keep `found` ≥ 0.5 (explored: 66/67);
   **at least 50%** of the gold-removed questions (explored: 45/72) and
   **at least 90%** of the other-window questions (explored: 12/12) are
   flagged.
8. **Not found, records.** Of J3's present questions picked right, **at
   least 95%** keep `found` ≥ 0.5 (explored: 57/58, 28/29); **at least 90%**
   of its absent questions are flagged (explored: 12/12, 6/6).
9. **`auto`.** The defaults resolve every R3 window to `log`, every P3 window
   to `prose`, every J3 array and every set F record array to `records`.
10. **The vote unchanged.** On set F, `vote` + `none` serves and refuses the
    same questions as the recorded run (`.scratch/locate/F-served.json` in
    the main checkout,
    [finding](../../findings/2026-09-27-locate-through-decide.md)): the same
    `locate_too_long` refusals, the same segment on every served question
    whose recorded winner led the next by more than one vote, one of the two
    where it led by one or tied.
11. **Short states.** On set F's present questions the vote serves, the
    defaults' right answers per family are at least the recorded vote's minus
    5 points: **logs ≥ 39/43, records ≥ 44/46, prose ≥ 56/67**.

### Reported, not asserted

P3's 1M group; by source, tier and sibling bin: top-1, level-1 accuracy,
shortlist recall (a target among the candidates), windows and prompt tokens,
wall time (median, p90); sentence-level pointer F1; `confidence` and `found`
beside a right and a wrong pick; `found`'s AUC per set and kind of absent
question; set F's absent questions under the defaults (flagged, per family);
J3's lexical and paraphrase questions apart; the fold's and `auto`'s host time
on the longest window. All of it goes in a finding with a README row, and ADR
0042's numbers come from it.

## Testing Decisions

A good test asserts what a caller or an operator can observe — the segment,
the pointers, `found`, the resolved enums, the refusals, the prefills
submitted — and holds the host arithmetic to the Python that measured it.

- **Golden cases (CPU, pure)** written by the reference into
  `crates/core/tests/fixtures/`: `compress.py golden` for `template_fold`
  (the first version's list: labels, times in every `TIME` shape, masked
  variables, the `SIM` boundary, the 600-character budget, long values,
  non-ASCII, repeats, affixes, empty and whitespace lines, `\r\n`, a
  bracket-opened line, records as spaced JSON, string elements); a new
  `golden` for the readings (rows and twins in, per-segment sum and end
  lifts, standardized scores, windows cut at empty segments and at record
  boundaries, merged, the shortlist), for `auto` (records arrays, content
  parts, arrays of strings, text), for every render of § The labelled
  `choice` (the fold's last `choice` over original lines included) and for
  the `found` rules. Synthetic inputs only, never a cluster line.
- **The endpoint over the mock (CPU, `/v1/decide`)**: no fields answer with
  the resolved kind and the shortlist; each valid combination answers with
  the shape above, `found` present exactly on the three measured routes; a
  not-found answer (null segment, empty pointers, ranking kept); every
  refusal before any prefill, `kind_mismatch` both ways; a target longer
  than a window read as several windows (text and records) and answered with
  an index into the original; a fold's level 2 asked only after level 1; the
  last request carrying `plain`, `none` and, for logs, the `noul`; the fan-out
  of kinds, methods, a `noul` and a `choice` over one state; the fold and the
  level-1 prefill shared; `pointers` at the threshold; `usage`, the log
  fields, `ignis_locates_total` with its `found` label, the OpenAPI document.
- **The vote is pinned**: today's `locate` tests ask for `method: "vote"`,
  `compression: "none"` by name and pass unchanged; the prompt-pinning test
  (`decide_locate_prompt.rs`) holds that render, which the shortlist's
  readings reuse.
- **End to end (GPU profile)**: on a committed synthetic fixture — logs with
  near-duplicate lines, a records array, and prose paragraphs with a title
  each, with one absent question each — the defaults through `/v1/decide`
  name every present question's target and flag the absent ones, and a
  fixture longer than one window is answered.
- **The Playground** (vitest): the selectors, defaults sending no field, the
  pointers and `found` shown, "not found" rendered; the dev mock's answers.

## Acceptance

1. **The host functions are the measured ones**: `template_fold`, the
   readings, the windows and merge, `auto`, the candidate renders and the
   `found` rules reproduce every golden case their Python reference writes.
2. **The calibration** carries the end heads, the sum heads and
   `LOCATE_WINDOW_KEYS` for the served 27B; the rows' room is reserved at
   load and in the VRAM plan.
3. **`/v1/decide` over the mock** passes § Testing Decisions' endpoint list.
4. **The vote is unchanged** under `vote` + `none`: today's tests pass asking
   for it by name, and the prompt-pinning test holds its render.
5. **End to end on the GPU** the fixtures' targets are named and their absent
   questions flagged by the defaults, including one past a window.
6. **Documented**: OpenAPI, the user README's "Finding a line or an item",
   `CONTEXT.md`, ADR 0017's amendment and ADR 0041's status; the load event;
   the Playground's selectors, pointers and `found` with their tests.
7. **Sets R3, P3 and J3 are registered** before any route is asked on them:
   the builders and the judge committed, every manifest hash in this banner.
8. **The acceptance holds**: rules 1-11 of § The acceptance run, each judged
   once, recorded as a finding with a README row; ADR 0042 accepted with its
   numbers. A failed rule is reported as failed and the owner decides.
9. `cargo test` passes workspace-wide, and the web tests pass.

## Out of Scope

- **`method: "copy"`** — the first version's default, a decode constrained to
  the candidates' text: a future study, **#279**. The route it would
  constrain (free generation over the fold) read 55 of R2's 58, as the
  shortlist does, and it needs leaf changes this ticket does not (the draw
  reported in the call that makes it, a vocabulary-wide permitted set).
- **Paragraphs as prose's unit** (`unit: "paragraph"`): a labelled `choice`
  over the heads' first 8 paragraphs picked a gold paragraph 35 of 36 times
  past 500K tokens (the sentence route's paragraph 32), but its instruction
  was written from HotpotQA's question ("Which paragraph helps answer this
  question: …"), which a caller does not send, and its pointer set was no
  better. A later value, once measured with the caller's instruction as sent.
- **`found` on the unmeasured routes** (`log` + `none`, `records` +
  `template_fold`, the vote), and any calibration of `found`.
- **The second hop** for prose (the heads read again with the first pointer
  in the instruction: +6 points of pointer F1 on P and P2): exploratory.
- **A fold by schema for records** (group by key set, values as rows): not
  tried; it could remove the records' first-question prefill.
- **Real record arrays and questions over several fields**: J3 is spec 18's
  synthetic kinds, one selecting value per question.
- **Reducing the rows on the device** (per-segment sums in the leaf instead
  of rows to the host) — a latency and memory optimisation.
- **Lifting `LOCATE_MAX_KEYS`** for the vote.
- **Content-parts states**, as for every `locate`.
- **The research branch's experiment hooks** (`IGNIS_LOCATE_MAX_KEYS_EXPERIMENT`,
  the row dumps, the control file, `cut_tail`, `answers`): they stay on
  `locate-long-context` and are never merged.

## Further Notes

- **Why the heads and not the copy.** Both resolve the instance; the
  shortlist does it with no token written, no decode lane and no leaf change,
  by using the heads where they are strong (narrowing a very long text to a
  handful of candidates) and a `choice` where it is strong (deciding among a
  handful). The owner preferred the heads, measured-better or equal being
  the condition (spec 23).
- **Why the check sees the original lines.** Over a fold's level-2 rows —
  values only — the "none" option won on present questions too (37 of 100
  kept): whether a row answers cannot be read from its values. Shown as their
  original lines, the same 16 rows keep every found answer and the pick is as
  good (101 vs 100 of R and R2's 108). In prose the paragraphs already carry
  that context, and checking a sentence alone was worse (a HotpotQA sentence
  often answers only with the other paragraph).
- **Rule 11 judges routes never measured on short states**: the `records`
  route and `found` on set F's 20-300-record arrays, short logs and short
  prose are unmeasured there (set F is kept unread as the guard).
- **Why enums and not booleans**: a third reading, compression or kind is a
  new value, echoed in the answer, never a flag contradicting another.
- **What changes from the routes the research measured**, and is therefore
  what the acceptance judges: the readings are ported from Python to the
  host; the `choice` over a fold's level 1 is asked over the first 5
  templates only; the window is 200,000 keys (spec 23 used 210,000 tokens
  for prose); the renders are the reference's; `found`'s log rule was chosen
  after seeing R4.
- **Prior art**: labelled selection over a shortlist is the retrieve-then-read
  shape of attention re-rankers (ICR, QRHead) with a single-token reader; the
  fold is Drain-style template mining (He et al., ICWS 2017); the literature
  pass is [the literature](../../findings/2026-09-28-zero-decode-locate-literature.md).

## References

- Findings: [locate at length](../../findings/2026-09-27-locate-at-length.md),
  [the instance is read while copying](../../findings/2026-09-27-the-instance-is-read-while-copying.md),
  [locating a line in real logs](../../findings/2026-09-28-locating-a-line-in-real-logs.md),
  [zero-decode locate on real logs](../../findings/2026-09-28-zero-decode-locate-exploration.md),
  [zero-decode locate for very long logs and prose](../../findings/2026-09-28-zero-decode-locate-very-long-logs-and-prose.md),
  [paragraphs at 1M, not found, JSON records](../../findings/2026-09-28-zero-decode-locate-paragraphs-not-found-records.md),
  [the literature](../../findings/2026-09-28-zero-decode-locate-literature.md),
  [locate through `/v1/decide`](../../findings/2026-09-27-locate-through-decide.md).
- Specs 18 (the `locate` wire, set F, the record generator), 19 (the vote),
  20, 21 and 23 (the research this ships), 04 (fan-out), 17 (layout L1).
- ADR 0041 (the vote, the rows readout), 0017 (metrics), 0030 (memory
  reserved at load), 0036 (OpenAPI), 0006 (CPU tests without a GPU), 0042
  (this spec's decision).
- Tools: `tools/locate-sets/compress.py` (the fold), `zd_logpipe.py` (the
  `log` route's levels, configuration `heads-end5+choice`), `zd_notfound.py`
  (`--raw-final`: the last `choice` over original lines, `none` and the
  `noul`; `--deleted`), `zd_windows.py` and `zd_prose.py multi` (the `prose`
  route), `zd_prose_nf.py` (prose's `none`, P3's absent builder),
  `zd_records.py` (the `records` route, `cut`, J3's builder),
  `zd_offline.py` and `zd_cache.py` (the readings), `folded_locate.py`
  (`--route choice`), `prosehay.py`, `r2set.py`, `z23_r4.py` and
  `z23_judge.py` (the builders and the judge the acceptance's follow),
  `served.py` (set F's).
