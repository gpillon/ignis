# 22 - locate over very long texts: heads narrow, a labelled choice decides

> **Registered 2026-09-28; revised 2026-09-28, before sets R3 and P3
> exist.** The first version made a constrained **copy** the default; the
> research that followed (spec 23 and its findings) measured a route that
> writes no token at all at least as well, and the owner chose it — this
> revision replaces the default, adds `kind`, and re-registers the
> acceptance below. The acceptance rules (§ The acceptance run) and the
> sets' seeds are fixed here and are not edited once R3's and P3's manifest
> hashes are recorded in this banner. A failed rule is reported as failed and
> the defaults do not ship until the owner decides; this spec does not
> pre-decide a fallback.

GitHub: #278. ADR: [0042](../../adr/0042-locate-copies-over-a-folded-state.md) (Proposed, revised).

`locate` (spec 18, ADR 0041) names the **segment** of a text `state` an
instruction asks for, today by one prefill's **head vote**, measured to 4,554
keys. The texts a `locate` is for are much longer — logs of hundreds of
thousands of lines, documents past the context — and there the vote finds the
right *kind* of line and loses the instance. This spec ships what the
research on very long logs and prose found (specs 20, 21 and 23 and their
findings): **attention heads narrow the text to a few candidates and a
labelled `choice` decides among them**, with no token generated. Logs are
first **folded** into templates and their values; prose is read by the heads
**window by window**; the kind of text is told by the fold itself. The caller
can choose everything by name, as three enums on the question: `kind`
(`auto`, `log`, `prose`), `method` (`shortlist`, `vote`) and `compression`
(`template_fold`, `none`).

One ticket, whose acceptance is numbered at the end.

## Problem Statement

A caller with a very long text wants the place that answers an instruction:
in a log, the one line — the failed write, the refused login, the pod that
restarted; in prose, the one or several passages that answer a question. The
texts are 16K tokens to past a million.

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
- **Nothing past the context is answered**, and the caller cannot choose a
  reading or a compression, nor get more than one segment back.

## Solution

Three fields on a `locate` question, all enums, all optional:

- **`kind`**: `"auto"` (default), `"log"` or `"prose"` — which reading the
  text gets. `auto` folds the target's first 2,000 segments with content and
  says `log` when at least half of them fall in templates of two or more,
  `prose` otherwise (logs measured 0.96-0.98, prose 0.04-0.07 on every fresh
  window; spec 23's HA).
- **`method`**: `"shortlist"` (default) or `"vote"`.
- **`compression`**: `"template_fold"` or `"none"`; its default follows the
  kind — `template_fold` for a log, `none` for prose.

| | `kind: log` | `kind: prose` |
|---|---|---|
| **`shortlist` + `template_fold`** (log default) | fold; the **end heads** read the templates and keep 5; a labelled `choice` picks one; its rows: the end heads keep 16 (all when ≤ 16); a labelled `choice` picks the row; unfold | refused: prose does not fold |
| **`shortlist` + `none`** | the end heads read the whole target in windows and keep 16 lines; a labelled `choice` picks one | (prose default) the **sum heads** read the target in windows and keep 16 sentences; a labelled `choice` over them *in their paragraphs*; every candidate at p ≥ 0.05 is a pointer |
| **`vote` + `none`** | today's `locate`, byte for byte, whatever the kind | the same |
| **`vote` + `template_fold`** | refused (read 28 of R2's 58) | refused |

- **`template_fold`** is `tools/locate-sets/compress.py`'s fold, as the first
  version of this spec described it: templates Drain-style, **level 1** one
  line per template with its slots' distinct values and `(xN)`, **level 2**
  the chosen template's lines as their values, values first and time last; a
  map back to the original segments. A 100K-token log folds to a median
  ~10K-token level 1; a 1M-token log to 25-34K.
- **The heads' reading** is a lift, as the vote's: each head's attention row
  at the scaffold, less the content-free twin's, **standardized over the
  candidates** and summed over the heads. The **end reading** (logs) scores a
  line by its last key, its separator and the next line's first key; the
  **sum reading** (prose) by all its keys. Candidates are kept in document
  order.
- **The labelled `choice`** is the endpoint's own `choice` over the
  candidates, each prefixed with its answer label; for prose each candidate
  is shown inside its paragraph (the paragraph's other lines unlabelled, as
  context). It generates nothing: it reads the labels' logits, as every
  `choice` does.
- **Windows.** A text the heads read that is longer than the calibrated
  window (`LOCATE_WINDOW_KEYS`, 200,000 keys on the served 27B) is cut into
  windows at segment boundaries — at an empty segment when one is within the
  window, as prose's paragraph breaks — each read as its own prefill with its
  own twin; scores are standardized within a window and merged. The text's
  raw length is then bounded by the request, not the context.

Measured (the numbers are spec 23's and its findings'):

| logs, top-1 | R (50) | R2 (58) | ~1M-token logs (41) | **R4, fresh (23)** |
|---|---|---|---|---|
| `shortlist` + `template_fold` | 45 | 55 | 36 | **21 (91.3%)** |
| labelled `choice` alone over the fold | 42 | 53 | 33 | 16 |
| the heads alone at both levels | 30 | 29 | 17 | — |
| served vote | 28 | 20 | — | — |
| median wall time of the first row | 0.8 s | 1.3 s | 5.0 s | 2.6 s |

| prose (HotpotQA gold sentences in a haystack) | ≤200K | 1M |
|---|---|---|
| a gold sentence in the sum heads' first 16 | 100% | 100% |
| the `choice`'s pick is a gold sentence (fresh P2) | 91.7% | 58% (7/12; 10/12 by paragraph) |
| paragraph pointer F1 (fresh P2, all) | 0.77 | |

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
5. As a caller, I want the default to be the route that measured best for my
   kind of text, and the kind told for me when I do not say it, so that I
   do not have to know the research to get the good answer.
6. As a caller, I want to name the kind, the method and the compression, so
   that I can trade accuracy, latency and context for my case.
7. As a caller, I want all three to be enums that name what they do, so that
   a new reading or compression later is a new value, not a contradicting
   flag.
8. As a caller, I want an unknown value refused with the accepted values
   named, so that a typo never runs the default silently.
9. As a caller, I want a combination that was measured bad or does not apply
   (`vote` with a fold, a fold of prose) refused by name, so that I never get
   an unmeasured answer.
10. As a caller, I want `kind` and `compression` refused on every other
    question type, so that a field I wrote is never ignored.
11. As a caller, I want the answer to name the kind, method and compression
    that produced it, so that a default I did not write is visible.
12. As a caller, I want `segment` to index the state I sent, whatever was
    folded or windowed, so that I map the answer back without the server.
13. As a caller, I want `pointers` — every candidate the `choice` gave at
    least 5% — so that I can take one answer or several.
14. As a caller, I want to know which line I get when several are the same
    text (identical, or identical but for their time), so that a repeated
    event gives a predictable answer: the first.
15. As a caller, I want `confidence` documented as what it is — the
    `choice`'s probability of its pick, the product over a fold's two levels
    — and never as a calibrated probability, so that I set thresholds
    knowingly.
16. As a caller asking several `locate`s over one state, I want the fold
    computed once and every prefill a later question can share claimed, so
    that the second question costs a question.
17. As a caller mixing `locate` with `noul` or `choice` over one state, I
    want every answer back in one request.
18. As a caller whose question needs context across lines, or names a line by
    its time alone, I want the documentation to say that folding removes
    that and `compression: "none"` keeps it.
19. As a Playground user, I want the Decide tab to offer the kind, method and
    compression beside a `locate` and show every pointer, so that I can
    compare them on my own evidence.
20. As an operator, I want to count `locate`s by kind, method and
    compression, and to see per question how many windows, templates, rows
    and candidates it read, so that a slow or wrong answer is attributable.
21. As a maintainer, I want the fold, the readings, the windowing, the merge,
    the candidate render and the `auto` rule to be pure host functions held
    to golden cases the Python reference writes, so that the served route is
    the measured route.
22. As a maintainer, I want the vote and every non-`locate` request unchanged
    bit for bit.
23. As the owner, I want the defaults judged once on fresh real logs and
    fresh prose with rules written here before the sets exist, and the vote
    and short states guarded, so that the new default is not tuned to its
    own test.

## Implementation Decisions

### The wire

- **`kind`** on a `locate`: `"auto"` (default), `"log"`, `"prose"`. Unknown:
  `kind_unknown`, naming the three. On any other type: `kind_unsupported`.
- **`method`** on a `locate`: `"shortlist"` (default) or `"vote"`. The field
  exists for `point` and `box` (`head`, `chain`); spec 18's refusal of it on
  a `locate` is lifted. An unknown value on a `locate` is `method_unknown`
  naming `shortlist` and `vote`; `shortlist` and `vote` on a `point` or `box`
  are `method_unknown` naming theirs; on every other type `method` stays
  `method_unsupported`.
- **`compression`** on a `locate`: `"template_fold"` or `"none"`, default by
  the resolved kind (`log` → `template_fold`, `prose` → `none`). Unknown:
  `compression_unknown`; on any other type: `compression_unsupported`.
- **Refusals, all before any prefill** (a 422, nothing reaches the engine):

  | code | when |
  |---|---|
  | `compression_unsupported` | `template_fold` with `kind: "prose"` (prose does not fold) or with `method: "vote"` (measured 28 of R2's 58) |
  | `locate_uncalibrated` | the load has no `locate` calibration — for either method: both read heads |
  | `locate_too_long` | `vote` only, as today: the target past `LOCATE_MAX_KEYS` |
  | `context_exceeded` | a level-1 text, a level-2 text or a `choice` prompt past the context (a fold's level-2 text past `LOCATE_WINDOW_KEYS` is windowed, not refused) |
  | `locate_too_few_segments` | fewer than two segments with content (unchanged) |
  | `locate_unsupported` | the loaded template cannot say where its tokens sit (unchanged) |

- **The answer** keeps `Answer::Locate`'s shape and gains four fields:

  ```json
  {"type": "locate", "kind": "log", "method": "shortlist", "compression": "template_fold",
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
    the first entry is the answer and its share is `confidence`.
  - `pointers` is every candidate of the last `choice` whose share is at
    least **0.05**, best first — always at least the pick. The threshold is
    the one chosen on P's dev split (spec 23 froze it).
  - **Several segments with one text** (a fold's row of lines identical but
    for their time): the first. `vote` + `none` is unchanged.
  - `vote` + `none`: today's answer plus `kind` (resolved, unused),
    `method`, `compression`, and `pointers` holding the winner alone.
- **`usage`**: `input_tokens` counts every prefill (windows, twins, levels,
  `choice`s); `output_tokens` is 0: nothing is generated.
- **Thinking** is refused as for every decision.

### `auto`

A pure host function: fold (`template_fold`, below) the target's first 2,000
segments with content, and say `log` when the segments in templates of two or
more are at least half of them. Computed once per target per request.
Arrays of records fold as spaced JSON (below) and are mostly `log` (median
share 0.94), not always (18 of 160 short record arrays read as prose); the
documentation says so.

### `template_fold`

Unchanged from the first version of this spec: `tools/locate-sets/compress.py`'s
`fold` with `values=True`, `summarize`, `level2` and `_common_affixes` at the
settings R2 was judged with (`SIM` 0.5, the `TIME`, `VARIABLE`, `LABEL` and
`TOKEN` patterns, a slot's values within a 600-character budget, else the
first 6 cut to 24 characters and `|+N`, `(xN)`, level-2 rows values-first with
their first time last), ported as a **pure host function in `ignis_core`**
with Python's arithmetic (code points, Unicode classes, first-seen cluster
order). It folds a string's lines as sent and an array's elements (a string
as its text, anything else as its JSON with a space after each `,` and `:`
outside strings); empty segments are left out. The map (template, row) →
original segments; the answer is the first. Kept as measured, and
documented: a bracket-opened line is read as a source label; level 1 drops
every time; folding removes lines' order and neighbours.

### The heads and their readings

- **The calibration** (`ignis_core::locate::LocateCalibration`) gains, per
  artifact: the **end heads** (the `log` reading), the **sum heads** (the
  `prose` reading) and **`LOCATE_WINDOW_KEYS`**. On the served NVFP4 27B:
  - end heads, in rank order (spec 23's `endheads.json`, chosen on sets A+B
    by single-head top-1 at a line's last key, next first key or separator):
    L47.h20, L47.h4, L51.h4, L47.h17, L47.h1, L51.h12, L51.h23, L47.h5,
    L47.h3, L47.h15, L39.h15, L47.h9, L47.h13, L51.h16, L39.h0, L39.h12,
    L43.h22, L43.h20, L47.h2, L39.h23, L55.h13, L43.h9, L43.h18, L51.h17,
    L55.h23, L43.h7, L51.h2, L55.h20, L47.h23, L35.h18, L43.h8, L43.h14;
  - sum heads: the vote's 32, as calibrated (ADR 0041);
  - `LOCATE_WINDOW_KEYS` 200,000 (spec 23 read prose in windows of at most
    210,000 tokens, logs' level texts to ~100,000 keys).
  `max_keys` (4,554) stays the vote's ceiling alone.
- **A reading is one prefill per window and its content-free twin** (the
  vote's render: layout L1, the kind text, the instruction, the forced
  `{"quote":"`; the twin's instruction `N/A`), the readout returning the 32
  heads' rows over the window's span (ADR 0041's rows readout). The rows'
  room is reserved at load for 32 heads x `LOCATE_WINDOW_KEYS` (25.6 MB of
  f32 on the served 27B; ADR 0030), and stated in the VRAM plan.
- **The host reading**, pure and held to golden cases from the Python
  reference (`tools/locate-sets/zd_offline.py`'s `zsum`, `zd_cache.py`'s
  `key_features`, `zd_windows.py`'s `sub_windows`, `zd_prose.py`'s merge):
  each head's softmax over the window's span; per segment the question's
  mass less the twin's at the segment's keys (sum), or at its last key, the
  keys between it and the next segment and the next segment's first key
  (end); each head's lifts standardized over the window's segments and
  summed over heads; windows' scores standardized and concatenated; empty
  segments and, for prose, lines that open with `# ` (titles) never
  candidates.
- **The shortlist** is the first K segments by that score, shown in document
  order: K = 5 templates at a fold's level 1, K = 16 rows at level 2 (all
  when ≤ 16), K = 16 lines or sentences without a fold.

### The labelled `choice`

- The endpoint's `choice`, as served, over the candidates labelled with its
  answer alphabet (spec 18's labelled route): a derived state holding the
  candidates as `label: text` lines, the instruction as sent, one option per
  label. No new primitive and no new render.
- **Prose candidates are shown in their paragraphs**: for each candidate,
  its paragraph (the segments between the empty segments around it), in
  document order, the candidates labelled and the paragraph's other lines
  indented and unlabelled — `zd_prose.py`'s `render`, held to golden cases.
- The `choice`'s probabilities are the candidates' shares.

### Fan-out, reuse and cost

- A request mixing kinds, methods and other primitives over one state
  answers every question. The fold is computed once per target; a fold's
  level-1 prefill (and its twin) is shared by every question over that
  target; a window's prefill is shared by every question over it — the
  scheduler submits a window's questions together, windows in order, so a
  window's prefix is retained while its questions run.
- Measured cost: a folded `log` question a median 0.8-2.6 s (5 s at ~1M
  tokens) with no long prefill; a `prose` question pays each window's
  prefill once (68 s per 200K-token window) and then ~4 s of head reading
  per window and ~0.5 s of `choice`. The documentation states both.

### Observability and documentation

- **Metrics** (ADR 0017): `ignis_decisions_total{type="locate"}` still counts
  one per question. A new counter
  `ignis_locates_total{kind="log|prose",method="shortlist|vote",compression="template_fold|none"}`,
  absent until the first `locate`. No answer mass. ADR 0017 is amended.
- **The request log**: per `locate` its resolved kind, method, compression,
  windows read, and for a fold the level-1 templates, the chosen template's
  rows and the candidates of each `choice`.
- **The load event** `ignis.decide.locate` names the calibration's heads and
  window.
- **OpenAPI** (ADR 0036): the three fields, the answer's new fields, the
  refusals, the defaults.
- **`docs/user/README.md`**, "Finding a line or an item": the kinds and
  `auto`, the defaults, the combinations with what each costs and measured,
  `pointers`, the first-of-identical rule, what `confidence` means, windows,
  the fold's limits, prose's first-read cost, and that there is no "not
  found" yet.
- **`CONTEXT.md`**: *Shortlist*, *End reading*, *Sum reading*, *End heads*,
  *Template fold*, *Level 1*, *Level 2*, *Window*, *Pointer*; *Locate*
  amended.
- **ADR 0042** accepted with the acceptance's numbers; ADR 0041's status
  names it.
- **The Playground's Decide tab** (#277): a `locate` card gets three
  selectors (kind, method, compression) in the shape of the point/box
  selector — the default choice sends no field; the answer panel names the
  resolved kind, method and compression and lists every pointer with its
  share. The dev mock answers each combination.

## The acceptance run (registered before sets R3 and P3 exist)

### Set R3 (logs)

As registered in the first version, seed **20261050**: a fresh capture of
the owner's cluster (`kubectl logs --since=6h --timestamps`, read only,
after every earlier capture, merged by `prodset.py timeline`) cut into 2
windows per tier of 16K / 50K / 100K / 200K tokens and 2 of ~1M, 4 targets
per window; the **full** LogHub logs (not the 2k samples), one window of 100K
tokens per system and, where the log holds it, one of 200K, sharing no line
with that system's 2k sample, 3 targets per window. Targets by r2set.py's
sibling rule, drawn round-robin over the bins 0, 1-5, 6-50, > 50; questions
written after reading the windows and before any answer, checked by
`r2set.py build`'s rules; a target no question can single out is dropped, not
replaced. At least 80 present questions and 12 windows at 100K tokens; a
shortfall the sources force is recorded before any run. Never committed; its
manifest sha256 enters this banner.

### Set P3 (prose)

`tools/locate-sets/prosehay.py --seed 20261110`, excluding every HotpotQA
question of sets A-F, P and P2; tiers 16K / 64K / 128K / 200K (four windows
each) and 1M (two windows); six questions per window. Its manifest sha256
enters this banner.

The builders and the judge (`r3_judge.py`) are committed, and both hashes
written here, **before any route is asked on either set**.

### The runs

On the served artifact under `make start`'s defaults, one `locate` per
request through `/v1/decide`:

1. **R3, no fields** (the defaults: `auto`, `shortlist`, `template_fold` for
   the logs), first, on a freshly started server.
2. **R3, `choice` alone over the fold** (`folded_locate.py --route choice
   --values`, the reported comparison).
3. **P3, no fields** (`auto`, `shortlist`, `none` for prose).
4. **Set F, `method: "vote"`, `compression: "none"`**, named explicitly.
5. **Set F, no fields.**

### The rules

1. **Logs.** The defaults name the target on **at least 85%** of R3's present
   questions.
2. **Logs, the heads' part.** The defaults' top-1 on R3 is **at least** the
   labelled `choice` alone over the same fold (run 2).
3. **Latency.** Over R3's 100K-token windows the median wall time of each
   window's first question under the defaults — the client's request time,
   the fold included — is **at most 3.0 s**.
4. **Prose.** On P3's windows up to 200K tokens the pick is a gold sentence
   on **at least 85%** of questions, and over all of P3 the paragraph-level
   pointer F1 averages **at least 0.75**. (The 1M group is reported, not
   asserted: spec 23 found it the weak point, and the research continues.)
5. **`auto`.** The defaults resolve every R3 window to `log` and every P3
   window to `prose`.
6. **The vote unchanged.** On set F, `vote` + `none` serves and refuses the
   same questions as the recorded run (`.scratch/locate/F-served.json` in
   the main checkout,
   [finding](../../findings/2026-09-27-locate-through-decide.md)): the same
   `locate_too_long` refusals, the same segment on every served question
   whose recorded winner led the next by more than one vote, one of the two
   where it led by one or tied.
7. **Short states.** On set F's present questions the vote serves, the
   defaults' top-1 per family is at least the recorded vote's minus 5 points:
   **logs ≥ 39/43, records ≥ 44/46, prose ≥ 56/67**.

### Reported, not asserted

P3's 1M group; by source, tier and sibling bin: top-1, level-1 accuracy,
shortlist recall (a target among the candidates), windows and prompt tokens,
wall time (median, p90); sentence-level pointer F1; `confidence` beside a
correct and a wrong pick; the fold's and `auto`'s host time on the longest
window. All of it goes in a finding with a README row, and ADR 0042's
numbers come from it.

## Testing Decisions

A good test asserts what a caller or an operator can observe — the segment,
the pointers, the resolved enums, the refusals, the prefills submitted — and
holds the host arithmetic to the Python that measured it.

- **Golden cases (CPU, pure)** written by the reference into
  `crates/core/tests/fixtures/`: `compress.py golden` for `template_fold`
  (the first version's list: labels, times in every `TIME` shape, masked
  variables, the `SIM` boundary, the 600-character budget, long values,
  non-ASCII, repeats, affixes, empty and whitespace lines, `\r\n`, a
  bracket-opened line, records as spaced JSON, string elements); a new
  `golden` for the readings (rows and twins in, per-segment sum and end
  lifts, standardized scores, windows cut and merged, the shortlist) and for
  `auto` and the prose render. Synthetic inputs only, never a cluster line.
- **The endpoint over the mock (CPU, `/v1/decide`)**: no fields answer with
  the resolved kind and the shortlist; each valid combination answers with
  the shape above; every refusal before any prefill; a target longer than a
  window read as several windows and answered with an index into the
  original; a fold's level 2 asked only after level 1; the fan-out of kinds,
  methods, a `noul` and a `choice` over one state; the fold and the level-1
  prefill shared; `pointers` at the threshold; `usage`, the log fields,
  `ignis_locates_total`, the OpenAPI document.
- **The vote is pinned**: today's `locate` tests ask for `method: "vote"`,
  `compression: "none"` by name and pass unchanged; the prompt-pinning test
  (`decide_locate_prompt.rs`) holds that render, which the shortlist's
  readings reuse.
- **End to end (GPU profile)**: on a committed synthetic fixture — logs with
  near-duplicate lines and records of one shape, and prose paragraphs with a
  title each — the defaults through `/v1/decide` name every question's
  target, and a fixture longer than one window is answered.
- **The Playground** (vitest): the three selectors, defaults sending no
  field, the pointers shown; the dev mock's answers.

## Acceptance

1. **The host functions are the measured ones**: `template_fold`, the
   readings, the windows and merge, `auto` and the prose render reproduce
   every golden case their Python reference writes.
2. **The calibration** carries the end heads, the sum heads and
   `LOCATE_WINDOW_KEYS` for the served 27B; the rows' room is reserved at
   load and in the VRAM plan.
3. **`/v1/decide` over the mock** passes § Testing Decisions' endpoint list.
4. **The vote is unchanged** under `vote` + `none`: today's tests pass asking
   for it by name, and the prompt-pinning test holds its render.
5. **End to end on the GPU** the fixtures' targets are named by the defaults,
   including one past a window.
6. **Documented**: OpenAPI, the user README's "Finding a line or an item",
   `CONTEXT.md`, ADR 0017's amendment and ADR 0041's status; the load event;
   the Playground's selectors and pointers with their tests.
7. **Sets R3 and P3 are registered** before any route is asked on them: the
   builders and the judge committed, both manifest hashes in this banner.
8. **The acceptance holds**: rules 1-7 of § The acceptance run, each judged
   once, recorded as a finding with a README row; ADR 0042 accepted with its
   numbers. A failed rule is reported as failed and the owner decides.
9. `cargo test` passes workspace-wide, and the web tests pass.

## Out of Scope

- **`method: "copy"`** — the first version's default, a decode constrained to
  the candidates' text: a later value of `method`. The route it would
  constrain (free generation over the fold) read 55 of R2's 58, as the
  shortlist does, and it needs leaf changes this ticket no longer does (the
  draw reported in the call that makes it, a vocabulary-wide permitted set).
- **"Not found"** — an answer that says no segment answers (a `none` label in
  the `choice`, or a threshold on its confidence): research first.
- **A kind for JSON records** — `auto` reads them as logs mostly; their own
  reading is research first.
- **Prose at ~1M tokens**, the registered weak point (spec 23's HP1) —
  paragraphs as the unit, paragraphs then sentences, fewer candidates:
  research first.
- **The second hop** for prose (the heads read again with the first pointer
  in the instruction: +6 points of pointer F1 on P and P2): exploratory.
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
- **Why enums and not booleans**: a third reading, compression or kind is a
  new value, echoed in the answer, never a flag contradicting another.
- **What changes from the routes spec 23 measured**, and is therefore what
  the acceptance judges: the readings are ported from Python to the host;
  the `choice` over a fold's level 1 is asked over the first 5 templates
  only; the window is 200,000 keys (spec 23 used 210,000 tokens); the
  prose render is the reference's.
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
  [the literature](../../findings/2026-09-28-zero-decode-locate-literature.md),
  [locate through `/v1/decide`](../../findings/2026-09-27-locate-through-decide.md).
- Specs 18 (the `locate` wire, set F), 19 (the vote), 20, 21 and 23 (the
  research this ships), 04 (fan-out), 17 (layout L1).
- ADR 0041 (the vote, the rows readout), 0017 (metrics), 0030 (memory
  reserved at load), 0036 (OpenAPI), 0006 (CPU tests without a GPU), 0042
  (this spec's decision).
- Tools: `tools/locate-sets/compress.py` (the fold), `zd_logpipe.py` (the
  `log` route, configuration `heads-end5+choice / L2 choice-end`),
  `zd_windows.py` and `zd_prose.py multi` (the `prose` route),
  `zd_offline.py` and `zd_cache.py` (the readings), `folded_locate.py`
  (`--route choice`), `prosehay.py`, `r2set.py`, `z23_r4.py` and
  `z23_judge.py` (the builders and the judge the acceptance's follow),
  `served.py` (set F's).
