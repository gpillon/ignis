# 22 - locate by copy, over a folded state

> **Registered 2026-09-28, before set R3 exists.** The acceptance rules
> below (§ The acceptance run) and set R3's seed are fixed here and are not
> edited once R3's manifest hash is recorded in this banner. A failed rule
> is reported as failed and the defaults do not ship until the owner
> decides; this spec does not pre-decide a fallback.

GitHub: #278. ADR: [0042](../../adr/0042-locate-copies-over-a-folded-state.md) (Proposed).

`locate` (spec 18, ADR 0041) names the **segment** of a text `state` an
instruction asks for, today by one prefill's **head vote**. That vote is
measured to 4,554 keys and, on real logs, finds the right *kind* of line but
not the instance among its near-duplicates. This spec ships what the
research on long real logs found (specs 20 and 21 and their findings): the
model **copies** the line under a constraint to the state's own lines, and
by default reads a **folded** state — near-duplicate lines folded into their
templates — first the template, then the instance, then unfolded to the
original line. The caller chooses both, as two enums on the question:
`method` (`copy`, the default, or `vote`) and `compression`
(`template_fold`, the default, or `none`).

One ticket, whose acceptance is numbered at the end.

## Problem Statement

A caller with a real log wants the one line that answers an instruction —
the failed write, the refused login, the pod that restarted — and the logs
this is for are 16K to 200K tokens long.

- **The served vote refuses past 4,554 keys** (`locate_too_long`), and
  lifting the ceiling does not help: on real logs it read 28 of 50 over
  4K-200K on set R and **20 of 58 (34.5%)** on the fresh set R2, falling
  with length ([locate at length](../../findings/2026-09-27-locate-at-length.md),
  [real logs](../../findings/2026-09-28-locating-a-line-in-real-logs.md)).
- **Its misses are near-duplicates.** At the copy scaffold the heads name
  the kind of line — the same exporter's other write failure, the same
  socket error of another session — and the miss rate climbs with the
  number of lines of the target's template (81% with six or more siblings
  on R2). Half of the attention row stays on the state at every length, so
  each line's share falls as the log grows; the heads that read the instance
  on more than half of R's questions fall from 51 of 384 at 4K tokens to one
  at 200K ([the instance is read while
  copying](../../findings/2026-09-27-the-instance-is-read-while-copying.md)).
- **Generating the line works and costs too much.** Having the model write
  the line out on the whole log read 86.2% of R2 and 92% of R, but the first
  question pays the whole log's prefill (a median 47 s on R2, 139 s at 200K
  tokens on R), and the answer is free text matched back to a line by string
  search — the parser `/v1/decide` exists to remove.
- **The caller cannot choose.** `method` is refused on a `locate` today, and
  there is one reading.

## Solution

Two fields on a `locate` question, both enums, both optional:

| | `compression: "template_fold"` (default) | `compression: "none"` |
|---|---|---|
| **`method: "copy"`** (default) | fold the target; copy a template line of level 1, then an instance row of level 2; unfold to the original segment | copy a segment of the whole state |
| **`method: "vote"`** | the head vote over level 1, then over level 2; unfold | today's `locate`, byte for byte |

- **`template_fold`** folds the target's lines, Drain-style, into
  **templates**: lines of one source label and one token count whose tokens
  agree on at least half their positions share a template, and every
  position where they differ becomes a slot. **Level 1** is one line per
  template, each slot showing its distinct values (`{v1|v2|...}`) and the
  count `(xN)`; **level 2** is the chosen template's lines as their values
  alone, values first and time last, exact repeats folded; a map takes a
  level-2 row back to the original segments. Level 1 is a median 7.4x
  shorter than the log (1.3x-51x). It is `tools/locate-sets/compress.py`'s
  fold, the one R2 measured, and nothing else.
- **`copy`** forces the vote's own scaffold `{"quote":"` and then decodes
  **only continuations of the candidate segments' text**, one constrained
  token at a time, and stops as soon as what it has written belongs to one
  segment's text alone — a median 26 tokens on real logs, where the whole
  line is 90. The written prefix *is* the answer: no string is matched.
- **`vote`** is kept, unchanged under `none`, and reads each level of a fold
  under `template_fold`. `LOCATE_MAX_KEYS` limits the text the vote actually
  reads, so a long log whose level-1 text fits can be voted.

On R2 the folded route read **55 of 58 (94.8%)** against 50 for generating on
the whole log, 28 for folded + vote and 20 for the served vote, answering in
a median 1.3 s (p90 3.9 s) instead of a long prefill. Every answer is still
an index into the **original** state, with the segment's value as the caller
sent it.

## User Stories

1. As an agent reading a 100K-token service log, I want to ask which line
   is the failed write and get its index in about a second, so that finding
   a line does not cost the log's whole prefill.
2. As that agent, I want the answer to be the instance I described — this
   session's socket error, not another's — so that I act on the right line
   among its near-duplicates.
3. As a caller, I want the default to be the method that measured best on
   real logs, so that I do not have to know the research to get the good
   answer.
4. As a caller, I want to choose the reading (`copy` or `vote`) and the
   compression (`template_fold` or `none`) by name, so that I can trade
   accuracy, latency and context for my case.
5. As a caller, I want both choices to be enums that name what they do, so
   that a third method or compression later is a new value, not a new flag
   that contradicts an old one.
6. As a caller, I want an unknown `method` or `compression` refused with the
   accepted values named, so that a typo never runs the default silently.
7. As a caller, I want `compression` refused on every other question type,
   so that a field I wrote is never ignored.
8. As a caller, I want `head`/`chain` refused on a `locate` and `copy`/`vote`
   refused on a `point` or `box`, so that each primitive's methods stay its
   own.
9. As a caller, I want the answer to name the `method` and `compression`
   that produced it, as a `point` names its method, so that a default I did
   not write is visible in what I got.
10. As a caller, I want `segment` to index the state I sent — the line of
    `split("\n")`, the array element — whatever was folded, so that I map
    the answer back without the server's help.
11. As a caller, I want to know which line I get when several lines are the
    same (identical, or identical but for their time), so that a repeated
    event gives a predictable answer: the first.
12. As a caller, I want `confidence` documented as what it is for each
    method — for a copy, the probability of the path it wrote among the
    candidates' continuations; for a vote, the heads' agreement — and never
    as a calibrated probability, so that I set my own thresholds knowingly.
13. As a caller asking a `vote`, I want the ceiling to apply to the text the
    vote reads, so that a long log folds to a level-1 text that fits and can
    still be voted.
14. As a caller, I want `copy` to be bounded by the context and not by the
    vote's ceiling, so that `copy` with `none` answers any state that fits.
15. As a caller with a state longer than the context, I want a folded
    `locate` to answer when its level texts fit, so that the log's raw
    length is not the limit.
16. As a caller on a load nobody calibrated heads for, I want `copy` to work
    and `vote` to be refused by name, so that the default does not depend on
    a calibration.
17. As a caller asking several `locate`s over one state, I want the fold
    computed once and the level-1 prompt shared, so that the second question
    costs a question, not a fold and a prefill.
18. As a caller mixing `locate` with `noul` or `choice` over one state, I
    want every answer back in one request, so that a folded question is just
    another question of the fan-out.
19. As a caller with an array of JSON records, I want `template_fold` to fold
    records of one shape into a template and find the record by its values,
    so that a list of thousands of tickets works like a log.
20. As a caller whose question needs context across lines ("the error right
    after the deploy") or names only a time, I want the documentation to say
    that folding removes that and `compression: "none"` keeps it, so that I
    choose knowingly.
21. As a Playground user, I want the Decide tab to offer the method and the
    compression beside a `locate`, starting on the endpoint's defaults, so
    that I can compare them on my own evidence.
22. As an operator, I want to count `locate`s by method and compression, so
    that I can see the mix a load serves and what it costs.
23. As an operator, I want the request log to say, per folded `locate`, how
    many level-1 lines it read, how many rows the chosen template had and how
    many tokens each copy drew, so that a slow or wrong answer is
    attributable.
24. As an operator, I want a load to say at start that `copy` is available
    and whether `vote` is, so that I know before the first request.
25. As a maintainer, I want `template_fold` to be a pure host function held to
    golden cases written by the Python reference that R2 measured, so that
    the served fold is the measured fold.
26. As a maintainer, I want the copy's constraint to be a pure host
    structure with table-driven tests, and the mock to report its draws and
    honour a wide set, so that the endpoint is covered without a GPU
    (ADR 0006).
27. As a maintainer, I want the leaf's new behaviour — the draw reported in
    the call that makes it, a set wider than 32 — held to an independent
    oracle on the GPU, so that a constraint silently dropped cannot pass.
28. As a maintainer, I want nothing to change for a request that is not a
    copy: the 32-id permitted path, `number`, `scalar` and the thinking
    close bit for bit, and no allocation or launch for a round without a
    copy lane.
29. As the owner, I want the defaults judged once on a fresh real-log set
    with rules written here before the set exists, and the vote's behaviour
    on set F held unchanged, so that the new default is not tuned to its own
    test.
30. As the owner, I want the short states the vote was accepted on (set F's
    logs, records and prose) guarded, so that the default that wins on long
    logs does not quietly lose on short ones.

## Implementation Decisions

### The wire

- **`method`** on a `locate`: `"copy"` (default) or `"vote"`. The field
  exists for `point` and `box` (`head`, `chain`); spec 18's refusal of it on
  a `locate` is lifted.
  - An unknown value on a `locate` is `method_unknown`, naming `copy` and
    `vote`; `head` and `chain` are unknown values there. `copy` and `vote`
    on a `point` or `box` are `method_unknown` naming `head` and `chain`.
  - On every other type `method` stays `method_unsupported`, its message now
    naming the three primitives that have one.
- **`compression`** on a `locate`: `"template_fold"` (default) or `"none"`.
  An unknown value is `compression_unknown`, naming both; `compression` on
  any other type is `compression_unsupported`.
- **All four combinations are valid.** Absent fields take the defaults and
  are never read as another value.
- **Refusals, all before any prefill** (a 422, nothing reaches the engine),
  except where marked:

  | code | when |
  |---|---|
  | `locate_uncalibrated` | `method: "vote"` on a load with no `locate` calibration (a `copy` is served on every load) |
  | `locate_too_long` | `vote` + `none`: the target past `LOCATE_MAX_KEYS`, as today. `vote` + `template_fold`: the level-1 text past it. **At runtime**, as that question's `Answer::Error`: the chosen template's level-2 text past it, which cannot be known before level 1 answers |
  | `context_exceeded` | `copy`: the prompt plus the copy's budget (its longest candidate's tokens and one) past the context; for level 2, at runtime, as the question's `Answer::Error` |
  | `locate_too_few_segments` | fewer than two segments with content, every method (unchanged) |
  | `locate_unsupported` | the loaded template cannot say where its tokens sit — the vote's alone, which reads keys; a `copy` needs only the tokenizer |

  `within`, `locate_needs_json_state` and the target refusals are
  unchanged.
- **The answer** keeps `Answer::Locate`'s shape and gains two fields:

  ```json
  {"type": "locate", "method": "copy", "compression": "template_fold",
   "segment": 1482, "value": "…the line as sent…", "confidence": 0.91,
   "ranking": [{"segment": 1482, "share": 0.91}]}
  ```

  - `segment` indexes the **original** target; `value` is that segment as
    the caller sent it.
  - **Several segments with one text.** A copy cannot tell identical
    segments apart, and a level-2 row folds lines identical but for their
    time. Either way the answer is **the first** of them (the lowest index),
    and the documentation says so. `vote` + `none` is unchanged: it may name
    any of them.
  - **`copy`**: `ranking` is one entry, the answer, with `share` equal to
    `confidence`. `confidence` is the product of the copy's **draws** — each
    the probability of the token within its own permitted set, exactly 1 for
    a set of one — so the probability, restricted at every step to the
    candidates' continuations, of the path the greedy decode wrote. Under
    `template_fold` it is the product over both levels. It is **not a
    calibrated probability**, and there is no "not found".
  - **`vote`**: `none` is today's (the winner's share of the votes, the top
    five). Under `template_fold`, `confidence` is the product of the levels'
    confidences, and `ranking` is the last level read's, each row or template
    named by its first original segment and each share multiplied by the
    earlier level's confidence — so, as for every `locate`, the first entry
    is the answer and its share is `confidence`.
  - A level with one row (or a fold with one template) is not asked; it
    counts as 1.
- **`usage`**: `input_tokens` counts every prefill (both levels, the vote's
  content-free twins); `output_tokens` counts the copy's draws (0 for a
  vote, as today).
- **Thinking** is refused as for every decision; the copy is greedy.

### `template_fold`

- **The reference is `tools/locate-sets/compress.py`** (`fold` with
  `values=True`, `summarize`, `level2`, `_common_affixes`) at the settings
  R2 was judged with: `SIM` = 0.5; the `TIME`, `VARIABLE`, `LABEL` and
  `TOKEN` patterns as written there; a slot's distinct values all shown when
  they fit a 600-character budget, else the first 6, each cut to 24
  characters, and `|+N` for the rest; `(xN)` on a template of N lines;
  level-2 rows values-first — each slot's common prefix and suffix cut, the
  values joined by `" | "`, the line's first time last (`" @ time"`), rows
  with equal values folded with `(xN)`.
- **A pure host function in `ignis_core`**, beside `ignis_core::locate`,
  ported and held to golden cases (§ Testing Decisions). Its arithmetic is
  Python's: lengths, cuts and affixes in **code points**, not bytes; `\d`,
  `\S` and `\b` with Unicode classes; clusters in first-seen order.
- **What it folds.** The target's segments that have content — empty and
  all-whitespace segments are left out, as they own no key for the vote, and
  are never an answer.
  - A string's lines, as sent (a `\r` stays; the fold's tokens ignore it).
  - An array's elements: a **string** element as its text; any other
    element as its JSON **with a space after each `,` and `:` outside
    strings** (Python's `json.dumps(element, ensure_ascii=False)`), keys in
    the order sent, numbers as the caller wrote them (the golden cases use
    integers and plain decimals, which both spell alike). *Not* the compact JSON
    the vote renders: the fold compares whitespace tokens, and a compact
    record is one token, so records would never fold.
- **Kept as measured, and documented as limits**, not repaired here:
  - a line that opens with a bracket is read as a source label (the
    cluster's `[pod]` prefix), so lines that open with a bracketed time do
    not fold — they are located as if unfolded;
  - level 1 drops every time, so a question that names a line by its time
    alone can only be answered at level 2;
  - folding removes the order and neighbours of lines, so a question that
    needs context across lines should ask `none`.
- **The map** takes (template, level-2 row) to the original segments, in
  order; the answer is the first.
- **Computed once per target per request** — per (`state`, `within`) — on
  the host before any prefill, and shared by every `template_fold` question
  over that target. It is deterministic, so a later request over the same
  state recomputes the same texts and claims their retained prefix.

### The prompts of a folded question

- Each level is asked as a `locate` over a **derived state**: the caller's
  `state` with its target replaced by the level's text — its rows joined by
  `\n`, one string — so the rest of a `within` state stays as the caller
  sent it. With no `within` (the case R2 measured) the evidence is the
  level's text alone.
- The kind text is the `line` kind text for both levels, whatever the
  original target was: the folded text is lines.
- Otherwise it is the `locate` render as it is: layout L1, the instruction
  as sent, the scaffold `{"quote":"`.
- **Two stages.** Level 1 is asked over the level-1 text. Its answer names
  a template; level 2 is asked over that template's rows; the row's first
  original segment is the answer. A fold with one template skips level 1; a
  template with one row skips level 2. A folded question is two internal
  requests in sequence; its second is submitted when its first answers,
  while the request's other questions run.

### The copy

- **The prompt is the vote's question prompt, byte for byte**: layout L1,
  the kind text, the instruction, the forced `{"quote":"`. No content-free
  twin. A `copy` and a `vote` over one target therefore share their whole
  prompt, and the prompt-pinning test covers both.
- **Each candidate's copy text** is the segment as the evidence renders it,
  which is what the model has to copy:
  - a line (and every row of a folded level): its JSON-escaped text,
    without the separators;
  - an array element that is a string: its JSON-escaped content, without
    its quotes;
  - any other element: its JSON as the evidence renders it.

  Each is tokenized **alone** with the loaded tokenizer (no special tokens)
  — the tokens the model would write after the scaffold's opening quote.
- **The copy tree** — a prefix tree over the candidates' copy tokens, each
  node knowing the segments whose tokens pass through it — is a pure host
  structure beside `ignis_core::constrained`.
  - The **permitted set** at a node is its children's tokens, plus the
    **terminators** where some candidate's tokens end at that node and
    others continue. The terminators are the loaded tokenizer's single
    tokens that spell `"` or `"}` (whichever it has), computed at load like
    the answer alphabet.
  - **The run settles** when every segment through the node it has reached
    has the same copy text — at once when one segment is left, which on
    real logs is a median 26 tokens (p90 46) into a line of 90. The answer
    is the lowest-index segment there.
  - A drawn **terminator** settles on the lowest-index segment whose copy
    text ends at that node. A text that is a proper prefix of another is
    reached this way.
  - A copy never ends on EOS (EOS is not in the tree) and never runs past
    the tree's depth plus one draw.
- **It is a constrained decode**, like a `number` (ADR 0034, GitHub #242):
  greedy, one lane, plain rounds on a speculative load (a verify round
  refuses a permitted set), the first draw made by the prefill, no residency
  after. Its reservation is the prompt plus its budget, the longest
  candidate's tokens and one.

### What the seam and the leaf need that #242 did not

#242's schedule is a function of the step index alone, at most 32 ids per
step, and a round **returns the token the previous call drew**. A copy's
next set depends on the token just drawn, and at its root it can be wider
than 32. Three changes, and nothing else moves:

1. **A constrained decode has two shapes**: #242's `Schedule`, or a
   **copy** (the copy tree). A request carries one of them or a decision
   read, never two. The scheduler asks a copy for the set of the draw each
   job makes, from the draws reported so far; when the reported draw
   settles the run, the next round carries no set, commits that draw and
   finishes the request (`stop`) — the schedule's last-round rule.
2. **The leaf reports the draw a call makes, in that call**: an appended
   output on `ignis_prefill_options` and `ignis_decode_options` (ADR 0016:
   a field append and a size bump), filled for the lanes and jobs that ask;
   `PrefillOutcome` and `DecodeOutcome` carry it across the `Compute` seam.
   What a round **commits** is unchanged — still the previous call's draw —
   so the lag stays where #242 documented it; only the host learns the draw
   one call earlier, one more token per copy lane in what the call returns.
   A call where no lane or job asks for it returns nothing new and waits for
   nothing new.
3. **A copy's permitted set may be as wide as the vocabulary**, and is
   honoured, never refused or truncated for its width: a per-lane
   **vocabulary bitmask** (one bit per id, 31,040 bytes on the 27B)
   appended to `ignis_sampling_params` beside the 32-id row. The mask kernel
   tests the bit; the restricted probability is the softmax over the mask's
   members. Its staging — one mask per decode lane
   (`IGNIS_DECODE_MAX_BATCH`) and per prefill job that can carry a set — is
   reserved at load (ADR 0030) and stated in the VRAM plan.

The 32-id path — `number`, `scalar`, the thinking close — is unchanged bit
for bit, and a round or prefill with no copy lane allocates and launches
nothing new. `MockCompute` reports its draws in the call that makes them and
honours a set of any width.

### The vote, and what `LOCATE_MAX_KEYS` limits

- `vote` + `none` is today's `locate`: the same render, the same content-free
  twin, the same heads, the same answer.
- `vote` + `template_fold` asks the vote at each level, each with its own
  content-free twin over that level's derived state.
- **`LOCATE_MAX_KEYS` limits the text the vote reads**: the target under
  `none`, each level's text under `template_fold`. The calibration table,
  the score room reserved at load and the ceiling (4,554 keys on the served
  27B) are unchanged. A `copy` is bounded by the context and never by it.

### Fan-out, reuse and loads

- A request mixing methods, compressions and other primitives over one state
  answers every question. `copy` and `vote` with one compression share their
  prompt up to the scaffold; `template_fold` questions over one target share
  its level-1 prompt; a `template_fold` prompt holds no state and so shares
  nothing with the other kinds over it (they keep sharing with each other).
- **Every load serves `copy`.** `vote` needs the load's `locate` calibration
  (ADR 0041), as today. The `ignis.decide.locate` event says both.

### Observability and documentation

- **Metrics** (ADR 0017): `ignis_decisions_total{type="locate"}` still
  counts one per question, whatever its stages. A new counter
  `ignis_locates_total{method="copy|vote",compression="template_fold|none"}`
  counts `locate` questions by what answered them — recorded by the handler,
  absent until the first `locate`, all four series exported once present
  (the decision family's exception). No answer mass (a copy is generated, a
  vote is read off attention). A copy's draws count wherever a `number`'s
  do. ADR 0017 is amended.
- **The request log**: a `locate`'s line names its method and compression,
  and for a fold the level-1 lines, the chosen template's rows, and the
  draws per level.
- **OpenAPI** (ADR 0036): `method` documented per primitive (`head`, `chain`
  for `point` and `box`; `copy`, `vote` for `locate`), `compression`, the
  answer's two new fields, the new refusals and the changed defaults.
- **`docs/user/README.md`**, "Finding a line or an item": the defaults, the
  four combinations with what each costs and measured, the answer's
  `method`/`compression`, the first-of-identical rule, what `confidence`
  means per method, the vote's ceiling and calibration, and the fold's
  limits (cross-line context, time-only questions, bracket-opened lines).
  The labelled-`choice` recipe stays as the route for a question the fold
  cannot serve and a vote-only load cannot reach.
- **`CONTEXT.md`**: *Copy* (the method), *Copy tree*, *Template fold*,
  *Level 1* and *Level 2*; *Locate*, *Constrained decode*, *Permitted set*
  (a copy's may be wider than 32), *Draw* (reported in the call that makes
  it) amended.
- **ADR 0042** accepted with the acceptance's numbers; ADR 0041's status
  names it (the vote is one of two methods).
- **The Playground's Decide tab** (#277, `web/src/decide/`): a `locate`
  card gets two selectors, method and compression, in the shape of the
  point/box selector (#260) — the default choice sends no field and says
  the endpoint will use `copy` / `template_fold`; each named value is sent.
  The answer panel names the method and compression that answered; for a
  copy it shows the confidence and no vote. The dev mock answers all four.

## The acceptance run (registered before set R3 exists)

### Set R3

A fresh real-log set nobody has asked, built by rule before any route runs
on it — r2set.py's rules, on new sources. Seed **20261050**.

- **Cluster**: a new capture of the owner's cluster, taken after R2's —
  `kubectl logs --since=6h --timestamps` of its running pods, read only,
  merged by time with `prodset.py timeline` — cut into 2 windows per tier of
  16K / 50K / 100K / 200K tokens, 4 targets per window. **Never committed**;
  only its manifest hash enters the repository.
- **LogHub**: the **full** logs LogHub publishes (downloaded into
  `.scratch/`, never committed) — not the 2k samples, which R2 used whole.
  Per system, one window of 100K tokens and, where the log holds it, one of
  200K, cut at seeded offsets and sharing no line with that system's 2k
  sample; 3 targets per window.
- **Targets by rule**: a line's siblings are the lines whose word sets,
  times, numbers, ids and hashes removed, have a Jaccard of at least 0.5 with
  it; lines with an exact twin after that removal are excluded; targets are
  drawn round-robin over the sibling bins 0, 1-5, 6-50, > 50.
- **Questions** written for the drawn targets after reading the windows and
  before any answer, checked by `r2set.py build`'s rules (lexical, combo,
  paraphrase only for targets without siblings); a target no question can
  single out is dropped, not replaced, with its reason recorded; at least 10
  absent questions (the target removed).
- **Size**: at least 80 present questions and at least 12 windows at 100K
  tokens. A shortfall the sources force is recorded before any run, and the
  rules apply as written to what exists.
- The builder (`r3set.py`, or r2set.py with a source for the full logs) and
  the judge (`r3_judge.py`) are committed, and R3's manifest sha256 written
  into this spec's banner, **before any route is asked on R3**. R, R2, L,
  L2 and sets A-D may be used to debug; R3 is asked once.

### The runs

On the served artifact under `make start`'s defaults (hq-e8-2b with the
residual window, chunk 1,024, DFlash2 loaded), one `locate` per request
through `/v1/decide` (`served.py ask` with `--method` and `--compression`):

1. **R3, `copy` + `template_fold`**, first, on a freshly started server — its
   first question per window meets no retained state.
2. **R3, `copy` + `none`.**
3. **R3, `vote` + `template_fold`** (reported).
4. **Set F, `vote` + `none`**, named explicitly.
5. **Set F, `copy` + `template_fold`.**

### The rules

1. **Top-1.** `copy` + `template_fold` names the target on **at least 90%**
   of R3's present questions.
2. **Non-inferiority.** `copy` + `template_fold`'s top-1 on R3's present
   questions is **at least `copy` + `none`'s minus 5 points**.
3. **Latency.** Over R3's 100K-token windows, the **median wall time of each
   window's first question** under `copy` + `template_fold` — the client's
   request time, the fold included — is **at most 3.0 s**.
4. **The vote unchanged.** On set F, `vote` + `none` serves and refuses the
   same questions as the recorded run (`.scratch/locate/F-served.json` in
   the main checkout, [finding](../../findings/2026-09-27-locate-through-decide.md)): the same
   `locate_too_long` refusals, and the same segment on every served question
   whose recorded winner led the next by more than one vote; where it led by
   one vote or tied, one of those two segments.
5. **Short states.** On set F's present questions the vote serves, `copy` +
   `template_fold`'s top-1 per family is at least the recorded vote's minus
   5 points: **logs ≥ 39/43, records ≥ 44/46, prose ≥ 56/67**.

### Reported, not asserted

`vote` + `template_fold` on R3; `copy` + `none` on F; `copy` +
`template_fold` on F's 45 present questions past the vote's ceiling; by
source and tier: top-1, level-1 accuracy (whether the answer's template
holds the target, from the reference fold), draws per level against the
target's unique-prefix length, level-1 compression, prompt tokens and wall
time (median, p90) of every route; the present/absent AUC of each method's
`confidence`; the fold's host time on the longest window. All of it goes in
a finding with a README row, and ADR 0042's numbers come from it.

## Testing Decisions

A good test asserts what a caller or an operator can observe — the segment,
the method that answered, the refusals, the rounds run — and holds the
device path to an **independent** oracle.

- **`template_fold` (CPU, pure), against golden cases.** `compress.py` gains
  a `golden` subcommand that writes, for a fixed list of inputs, the input
  lines, the level-1 rows, each template's members and each template's
  level-2 rows with their segments, into
  `crates/core/tests/fixtures/template_fold/`. The Rust fold reproduces them
  exactly. The inputs are synthetic — `logs.py` and `records.py` output at
  fixed seeds, and hand-written edge cases — and never a LogHub or cluster
  line: labels, times in every `TIME` shape, masked variables, the `SIM`
  boundary, a slot past the 600-character budget (`|+N`), a value past 24
  code points, non-ASCII text and digits, one-value slots, exact repeats and
  repeats differing only in time, affixes cut to nothing, empty and
  whitespace lines, `\r\n`, a bracket-opened line, records of two shapes
  rendered as spaced JSON, string elements.
- **The copy tree (CPU, pure), table-driven**: permitted sets at the root
  and below; settling at the first prefix one text owns; identical texts
  settling on the lowest index; a text that is a proper prefix of another,
  reached by a terminator; texts that share characters but not tokens;
  copy texts of a line, a string element and an object element, escapes
  included; whitespace segments left out; a node wider than 32; the budget;
  the confidence's product.
- **The scheduler (CPU)**: a copy request's prefill carries the root set;
  every round's set is the one the reported draws lead to; the run finishes
  one round after the settling draw, never on EOS; a batch mixing a copy
  lane with a free lane, a `number` and a thinking close leaves their jobs
  as they were; a copy on a speculative load runs plain rounds.
- **The endpoint over the mock (CPU, `/v1/decide`)**:
  - no fields answer `method: "copy"`, `compression: "template_fold"`; each
    of the four combinations answers with the shape above;
  - whatever the mock draws, the answer is the one segment the draws lead
    to, and the rounds are the draws to settle plus one; forced steps report
    1; identical lines answer the first;
  - every refusal in the wire section, before any prefill, and the two
    runtime ones as that question's `Answer::Error` while its siblings
    answer;
  - `vote` refused on an uncalibrated load while `copy` is served there;
    `vote` + `none` refused past the ceiling where `vote` + `template_fold`
    is served because its level-1 text fits;
  - a fan-out of all four combinations, a `noul` and a `choice` over one
    state; the fold computed once per target, and a second `template_fold`
    question claiming the first's level-1 prefix;
  - `usage`, the request log's fields, `ignis_locates_total`, the OpenAPI
    document's new fields and values.
- **The vote is pinned.** Today's `locate` tests ask for `method: "vote"`,
  `compression: "none"` by name and pass unchanged, and the prompt-pinning
  test (`decide_locate_prompt.rs`) holds that render byte-identical.
- **The leaf (GPU profile), against the readout as oracle**, under BF16 and
  hq-e8-2b:
  - the draw a prefill or a round reports is the token the next round
    commits, on every lane of a batch mixing a copy lane with free and
    32-id lanes;
  - a set of 1,000 ids and one of the whole vocabulary but one are honoured:
    every draw is a member;
  - the greedy draw equals the argmax of a readout (ADR 0034) over the same
    ids at the same position, and its reported probability equals that
    readout's softmax over them, to float accumulation;
  - `permitted_decode_gpu.rs` (the 32-id path) passes unchanged; the VRAM
    plan's test states the new bytes.
- **End to end (GPU profile)**: on a committed synthetic fixture of logs
  with near-duplicate lines and records of one shape, `copy` +
  `template_fold` and `copy` + `none` through `/v1/decide` name every
  question's target, each level's draws at most its unique prefix plus one.
- **The Playground** (vitest): the two selectors, their defaults sending no
  field, the named values sent, the answer's method and compression shown,
  a copy's panel without a vote; the dev mock's four answers.

## Acceptance

1. **The fold is the measured fold.** `template_fold` is a pure host
   function that reproduces every golden case `compress.py golden` writes.
2. **The copy tree** passes its table-driven CPU tests.
3. **The seam carries a copy.** The scheduler's CPU tests pass; the mock
   reports its draws and honours a set of any width.
4. **The leaf** reports each draw in the call that makes it and honours a
   set as wide as the vocabulary, held to the readout under both KV formats;
   the 32-id path is unchanged and a call with no copy lane allocates and
   launches nothing new; the masks' reservation is in the VRAM plan.
5. **`/v1/decide` over the mock** passes § Testing Decisions' endpoint list:
   the defaults, the four combinations, every refusal, the fan-out, `usage`,
   the log, the metric.
6. **The vote is unchanged** under `vote` + `none`: today's tests pass
   asking for it by name, and the prompt-pinning test holds its render.
7. **End to end on the GPU**, the fixture's targets are named by `copy`
   under both compressions.
8. **Documented**: OpenAPI, the user README's "Finding a line or an item",
   `CONTEXT.md`, ADR 0017's amendment and ADR 0041's status; the load event
   names both methods; the Playground's selectors and answer panel with
   their tests.
9. **Set R3 is registered** before any route is asked on it: the builder and
   the judge committed, the manifest hash written into this spec's banner,
   the questions checked.
10. **The acceptance holds**: rules 1-5 of § The acceptance run, each judged
    once, recorded as a finding with a README row; ADR 0042 accepted with
    its numbers. A failed rule is reported as failed and the owner decides.
11. `cargo test` passes workspace-wide, and the web tests pass.

## Out of Scope

- **A `found` flag or abstention.** A sibling `noul` separated absent
  questions at an AUC of 0.93 on R and 0.79 on R2, over 10 absent questions
  each: not yet a flag. `confidence` is reported beside present and absent
  questions for that later work.
- **Several lines as one answer**, and spans finer than a segment.
- **Questions that need context across lines** under `template_fold` —
  folding removes the order by design; `none` keeps it.
- **Other compressions** — a better label rule, semantic clustering, another
  budget: each is a new `compression` value with its own measurement.
  `template_fold`'s parameters are not tuned here.
- **Lifting `LOCATE_MAX_KEYS`** — real logs put the vote's ceiling where set
  D did.
- **The end-marker reading** of spec 20 (a small gain for the vote at its
  ceiling).
- **A ranking for `copy`** beyond its answer (a beam over the tree), and a
  copy that samples.
- **Writing forced runs as prompt** (jump-forward decoding: a node with one
  child needs no decision, only a token): a latency optimisation, not
  needed for rule 3.
- **Content-parts states**, as for every `locate`.
- **The research branch's experiment hooks** (`IGNIS_LOCATE_MAX_KEYS_EXPERIMENT`,
  the row dumps, the control file): they stay on `locate-long-context` and
  are never merged. Nothing here needs them.

## Further Notes

- **Why enums and not booleans.** `fold: true` and `copy: true` would be two
  flags whose four combinations are an accident of their spelling, and a
  third method or compression would be a third flag contradicting them. An
  enum names what runs, is echoed in the answer and extends by a value.
- **What changes from the route R2 measured**, and is therefore what the
  acceptance judges:
  - R2's copy was **free generation** through `/v1/chat/completions`, its
    quote matched to a line by exact, contained or word-overlap search —
    at level 1 the model often wrote one value of a `{a|b|c}` slot. The
    served copy is **constrained** to the rows' text and stops at the first
    prefix one row owns.
  - The scaffold is **forced** as `{"quote":"`, the vote's, where the free
    route wrote its own opening; the prompt is otherwise the same L1 render
    and kind text.
  - Arrays fold as spaced JSON; R2 held logs only.
- **The mechanism is the measured one.** The heads settle on the instance
  where the written prefix stops matching every other line (rho 0.59 on R2,
  spec 21's H4b), which is exactly where the copy tree stops the run.
- **Prior art.** A decode constrained to a prefix tree of a closed set's
  tokenized names is GENRE's (De Cao et al., *Autoregressive Entity
  Retrieval*, arXiv 2010.00904); the fold is Drain-style template mining
  (He et al., *Drain*, ICWS 2017); LogHub is Zhu et al.'s collection
  (arXiv 2008.06448).
- **Cost expected.** A folded question is the fold (host, milliseconds), a
  level-1 prefill of a median ~10K tokens for a 100K-token log, a handful of
  constrained rounds, and a short level-2 prefill and rounds. A copy holds a
  decode lane for those rounds, which a vote never did.

## References

- Findings: [locate at length](../../findings/2026-09-27-locate-at-length.md),
  [the instance is read while copying](../../findings/2026-09-27-the-instance-is-read-while-copying.md),
  [locating a line in real logs](../../findings/2026-09-28-locating-a-line-in-real-logs.md),
  [locate through `/v1/decide`](../../findings/2026-09-27-locate-through-decide.md).
- Specs 18 (the `locate` wire, layout L1, set F), 19 (the vote), 20 and 21
  (the research this ships), 06 and 10 (the constrained decode, `number`,
  `scalar`), 04 (fan-out), 17 (layout L1).
- ADR 0034 (readout and constrained decode), 0041 (the vote), 0016
  (appended ABI fields), 0030 (memory reserved at load), 0017 (metrics),
  0036 (OpenAPI), 0006 (CPU tests without a GPU), 0042 (this spec's
  decision).
- Tools: `tools/locate-sets/compress.py` and `folded_locate.py` (the fold
  and the route R2 measured), `r2set.py` and `r2_judge.py` (R2's builder and
  judge), `served.py` (set F's), `prodset.py` (the cluster capture).
