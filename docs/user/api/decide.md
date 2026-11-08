---
title: "Decisions: a System One-compatible API"
description: "POST /v1/decide and POST /v1/systemone: the System One API's typed decisions, answered from one prefill of the model's distribution, plus locating a line or an item."
sidebar:
  order: 10
  label: "Decisions (System One)"
---

ignis serves the **System One API**, the typed-decision API the Jev model is
served under, on its own model. The same handler answers on two routes,
**`POST /v1/systemone`**, System One's own, and `POST /v1/decide`, so a client
written for System One reaches ignis by changing its base URL. The wire shape
is copied, not invented: one `state`, a map of `questions`, typed answers with
a probability on each, and System One's own error statuses (a request body
that fails validation is a `422`). The one field that differs is `model`: leave
it out or name the model ignis serves, since a Jev model name is refused with
`422 model_not_found`. A decision runs in the `agent` lane class where every
other route defaults to `interactive`, since it holds no residency and takes no
decode lane; a caller whose single decision really is interactive says so with
an `@interactive` suffix on `model`. A fan-out of two or more questions keeps
the longest common prefix of its rendered questions until the fan-out ends;
questions of different kinds do not share their system text, so they have
none.

System One's three primitives (`noul`, `choice`, `score`) are read from the
logits of named answer tokens at a single position: **nothing is generated**,
they cost one prefill, and `usage.output_tokens` is 0. ignis adds five of its
own: `number`, `scalar`, `point` and `box` generate one constrained token per
step, so they cost a prefill plus a round per token (a `point` or `box` read
off the calibrated attention heads costs the prefill alone), and `locate` names
a line or an item ([below](#finding-a-line-or-an-item)).

`number`, `scalar`, and a `point` or `box` answered by the digit chain are
constrained decodes: each step draws
from a permitted set of at most 32 tokens, and a set the leaf cannot honour is
refused, never truncated, since a constraint silently narrowed is a wrong
answer that looks like a right one. `number` reads a number digit by digit into
a field of fixed width, and `point` and `box` two or four of those; a `point`
is a digit chain only when asked for with `"method": "chain"` or on a load with
no calibrated pointing head, and a `box` may be asked for with
`"method": "head"` instead. `scalar` adds the closing brace to the alphabet, so
a number that is complete closes its own object instead of filling a field, and
its `digits` is a maximum. The first token of a run is drawn by its prefill,
not by its first decode round, because a round returns the token the previous
call made ready.

The body is one `state` and a map of `questions`:

```bash
curl http://127.0.0.1:8000/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":"Help! My payouts have been failing for 3 days.",
       "questions":{"queue":{"type":"choice",
                             "instructions":"Which queue should this ticket land in?",
                             "criteria":{"payouts":"Money leaving the account",
                                         "billing":"Money coming in",
                                         "fraud":"Suspected fraud or account takeover"}}}}'
```

Object key order is preserved on the way in and out: options are answered in the
order the body declares them. A `state` that is an image — or an image plus
words — is what `point` and `box` answer against, in the image's own pixels.

Why the endpoint exists, what its confidence means, and the one failure nothing
in the response can show (the **answer mass**, charted on the metrics listener):
[ADR 0034](../../adr/0034-the-leaf-answers-without-generating.md). The Playground's
**Decide** tab builds these requests and shows the full distribution behind each
answer.

## Finding a line or an item

To ask *which* line of a log, element of a list or sentence of a text answers an
instruction, ask a `locate` (GitHub #275, #278). It generates nothing and
writes nothing into the state: calibrated attention heads narrow the text to a
few candidates, and a labelled `choice` decides among them. The text may be far
longer than the context — a log of a million tokens, a document, an API's
answer of thousands of records:

```bash
curl http://127.0.0.1:8000/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"state":{"service":"billing","log":"09:21 INFO gateway: GET /v1/orders 200\n09:21 ERROR billing: provider returned 503\n09:22 INFO auth: user 61 signed in"},
       "questions":{"cause":{"type":"locate",
                             "instructions":"Which line says the card processor was unavailable?",
                             "within":"/log"}}}'
```

```json
{"type": "locate", "kind": "log", "method": "shortlist", "compression": "template_fold",
 "found": 0.97,
 "segment": 1, "value": "09:21 ERROR billing: provider returned 503", "confidence": 0.91,
 "ranking": [{"segment": 1, "share": 0.91}, {"segment": 0, "share": 0.05}],
 "pointers": [{"segment": 1, "value": "09:21 ERROR billing: provider returned 503", "share": 0.91},
              {"segment": 0, "value": "09:21 INFO gateway: GET /v1/orders 200", "share": 0.05}]}
```

- **The target** is the `state`, or the part `within` names (a JSON Pointer): a
  string, whose segments are its lines split on `\n` exactly, or a non-empty
  array, whose segments are its elements. `segment` numbers them from 0, the
  way `split("\n")` or the array index would — always into the state you
  sent, whatever was folded or windowed — and `value` is the segment as you
  sent it. A segment's keys are the prompt tokens holding most of its bytes,
  found from the byte range recorded as the state was written, never by
  searching the text, so two identical lines are two segments; an empty or
  all-whitespace one keeps its index and owns no key. The reading prompt puts
  the evidence alone in the system message (layout L1) and reads at the copy
  scaffold `{"quote":"`.
- **`kind`** names the reading: `log`, `prose` or `records`, or `auto` (the
  default). `auto` says `records` for an array of two or more JSON objects;
  otherwise it folds the target's first 2,000 non-blank segments into
  templates and says `log` when at least half of them fall in templates of
  two or more, `prose` otherwise. The answer names what it resolved.
- **`compression`**: `template_fold` (the default for a log) folds the log
  into templates and their values first, so nothing near its length is ever
  prefilled; `none` (the default for prose and records) reads the text as it
  is.
- **`method`**: `shortlist` (the default) or `vote`, the head vote served
  before GitHub #278, unchanged — one prefill over at most 4,554 tokens on the
  served artifact (`locate_too_long` past that), no `found`, `pointers` its
  winner alone. Its `confidence` is how much the heads agree, not a calibrated
  probability.

What each combination does, costs and measured
([spec 22](../../specs/decide/22-locate-by-copy-over-a-folded-state.md); the
acceptance run's numbers are in
[the shortlist's acceptance](../../findings/2026-09-28-locate-shortlist-acceptance-partial.md)):

| | reads | cost | measured (exploratory) |
|---|---|---|---|
| `log` + `template_fold` (default) | the end heads keep 5 templates, a `choice` picks one; they keep 16 of its rows, a last `choice` over those rows' original lines picks the line; `found` | a few short prefills: 0.7-2.6 s median at 100K-1M tokens | 21 of 23 on a fresh cluster capture of 100K to 1M tokens |
| `log` + `none` | the end heads over the whole log, window by window, keep 16 lines; a `choice`; no `found` | each window's prefill | — |
| `prose` + `none` (default) | the sum heads keep 16 sentences; a `choice` over them inside their paragraphs; `found` | first question: each window's prefill (45-68 s per 200K tokens); then 1-4 s per window | a gold sentence picked 91.7% up to 200K tokens, 58% at 1M |
| `records` + `none` (default) | the end heads keep 16 records; a `choice` over them as one-line JSON; `found` | as prose | 58 of 60 up to 3,500 records, 29 of 30 at 10,000 |
| `records` + `template_fold` | the log route over the records as one-line JSON; no `found` | no long prefill: 1.3-1.7 s median | 52 of 60 |
| `vote` + `none` | the head vote | one prefill | 93.6% on short states |

Refused before any prefill: an unknown value (`kind_unknown`,
`method_unknown`, `compression_unknown`, each naming the accepted ones); a fold
under `vote` or of prose (`compression_unsupported`); `kind: records` on a
state that is not an array of objects, or `log`/`prose` on one
(`kind_mismatch`); `kind` or `compression` on any other type
(`kind_unsupported`, `compression_unsupported`); one segment longer than a
window (`locate_segment_too_long`); a load whose artifact nobody calibrated
(`locate_uncalibrated` — the load says which at start, `ignis.decide.locate`,
naming its heads and window); a fold's level-1 text past the context
(`context_exceeded`). Content-parts states are refused too. A step rendered
from an earlier step's answer — a fold's level 2, every `choice` — is known
only after the first prefill, so a fault there (say a `choice` over sixteen
lines too long for the context) is that question's `error` answer with its
code, beside its siblings' answers, not a 422.

**`found`** — on `log` + `template_fold`, `prose` + `none` and `records` +
`none` — says whether the text answers at all. The last `choice` is asked
again, in the same request, with one more option "no line (sentence) of the
evidence answers the criterion", and for a folded log with a yes/no "Is there
a line in the evidence that answers this question: …". `found` is `1 -
p(none)` (averaged with the yes/no's `p(yes)` for a log). **Below 0.5** the
answer names nothing: `segment`, `value` and `confidence` are `null` and
`pointers` is empty, but `ranking` still lists the candidates — apply your own
threshold, or take the best guess knowingly. The other routes always name a
segment and carry no `found`.

**`confidence`, `ranking` and `pointers`** are the last `choice`'s
probabilities — under a fold, times the probability of the template the first
`choice` picked — **never calibrated probabilities**. `ranking` is the
candidates by that share, at most five, the pick first; `pointers` is every
candidate at 0.05 or more, the pick always among them: one answer, or the
several sentences a two-part answer needs.

- **Several lines identical but for their time** are one row of the fold; the
  answer is the first of them.
- **Folding removes order and neighbours.** Level 1 drops every time, a
  line's neighbours are not read, and a bracket-opened line (`[svc-a] …`) is
  read as a source label — unless the bracket holds a time
  (`[Sun Dec 04 04:47:44 2005] …`), which is then the line's timestamp. A question that needs context across lines, or
  names a line by its time alone, is better asked with `compression: "none"`.
- **Windows.** A text the heads read that is longer than 200,000 tokens is cut
  at segment boundaries — at paragraph breaks where there are any — and each
  window is read as its own prefill, with the window alone as its state. A
  window's prefix is kept while the questions over it run, so a second
  question over the same text costs its own short prefills, not the text's.
- **Every prefill is billed**: `usage.input_tokens` counts windows, their
  content-free baselines, a fold's levels and every `choice`; `output_tokens` is 0.
- The Playground's **Decide** tab asks one (GitHub #277, #278): it counts the
  segments `within` cuts before you send, offers the kind, method and
  compression, and shows the answer's pointers and `found`.

A `choice` over labelled segments is what the shortlist asks in its last step,
and you can still ask it yourself over a short text (Jev's "line search"):

- Prefix every non-empty line of a string `state` with its label and `: `, or
  turn an array into an object from label to element, in order.
- Declare the `choice`'s options under the same labels, in the same order, each
  with no description (`null`).
- Use the endpoint's own label order: `A`-`Z`, `a`-`z`, `0`-`9`, then the
  uppercase bigrams this tokenizer reads as one token (`AA`, `AB`, … — not
  `BQ`, which is two). The endpoint shows the model each option under the label
  at its position in that order, so options named with it put the same label in
  the state and in the answer.
- At most 256 segments, the endpoint's option ceiling — past about 16 to 32
  near-duplicates it degrades, which is why the shortlist narrows first.
