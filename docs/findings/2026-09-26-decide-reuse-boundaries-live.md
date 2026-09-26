# Reuse boundaries on /v1/decide, live

- Kind: experiment
- Status: current
- Observed: 2026-09-26
- Last verified: 2026-09-26
- Scope: serving / `/v1/decide` parts-state reuse, fan-out, retained slots
- Related: GitHub #270, spec `docs/specs/decide/16-reuse-boundaries.md` (acceptance 7), ADR 0029 (amendment 2026-09-26), `docs/findings/2026-09-26-prefix-reuse-prior-art-for-decisions.md`
- Superseded by: none

## Question

Does a `/v1/decide` `state` given as content parts now reuse what it repeats —
a static text head across requests, and the whole state across a fan-out's
questions — and does it meet spec 16's live bounds?

## Evidence

`scripts/decide-reuse-repro.py` (the issue's repro, kept in the repository and
extended with a reuse-marker case G), against a live server with the issue's
flags: 27B `qwen3_8_27b_nvfp4full-v2`, `--kv-format hq-e8-2b --max-context
262144 --prefill-chunk 1024 --kv-host-pool-bytes 8G --vision --spec dflash2
--draft-tokens 7 --metrics`, prompt reuse on, release build, RTX 5090, nothing
else on the card. Each case sends 5 requests one after another; every case has
its own text and its own picture, so no case is served by what an earlier one
left behind. Median of requests 2..5, in ms.

| case | state | baseline eae051d | final, run 2 | final, run 3 |
|---|---|---|---|---|
| A | JSON `{mission: LONG}` | 107 | 85 | 79 |
| B | parts `[LONG]` | 472 | 64 | 64 |
| C | parts `[SHORT, image]` | 70 | 59 | 62 |
| D | parts `[LONG, image]` | 539 | 78 | 93 |
| E | D with 4 questions | 1,958 | 203 | 205 |
| F | C with 4 questions | 348 | 226 | 223 |
| G | D with a marker on LONG — 2nd request | 560 | 98 | 76 |
| E | first request | 1,935 | 485 | 484 |
| D | first request | 557 | 393 | 379 |

The runs went: an earlier final-code run before the part-end trims below, the
baseline, final run 2, the experiment below, final run 3 — one server at a
time, same card, same afternoon. The baseline is ~20% slower even where nothing
is reused (A, C): with a server up the card sat at 31.1 of 32.6 GiB, where
Windows paging makes timings drift. Within one run the cases are comparable;
across runs the same work moved by ±20 ms (D and G do the same work: 78 and 97
in run 2, 93 and 75 in run 3).

`ignis_retained_reused_tokens_total{kind="prefix"}` grew in B (+8,256), D
(+8,256), E (+51,456) and G (+11,008); on the baseline it grew only in A.
`ignis_retained_slots{state="in_use"}` rose by one per case with a kept
boundary and never by a fan-out's head.

Spec 16's bounds on the final build:

| bound | run 2 | run 3 |
|---|---|---|
| B <= 1.3 x A | PASS (64 vs 111) | PASS (64 vs 103) |
| D <= C + 30 ms | PASS (78 vs 89) | FAIL (93 vs 92) |
| E <= 2 x D | FAIL (203 vs 156) | FAIL (205 vs 186) |
| E first <= 1.5 x D first | PASS (485 vs 590) | PASS (484 vs 569) |
| G second <= C + 30 ms | FAIL (98 vs 89) | PASS (76 vs 92) |
| prefix reuse grows in B, D, E | PASS | PASS |

Raw output: `.scratch/reuse-boundaries-270/` in the worktree that ran it
(baseline, two final runs, an earlier run before the part-end trims below, and
the experiment below).

**Where E's time goes.** E is one sequenced question plus three followers:
203 ≈ D (78) + 125. Two things keep the followers from being "tails":

1. **The head ends before the image.** The four questions share
   `<|vision_end|>\n{"criterion":"Question ` after the picture — a handful of
   tokens — and a head is whole 64-token pages that never end inside a media
   item (#193). Its floor lands in the image's placeholders, walks back before
   them, and every follower prefills the image again. F's head, by luck of
   geometry (its picture starts at token ~51), does reach past its image, which
   is why F dropped from 348 to ~225 with no other reuse available to it.
2. **Per-question fixed cost.** F with its head past the image still costs
   ~55 ms per follower. Every question re-acquires and re-renders its image
   and prompt on the CPU, in series, before the first submit (a 320x208 PNG's
   `prepare_media` is ~11 ms, a ~2.7K-token encode ~5 ms), and on the engine
   each decision publishes its generation opener's page — a head no decision
   ever claims (#238 refuses its checkpoint) — which costs a chunk split and
   makes each follower multi-tick, and the scheduler serves one multi-tick
   prefill at a time.

An experiment on the final build that stopped decisions publishing the opener's
page (not kept) gave A 61, B 52, C 60, D 73, E 187, F 180, G 72: every bound
passed except E <= 2 x D (187 vs 146).

**What reporting part ends costs.** Each reported part end is the prompt head
tokenized once more to check it is an exact token prefix (spec 16: fail
closed). On D's prompt that measured ~5 ms per end on the CPU; reporting all
three ends of `[LONG, image, question]` added ~17 ms per question. The final
build reports no end for a message's last part (the question — nothing is cut
there) and asks for ends only on the first question of each system text (the
others share the state and claim the fan-out's head, which covers them): ~10 ms
per decide request for D's shape.

## Finding

A parts `state` is now reused as a JSON one is: B and D drop from ~500 ms to
under 100 ms per repeated request, a reuse marker makes the second request fast
(G), and a four-question fan-out over a repeated image state drops ~10x (E,
1,958 to ~205 ms). The spec's live bounds hold except **E <= 2 x D**, which
fails by ~50 ms in every run, and the two "<= C + 30 ms" bounds, which sit inside
this card's run-to-run noise and flip between runs.

E's bound cannot be met by where state is kept alone: the fan-out's head
reaches past a picture only when the question text its questions share after it
crosses the next page boundary, and even a follower that resumes past its image
costs ~40–55 ms of per-question work that #270 does not touch.

## Implications

- Spec 16's acceptance 3 ("the head runs through the images") holds only for
  that geometry; a test that asserts it (`decide_reuse_boundaries.rs`) builds
  the geometry on purpose, and a sibling test pins the common case, the head
  walking back before the image.
- The cheapest measured win left for fan-outs is not a reuse boundary: stop a
  decision publishing its opener's page (~10–20 ms per question here, and a
  follower no longer needs two prefill steps), and render a fan-out's image once
  instead of once per question.
- A head that ends inside an image would need a claimant able to resume
  mid-item — the encoder output for the remaining placeholders, which #243's
  lingering embedding cache may already hold — and #193 rules that out today.

## Limits and unknowns

- One baseline run and two final runs, 5 requests per case, on a card at its
  VRAM ceiling: the ±20 ms run-to-run noise is larger than the margin of either
  "<= C + 30 ms" bound.
- The game agent's own shape (`[mission][situation][label][image]x3`, two
  questions) was not run; its part ends are ~8 per question, each a head
  tokenization, so the part-end cost there is larger than D's.
- The opener-page experiment was measured once and is not in the code.

## Follow-ups

- Owner decision: revise E's bound, or take one of the three levers above
  (no opener publish for decisions; one render per fan-out image; mid-image
  claims).
- Cheaper exact part ends: tokenize only from the last special token before a
  part end (segments between special tokens tokenize independently), which
  makes every end after an image free.
