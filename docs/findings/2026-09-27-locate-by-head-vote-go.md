# Locate by a head vote is a go on set D

- Kind: experiment
- Status: current
- Observed: 2026-09-27
- Last verified: 2026-09-27
- Scope: `/v1/decide` / attention readout over a text state — spec 19 track L, spec 18's rule 2 on set D
- Related: [GitHub #276](https://github.com/gpillon/ignis/issues/276), [#275](https://github.com/gpillon/ignis/issues/275) (spec 18 phase B); [spec 19](../specs/decide/19-a-span-read-from-attention.md) § Phase 3, track L; [spec 18](../specs/decide/18-locate-by-attention.md); [phase 0: a vote of heads finds the line](2026-09-27-a-head-vote-finds-the-line.md); [spec 18's no-go](2026-09-26-locate-attention-no-go.md)
- Superseded by: none

## Question

Spec 18 phase A's single head read the line a text instruction asks for 18
points behind the labelled `choice` on set C, and its rule said no-go. Spec
19's phase 0 found a reading that did far better on the development sets —
a vote of the heads with the most training hits — and registered it, with
spec 18's rule 2 verbatim, as track L's one check on set D, which nothing
had read (spec 19, commit `1d1d999`; the owner confirmed the rule before the
scores were computed). Does it pass?

## Evidence

- **Set D** (seed 20261013, 240 questions: logs, JSON records, HotpotQA
  prose; 201 present), generated with A-C on 2026-09-26 and unread until now.
- **The dump**: spec 18's harness unchanged
  (`attention_head_locate_gpu.rs`, served NVFP4, consumed hq-e8-2b keys; the
  capture's self-check passed on every layer of all 960 prefills), 723 s.
- **The labelled route**: `labelled.py` through a live `make start` server,
  the configuration spec 18 used on C; 188 questions asked (155 present with
  at most 256 segments, 33 absent).
- **The reading**: `score.py check --choice vote-choice.json` — the vote of
  the 32 heads fitted on A+B, copy scaffold, content-free baseline — with
  rules 2-4 as spec 18 computes them. Output in
  `.scratch/locate/phase0/D-check.json`; spec 18's own rule-1 choice (R1,
  L39.h12) scored beside it as a reference (`D-check-r1.json`).

| D | head vote (registered) | labelled `choice` | R1, spec 18's choice |
|---|---|---|---|
| **present, ≤ 256 segments** | **146/155 = 94.2%** | 142/155 = 91.6% | 118/155 = 76.1% |
| **paraphrase** | **73/78 = 93.6%** | 69/78 = 88.5% | 63/78 = 80.8% |
| lexical | 73/77 = 94.8% | 73/77 = 94.8% | — |
| logs / records / prose | 42/43, 44/45, 60/67 | 40/43, 44/45, 58/67 | — |
| all present (201) | 185 = 92.0%, top-3 98.0% | up to 256 segments only | 150 = 74.6% |
| by length: ≤ 2.8K / ≤ 4.6K / ≤ 17.7K keys | 92.7 / 97.8 / 84.8% | — | 75.2 / 78.3 / 69.6% |
| present / absent AUC of the confidence | 0.73 (0.72 on the 188) | 0.86 | — |

- **Rule 2**: the vote is 2.6 points *ahead* of the labelled route overall
  and 5.1 ahead on the paraphrase half; the bars are 5 and 10 behind. **Go.**
- **Rule 3** (the length ceiling): the longest tier within 5 points of the
  shortest is the 4.6K one (the 17.7K tier reads 84.8% against 92.7%), so
  `LOCATE_MAX_KEYS` = **4,554**.
- **Rule 4** (the floors, on the questions within the ceiling): logs at
  least **41 of 43**, records at least **43 of 45**, prose at least **58 of
  67**.
- **Cost** (medians over the 188): the labelled `choice` 294 ms and 1,831
  prompt tokens, a `noul` over the unlabelled state 230 ms and 1,385 tokens;
  on logs 604 against 411 ms, 4,501 against 2,814 tokens. The vote adds the
  content-free prefill, a median 55 tokens once the state's prefix is
  shared.
- Spec 18's own reading, scored on D for reference, repeats its C result:
  76.1% against 91.6%, no-go.

## Finding

On the set that judged it, the head vote registered after phase 0 finds the
line, record or sentence a text instruction asks for on 94.2% of the
questions the labelled route can answer, against the labelled `choice`'s
91.6%, with no labels in the state — lexical questions as well as the labels
(73/77 each), paraphrases better (93.6 against 88.5%). Spec 18's rule 2 says
**go**. Its length ceiling is 4,554 keys; beyond it the vote still reads
84.8% of 17.7K-key logs, where the labels cannot go. Its confidence separates
absent questions worse than the labelled route's (AUC 0.73 against 0.86).

## Implications

- **Spec 18's phase B (#275) can go ahead, with the vote as its reading**
  in place of R1: 32 heads over L35-L63, per-segment shares at the copy
  scaffold less the content-free prefill's, each head's winner a vote. That
  is a head *set* on the seam, as ADR 0039 made for `box`; the heads are a
  calibration constant keyed to the artifact, with the procedure of spec 19
  phase 0 to recalibrate. `LOCATE_MAX_KEYS` and the floors are the ones
  above; phase B's acceptance runs on fresh sets.
- A `locate` shipped this way costs one prefill of the plain state — shared
  with any other question over it — plus about 55 tokens for the
  content-free baseline, against the labelled route's 32% more prompt (60%
  on logs) and no shared prefix.
- A `found` flag is still not warranted from the vote's agreement (AUC
  0.73).

## Limits and unknowns

- One check set, 155 comparable questions (one question is 0.65 points);
  the margin to the bar is 7.6 points overall.
- The prose family is HotpotQA with a weaker paraphrase split (spec 18);
  logs and records are synthetic.
- One artifact, the served render, consumed hq keys, fresh prefills: a
  fan-out that claims the state from an earlier request reads the same keys
  from the cache, where all but the residual window is decoded.
- The engine cost of reading 32 heads over up to 4.6K keys was not measured;
  spec 14's fused readout for 96 image heads is the nearest number (+0.14-0.4
  ms).

## Follow-ups

- Reopen #275 (spec 18 phase B) with the vote as its reading, the ceiling
  and floors above, and its acceptance on fresh sets.
