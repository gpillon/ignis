# Locate through `/v1/decide` holds D's floors on a fresh set

- Kind: experiment
- Status: current
- Observed: 2026-09-27
- Last verified: 2026-09-27
- Scope: `/v1/decide` `locate` (spec 18 phase B, GitHub #275) — the served head vote over a text state, on the served artifact, set F
- Related: [GitHub #275](https://github.com/gpillon/ignis/issues/275); [spec 18](../specs/decide/18-locate-by-attention.md) (its acceptance 8, registered before set F existed); [ADR 0041](../adr/0041-the-readout-reads-a-text-span.md); [the go on set D](2026-09-27-locate-by-head-vote-go.md); [how the vote was found](2026-09-27-a-head-vote-finds-the-line.md); [`tools/locate-sets/served.py`](../../tools/locate-sets/served.py)
- Superseded by: none

## Question

Spec 19's head vote passed spec 18's go rule on set D **from the harness's
dumps**: fresh prefills, every head read by the test-only tap, the reading
in `score.py`. Phase B ships it — the leaf reads 32 heads' rows over a text
span, the host votes, the baseline prefill claims the question's retained
state. Does the served endpoint hold D's per-family floors on a set nothing
has read, as spec 18's acceptance 8 requires (registered, with the set's
seed, before the set was generated: spec 18's banner, commit `337a67f`)?

## Evidence

- **Set F**: `generate.py --seed 20261014 --exclude A B C D`, 240 questions
  (80 logs, 80 JSON record arrays, 80 HotpotQA prose; 201 present).
- **The run**: `make start` on branch `locate-spans-276` (served artifact,
  hq-e8-2b with the residual window, prefill chunk 1,024, DFlash2 loaded),
  then `served.py ask`: one `/v1/decide` request per question, one `locate`
  each, no `within`. `served.py judge` applies the rule. Raw output in
  `.scratch/locate/` (`F-served.json`, `F-acceptance.json`).
- **Beside it**: `labelled.py` on F through the same server (the labelled
  `choice`, and a `noul` over the unlabelled state), `F-labelled.json`.

| F, present questions a `locate` serves | top-1 | D's floor | |
|---|---|---|---|
| logs | 41/43 = 95.3% | 41/43 = 95.3% | pass (at the floor) |
| records | 46/46 = 100% | 43/45 = 95.6% | pass |
| prose | 59/67 = 88.1% | 58/67 = 86.6% | pass |
| **all** | **146/156 = 93.6%** | | **PASS** |

- **Splits**: lexical 73/76, paraphrase 73/80 (prose paraphrase 28/34, the
  weaker split by construction).
- **By length** (keys of state): 60-line logs 21/21, 250-line logs 20/22,
  20-record arrays 24/24, 80-record arrays 22/22. Refused with
  `locate_too_long` before any prefill: the 26 1,000-line logs, the 26
  300-record arrays and one 250-line log past 4,554 keys (45 present, 8
  absent) — the endpoint does not answer a length it was not measured on.
- **Top-3** 155/156; **confidence** (the winner's share of the votes) median
  0.625 on hits, 0.375 on misses and on absent questions; present/absent AUC
  **0.79** (D's dumps: 0.73). Errors: none.
- **Against the labelled route** on the same 156 questions: 146 against 143.
- **Cost** (medians over the 187 served questions the labelled route could
  label): a `locate` 259 ms — the question's prefill and its content-free
  twin's, which claims the question's retained state — against the labelled
  `choice`'s 274 ms and a `noul` over the unlabelled state's 222 ms.
  `usage.input_tokens` counts both of a `locate`'s prompts in full (2,674
  median, against the labelled prompt's 1,828), though the twin prefills only
  its tail after the claimed state.

## Finding

Served through `/v1/decide`, the head vote finds the line, record or
sentence an instruction names on 93.6% of a fresh set's present questions
within its measured length, every family at or above the floor rate set D
fixed — logs exactly at it, records and prose above — so spec 18's
pre-registered acceptance 8 passes. It reads three more of those questions
than the labelled `choice` does, without writing a label into the state, in
less wall time than the labelled `choice` takes; and it refuses, rather than
guesses, every target longer than the ceiling it was measured to.

## Implications

- `locate` ships as spec 18 phase B describes it, with the vote as its
  reading (ADR 0041): #275's acceptance 6 is met.
- The labelled `choice` recipe (`docs/user/README.md`, "Finding a line or an
  item") is no longer the only way to locate: it remains the route past
  `LOCATE_MAX_KEYS` and on an artifact nobody calibrated.
- A `found` flag is still not warranted: the confidence separates absent
  questions at an AUC of 0.79, short of the labelled route's 0.86 on D.

## Limits and unknowns

- One fresh set; logs sit exactly on their floor (one more miss would have
  failed the family). 156 questions: one question is 0.64 points.
- One question per request: a fan-out's followers claim the state from the
  first question's retained prefix, which is the path the baseline already
  takes here, but a fan-out of many `locate`s over one long state was not
  timed.
- The prose family is HotpotQA with the weaker paraphrase split; logs and
  records are synthetic.
- Wall times are medians on an otherwise idle server with DFlash2 loaded;
  the labelled route ran after the `locate` run, on the same load.
