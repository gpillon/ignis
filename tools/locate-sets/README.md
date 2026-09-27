# Locate sets, the attention-tap harness and the scorer

The tools behind spec 18 phase A (`docs/specs/decide/18-locate-by-attention.md`,
GitHub #274): which heads of the served 27B point into a **text** state, which
reading of them to use, and whether that is good enough for `locate` to ship.
Kept as they were run, so that recalibrating for a new artifact is a repeat and
not a new study, the way `tools/pointing-scenes/` is for `point`.

| File | What it is |
|---|---|
| `generate.py` | Writes one set: 240 questions, 80 per family, into `<out>/manifest.json`. Deterministic per `--seed` and the sets named by `--exclude` (whose HotpotQA questions it does not reuse). |
| `logs.py` | Synthetic service logs of 60, 250 and 1,000 lines (about 1.1K, 4.4K and 17.7K state tokens): one ERROR target among 3 to 8 ERROR distractors of the same shape, the rest routine traffic. |
| `records.py` | JSON arrays of 20, 80 and 300 records (about 0.9K, 3.4K and 12.9K tokens) with 5 to 8 fields: employees found by city, tickets by title, products by name. |
| `prose.py` | HotpotQA distractor dev (CC BY-SA 4.0), one line per sentence of the ten paragraphs, an empty line between paragraphs; correct on any gold supporting sentence. Downloads the dev file into `.scratch/` (never committed): the official URL first, the Hugging Face copy (`hotpotqa/hotpot_qa`) when it does not answer. |
| `common.py` | The word rules behind the lexical / paraphrase split, checked on every generated question. |
| `score.py` | Scores the harness's dumps: R1, R2 and R3, each with and without the content-free baseline, per scaffold; `dev` cross-validates on A+B and applies rule 1, `check` judges the choice on C and applies rules 2-4. Needs NumPy. |
| `profiles.py` | Spec 19's phase 0 (`docs/specs/decide/19-a-span-read-from-attention.md`, GitHub #276): reads a dump as *profiles* -- per head, the weight, the lift over the content-free prefill and the prior alone -- and reports token-level localisation, where each head's peak sits against the target, span-aligned averages, line offsets, head votes, a conditional logit over every head, the top-k ceiling and where lexical misses land. `extract` once per set, then `report`. Needs NumPy and SciPy, and `tokenizers` for the rare-word keys. |
| `spans.py` | Spec 19's span sets (phase 1): 300 questions whose golds are **character spans** -- a value inside a log line (`logvalue`) or a record (`recvalue`), SQuAD 2.0 answers (`squad`), HotpotQA answers inside a gold sentence (`hotspan`), several or no whole lines (`logmulti`), every supporting sentence (`hotfacts`). Downloads SQuAD 2.0 dev into `.scratch/` beside HotpotQA. |
| `generation.py` | The generation route, spec 19's comparator for spans: the same render through `/v1/chat/completions`, greedy, thinking off, and the answer searched for in the state (exact, else case- and space-folded). |
| `spanscore.py` | Scores the span harness's dumps on the key grid: gold spans onto keys, one head / vote / summed profile / segment-then-key readings chosen in CV (and nested), span F1, the generation route on the same keys, where the full row's mass goes (Q4), and the forced quote's queries. `extract` per set, then `report`. |
| `labelled.py` | The labelled `choice` route (Jev's line search) through a running server's `/v1/decide`, for rule 2's comparison, with a `noul` over the unlabelled state timed beside it. Measured, never shipped. |
| `test_score.py` | The scorer on a synthetic dump whose answer is known (`python tools/locate-sets/test_score.py`). |
| `test_spans.py`, `test_generation.py`, `test_spanscore.py` | The span sets' promises (no download), the generation route's parsing and search, and the span scorer on a synthetic dump with planted heads. |
| `test_profiles.py` | Phase 0's tables on a synthetic dump with planted heads -- an initiator, a terminator, a prior, an inside head (`python tools/locate-sets/test_profiles.py`). |
| `test_sets.py` | The generators' promises — the split holds on every question, one in six absent, a seed writes the same set — and the labelled route's state (`python tools/locate-sets/test_sets.py`; no download). |

The harness is `crates/server/tests/attention_head_locate_gpu.rs`; the prompt
and segment map it measures with are the pure functions of
`crates/server/tests/support/locate.rs`, held to tables by
`crates/server/tests/locate_segments.rs`.

## The split

- **Lexical**: the question shares a rare word with its target, one that
  appears in the target and in no other segment.
- **Paraphrase**, for logs and records: the question shares no content word
  with the target at all. Words are compared by their first five characters,
  so a suffix change does not pass.
- **Paraphrase**, for prose: the question shares no *rare* word with any gold
  sentence. HotpotQA's questions name their entities, and only 12 of its 7,405
  share no content word with any gold sentence, so the stricter rule cannot
  fill 40 questions. What it keeps is the point of the split: a rare-string
  matcher cannot single the gold sentence out.
- **Absent**: one question in six (13 of 80 per family) has its target
  removed — a log line replaced by routine traffic, a record by a filler
  record, the gold sentences dropped. Absent questions have no right segment;
  top-1 counts present questions, and absent ones measure only whether
  `confidence` separates the two.

## The sets

| Set | Command | Role |
|---|---|---|
| A | `generate.py --seed 20261010 --out .scratch/locate/A` | development |
| B | `generate.py --seed 20261011 --out .scratch/locate/B --exclude .scratch/locate/A` | development |
| C | `generate.py --seed 20261012 --out .scratch/locate/C --exclude .scratch/locate/A .scratch/locate/B` | check: go/no-go, reading confirmation, length ceiling, floors |
| D | `generate.py --seed 20261013 --out .scratch/locate/D --exclude .scratch/locate/A .scratch/locate/B .scratch/locate/C` | phase B's acceptance — used once, not yet used |

A set used to choose something is spent for judging it. A and B choose; C
judges once; D is phase B's.

The prose questions depend on the HotpotQA file's row order as well as the
seed: A-D were drawn from the Hugging Face copy, sha256 `c20b638c…4e972f7c6`
(in full in `prose.py`).

Phase A ran on 2026-09-26 with A, B and C and ended in a **no-go**: rule 1
chose R1 (L39.h12) on the copy scaffold with the baseline, and on C it read
73.3% of the questions the labelled route answers at 91.3%
(`docs/findings/2026-09-26-locate-attention-no-go.md`). A, B and C are spent;
D was never run and is free for a fresh study, which is what any follow-up
would be: spec 19 keeps it for its track L check.

## Running phase A

1. **Generate** the four sets with the commands above.
2. **Dump every head** of A, B and C, with the GPU free
   (`make gpu-status`, `docs/agents/testing.md`):

   ```text
   IGNIS_LOCATE_SET=<abs path>/.scratch/locate/A IGNIS_LOCATE_OUT=<abs path>/.scratch/locate/dumps \
   cargo test -p ignis-server --features cuda,attn-tap --test attention_head_locate_gpu \
     -- --ignored --test-threads=1 --nocapture
   ```

   Absolute paths: cargo runs a test in its crate's directory. The KV is
   hq-e8-2b with the consumed keys by default (`IGNIS_LOCATE_KV=bf16` for a
   BF16 pool); `IGNIS_LOCATE_RESUME=1` continues a dump a crash cut short.
   Each set writes `<set>-hq.bin` (every head's scores, f16), `.jsonl` (one
   row per question) and `.json` (the rest, including the endpoint's answer
   alphabet and the first render of each scaffold).
3. **Choose on A+B**: `python tools/locate-sets/score.py dev --dumps <A-hq.json> <B-hq.json> --out dev.json`.
4. **The labelled route on C**, through the served artifact (`make start`,
   then `make stop`):
   `python tools/locate-sets/labelled.py --set .scratch/locate/C --alphabet <C-hq.json> --out C-labelled.json`.
5. **Judge on C**: `python tools/locate-sets/score.py check --choice dev.json --dumps <C-hq.json> --labelled C-labelled.json --out check.json`.
6. **Record**: the finding, and — on a go — `LOCATE_MAX_KEYS` and the floors
   in spec 18 before phase B's acceptance runs; on a no-go, spec 18 marked
   NOT IMPLEMENTED.

## Spec 19: the head vote (phase 0)

Spec 19 reads the same dumps again (`docs/findings/2026-09-27-a-head-vote-finds-the-line.md`):

1. **Extract** each development set once (a few minutes each; the tokenizer
   is the artifact's `tokenizer.json`, used only to find the rare-word keys
   of lexical questions):

   ```text
   python tools/locate-sets/profiles.py extract --dump <dumps>/A-hq.json --manifest <sets>/A/manifest.json      --tokenizer <tokenizer.json> --out <phase0>
   ```

2. **Report** on A+B (`--parts` picks tables; `combinations` is the slow one,
   about half an hour):

   ```text
   python tools/locate-sets/profiles.py report --sets <phase0>/A <phase0>/B --dumps <A-hq.json> <B-hq.json>      --manifests <A/manifest.json> <B/manifest.json> --out <phase0>/quick.json --parts quality,peaks,votes
   ```

3. **The vote** is `score.py`'s reading `vote`: its choice file is `score.py
   check`'s `--choice` (`{"choice": {"kind": "vote", "scaffold": "s2",
   "baseline": true, "method": {"reading": "vote", "heads": [...]}}}`), and
   `check` applies spec 18's rules 2-4 to it unchanged. The heads registered
   for set D are in spec 19, track L.

| Set | Role in spec 19 |
|---|---|
| A, B | development: every choice |
| C | replication only (spent by spec 18; read with the frozen vote after its registration) |
| D | track L's check, once: **go**, 146/155 against the labels' 142 (`docs/findings/2026-09-27-locate-by-head-vote-go.md`) |

## Spec 19: spans (phase 1)

The harness is `crates/server/tests/attention_span_locate_gpu.rs` (the
shared capture code in `support/locate_tap.rs`); the span kind text is
`support/locate.rs::SPAN_KIND`, the same words `generation.py` asks with.

| Set | Command | Role |
|---|---|---|
| E1 | `spans.py --seed 20261020 --out .scratch/locate/E1 --cache .scratch/locate --exclude .scratch/locate/A .scratch/locate/B .scratch/locate/C .scratch/locate/D` | development |
| E2 | `spans.py --seed 20261021 --out .scratch/locate/E2 --cache .scratch/locate --exclude <A-D> .scratch/locate/E1` | development |
| E3 | `spans.py --seed 20261022 --out .scratch/locate/E3 --cache .scratch/locate --exclude <A-D> .scratch/locate/E1 .scratch/locate/E2` | track S / P check, once, after its rule is registered |

SQuAD's split is classified like prose's (a rare word shared with the
answer's paragraph, against the article's neighbouring paragraphs), so its
paraphrase half is the weaker one. SQuAD 2.0 dev is sha256
`80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8`.

1. **Dump** a set (GPU; about three times spec 18's harness per question):

   ```text
   IGNIS_LOCATE_SET=<abs>/.scratch/locate/E1 IGNIS_LOCATE_OUT=<abs>/.scratch/locate/dumps \
   cargo test -p ignis-server --features cuda,attn-tap --test attention_span_locate_gpu \
     -- --ignored --test-threads=1 --nocapture
   ```

2. **The generation route** on the same set, through `make start`:
   `python tools/locate-sets/generation.py --set .scratch/locate/E1 --out E1-generation.json`.
3. **Extract, then report**:

   ```text
   python tools/locate-sets/spanscore.py extract --dump <dumps>/E1-hq.json --manifest <E1>/manifest.json --out <spans>
   python tools/locate-sets/spanscore.py report --sets <spans>/E1 <spans>/E2 --dumps <E1-hq.json> <E2-hq.json> \
     --manifests <E1/manifest.json> <E2/manifest.json> --generation <E1-generation.json> <E2-generation.json> --out spans.json
   ```

## Recalibrating for a new artifact

Phase A again, with fresh seeds and the same rules:

1. **Sets.** Generate new development, check and acceptance sets like A-D
   with seeds nobody has used, each excluding the ones before it.
2. **Dumps.** Run the harness over the development and check sets on the new
   artifact, with the served render and the consumed hq keys (the defaults).
   A capture that fails its self-check is not a measurement: the test fails
   after writing the dump.
3. **Choose** with `score.py dev` on the development dumps. The reading,
   scaffold and baseline may come out differently for another model; the
   rules decide, not the previous answer.
4. **Judge** with `labelled.py` and `score.py check` on the check set, and
   write the new `LOCATE_MAX_KEYS` and floors down before the acceptance set
   runs.
5. **Record** the heads (or set) and the choice as a new row of the
   calibration table keyed to the artifact's content hash (spec 18,
   "Calibration is a constant keyed to the artifact"), and the result as a
   finding.
