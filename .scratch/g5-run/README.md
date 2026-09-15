# G5 gate records (GitHub #156)

The phase 5 gate: ignis with DFlash2 against the reference with DFlash2-7,
committed tok/s at 24K / 98K / 196K pooled over two launches per engine,
greedy spec-on/spec-off equivalence, canary self-consistency, and one
informational G4 trace replay per engine
(`.scratch/runtime/specs/05-speculative-decoding.md` §Gate G5).

The verdict itself lives in `.scratch/REVIEW-2026-09-05.md` §6 Phase 5.

## The session

One GPU-exclusive session (ADR 0006), driven end to end by
[`run-g5.ps1`](run-g5.ps1), session id **`g5-20260914T140231Z`** (`session-id.txt`), against
`F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer` on a free RTX 5090.

**Tree under measurement.** Branch `issue-156-g5-gate-run` at `aae467e`:
main `0801216`, the spec branch `issue-65-g4-gate-run-2` merged in
(`bea1238`), capture-mode equivalence (`620b245`, `9d84eef`) and the #159
depth-cell fixes (`aae467e`). GPU profile on that tree: `gpu-profile.log`,
exit 0, first stage of this session (16:03–16:18).

**Reference at the pinned commit.** `ninfer-serve.exe` was built 2026-09-02;
the pin `a00648cb` is dated 2026-09-03. `build-ninja.ps1` at the pin (clean
tree) reported `ninja: no work to do`, so the binary matches the pinned
sources.

**Both engines at hq-e8-2b, matched capacity**, `--max-context 262144`,
`--prefill-chunk 1024`, CUDA graphs and prefix reuse on:

```
ninfer-serve.exe <artifact> --model-id qwen3.8-27b-nvfp4full-v2 --host 127.0.0.1 --port 8080 `
  --kv-dtype hq-e8-2b --max-context 262144 --kv-capacity 465984 --max-concurrency 8 `
  --prefill-chunk 1024 --spec dflash2 --draft-tokens 7
ignis-server.exe --artifact <artifact> --bind 127.0.0.1:8000 --kv-format hq-e8-2b `
  --max-context 262144 --prefill-chunk 1024 --request-timeout 1800 --spec dflash2 --draft-tokens 7
```

Each launch's resolved capacity is in `session.log` (`kv:` lines).

**Order.** ignis with speculation off (the equivalence capture only, before
any measured launch); reference 1, ignis 1, reference 2, ignis 2 (the `g5`
depth cells each; the canary and the spec-on capture on ignis 2; the G4
trace replayed last on each engine's second launch); then the equivalence
comparison and `g5-gate`.

## Earlier attempts (`attempts/`)

- **1** (`g5-20260914T115716Z`): GPU profile green (exit 0, 877 s), then the
  driver died writing `session.log`, locked by the log watcher. No engine
  launched.
- **2** (`g5-20260914T121258Z`): spec-off capture fine; reference launch 1's
  `g5` record could not decide the gate — 98K/196K could not be cut from the
  65,536-id corpus, the 24K sample stopped at EOS after 10 of 512 tokens and
  was accepted, and the reader counted 5 SSE chunks where the engine
  generated 10. Stopped by hand; filed as **#159**, fixed in `aae467e`
  (tiled corpus with a distinct offset per depth, a decode instruction on
  every prompt and a void below the committed budget, `usage.completion_tokens`
  as the token count).

The equivalence harness itself was changed before any run (`620b245`): two
27B engines do not fit on the card at once, so each side is captured against
its own launch and the two files are compared.

## Records

Session id `g5-20260914T140231Z` on every record.

| file | engine | launch | 24K | 98K | 196K |
|---|---|---:|---:|---:|---:|
| `ninfer-launch1-g5.json` | reference | 1 | 129.4 | 92.9 | 106.4 |
| `ignis-launch1-g5.json` | ignis | 1 | 84.3 | 67.9 | 117.2 |
| `ninfer-launch2-g5.json` | reference | 2 | 130.1 | 95.6 | 115.7 |
| `ignis-launch2-g5.json` | ignis | 2 | 81.1 | 72.0 | 118.4 |

Committed tok/s, C=1, cold, 512 tokens each (every sample `finish=length` /
`output_limit` at 512). The reference's own request log agrees with the
bench to 0.2% (launch 1: decode 129.6 / 93.0 / 106.5).

**Tokens per verify round** (512 / rounds; ignis from `ignis.request.done`
`spec.rounds`, the reference from its request log):

| launch | 24K | 98K | 196K |
|---|---:|---:|---:|
| reference 1 | 4.09 | 3.75 | 6.08 |
| ignis 1 | 2.59 (314/1371 accepted) | 2.88 (334/1241) | 6.24 (430/568) |
| ignis 2 | 2.59 (314/1371, identical) | 2.88 (334/1241) | 6.24 (430/568) |

Round cost is level (24K: ignis ~30 ms/round, reference ~32 ms). The gap at
24K/98K was first read as drafter acceptance; corrected 2026-09-15 (#160):
the two engines generated different text there (see "Reading the depth
FAILs").

**G4 trace under speculation** (informational, no threshold):

| file | main | sub | needles |
|---|---:|---:|---|
| `ninfer-launch2-g4trace.json` | **62.9** tok/s (record: 20.3) | **57.1** tok/s (record: 12.9) | 64K, 128K RETRIEVED |
| `ignis-launch2-g4trace.json` | 42.3 tok/s | 61.1 tok/s | 64K, 128K RETRIEVED |

One replay per engine, speculation on both sides, 11/11 requests.
**The reference's recorded figures are chunk-counted**: trace replay
requests do not ask for the usage chunk (`trace.rs`, `include_usage:
false`), so #159's fix does not reach them and each multi-token DFlash2
chunk counts once (req-001: 5,165 recorded against the server's
`gen=16000`). The bold figures re-derive the reference with the server log's
own `gen=` per request over the record's decode times: main 16,000 tokens /
254.2 s, sub 49,374 / 865.0 s. ignis streams one token per chunk, so its
record stands. Informational only: ignis main 0.67, sub 1.07 of the
reference. The replay path's gap is added to #159.

**Other cells.**

| file | what it is |
|---|---|
| `ignis-launch2-canary.json` | Canary self-consistency, speculation on: 4/4 `sane=true deterministic=true`, **PASS**. |
| `equivalence-spec-off.json`, `equivalence-spec-on.json` | The two captures: ignis spec-off launch, ignis launch 2. |
| `equivalence.json` | `g5-equivalence` over them: rust-hello and explain-reverse identical; **rust-sort diverges @44, math-greedy @33** — exit 1. |
| `g5-verdict.json` | `g5-gate` pooled over the four launches. |

**Session id on every record — not quite.** The `g5`, `g4` and capture
records and the verdict carry `g5-20260914T140231Z`; `ignis-launch2-canary.json`
(`canary` takes no `--session`, as in G4 run 2) and `equivalence.json` (a
list of per-canary comparisons; its two input captures carry the session)
do not.
| `*-server.log`, `*.log` | Each launch's engine log and each bench step's own output. |

## Verdict

**G5: FAIL.**

| cell | result | verdict |
|---|---|---|
| committed tok/s @24K | 82.7 vs 129.8, ratio **0.637** | FAIL → #160 (closed), #161, #162, #173 |
| committed tok/s @98K | 69.9 vs 94.2, ratio **0.742** | FAIL → #160 (closed), #161, #162, #173 |
| committed tok/s @196K | 117.8 vs 110.8, ratio **1.063** | PASS |
| greedy equivalence | 2/4 canaries diverge (@44, @33) | FAIL → #161 |
| canary self-consistency | 4/4 | PASS |
| G4 trace under speculation | main 42.3 vs 62.9, sub 61.1 vs 57.1 (reference re-derived from its log) | informational |

**Reading the depth FAILs.** Both ignis launches agree to the round (198 /
178 / 82 rounds, identical accepted counts), so the miss is not launch noise.
A verify round costs the same on both engines; at 24K and 98K ignis commits
roughly half the reference's tokens per round on the same prompts, at 196K
the two match. This was recorded as a drafter-state defect (#160). The
2026-09-15 diagnosis says otherwise: at 24K/98K ignis hq-e8-2b picks a
different greedy first token, so the engines generate different text and
acceptance follows the content. ignis bf16 generates the reference's text
token for token and drafts better on it (4.30 vs 4.09), and over 20
token-identical real prompts ignis pools 0.965 of the reference's
tok/round. The FAIL stands; the misses are the hq early-token divergence
(#161, #173) and the cell accepting mismatched text (#162).

**Reading the equivalence FAIL.** Both divergences are single word choices
that read like near-ties, the class the owner accepted on 2026-09-14 (#153,
#155) — but this harness records no logits, so the cell cannot classify
them; #161 carries the check.

**`spread_flagged` on every depth is the instrument's, not the engines'.** A
C=1 cell has one sample per launch, so within-launch spread is 0.0% by
construction and any across-launch difference trips ADR 0021's flag. The
across-launch spreads themselves are small (ignis 0.5–2.9%, reference
0.3–4.2%).

**Nothing waived.** Every miss is filed: #159 (instrument, fixed here),
#160 (closed as not the drafter), #161; follow-ups #162, #173.
