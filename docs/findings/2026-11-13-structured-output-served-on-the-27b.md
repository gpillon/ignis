# Structured output served on the 27B: plain traffic decodes as before it, the requests beside a constrained one keep their rate, and a described schema's answer decodes at 98% of the same prompt free

- Kind: experiment
- Status: current
- Observed: 2026-11-13
- Last verified: 2026-11-13
- Scope: the server's structured output on the 27B with DFlash2 on (the integration branch of #287 at e4176765)
  against c8358731, on the RTX 5090: plain agentic traffic, the requests beside a constrained one, and a
  constrained request's own decode
- Topic: structured-output
- Related: https://github.com/gpillon/ignis/issues/595,
  https://github.com/gpillon/ignis/issues/287,
  [ADR 0089](../adr/0089-a-grammar-checks-each-sampled-token-and-repairs-a-miss-and-the-model-is-told-the-schema.md),
  [the grammar measured on the 27B](2026-11-12-structured-output-grammar-measured-on-the-27b.md)
- Superseded by: none
- Summary: RTX 5090, DFlash2: plain traffic decodes at 100.0% of c8358731 at one lane, requests beside a constrained one at 99.4% of beside a free one; a described schema's answer at 98% of its prompt free (0.41 misses/100 tokens), 76% off.

## Question

Served, on the 27B with DFlash2 on, after #287's tickets landed: does traffic
with no structured-output request decode as fast as on the build the work
started from (c8358731)? Do requests decoding beside a constrained one keep
their rate? What does a constrained request cost itself, with the schema
described to the model and with the description off? And are greedy answers
the same with DFlash2 on and off? The #287 spike measured the grammar and
estimated the decode cost from miss positions; this measures it served.

## Evidence

**Setup** (2026-11-13, 04:04-04:33Z, 29 minutes of card): RTX 5090 (Windows
11, WDDM, i9-10900K), `make start LOOP_GUARD=off IDLE_UNLOAD=0`: the 27B
NVFP4 artifact (`qwen3_8_27b_nvfp4full-v2`), KV `hq-e8-2b`, DFlash2 with 7
draft tokens, 8 decode lanes. The integration branch at e4176765
(`ignis-server.exe` sha256 `a1d44290...8827`) and main at c8358731 (sha256
`ba10aec7...d56e`, built from 3a3135f8; the two differ in docs only). Every
record, script and server log is in `.scratch/287/595/`: `protocol.md` (the
A/B's protocol, fixed before main's first launch), `run-launch.sh`,
`pool.py`, `client595.py` (the neighbours' stream and the own-cost client),
`legsum.py`, `leg4-rounds.py`, one directory per launch.

**Carried forward from the spike** (the
[grammar finding](2026-11-12-structured-output-grammar-measured-on-the-27b.md)
holds the method and the detail): a check takes 0.2 µs (1.0 µs p99), a
repair's permitted set 34 µs on a state's first visit (0.71 ms p99) and under
1 µs from the cache, a schema compiles in at most 2.4 ms. The check-first
emulation on the 27B (greedy, thinking off, the 19-case corpus):

| | schema absent | schema in the prompt |
|---|---|---|
| misses per 100 tokens | 4.78 | 0.51 |
| misses after the first token, per 100 | 4.16 | 0.04 |
| conforming | 17/18 (one runaway cut at 512) | 17/17 |
| estimated decode against free, 4.41 / 5.43 tokens a round (mean) | 0.925 / 0.904 | 0.991 / 0.989 |

**Plain traffic against c8358731.** `ignis-bench replay` of
`bench/traces/g4-load-trace.jsonl` (11 recorded agent chat bodies, 1 main
and 10 subs, 21-83 KB with 4 tools each, arriving 2 s apart) `--as-recorded`,
greedy: cell *several* at `--conc 8` on a cold cache, then cell *one* at
`--conc 1`; one bench binary for every launch, one measurement per launch per
cell. Launches alternated, branch 1, main 3, branch 2, main 4, and pooled per
build by `g4-gate`'s rule (decoded tokens over decode time, the two launches'
requests together):

| cell | class | main (launch 3, 4) | main pooled | branch (launch 1, 2) | branch pooled | branch / main |
|---|---|---:|---:|---:|---:|---:|
| several | main | 152.49, 144.76 | 148.75 | 152.56, 150.98 | 151.72 | 1.0199 |
| several | sub | 96.35, 96.95 | 96.65 | 94.86, 96.70 | 95.77 | 0.9909 |
| several | all | 124.65, 120.34 | 122.53 | 121.89, 124.25 | 123.11 | 1.0047 |
| one | main | 181.08, 181.56 | 181.32 | 181.30, 181.50 | 181.40 | 1.0004 |
| one | sub | 228.49, 229.18 | 228.83 | 228.81, 228.84 | 228.83 | 1.0000 |
| one | all | 203.91, 204.49 | 204.20 | 204.17, 204.31 | 204.24 | 1.0002 |

- **The one cell did the same work on every launch**: the same tokens per
  request, and the server ran 4,096 rounds and accepted 10,586 drafts on each.
  Its spread across launches is at most 0.15%.
- **The several cell's subs** produced the same 9,953 tokens on all four
  launches; the main request did not (14,100 to 16,000 tokens, main's two
  launches as far apart as the builds), because greedy output depends on the
  lane mix each round sees. A ratio over different tokens is not a speed
  comparison; the main class and the aggregate are reported, not decisive.
- **Drift between sessions.** Main's launches 1 and 2 (01:43Z, an earlier
  session, the protocol's baseline) decode the one cell at 196.15 tok/s, 3.9%
  below main's launches 3 and 4 (204.20) on identical tokens; pooled against
  them the branch would read 1.041. The alternated pair removes that drift.
- Each launch logged its 30 `ignis.request.done` lines (8 canaries, 11 + 11
  replayed); the card held P0 at a median SM clock of 2,857 MHz, VRAM peak
  30,696 of 32,607 MiB.

**DFlash2 on against off.** `ignis-bench g5-equivalence capture` on a launch
with `SPEC=off` and one with DFlash2 on (the branch's bench, 4 canaries, 64
tokens, greedy), then compared. On the branch `math-greedy` is equal and
`rust-hello`, `rust-sort` and `explain-reverse` diverge at positions 3, 10
and 33, each between two wordings of the same answer ("the program's entry
point" against "the entry point of a Rust program"; `[1, 2, 3]` inline
against a code block; "the last element becomes the first" against "the last
becomes the first"). The same client against c8358731 captured the same
tokens as the branch on both sides, token for token, so the same three
divergences. `ignis-bench canary` on the branch with DFlash2 on: 4 of 4 sane
and deterministic, self-consistency PASS, every output identical to main's.

**The requests beside a constrained one.** The several cell's replay at
`--conc 7`, and beside it a stream of one request at a time cycling through
the corpus from just before the replay starts until it ends: constrained
(`json_schema` strict, the schema described), or the same prompt free (the
described block sent as a trailing system message, no `response_format`; its
`prompt_tokens` equal the constrained request's on every case). The stream is
the eighth lane, so neither side meets `engine_full`. Launches alternated
constrained 1, free 1, constrained 2, free 2. The neighbours' decode rate, the
replay's own figure, pooled per condition as above:

| class | constrained beside (1, 2) | pooled | free beside (1, 2) | pooled | constrained / free | no stream, `--conc 8` (the A/B's branch) |
|---|---:|---:|---:|---:|---:|---:|
| main | 126.57, 125.37 | 125.97 | 135.53, 123.89 | 129.46 | 0.973 | 151.72 |
| sub | 83.93, 85.83 | 84.91 | 82.52, 85.23 | 83.86 | 1.012 | 95.77 |
| all | 104.90, 104.64 | 104.77 | 106.81, 103.98 | 105.38 | 0.994 | 123.11 |

The constrained stream ran 208 requests on each launch (27,925 and 27,884
tokens, every answer parsing, 107 misses each, 0.38 per 100 tokens, 287 and
288 tok/s); the free stream 168 and 187 (281 and 287 tok/s). The neighbours'
tokens differ between launches (24,185 to 24,893), as in the A/B's several
cell.

**A constrained request's own cost.** One request at a time, greedy,
`enable_thinking: false`, `max_tokens` 512, streamed: the 19 corpus cases of
`crates/grammar/tests/corpus/cases.json` (2 `json_object`, 17 strict
`json_schema`), three passes in rotated order, after two discarded warm-up
requests. Four variants of a case: *free*, the prompt alone; *told*, the
prompt and the described block as a trailing system message, unconstrained
(the same prompt tokens as *described* on 17 of 17 cases); *described*,
`response_format` with the schema described (the default); *off*,
`response_format` with `ignis_describe_schema: false`. A `json_object` case
has free and described only. The decode rate is replay's (completion tokens
minus one over first to last streamed token); the rounds, tokens and misses
are the server's `done` lines (client and server agree on every token count),
the round time is `done` less `ttft`:

| variant | requests | tokens | decode tok/s | tokens a round | ms a round | misses per 100 tokens | answers that parse | cut at 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| free | 57 | 14,910 | 242.8 | 3.91 | 16.07 | | 3 | 15 |
| told | 51 | 7,730 | 423.8 | 6.83 | 16.11 | | 30 | 0 |
| described | 57 | 7,598 | 411.7 | 6.68 | 16.11 | 0.41 | 57 | 0 |
| off | 51 | 8,333 | 319.7 | 5.19 | 16.12 | 4.55 | 48 | 3 |

- **Described against the same prompt free** (*told*), the 51 `json_schema`
  pairs: 0.980 of its decode rate (415.3 against 423.9 tok/s, the server's
  figures), per-case median 0.989 (0.893 to 1.008). On the 27 pairs whose
  answers are the same text: 1.002, and 0.9999 of the round time.
- **Off**: 0.759 of *told* and 0.776 of *described*; its 3 cuts are the
  recursive `ui_component` schema running to `max_tokens` on every pass, as in
  the spike.
- **Free** answers in prose or fenced Markdown (3 of 57 parse as JSON, 15 run
  to the cap) and draft worse, 3.91 tokens a round: the constrained answers
  decode 1.70 (described) and 1.32 (off) times as fast. A free prompt is not a
  speed baseline for a JSON answer.
- **Where the described misses fall**: 26 of the 31 are on the 20 requests
  whose *told* or *free* twin opens with a code fence, one on each except
  `task_list`'s three passes, three on each; the other 5 on 5 requests whose
  twin opens with `{`.
- **Repairs**: the server's `ignis_structured_repair_seconds` after the
  own-cost launch: 418 repairs (both warm-ups included), 0.27 ms on average,
  269 at most 0.25 ms, 410 at most 1 ms, all at most 2.5 ms.
  `ignis_structured_misses_total` and `_repairs_total` agree with the `done`
  lines.

## Finding

- **Observed: plain traffic decodes as on c8358731.** Pooled over the
  alternated launches the branch is at 1.0002 of main at one lane (1.0004
  main, 1.0000 subs, same tokens and same rounds), and at 0.9909 to 1.0199 at
  eight: at least 99% in every cell, per class and in all.
- **Observed: greedy answers with DFlash2 on differ from those with it off on
  3 of 4 canaries, on c8358731 and on the branch alike, token for token**: the
  branch changes neither side, and the check as the ticket words it (on equals
  off) does not hold on c8358731 either. The canary suite passes.
- **Observed: requests beside a constrained one keep their rate**: 0.994 of the
  same replay beside the same prompts free, both constrained launches between
  the two free ones, and each class's ratio (0.973 main, 1.012 subs) inside its
  own launches' spread (9% and 3%). Against the A/B's replay with no stream
  (at `--conc 8`) both run about 15% slower: the eighth lane's work,
  constrained or free alike.
- **Observed: a round costs the same with a grammar and without**, 16.1 ms at
  one lane in every variant, so a constrained answer's speed is its tokens a
  round, which misses set: a miss ends the verified run. With the schema
  described, 0.41 misses per 100 tokens and 98% of the same prompt free,
  100.2% where the answer is the same text; with the description off, 4.55
  per 100 and 76%.
- **Inferred: most described misses are the first token**: the 20 requests
  whose unconstrained twin opens with a code fence each repair that fence on
  the prefill's own draw, which costs no drafts; that leaves 11 later misses,
  0.14 per 100 tokens, against the spike's 0.04 (the spike appended the schema
  to the prompt; the server describes it in its own block). The rest of the 2%
  is the answers that differ: the fence's tokens, which draft well, and those
  later misses.
- **Inferred: off costs more than the spike's estimate** (24% against 8-18%)
  because the estimate assumed 4.41 to 5.43 tokens a round, and a JSON answer
  drafts at 6.8 at one lane: each cut run loses more.

## Implications

- Nothing in the plain or verify round needs fixing: the plain-traffic
  criterion holds, and a constrained request costs its neighbours nothing
  measurable.
- Describing the schema by default is confirmed served: off is about a quarter
  slower and runs the recursive schema away.
- DFlash2's greedy equivalence on the canaries diverges at near-ties on
  c8358731 already; a gate that requires on to equal off fails on main as on
  the branch. A comparison of the branch's captures with main's, as here, is
  what shows the branch changes no greedy answer.
- A served comparison of a constrained answer's speed takes the same prompt
  unconstrained (the described block sent as a system message) as its
  baseline, not the bare prompt.

## Limits and unknowns

- One card, one model, greedy only. The own-cost leg and the stream ran with
  thinking off; temperature sampling, thinking on, tools beside a schema and
  the Responses API were not measured served.
- Two launches per build or condition. The several cell's and the
  neighbours' greedy tokens vary with the lane mix, so their classes compare
  different tokens; the neighbours' main class spreads 9% across the free
  launches, and 0.973 there is inside it.
- The neighbours ran at `--conc 7` beside the stream, the A/B at `--conc 8`
  with none; the no-stream column is a reference, not the comparison.
- The first-token attribution of the described misses is inferred from the
  unconstrained twin, not from the miss's position, which the `done` line
  does not carry.
- The client's decode window has a round's granularity (one streamed chunk a
  round); the server's figures agree within 0.8% per variant.

## Follow-ups

- A loop inside a constrained answer: #584 (`ui_component` with the
  description off runs to `max_tokens`).
