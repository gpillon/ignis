# Held tools served on the 27B: every held call valid, plain traffic decodes as on main, and the operator's switch changes no call of an agentic replay, so its default stays `as-sent`

- Kind: experiment
- Status: current
- Observed: 2026-11-15
- Last verified: 2026-11-15
- Scope: held tools on the 27B with DFlash2 on (the integration branch of #582 at 47b7e868) against main at
  41f46165, on the RTX 5090: a corpus of held-tool agent requests beside their free twins, plain agentic
  traffic, the operator's switch on the same traffic, and `parallel_tool_calls: false` on the 27B and Flash-Next
- Topic: structured-output
- Related: https://github.com/gpillon/ignis/issues/617,
  https://github.com/gpillon/ignis/issues/582,
  [ADR 0094](../adr/0094-a-held-tools-call-is-checked-as-the-model-writes-it-in-its-own-order-and-a-value-ends-only-where-the-call-continues.md),
  [structured output served on the 27B](2026-11-13-structured-output-served-on-the-27b.md)
- Superseded by: none
- Summary: RTX 5090, DFlash2: 50 of 50 held calls valid (1 miss, repaired), plain traffic at main's cost a round (1.0003 at one lane); `held` changed none of an agentic replay's 11 deliveries at +0.1% a round: default stays `as-sent`.

## Question

Served, on the 27B with DFlash2 on, after #582's tickets landed: does every call to a held tool come back
valid under its schema, across the kinds of schema and request an agent sends? How often is a miss repaired,
and what do the repairs write? Does traffic with no held tool decode as fast as on main? What does holding
cost the requests that use it? Should the operator's switch, `model.tool_arguments`, default to `held`? And
does `parallel_tool_calls: false` give one call per turn on the served 27B and the served Flash-Next?

## Evidence

**Setup** (2026-11-15, 06:25-07:12Z): RTX 5090 (Windows 11, WDDM, driver 610.88), the 27B NVFP4 artifact
(`qwen3_8_27b_nvfp4full-v2`), KV `hq-e8-2b`, DFlash2 with 7 draft tokens, 8 decode lanes; Flash-Next
`qwen3_8_flash_next_trellis_a25-v2`. The integration branch at 47b7e868 (`ignis-server.exe` sha256
`7285f9a9...7f95`) and main at 41f46165 (`0cbde3e9...6ad4`, run from a checkout at 1f0fce2e, which changes
the glossary only). Both release builds were made with `UI=0`, every launch was `make start UI=0
LOOP_GUARD=off IDLE_UNLOAD=0` (metrics on, the default), and the host's user configuration was the same file
for every launch (copied into each launch's record). One `ignis-bench` binary served every leg. Records,
scripts and server logs: `.scratch/582/617/` (`ab/` for the A/B, `switch-*` for the switch leg, the test logs
beside them).

**The corpus** (`held_tools_gpu`, `--test-threads=1`, 97 s): 30 requests on chat completions and 20 on the
Responses API, each served held and beside its free twin (the same request with the tool not held, so the
same prompt where the template renders `strict` the same), greedy. It covers strict schemas with enums,
integers, nested objects, nullable and many parameters, a long string value (a 470-token script), parallel
calls, `tool_choice` `"required"` and named with thinking off and on, `response_format` beside tools, the
Responses default (tools sent without `strict`, normalized), and 27 cases built to make the model miss: a value
outside an enum (4) or a bound (4), a word for a number (4), a value outside a pattern (2) or a format (2), a
required parameter the prompt leaves out (6), a parameter the tool does not declare (5).

| | requests | calls valid | misses | repaired |
|---|---:|---:|---:|---:|
| held | 50 | 50 of 50 | 1 | 1 |
| free twins | 50 | 49 of 50 | — | — |

- The cases built to miss drew no miss of any kind: the 27B wrote a value inside the enum, the bound, the
  pattern and the format, and a value for every required parameter, held or free. No early close was
  repaired because none was attempted (0 of 0).
- The one miss is the Responses default's nested object: `copy_dir`'s `options` declares two optional members,
  which normalization makes required (nullable). The free twin wrote `{"recursive":true}`; the held call tried
  to close the object there, and the repair wrote `"depth":1`, a value the prompt never gave. The call is
  valid under the normalized schema.
- The free twin that failed asked for ten renames; it called `read_file`, a tool the request did not declare,
  where the held call made the ten edits.
- The call dialect case (the model asked to write a file explaining its own tool-call format, with an example
  call inside it) is served valid, and shows the residue ADR 0094 records. The held call wrote `content`
  first; the example's own `<parameter=mode>` line opened a parameter not yet written, so `content` ended at
  the example's value and `mode` took the example's; the example's `</function>` was not followed by
  `</tool_call>`, so it closed nothing, and the example's tail, with the model's own last lines, became the
  `path` value. The free twin, from the same prompt, described a JSON format instead and was unaffected.
- The cost on held traffic, the same requests held and free (they do different work past the first point
  where they part, so a reading, not a gate): decode rate held / free 1.025 on chat completions, 0.990 on
  the Responses API, 1.014 over all 100 requests; 0.189 and 0.190 rounds a token, 60.3% and 60.4% of drafts
  accepted.

**Plain traffic against main.** `ignis-bench replay` of `bench/traces/g4-load-trace.jsonl` (11 recorded
agent chat bodies, 1 main and 10 subs, 4 tools each, none strict), `--as-recorded`, greedy: cell *several*
at `--conc 8` on a cold cache, then cell *one* at `--conc 1`; launches alternated main 1, branch 1, main 2,
branch 2 (06:36-06:48Z), the #595 protocol. `pool.py`'s verdict, verbatim:

| cell | class | main | branch | ratio | |
|---|---|---:|---:|---:|---|
| several | main | 162.80 | 151.02 | 0.9276 | FAIL |
| several | sub | 100.53 | 100.20 | 0.9968 | PASS |
| several | all | 129.74 | 122.59 | 0.9449 | FAIL |
| one | main | 187.79 | 187.81 | 1.0001 | PASS |
| one | sub | 233.55 | 233.66 | 1.0005 | PASS |
| one | all | 209.99 | 210.05 | 1.0003 | PASS |

with "token counts differ: several req-001: [14126, 14325, 7599, 16000]". The several cell's main class is
one request, and its text differs on every launch, main's two included (155.5 and 170.8 tok/s at 14,126 and
14,325 tokens on main). The server's own record of each request:

| launch | several/main tokens | tokens a round | ms a round | several/sub ms a round | one/main ms a round |
|---|---:|---:|---:|---:|---:|
| main 1 | 14,126 | 3.355 | 22.02 | 43.91 | 17.444 |
| main 2 | 14,325 | 3.759 | 22.48 | 43.96 | 17.449 |
| branch 1 | 7,599 | 3.579 | 26.15 | 43.94 | 17.445 |
| branch 2 | 16,000 | 3.289 | 21.45 | 44.21 | 17.444 |

The subs did the same work on every launch (9,953 tokens in 2,573 rounds), as did the whole one cell.

**The operator's switch on the same traffic** (06:53-07:12Z): the same trace and cells, six launches
alternated `as-sent` 1, `held` 1, `strict` 1, `as-sent` 2, `held` 2, `strict` 2, the switch on the command
line; `strict` is the switch at its default with the trace's tools rewritten `strict: true`, the cost on held
traffic by the client's own choice. Each mode's first launch kept every delivery of a third, one-at-a-time
pass, and `/metrics` was read around each cell.

| | several/sub | one/main | one/sub | one/all | one cell, ms a round |
|---|---:|---:|---:|---:|---:|
| `as-sent` (tok/s) | 99.66 | 188.03 | 233.87 | 210.26 | 17.17, 17.17 |
| `held` / `as-sent` | 1.0057 | 1.0003 | 0.9982 | 0.9993 | 17.19, 17.20 |
| `strict` | different prompt | | | | 17.17, 17.18 |

- Under `held` all 11 requests were held (`ignis_structured_requests_total{type="tool"}` 11 a cell), every
  call was valid (`ignis_tool_calls_total{outcome="valid"}`, no other outcome), no miss was counted, and all 11
  deliveries were the text and calls `as-sent` delivered. The one cell's tokens were the same in both modes
  (14,671), in 4,095 rounds against 4,096.
- Under `as-sent` too every call of the trace was valid: there was nothing for `held` to repair.
- `strict` renders every tool `strict: true` in the prompt, so its deliveries part from `as-sent`'s before any
  miss (1 of 11 the same) and its rates are of other work; a round costs the same.

**One call per turn.** A three-file read, as a client that sends each call's result back, with
`parallel_tool_calls` absent (the control) and `false`:

- the 27B, on chat completions and the Responses API, with strict tools held and with `strict: false` tools
  free: the control makes the 3 calls in one turn, then answers; `false` makes one call a turn for 3 turns,
  then answers.
- Flash-Next, the same two surfaces: the control makes 3 calls in one turn; `false` makes one call a turn,
  on the Responses API reading the first file again in a fourth turn. The test ran with
  `--model-host-margin-bytes 1G` in its flags (the host's own configured margin) because the default 4 GiB
  left the load 736 MiB short of RAM beside the machine's other work.

**The existing GPU legs** on the same tree, after `make gpu-status`: `repair_gpu`, `verify_cut_gpu`,
`permitted_decode_gpu` and `structured_output_gpu` (#287's) pass.

## Finding

Observed:

- Every held call in the corpus came back valid, 50 of 50, with one miss, repaired; one free twin of 50 called
  a tool the request did not declare.
- The 27B rarely needs holding: 27 cases written to provoke a miss drew none, held or free, and the agentic
  replay's calls were all valid without it.
- Traffic with no held tool costs the same a round on the branch as on main: the one cell at 1.0001-1.0005
  on the same tokens, the several cell's subs at 0.9968 on the same tokens and rounds. The several cell's main
  class fails `pool.py`'s gate at 0.9276 on four different texts of one request.
- `held` on the agentic replay changed no delivery and cost 0.1% a round at one lane (17.19 against 17.17 ms),
  0.3% on the subs' rounds.
- `parallel_tool_calls: false` gives one call per turn on the served 27B, held and free, and on the served
  Flash-Next.

Inferred:

- The several/main FAIL is the request's text, not the build: its rate follows how many drafts its text has
  accepted and how much of its life it decoded alone (branch 1 stopped at half length, so most of its rounds
  were shared), and the branch's 16,000-token launch has the cheapest rounds of the four. A ratio over
  different tokens is not a speed comparison, as #595's protocol already says.
- The switch's default stays `as-sent`. ADR 0094 made a change conditional on `held` being no slower with
  sensible repairs; it is no slower, but on the owner's traffic it found nothing to repair, and the one repair
  of the corpus invented a value a free call would have left out. With no gain measured, OpenAI's defaults
  stay. Free calls are now counted (`ignis_tool_calls_total{outcome="invalid"}`): an operator who sees invalid
  ones can switch to `held` live.

## Implications

- Held tools can ship as they are: valid calls with DFlash2 on, plain traffic unchanged a round, holding's own
  cost within a percent.
- A held call's repairs on a normalized schema can write values the client did not ask for (a nullable member
  given a value instead of `null`); that is normalization's contract, as on OpenAI's Responses API.
- The several cell's main class no longer reproduces between launches of one build (#595's did, at its
  16,000-token cap), so an A/B on this trace must compare it per round or at the same tokens, not by `pool.py`'s
  ratio alone.

## Limits and unknowns

- One card, one artifact, greedy. The corpus is 50 requests written for this ticket; the agentic replay is 11.
  Neither drew an early close, so how an early close is repaired on the 27B is still unmeasured.
- Two launches per build in the A/B: they settle the one cell and the several cell's subs, not a per-launch
  difference below the several cell's spread.
- The Flash-Next leg ran with a 1 GiB host margin, not the 4 GiB default.
- Main was started with its Playground build older than its sources: its `web/dist/index.html` was dated
  just before the exe for the A/B and dated back after (`UI=0` serves none of it).
- Why the several cell's main request stopped reproducing since #595 was not looked into.

## Follow-ups

- Flash-Next holds tool calls: GitHub #581.
