# GitHub #137 — live evidence (2026-09-12)

The bench's SSE reader used to count a token only from
`choices[0].delta.content`, so a turn that generated on the thinking channel
read back as a request that produced nothing. This is the before/after,
measured **in one session** on a free RTX 5090.

## The runs

```
ignis-server --artifact qwen3_8_27b_nvfp4full-v2.ninfer \
             --model qwen3.8-27b-nvfp4full-hq-e8-2b --bind 127.0.0.1:8000
```

started with its own defaults — `enable_thinking` on, `hq-e8-2b` KV — and
driven by `ignis-bench replay` on `probe-prompts.jsonl` (1 main + 2 sub, 256
tokens each, streaming, with no `enable_thinking` on the request so the
server default applies). Those three prompts are written for this probe, not
recorded from a working session: a recorded load trace is never committed
(`.gitignore`, GitHub #118).

`replay-before.json` is the same probe through the **pre-fix** bench binary
against the same live server process, 38 seconds later.

## ignis, before and after

Per-request, with the engine's own figures from `server-requests.jsonl`
beside each leg (the after leg is request ids 8–10, the before leg 11–13 —
the same three prompts, so the two legs' server-side figures agree with each
other and only the *reading* differs):

| request | bench before | server, before leg | bench after | server, after leg |
|---|---|---|---|---|
| `main` | 0 tok / ttft 4363.3 ms | 256 tok / ttft 76 ms | 255 tok / ttft 73.7 ms | 256 tok / ttft 73 ms |
| `s1` | 111 tok / ttft 1808.6 ms | 213 tok / ttft 87 ms | 210 tok / ttft 94.7 ms | 213 tok / ttft 87 ms |
| `s2` | 0 tok / ttft 4276.8 ms | 256 tok / ttft 87 ms | 254 tok / ttft 93.7 ms | 256 tok / ttft 89 ms |

Before, two of the three requests reported `n_tokens = 0` and
`ttft_ms == total_ms` (the "no content chunk: nothing to measure" fallback)
while the engine's own log for that very leg recorded 256 tokens at a 76 ms
ttft. The one request that did reach the answer channel timed its ttft 1.8 s
late: it had been thinking for that whole time. After, every request lands on
the engine's figures — `main` reads 59.8 tok/s against the server's 58.7.

The residual 1–3 token gap is the chunk-counting approximation this reader
has always made: a chunk carrying no text on either channel is not counted
(the leading `role` chunk, the decoder holding back a partial character, the
trailing tool-call chunk). It is 1.2% rather than 100%.

## The reference reads the same way

`ninfer-raw-sse.txt` is the reference engine's own stream for one canary
prompt, thinking on: 40 `delta.reasoning_content` chunks for `max_tokens:
40`, one token each, plus a leading `{"content": "", "role": "assistant"}`
chunk that carries no token. Same channel name, same framing, one chunk per
token — so the fixed reader counts both engines by the same rule, which is
what the G4 per-class ratio depends on.

`replay-reference.json` is the same probe replayed against that reference
(`--max-concurrency 1`, so the two sub requests queue and their ttft is
queueing, not prefill): 256 / 240 / 256 tokens, no zero-token reading.

## Canary

`canary-after.json`: the canary suite against the thinking-enabled ignis
server, 4/4 `sane=true deterministic=true`, `self-consistency: PASS` — with
no `--enable-thinking false` workaround, which is what the same suite needed
before (`.scratch/g4-logs/ignis-canary.json`, 3/4 "empty output").

## What is not measured here

No `g4`, `g3` or `ttft` record is part of this evidence.

- `g4`'s per-class throughput cell *is* this measurement: `g4::measure`
  calls `client::replay`, the same function this probe runs, through the
  same reader. Producing a full `g4` record would have added its 64K/128K
  needle cells, which need a corpus and a larger `--max-context` and measure
  nothing this ticket touches.
- `g3` and `ttft` cannot exercise the thinking channel at all: both send
  `enable_thinking: false` on every request by design (`g3.rs`, `ttft.rs`),
  as does G4's own needle cell (`g4.rs`). They are unaffected by this bug and
  by its fix.
