# 01 — trace replay harness (JSONL load trace)

GitHub: #19

Re-send a recorded "1 main agent + N subagents" load trace (JSONL) against
the running engine (`docs/design/ignis-v1.md` §4, §6). Drive the scheduler
with a realistic concurrent load; collect per-request ttft / tok-s. The
reference is recorded with the **same** harness so comparisons are
apples-to-apples (eager CUDA graph on both sides).

## Acceptance

- The harness replays the recorded trace and produces per-request ttft /
  tok-s.
- A realistic "1 main + ~10 subagents" concurrent load is driven into the
  scheduler.

Delivered (commit 968a2c1): `HttpEndpoint` transport in `crates/bench`
(`client.rs`) — `reqwest` 0.21 blocking client (one `Client` shared across
the driver's worker threads, so it stays `Send + Sync` for the existing
`Endpoint` seam): `POST /v1/chat/completions` (streaming SSE for per-token
timing when the trace line is streaming; a single JSON body otherwise,
where ttft == total) + `GET /v1/models` readiness probe; `cmd_replay` /
`cmd_canary` pre-flight `list_models()` for an "engine reachable + model
loaded" check. The "1 main + N subagents" load drives the **existing**
bounded-concurrency `replay` driver (reused, not reinvented). Tests are
CPU-only and in-process: an axum mock engine (the server's wire shape —
SSE `chat.completion.chunk` + `[DONE]`, non-streaming `chat.completion`
JSON, OpenAI error bodies) served on a random `127.0.0.1` port; 37/37
`ignis-bench` tests green (28 unit + 5 `http_endpoint` + 2
`replay_load` + 2 JSON round-trip), asserting per-request ttft > 0 /
tok_s > 0 and the driver's concurrency bound (`peak_in_flight` within
`[2, max_concurrency]`, 11/11 requests completed).

Note: no recorded trace existed anywhere in the repo, so the fixture is a
documented **synthetic** `tests/fixtures/main_plus_10.jsonl` (1 main + 10
subagents sharing a system + tools prefix, staggered `t_arrive_ms`
0..1000) — a realistic load shape, **not** a reference recording. The ADR
0007 99% gate still needs a *recorded* reference run (bench-02, GPU); the
synthetic fixture is its placeholder. `HttpEndpoint` sends one user
message per trace prompt (the trace format is prompt-based; a future
multi-turn / system-role trace would extend the request-body builder).
