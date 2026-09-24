# server 08 — A thinking budget by default

GitHub: #265

## Problem Statement

A coding agent (qwen-code, opencode, Claude-Code-style clients) talks to ignis with
thinking on and usually no effort, or the OpenAI vocabulary's `high`. Both resolve
to Qwen3.8's `xhigh`, and at `xhigh` the model over-deliberates:

- on the 2026-09-24 effort sweep, 7 of 16 checkable coding tasks were still
  reasoning at 16K tokens and never answered;
- the whole sweep passed 8/16;
- median wall time was 125 s per task.

The agent sees a turn that burns its whole `max_tokens` and returns no content.
Thinking is 73.5% of real agent output, so this is also where throughput goes.

The thinking budget built on 2026-09-24 (spec server/04 §"Thinking budget",
commit `ae5d326`) fixes this when a request asks for it: an 8K budget at `xhigh`
passed **15/16**, every forced close answered correctly, and median wall time halved.
But the server default is still "no budget", so no client that doesn't know the ignis
extension ever gets it.

## Solution

The server ships with a measured thinking budget on by default:

- A request that asks for nothing gets the budget.
- A request can set its own budget, or opt out.
- The budget always leaves the answer room inside `max_tokens`.
- Every forced close is visible, in the response and in telemetry, so an operator can
  tell how often the budget bites and tune it.

## User Stories

1. As a coding-agent user, I want a turn at the default effort to always end with an answer, so that my agent never stalls on a turn that reasoned until `max_tokens`.
2. As a coding-agent user, I want the default to keep `xhigh`'s reasoning quality up to the budget, so that I do not trade correctness for speed without knowing it.
3. As an API client, I want `thinking_budget` on a request to override the server default, so that a hard task can think longer.
4. As an API client, I want to opt out of any budget on one request, so that an evaluation or a research prompt can reason without a forced close.
5. As an API client, I want an explicit opt-out value, so that "no budget" does not require picking an arbitrarily large number.
6. As an API client with a small `max_tokens`, I want the budget to stop early enough that the answer still fits, so that a forced close is not immediately followed by a `length` cut.
7. As an API client, I want to know when my response's reasoning was closed by the budget, so that I can retry with a larger budget when it matters.
8. As an API client that does not know the extension, I want nothing in the standard OpenAI fields to change shape, so that my parser keeps working.
9. As an operator, I want to set the default with a flag and an environment variable, so that I can tune it per deployment.
10. As an operator, I want to turn the default budget off entirely, so that I can reproduce unbudgeted behaviour.
11. As an operator, I want a counter of forced closes, so that I can see how often the budget bites on my traffic.
12. As an operator, I want the request log to say whether a request's close was forced and after how many reasoning tokens, so that I can inspect individual turns.
13. As an operator, I want the server to refuse a default budget it cannot apply (a tokenizer that splits `</think>`) loudly at startup, so that I do not believe a budget is active when it is inert.
14. As the project owner, I want the default value chosen from a measurement on agent-shaped traffic, not only from the 8-task sweep, so that the default is the measured-better one.
15. As the project owner, I want the default and its evidence recorded in a finding, so that a later change of model or template re-measures instead of guessing.
16. As a Playground user, I want the Playground's own requests to follow the same server default, so that what I try matches what agents get.
17. As an API client, I want a budget on a request with thinking off to be inert rather than an error, so that one client configuration works for both modes.
18. As an API client, I want the forced close text to be the model card's own hand-off, so that the answer that follows is written in the model's usual register.
19. As a maintainer, I want the budget to keep forcing through the constrained-decode seam only, so that no new leaf entry point or second forcing mechanism exists.
20. As a maintainer, I want the default to live in the server's configuration next to the thinking defaults, so that there is one place thinking policy is set.

## Implementation Decisions

- **The default value.**
  - `--thinking-budget` / `IGNIS_THINKING_BUDGET` gets a shipped default, **8192
    reasoning tokens** as the starting candidate.
  - The value is fixed by the acceptance measurement below. If that measurement favours
    another value (6K, 12K), that value ships.
  - An explicit `off` value (flag and env) restores today's behaviour.
- **Two levels, the request wins.**
  - The CLI / env value (`--thinking-budget`, `IGNIS_THINKING_BUDGET`) is the server's
    default budget.
  - A request's `thinking_budget` always overrides it, upward or downward, including
    turning it off.
  - The server does not cap a request's budget: `max_tokens` already bounds it.
- **Opt-out per request.** `thinking_budget: 0` means *no budget for this request*.
  This changes today's contract, where 0 is a 400.
  - absent / `null` → the server default;
  - a positive integer → that budget;
  - `0` → none;
  - anything else → 400 naming the field.

  The spec server/04 wire table is amended to match.
- **Answer room.** The effective budget is `min(budget, max_tokens − answer_reserve)`
  whenever `max_tokens` is set, so the close plus an answer fits.
  - `answer_reserve` is a server constant, measured from the sweep's answer lengths;
    ~1,500 tokens covers the longest answer seen.
  - A request whose `max_tokens` leaves no room below the reserve gets no budget, and
    no forcing at token 0.
- **Visibility.**
  - Non-streaming: the response carries an ignis extension field saying the close was
    forced, with the reasoning-token count at which it happened.
  - Streaming: the same field on the final chunk.
  - No standard OpenAI field changes.
  - The request log (`ignis.request.done`) gains the same two attributes.
  - Prometheus gains a forced-close counter, created lazily like the decision
    counters (ADR 0017 departure already accepted for #241).
- **Startup.** When a default budget is configured and the loaded tokenizer yields no
  close sequence, the server refuses to start. Today it only warns. A per-request
  budget with no close sequence stays inert.
- **Effort stays as resolved by server/04:** `high` → `xhigh`, unset → the template
  default `xhigh`. The budget is what makes `xhigh` safe; the effort default is not
  changed by this spec.
- **Seam:** unchanged. The scheduler forces the close through the permitted-set path
  of the constrained decode (`ignis_core::thinking_budget`). This spec touches only
  the server's resolution, reporting and defaults.

## Testing Decisions

- A good test drives the HTTP boundary and reads what a client or an operator sees:
  status, body fields, SSE chunks, `/metrics`, the request log. It never inspects
  scheduler internals.
- **HTTP (CPU, mock compute):** extend `crates/server/tests/openai_http_thinking.rs`
  (prior art: `a_thinking_budget_reaches_the_decode_rounds_only_while_thinking`).
  Cover:
  - default applied / overridden / `0` opt-out / thinking off inert / malformed 400;
  - the answer-room clamp against `max_tokens`;
  - the forced flag present exactly when the mock's decode rounds carried the close.
- **Config:** `config.rs` unit tests for the flag/env default, `off`, and the startup
  refusal (prior art: the `reasoning_effort` flag/env tests).
- **Metrics:** the metrics contract test (prior art: the #241 decision counters) sees
  the counter appear after the first forced close.
- **Scheduler (CPU):** `crates/core/tests/thinking_budget.rs` gains a case where the
  effective budget is clamped below the requested one.
- **Acceptance measurement (GPU, once, owner-run gate style):**
  - the 8-task effort sweep (`.scratch/effort-2026-09-24/effort.py`), 2 seeds, at
    budgets {off, 6K, 8K, 12K};
  - plus one recorded qwen-code session replayed through the bench trace harness
    (bench/sim, #118), measuring turns that end without content, reasoning tokens per
    turn, and wall time.

  The shipped default is the value with the best pass rate at no worse median wall
  time than 8K. The result is recorded as a finding (update
  `2026-09-24-reasoning-effort-on-coding-tasks.md` or supersede it).

## Out of Scope

- Changing the default `reasoning_effort` (`medium` measured 13/16 at ~3.5× less
  wall time; a separate owner decision).
- An adaptive budget (per task class, per lane, per conversation length).
- Budgets on `/v1/decide` (a decision generates no reasoning).
- Playground UI controls for the budget: their own spec, playground/04
  "thinking budget control", and their own ticket. This spec only guarantees the
  per-request field that UI will send.
- A second close text per effort or per language.

## Further Notes

- Evidence:
  - `docs/findings/2026-09-24-reasoning-effort-on-coding-tasks.md` (sweep with and
    without the budget);
  - `docs/findings/2026-09-24-copy-drafting-on-agent-traces.md` (thinking is 73.5% of
    agent output).
- The forced rounds run the whole batch non-speculatively for as many rounds as the
  close has tokens (~25). The 8-lane throughput cost of that is expected to be noise.
  The acceptance run should confirm it on the replayed session.
