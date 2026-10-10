# server 13 — A turn that ends inside its reasoning

GitHub: #315 · ADR 0048

## Problem Statement

2026-10-10, opencode 1.18.35 on Flash-Next (`variant: max`). The fourth turn
of a session reasoned for 12,939 tokens. It wrote a whole `<tool_call>` block
inside the reasoning and drew its EOS with the block still open. The request
log (`chatcmpl-4`):

```
finish_reason: stop   tokens: 12939   thinking_budget: 29952
thinking_closed: false   thinking_forced: false
WARN generation produced reasoning but no content or tool call
     (token budget exhausted before an answer began)
```

- The server channels output on `</think>`, so the whole turn was
  `reasoning_content`. The tool-call scanner reads only the content channel
  (`split_reasoning_and_tools`), so the call was never seen. opencode got a
  `stop` with no content and no call, showed thinking only, and left its
  loop.
- **The WARN was wrong.** It blames the token budget for every all-reasoning
  turn. This one ended on the model's own EOS, 17,000 tokens short of the
  budget.
- **The thinking budget cannot catch it.** Its forced close is a permitted
  set on the *next* draw, one round late. By the time the host sees the EOS,
  the leaf has fed it to the model (ADR 0048). The same hole sits inside the
  budget's own lag: an EOS drawn as the one free token between the budget
  and the forced close ends the turn in its reasoning.
- Frequency, from opencode's database: this shape is 1 of 115 Flash-Next
  turns and 0 of 413 on the 27B. It is model behaviour (a long, degrading
  reasoning that acts without closing its block) and is not reproducible on
  demand. The sampler had no penalty on `</think>`; the block was simply
  never closed.

## Solution

1. **The WARN names the ending, not the budget.** An all-reasoning turn is
   reported with a cause read off how it ended:
   - a `stop` / `completed`: the model ended its turn (EOS) before any answer;
   - a `length` / `incomplete`: the output limit was reached before any
     answer;
   - a `cancelled`: the client cancelled before any answer.
2. **An EOS drawn inside an open reasoning block becomes `</think>`**
   (ADR 0048, the **reasoning redirect**). The model continues after the
   close on its own. In the incident that would have been the call it had
   just written, now on the content channel.
3. **The request log says when it happened**: `reasoning_redirected_at`, the
   output index of the redirected `</think>`, present only on a request it
   happened to.

## Implementation Decisions

- **The WARN's cause is one pure function** in `crates/server/src/api.rs`:
  `all_reasoning_cause(ending: &str) -> &'static str`, called by
  `report_if_all_reasoning_no_content`. The three call sites pass what they
  pass today: chat's `finish_reason`, Responses' `status`. The WARN carries
  the cause as its text and as a `cause` field. It never mentions the token
  budget: whether the budget closed the block is already on the request log
  (`thinking_forced`).
- **ABI (ADR 0016).**
  - `struct ignis_sampling_params` appends `int32_t reasoning_close_id`:
    `-1` = no redirect.
  - `struct ignis_decode_options` and `struct ignis_prefill_options` each
    append `int32_t *out_reasoning_redirected`. Per lane, it is `1` when this
    call redirected that lane's draw, else `0`; null asks for nothing.
  - The sizes bump. `crates/core/src/step.rs` changes 1:1 in the same commit.
  - The degenerate `ignis_prefill` / `ignis_decode` entry points refuse a
    `reasoning_close_id >= 0` rather than ignoring it, as they do a permitted
    set.
- **One host function decides the cut and the redirect**, shared by both
  leaves and free of CUDA. Given a round's committed run (anchor plus
  accepted drafts), the target's tokens, the lane's stop ids and
  `reasoning_close_id`, it returns:
  - the committed length;
  - the next pending token;
  - whether it redirected.

  A stop id in the run cuts the run before it while the block is open, and
  inclusively (today's rule) once it is closed. A `</think>` committed
  earlier in the run closes the block for the rest of the round. Plain decode
  and the prefill's draw are the zero-draft case of the same function.
- **Where it applies.** Every place a leaf assigns `pending_token`:
  - 27B (`kernel/src/step.cu`): the prefill's successor, the plain decode
    round, the verify round's commit loop;
  - Flash-Next (`kernel/src/flash_next/program.cu`): its prefill successor,
    its decode round, its verify round (test drafts and the MTP head).

  The verify round's GDN fold and KV-ring invalidation already take the
  host's committed length, so a cut before an accepted EOS discards that
  column the way a rejected draft is discarded.
- **The flag's source.** `DecodeParams` gains `starts_in_reasoning: bool`
  (default `false`). The server sets it from
  `TemplateProvider::decoder_starts_in_reasoning` on chat completions and
  `/v1/responses`; `/v1/decide`, the bench and every internal caller leave it
  `false`. The scheduler hands the leaf `reasoning_close_id =
  close.think_end()` for a request that starts in its reasoning, has no
  `closed_at` yet, and runs with a configured `thinking_close`. Otherwise it
  passes `-1`. `ignore_eos` passes no stop ids, so there is nothing to
  redirect.
- **The runtime** (`crates/runtime`) carries the flag on `DecodeLane` and the
  prefill call, and returns the redirect flag on `DecodeOutcome`. The
  scheduler records the output index on the request, beside `BudgetState`.
  `BudgetState::commit` then sees the `</think>` like any other: a redirect
  inside the budget's lag is a natural close there and stops the forcing.
- **The mock compute** emulates the rule: with `eos_after` reached and the
  job carrying a close id, it emits that id, reports the redirect and keeps
  generating. That puts the scheduler, the server and the HTTP surface on
  the CPU gate.
- **No metric family.** The request log field is enough to find these
  turns. A `/metrics` counter would also need the Playground Monitor and
  ADR 0017.

## Testing Decisions

- **Pure functions first.** `all_reasoning_cause` gets a unit test. The
  leaf's cut-and-redirect function gets a C++ unit test in the kernel's CTest
  executable (`kernel/build.ps1 -Test`). It must cover: no stop, a stop with
  the block closed (inclusive cut, unchanged), a stop as the next pending
  with the block open (redirected), an accepted-draft stop with the block
  open (cut before, redirected), and a `</think>` earlier in the run
  disabling the redirect.
- **Core, on the mock**: the flag is set and cleared as specified, and the
  budget-lag case is covered.
- **Server HTTP, on the mock** (`eos_after`): chat streaming and not,
  `/v1/responses`.
- **GPU profile, once at the end, card free first (AGENTS.md).** On each
  leaf, a request whose draw is constrained to the EOS (a permitted set of
  one) with its block open commits `</think>` and keeps generating. The same
  request with the block closed ends `stop`. Check it on the prefill's draw
  and on a decode round. The verify-round cut is covered by the CTest unit.
  A forced accepted-draft EOS on the GPU would need a hand-made drafter.

## Acceptance

1. **WARN cause.** `all_reasoning_cause` maps `stop`/`completed`,
   `length`/`incomplete` and `cancelled` as in Solution 1, with a neutral
   text for anything else. No variant mentions the token budget. The WARN
   carries it as text and as a `cause` field.
2. **ABI.** `reasoning_close_id` on `ignis_sampling_params` and
   `out_reasoning_redirected` on the decode and prefill options, with the
   header's doc comment, the size bump and the 1:1 Rust binding. The
   degenerate entry points refuse a set close id.
3. **The cut-and-redirect function** exists once, is used by both leaves on
   every path that assigns a successor, and passes its CTest unit (all five
   cases in Testing Decisions).
4. **The flag.** `DecodeParams::starts_in_reasoning` is set by chat
   completions and `/v1/responses` from the template. The scheduler passes
   the close id exactly while a request that starts in its reasoning has no
   `closed_at`, and `-1` otherwise (thinking off, after the close,
   `ignore_eos`, no configured close). Unit tests in core.
5. **Mock emulation.** `MockCompute` redirects an `eos_after` that falls
   inside an open block, reports it, and keeps generating.
6. **HTTP.** On the mock, a thinking request whose EOS falls inside its
   reasoning returns:
   - reasoning, then content after the close;
   - no all-reasoning WARN;
   - a `finish_reason` / `status` from how the answer ended.

   This holds for chat streaming, chat non-streaming and `/v1/responses`. A
   thinking-off request with the same EOS ends `stop` as before.
7. **Budget lag.** A budgeted request whose lag token is the EOS is
   redirected, and its forcing stops. Asserted in core, on the mock.
8. **Request log.** `ignis.request.done` carries `reasoning_redirected_at`
   (the output index) on a request that was redirected, and nothing on one
   that was not.
9. **GPU.** The profile checks in Testing Decisions pass on the 27B and on
   Flash-Next, under `IGNIS_GPU_PROFILE=1`.
10. **Docs.**
    - CONTEXT.md gains **Reasoning redirect**.
    - `docs/user/README.md`'s thinking section says what a turn that ends
      inside its reasoning now does.
    - Spec 08 notes the lag hole and points here.
11. `cargo test` passes workspace-wide (baseline at a55682f:
    `ignis-server --lib` 591/591), and the kernel CTest stays green.

## Out of Scope

- **Recovering a tool call written inside the reasoning** (ADR 0048,
  rejected).
- **Redirecting `<tool_call>` drawn inside the reasoning.** False positives
  on a model that writes the token while reasoning about the format. Revisit
  only on a measured case that the EOS rule does not cover.
- **Why opencode's `max` variant arrived with a budget.** On every request
  of the incident session the server logged `thinking_budget: 29952`, which
  a request at `max` drops (`thinking::resolve_thinking_budget`). Either
  opencode did not send `reasoning_effort: "max"`, or it did not arrive as
  such. It is not this ticket's cause: the budget was not reached.
- **opencode's `small_model`** naming the 27B on the same server. Each
  session's title request then switches the model (spec model-switch/01).
  That is client configuration.

## Further Notes

- Evidence: opencode session `ses_eda64130cffe6qQxF3Fe2ROBs9`, message
  `msg_125a5aeec001cE8PwncPzV1SSq`. The reasoning ends `…Vado.\n\n<tool_call>
  \n<function=bash>\n<parameter=command>\necho skip\n…</tool_call>`, then EOS.
  The server log of that run is not tracked; the request log lines quoted
  above are the durable part.
- An end-to-end reproduction is not an acceptance criterion. The behaviour
  is a 1-in-115 sample at 57K context, and opencode does not keep the request
  body.
