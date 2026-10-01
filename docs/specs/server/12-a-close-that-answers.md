# server 12 — A close that answers: the budget's hand-off and the model card's sampling

GitHub: #297

## Problem Statement

The thinking budget (spec 08) closes a reasoning block that has spent 6,144
tokens by forcing the model card's hand-off sentence and `</think>`. On
coding tasks that is what makes a turn end with an answer. In an agent's tool
loop, on 2026-10-01, it made two sessions loop instead:

- an opencode reviewer subagent of a scan job (lane 5): 32 forced turns in a row, each
  6,249–6,255 tokens, ~80 tokens after the close (a one-line stub and two
  `glob` calls), the prompt growing 6,306 tokens a turn to 192K;
- a second agent session that morning (lane 2): 15 forced turns, the prompt
  to 337K, where every retry then failed with #296's `bad_alloc`.

What the model is doing when it is cut: the whole assessment inside the
reasoning (9 library groups × 8 rubric criteria), still in group 1 at 6,144.
After the forced close it has nothing ready to write, so it defers with a
tool call ("let me verify …", `todowrite`, `grep`). The next turn restarts
the analysis and is cut again. The server serves greedy, seed 0, to a client
that sends no sampling fields (opencode sends none), so when the tool result
is the same, the next turn is the same turn: the deferral repeats verbatim.

Measured on a captured opencode turn
([finding](../../findings/2026-10-01-a-close-that-answers.md)):

- a bigger budget does not help: 16,384 defers the same way; no budget
  reasons past 32K and never answers;
- removing past reasoning from the history does not help either: with
  sampling, all three history conditions answer at a similar rate;
- **the close text is the lever**: with a close that says to write the final
  answer now, the turn answers greedy and 8 of 8 sampled (against 6 of 8 and
  never greedy for the current close), and the reviewer end to end answers at
  its first forced turn (1m19s greedy) where the current close took three
  forced turns and two deferrals;
- **greedy is what turns a deferral into a loop**: it repeats the deferral
  verbatim while the tool result stays the same; under the model card's
  thinking sampling a forced turn answers about 7 times in 10 even with the
  current close, so a long run of identical deferrals stops being likely.

There is a second cost to the greedy default. A client that sends `top_p` or
`top_k` without `temperature` — opencode does exactly that unless the model
declares `"temperature": true` — is answered `400 invalid_sampling_parameter`
today, because the absent temperature resolves to 0.

## Solution

1. **The close.** The forced hand-off becomes:

   ```text
   \n\nMy thinking time is over. I must now write the complete final answer from what I already have, without calling any more tools.\n</think>\n\n
   ```

2. **The model card's sampling by default.** A chat completion or Responses
   request that leaves a sampling field unset takes the Qwen3.8 model card's
   value for its mode:

   | mode | temperature | top_p | top_k | presence_penalty | frequency_penalty |
   |---|---:|---:|---:|---:|---:|
   | thinking (`enable_thinking` resolved true) | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
   | non-thinking | 0.7 | 0.80 | 20 | 1.5 | 0.0 |

   The thinking row is also what the artifact's own `generation_config.json`
   declares (`do_sample: true`, 1.0 / 0.95 / 20). `min_p` 0 and
   `repetition_penalty` 1.0 are neutral and need nothing.

3. **A fresh seed when none is sent.** A request without `seed` draws with a
   seed of its own, so two identical requests are two independent samples
   (the OpenAI semantics). A request that sends `seed` keeps it.

4. **History reasoning is unchanged.** The template keeps an in-flight tool
   loop's reasoning (spec 04, `preserve_thinking`); it was measured and is
   not the lever.

## Implementation Decisions

- **Where the defaults live.** `SamplingRequestFields::resolve`
  (`crates/server/src/api.rs`) takes the request's mode and fills each unset
  field from that mode's row. Both callers — chat completions and
  `/v1/responses` (`responses/input.rs`) — resolve the thinking controls
  first and pass `ThinkingOptions::enable_thinking`. Nothing in the core
  changes: `DecodeParams::default()` stays greedy for the bench, `/v1/decide`
  and every internal caller, which build their parameters themselves.
- **Greedy is still one field away.** A request whose `temperature` is
  explicitly `0` is greedy: the fields it did not send take the neutral
  values (`top_p` 1, `top_k` 0, penalties 0), and a non-neutral field it did
  send is still refused with `400` as today. Only an *explicit* greedy
  request can meet that refusal; a request that sends `top_p` alone is now
  served.
- **Each field defaults on its own.** A request that sends only
  `temperature: 0.6` gets `top_p` 0.95 and `top_k` 20 of its mode. The row is
  a set of defaults, not a preset the client opts into.
- **The seed** comes from `std::collections::hash_map::RandomState` (no new
  dependency): per-process random keys, a fresh hasher per request.
- **The close** stays one constant (`THINKING_CLOSE_TEXT`), still opening with
  the line break the one-round lag needs and still containing the single
  `</think>` token; the startup check that refuses a tokenizer without a
  single-token `</think>` is unchanged.
- **Order of the refusals.** The thinking controls are resolved before the
  sampling fields now, so a request wrong in both is answered with the
  thinking error.
- **Reproducible mock streams.** The mock compute mixes the request's seed
  into every token, and the HTTP tests that pin an exact stream assumed the
  old seed 0. Their harnesses say so with `Server::with_seedless_seed(0)`, the
  way a fixed `wall_clock` pins a test's timestamps; a served build never sets
  it.
- **The Responses object echoes the sampling as decimals.** `temperature` and
  `top_p` were echoed as the widened `f32`, so the new default `top_p` read
  0.949999988079071 on one path and 0.95 on another; they are echoed as the
  decimal the value was written as.
- **The bench** sends `temperature: 0, seed: 0` explicitly
  (`crates/bench/src/client.rs`) and is unaffected. The GPU tests that rely on
  greedy — the hq canary's determinism check and the `openai_http_gpu.rs`
  coherence checks — relied on the unset temperature and now send
  `temperature: 0` themselves.

## Testing Decisions

- **Server, over a recording compute** (the `sampling_http.rs` stand-in that
  sees the `DecodeParams` each job carries): the two rows by mode, a field
  overriding its default alone, explicit greedy with neutral filters, the
  explicit-greedy refusal, `top_p` alone served, distinct seeds for two
  seedless requests and an explicit seed passed through — chat streaming and
  not, and `/v1/responses`.
- **The close**: the forced close a budgeted request emits on the mock ends
  its reasoning with the new sentence; the Playground's dev mock uses the
  same text.
- **GPU, once at the end** (card free first, AGENTS.md): the synthetic
  reviewer task of the finding, opencode 1.18.34 with the scan job's
  provider config (no sampling fields), against the served build.

## Acceptance

1. **The close.** A request that reaches its budget emits
   "My thinking time is over. I must now write the complete final answer
   from what I already have, without calling any more tools." before
   `</think>`; asserted on the mock and in the Playground mock.
2. **Thinking defaults.** A thinking request with no sampling fields reaches
   the compute with temperature 1.0, top_p 0.95, top_k 20, presence and
   frequency penalties 0 — chat (streaming and not) and `/v1/responses`.
3. **Non-thinking defaults.** The same request with `enable_thinking: false`
   reaches it with 0.7, 0.80, 20, presence 1.5, frequency 0.
4. **Field by field.** An explicit value replaces its own default and no
   other.
5. **Explicit greedy.** `temperature: 0` alone reaches the compute greedy
   with neutral filters; `temperature: 0` with a non-neutral `top_p`,
   `top_k` or penalty is still `400 invalid_sampling_parameter`.
6. **`top_p` alone is served.** A request with `top_p` and no `temperature`
   answers `200`.
7. **Seeds.** Two seedless requests reach the compute with different seeds;
   an explicit seed reaches it unchanged.
8. **End to end.** On the served model, the synthetic reviewer task with
   the scan job's provider config writes its full report at its first forced
   turn, with no deferral.
9. **Docs.** `docs/user/README.md` states the defaults by mode, the fresh
   seed, explicit greedy, and the new close; spec 04's close block shows the
   new text; the finding and its index row land.
10. `cargo test` passes workspace-wide; `npm --prefix web test` and the web
    build stay green.

## Out of Scope

- Removing reasoning from a tool loop's history (measured; not the lever).
- A budget per `reasoning_effort` (a bigger budget defers the same way).
- Logging the resolved sampling on the request log. The incident could not
  answer "what sampling did this request run with" from the server side;
  worth its own ticket.
- `min_p` and `repetition_penalty` as wire fields.

## Further Notes

- **Why the close bans tools and still lets one through.** The sentence is
  read by the model, not enforced: on a turn whose evidence was genuinely
  missing (the module READMEs), one sampled run still called the tool it
  needed. What the sentence removes is the stub-and-defer turn.
- **Why the default is the model card and not a server flag.** ignis serves
  one model family; the card is the authority on its sampling, and the
  artifact already ships the thinking row. An operator who wants greedy
  sends `temperature: 0`.
