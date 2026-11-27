---
status: accepted
---

# An end of turn drawn inside an open reasoning block closes the block instead

A stop id that a request draws while its reasoning block is still open becomes the block's `</think>`, and the
model goes on to answer. A turn that ends there has no answer: the server channels output on `</think>` and reads
tool calls only from the content channel, so the client gets a turn that is all reasoning and its agent loop stops,
when what the model wanted at that draw was to stop thinking and act. The leaf decides it, wherever it assigns a
lane's successor (the prefill's draw, a plain decode round, a verify round's next pending token): only the leaf
holds a successor before it is fed, from a copy each round already waits for. The host sees a token one round late,
after the hybrid models' recurrent state has absorbed it, and there is no rollback; for the same reason the
thinking budget's forced close, a permitted set on the next draw, cannot catch it.

- Only the stop set is redirected. A `<tool_call>` drawn inside the reasoning stays reasoning text: intercepting it
  would close the block whenever the model writes the token while reasoning about the format, and the turn that
  showed the problem ended on its stop id anyway.
- It applies to a request that starts inside its reasoning block, until the scheduler's thinking-budget state
  records the block's close. A request with `ignore_eos` has no stop ids and nothing to redirect.
- The scheduler's view lags by the round it cannot see, so the leaf decides "open at that position" over the
  round's own tokens: a model that closes its own block and then ends an empty answer is never given a second
  `</think>`.
- In a verify round, whose cut is decided on the host before the state uses it, an accepted draft that is a stop id
  cuts the run before it and the `</think>` is the next pending token. Nothing else is cut or committed differently.
- The leaf says when it redirected, and the request log records it beside `thinking_closed`, so a turn closed this
  way can be told from one the model closed itself.
- There is no server flag: a turn that would end inside its reasoning is a turn with no answer, and there is no
  client for which that outcome is the point.

The shape is the model's behaviour, a long, degrading reasoning that acts without closing its block, not a
regression in the server, and it is not reproducible on demand: in one agent's session database it is 1 of 115
Flash-Next turns and 0 of 413 on the 27B.

## Considered Options

- **Salvage the tool call out of the reasoning text**, on the server. The call in the turn that showed the problem
  was nothing like what the reasoning had planned; promoting it would have run garbage, and reasoning routinely
  contains hypothetical calls.
- **Continue on the host after the stop.** The stop id has already been fed, and on a hybrid model there is no
  position to step back to without a checkpoint at every token.
- **Re-submit the turn from the server**, with the generated tokens and `</think>` as a new prompt. Prompt retention
  holds the prompt's checkpoint, not the generated tokens, so the continuation prefills the whole reasoning again,
  tens of seconds on Flash-Next for the turn that showed the problem, and one HTTP request becomes two engine
  requests. ADR 0052 chooses it for a reasoning loop, where no fix inside the leaf exists: there the continuation
  keeps only the text before the loop, and the prompt comes back from its checkpoint.
- **A device-side deny mask on the stop ids while the block is open.** It needs a mask inside the captured decode
  graph and the verify path, and it closes nothing: the model keeps reasoning past the point where it wanted to
  stop, often until the budget forces the close.

## Consequences

- Output differs from the reference only on a turn that would have ended inside its reasoning. A greedy comparison
  that runs with `ignore_eos`, as the bench does, is unaffected; any comparison that crosses such a turn must say
  so. In ADR 0037's vocabulary this is an Ignis-patched behaviour, of the leaf and not of a vendored file.
- The model continues after the `</think>` on its own, usually with the call it had written in its reasoning, now
  on the content channel where the scanner reads it. The reasoning keeps whatever the model wrote in it.
- The thinking budget's lag closes by the same rule: a stop id drawn in the one free draw between the budget and the
  forced close is redirected, and that `</think>` is a natural close inside the lag, which already stops the
  forcing.
- The step ABI grows under ADR 0016, by appends with their size bump and the Rust binding in the same commit: the
  close id on the sampling parameters, and the redirected flag as an output of the decode and prefill options.
