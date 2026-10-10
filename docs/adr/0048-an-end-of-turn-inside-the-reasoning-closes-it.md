# ADR 0048 — an end of turn drawn inside an open reasoning block closes the block instead

## Status

Accepted (2026-10-10, owner). GitHub #315, spec
`docs/specs/server/13-a-turn-that-ends-inside-its-reasoning.md`. Extends the
step ABI under ADR 0016 (a field append and a size bump on
`ignis_sampling_params`, an output append on the decode and prefill options).

## Context

On 2026-10-10 an opencode turn on Flash-Next at `max` effort reasoned for
12,939 tokens. It wrote a whole `<tool_call>` block *inside* its reasoning and
then drew its EOS with the block still open. The request log said
`finish_reason: stop`, `thinking_closed: false`, `thinking_forced: false`;
the budget (29,952) was nowhere near. The server channels output on
`</think>`, so everything was `reasoning_content`, and the tool-call scanner
reads only the content channel. The client got a turn that was all reasoning
and its agent loop stopped. In opencode's database this shape is 1 of 115
Flash-Next turns and 0 of 413 on the 27B. It is the model's behaviour (a
long, degrading reasoning that acts without closing its block), not a
regression in the server, and it is not reproducible on demand.

What the model wanted at that EOS is clear: it was done thinking and wanted
to act. A turn that ends there is wasted, whatever happens next.

Why the thinking budget's forced close (spec server/08) cannot catch it:
the close is a **permitted set** on the *next* draw, one round late. A round
returns the token the previous call drew, so the host sees the EOS only
after the leaf has already fed it to the model. On the hybrid models the GDN
state has then absorbed it, and there is no rollback. The same hole exists
inside the budget's own lag: if the one token drawn freely between the
budget and the forced close is the EOS, the request ends in its reasoning
anyway.

The leaf's C++ host code, though, holds every successor *before* it is fed:
`seq->pending_token` is assigned from a device-to-host copy that each round
already synchronizes on. The verify round's cut is decided on the host too,
before the GDN fold and the KV ring invalidation that use it.

## Decision

- **A stop-id draw inside an open reasoning block becomes `</think>`.** Each
  lane carries `reasoning_close_id`: the `</think>` id while its reasoning
  block is open as far as the host knows, `-1` otherwise. Wherever the leaf
  assigns a successor — the prefill's draw, the plain decode round, a verify
  round's next pending token — a successor that is one of the lane's
  `stop_ids` is replaced by `reasoning_close_id` when the block is open at
  that position. In a verify round, an accepted draft that is a stop id cuts
  the run *before* it, and the replaced token is the next pending. Nothing
  else is cut or committed differently.
- **"Open at that position" is decided by the leaf over this round's
  tokens.** The host's flag lags by the round it cannot see yet. The leaf
  clears it for the rest of the round when it commits a `</think>` (the
  anchor fed this round, or an accepted draft). So a model that closes its
  own block and then ends an empty answer is never given a second
  `</think>`.
- **Only the stop set is redirected.** A `<tool_call>` drawn inside the
  reasoning stays reasoning text. Intercepting it would close the block
  whenever the model writes the token while reasoning *about* the format,
  and the observed turn ended on its EOS anyway.
- **The scheduler sets the flag from what it already tracks.** A request
  that starts inside its reasoning block (the server's
  `decoder_starts_in_reasoning`) carries it until its `BudgetState` records a
  `closed_at`. A request with `ignore_eos` has no stop ids and nothing to
  redirect.
- **The leaf says when it redirected**, one flag per lane on the decode and
  prefill options (null asks for nothing). The request log records it beside
  `thinking_closed`, so a turn whose block was closed this way can be told
  from one the model closed itself.
- **No server flag.** A turn that would end inside its reasoning is a turn
  with no answer. There is no client for which that outcome is the point.

## Alternatives rejected

- **Salvage the tool call out of the reasoning text** (server-side). The
  call in the incident was `echo skip` with a 5-second timeout, nothing like
  what the reasoning had planned. Promoting it would have run garbage, and
  reasoning routinely contains hypothetical calls.
- **Continue on the host after the `Stop`.** The EOS has already been fed
  (above). On a GDN model there is no position to step back to without a
  checkpoint at every token.
- **Re-submit the turn from the server** with the generated tokens plus
  `</think>` as a new prompt. Prompt retention holds the prompt's checkpoint,
  not the generated tokens, so the continuation re-prefills the whole
  reasoning: tens of seconds on Flash-Next for this turn, and two engine
  requests per HTTP request.
- **A device-side deny mask on the EOS while the block is open.** It needs a
  mask inside the captured decode graph and the verify path. And it does not
  close anything: the model keeps reasoning past the point where it wanted
  to stop, often until the budget forces the close.

## Consequences

- **Reference parity.** Output differs from the reference only on a turn
  that would have ended inside its reasoning. A greedy comparison that runs
  with `ignore_eos`, as the bench does, is unaffected. Any comparison that
  crosses such a turn must say so (ADR 0037's vocabulary: this is an
  **Ignis-patched** behaviour of the leaf, not of a vendored file).
- The model continues after the `</think>` on its own: usually a line break
  and the call it had written in its reasoning, now on the content channel
  where the scanner reads it. The reasoning keeps whatever the model wrote
  in it.
- The thinking budget's lag hole closes by the same rule: an EOS drawn in
  the lag is redirected, and the redirected `</think>` is a natural close
  inside the lag, which already stops the forcing.
- ABI: `ignis_sampling_params` grows by one `int32_t` (and its size). The
  decode and prefill options grow by one output pointer each. The Rust
  binding changes in the same commit (ADR 0016).
