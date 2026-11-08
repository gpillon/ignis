# A reasoning-loop guard: a normalized compression score finds the loops, a rewind ends most of them

- Kind: experiment
- Status: corrected
- Observed: 2026-10-14
- Last verified: 2026-10-15
- Scope: server / reasoning-loop guard on chat completions and `/v1/responses`, HTTP and WebSocket (`crates/server/src/loop_guard.rs`, config v2 `model.loop_guard*`); serving / sampled reasoning on qwen3.8-27b, agentic contexts of 79k and 138k tokens, cross-turn prompt reuse; Playground reasoning view and Responses WebSocket transport
- Topic: reasoning-and-context
- Related: [A loop persists](2026-10-16-a-loop-persists.md) (the trigger and the cut it led to, ADR 0053), [ADR 0052](../adr/0052-a-reasoning-loop-is-taken-back-and-nudged.md), [Reasoning loops are the model's](2026-10-12-reasoning-loops-are-the-models.md), [ADR 0048](../adr/0048-an-end-of-turn-inside-the-reasoning-closes-it.md), [ADR 0046](../adr/0046-one-declaration-per-config-field.md), [#288](https://github.com/gpillon/ignis/issues/288)
- Superseded by: none
- Corrected by: [a loop persists](2026-10-16-a-loop-persists.md)
- Summary: A zlib score over the reasoning, code removed and digit runs folded, finds counting and template loops, and a rewind restores the prompt's checkpoint (0.08-0.10 s at 79k-138k); the guard now acts at 0.265 on 4 checks in a row.

> **Corrected 2026-10-16 (ADR 0053).** The guard no longer acts at the first
> check under 0.20: it acts at 0.265 on the fourth low check in a row, and
> only on a rewind that keeps something ([a loop
> persists](2026-10-16-a-loop-persists.md)). The score, the gate and the
> rewind below stand.

## Question

The [previous finding](2026-10-12-reasoning-loops-are-the-models.md) showed that about a
third of first turns on a long enumeration loop, and that the loop is the model's. The
model cannot be fixed from ignis. Can the server catch a loop while the reasoning
streams, take it back and get the model out of it, without disturbing legitimate
reasoning? And which signal finds the loops, including counting loops
(`the 321 AD, the 322 AD, …`) and template loops whose slots vary
(`the election of Clement XII, the construction of the Sistine Chapel (1508–1512)`)?

Then, on 2026-10-15: does the repaired history re-tokenize into what the model saw, does
the guard stay quiet on agentic contexts of 50k tokens or more, what does a rewind cost
there, and what does the next turn after a rewind reuse?

## Evidence

**The guard as the confirm run measured it** (`f19d6f4`):
- It is a tokio task between a chat request's engine event stream and its consumer, so
  it runs off the scheduler thread. At `f19d6f4` it covered chat completions only, and
  the reasoning only. `/v1/responses`, HTTP and WebSocket, was wired at `4daf4be3` and
  measured on the agentic run below.
- **Detection.** Every 250 raw reasoning characters it scores the last `window`
  characters of the **normalized** reasoning:
  - fenced code is removed, and a block still open is never scored;
  - inline code is removed;
  - every run of ASCII digits becomes `0`.
  The score is the zlib-9 compress ratio (`miniz_oxide`). A check fires when that
  score is under `ratio` **and** the zlib ratio of the last `window` raw characters is
  under `raw_gate`.
- **The cut.** The cut is the raw start of the window of the last earlier check that
  scored at least `clean`, snapped to a line break and never before the previous cut.
  If the kept text would end inside a ``` block, the cut moves before that block's
  fence.
- **The rewind.** The rewind is a cancel and a resubmit: the original prompt, plus the
  kept tokens, plus a nudge in the model's own voice ("Wait, I'm starting to repeat
  myself. …"). A KV truncation is not possible here: the hybrid model's recurrent
  state cannot go back to a position without a snapshot.
- **What the client sees.** The client receives `[LOOP DETECTED, REWINDING n TOKENS /
  m CHARS]` inside the reasoning. The Playground strikes through the `m` characters
  before the tag (`web/src/ui/rewinds.ts`). The tag's hash (`· h`) came later, with
  ADR 0052's history repair.
- **Retries and close.** After `retries` nudges, the next fire closes the reasoning
  instead.
- **Config.** All knobs are live config v2 fields (`PATCH /v1/config`). At `f19d6f4`,
  `loop_guard` defaulted to `off`; ADR 0052 makes it `rewind`, and makes the cadence,
  the snap, the room a continuation needs and its submit wait fields too:

| field | default |
|---|---|
| `loop_guard` | `off` (`log` and `rewind` are the other modes) |
| `loop_guard_normalize` | on |
| `loop_guard_window` | 1,000 characters |
| `loop_guard_ratio` | 200 (permille) |
| `loop_guard_raw_gate` | 300 (0 = off) |
| `loop_guard_clean` | 400 |
| `loop_guard_margin` | 0 |
| `loop_guard_retries` | 2 |

**Labels.** The 98 Rome runs of the previous finding and of the prototype's two arms
were labelled by reading their tails. 40 have an acute loop. 26 are **harmful**: an
acute loop followed by a forced close at the thinking budget, an EOS inside the
reasoning, or `length` with no answer. Those 26 runs are 23 of the 82 unguarded runs
(28%), and 8 of 32 (25%) in the two unguarded arms at the serving default on ignis. The
rules and per-run labels are in `research/labels.json`.

**Negatives:**
- 4,854 Qwen reasoning parts from the owner's opencode sessions (1,420 long enough to
  score);
- 12,936 Qwen thought parts from qwen-code sessions, 47M characters (5,655 scored);
- the 47 Rome runs with no loop.

All three are agentic coding except the Rome runs.

**Signals, simulated online** (check every 250 raw characters, the open fence stripped):

| signal | harmful caught | acute caught | Rome clean min | opencode floor | qwen-code parts that fire |
|---|---:|---:|---:|---:|---:|
| prototype: raw text, 3,000 chars, < 0.12 | 19/25 | 22/38 | 0.338 | 0.163 | 0 |
| normalized, 1,000 chars, < 0.20 | 24/25 | 34/38 | 0.257 | 0.273 | 7 |
| normalized < 0.20 and raw 1,000 < 0.30 (default) | 24/25 | 33/38 | 0.257 | 0.274 | 2 |
| normalized, 1,500 chars, < 0.20 | 24/25 | 34/38 | 0.295 | 0.229 | 7 |
| normalized + capitalized words folded, < 0.20 | 25/25 | 34/38 | 0.177 (fires) | 0.264 | 7 |

The 25 and 38 count runs after merging identical runs: some arms share up to 26k
characters per seed. Of the 7 qwen-code parts that fire ungated, 5 are numeric tables
(Fibonacci, hex/binary, primes, fp16 encodings), where the digits carry the
information. Their raw ratios are 0.307-0.384, and the gate keeps them out. The other 2
are templated text (an issue-status checklist, quoted log lines) at raw 0.201 and
0.218, and they pass the gate.

Tried and dropped:
- **Novelty against the reasoning's own history** (the window compressed with the
  previous 32K characters as a dictionary) and **the share of repeated lines.** Real
  reasoning re-drafts code it wrote earlier, which puts its floor at 0.015-0.08.
- **Adding the old raw rule** (normalized OR raw 3,000 < 0.12). It catches nothing more.
- **Logit entropy was not built.** It measures how predictable the next token is, and
  counting loops and code are both predictable. Logprobs are not served either
  ([#288](https://github.com/gpillon/ignis/issues/288)).

**Parity.** The Rust scorer agrees with an independent Python version, fed prefix by
prefix, on all 4,874 texts: the normalized text, every check window, the cadence.
miniz and zlib differ by at most 0.015 in the 0.15-0.30 band. Rust's cuts are never
later than Python's (median 96 characters earlier), because Rust maps the window start
to the raw text exactly.

**Prototype arm** (2026-10-13, raw 3,000 / 0.12, the old cut rule, 16 seeds). The guard
acted on 3 of 6 loops, and 3 of 3 left the acute loop.

**Confirm run.** It ran on 2026-10-14 with the defaults above, `f19d6f4`, serving
defaults (`yarn:2`, 524,288, `hq-e8-2b`, DFlash2) and the Playground's Rome body: seeds
101-132 non-streaming, then 133-140 streaming. None of its data was used to choose a
threshold.

| | |
|---|---|
| rewinds | 25, **0 on legitimate reasoning**: verbatim 10, template 7, counting 4, "?"-degenerate 4 |
| runs with an acute loop | 20 of 40; 25 episodes caught, 8 missed |
| cut before the acute onset | 25/25; span median 2,144 characters (1,071 tokens), max 8,828 |
| after the first rewind | 9 of 16 escaped for good; 7 relapsed (2 then closed by the guard) |
| harmful | **4/40 (10%)**: 102 and 124 clear; 101 and 123 escaped the loop but met the budget |
| answered or called a tool | 40/40 |
| streaming (8 runs) | well-formed, `[DONE]` last, one usage chunk; tag and nudge in one reasoning delta |
| continuation | prompt 4,416-13,867 tokens; TTFT 369-1,744 ms: it restored the 1,984-token prompt checkpoint and prefilled only the kept text and the nudge, 3-12 chunks (see the agentic run's cost below) |

- **Harmful, compared.** 4/40 here against 23/82 in all unguarded runs: Fisher
  two-sided p = 0.035. Against the 8/32 in the two arms at the same configuration:
  p = 0.12.
- **Misses** score 0.21-0.31 normalized. Run 124's counting loop over papal reign
  ranges ("683–701? 701–709?") scores 0.09 normalized but 0.33 raw, so the gate
  stopped it.
- **Thin margins.** Three caught episodes fired at raw 0.293-0.295, just under the
  0.30 gate.
- **`usage.completion_tokens`** counts the struck tokens and the tags, so it exceeds
  `max_tokens` in 6 runs.
- **Loops in the answer.** 4 of the 18 runs that answered in content end that answer
  in a verbatim loop. The guard does not watch content.

**The nudge's line breaks and the tokenizer** (2026-10-15). History repair gives the
model back the text it saw. Whether that text re-tokenizes into the tokens the
continuation's prompt carried was measured with `crates/server/examples/loop_retokenize.rs`
on the 27B's tokenizer, over the 48 runs above (`.scratch/loop-rewind/out/rewind/`,
`.scratch/loop-rewind/confirm/out/rewind2/`): 4,165 cuts at the end of a line-break run
after the first 1,000 characters, and the 18 real cuts. At each cut,
`kept tokens ++ encode(sentence)` against `encode(kept text + sentence)`. The generated
tokens are stood in for by the tokenizer's own split of the reasoning, which was
prefix-stable at every cut.

| Sentence after the kept text | Line-break cuts identical | Real cuts identical |
|---|---|---|
| As written, opening with `\n\n` | 0 / 4,165 | 5 / 18 |
| Without its leading line breaks after a kept line break | 4,165 / 4,165 | 18 / 18 |

The figures are the same for both nudges and for the close.

- As written, the kept text's last newline token and the sentence's opening `\n\n`
  re-tokenize as one: `"." "\n\n" | "\n\n" "Wait"` becomes `"." "\n\n\n\n" "Wait"`.
  The next turn's prompt then leaves the retained state at the cut.
- The 5 real cuts that matched are the ones whose kept text does not end in a line
  break.
- The guard now drops the sentence's leading line breaks when the kept text ends in
  one (`loop_guard::sentence_after`). That applies to the model's view and to what
  the client receives after the tag.
- What the model reads changed with it. A kept text ending in one `\n` used to be
  followed by three line breaks before the nudge, and now by that one.

**Agentic contexts of 79k and 138k tokens** (2026-10-15). Branch `experiment/loop-rewind`
at `dbe03653`, CUDA release build, the owner's serving flags (`yarn:2`, 524,288,
`hq-e8-2b`, DFlash2/7, chunk 1,024, 8G host pool), idle unload off, the loop guard at its
defaults (`rewind`). One GPU hold.

Two request bodies came from real opencode sessions, cut at a real user turn. Their
tools have the session's names and permissive schemas, and they use the Playground's
settings: temperature 1, `top_p` 0.95, `xhigh`, `max_tokens` 16,384, class
`interactive`. The bodies stay in the worktree's `.scratch/loop-gpu/`, untracked.

| body | source | prompt tokens |
|---|---|---|
| A | a debugging session the 27B ran on ignis | 78,773, or 78,809 with one sentence appended asking for the whole analysis before any edit |
| B | an infrastructure session a llama.cpp-served Qwen3.8-27B ran | 137,742, or 137,772 with the same sentence |

*False rewinds.* Run 1 used body A with the analysis sentence, 8 seeds, concurrency 4.
Checks are counted as 1 + (reasoning chars − 1,000) / 250: the first check comes when a
window is full, then one every 250 raw characters. The lowest score of each run comes
from the research scorer, fed prefix by prefix.

| seed | reasoning chars | checks | lowest normalized | lowest raw | completion tokens | finish |
|---|---:|---:|---:|---:|---:|---|
| 1001 | 33,536 | 131 | 0.486 | 0.486 | 9,681 | tool_calls |
| 1002 | 39,583 | 155 | 0.491 | 0.447 | 10,991 | tool_calls |
| 1003 | 53,222 | 209 | 0.472 | 0.477 | 14,439 | tool_calls |
| 1004 | 16,259 | 62 | 0.517 | 0.517 | 4,268 | tool_calls |
| 1005 | 37,443 | 146 | 0.509 | 0.512 | 10,280 | tool_calls |
| 1006 | 40,522 | 159 | 0.484 | 0.505 | 12,179 | tool_calls |
| 1007 | 25,349 | 98 | 0.521 | 0.521 | 7,214 | tool_calls |
| 1008 | 34,374 | 134 | 0.454 | 0.454 | 9,239 | tool_calls |

- Totals: 280,288 characters, 1,094 checks, 0 rewinds, and no `loop_guard` event in
  the log.
- Eight more seeds ran under a stricter setting and never fired:
  - B, 4 seeds at ratio 0.35 with the raw gate off;
  - A, 3 seeds at ratio 0.50 with the raw gate off;
  - B, 1 seed at the defaults.
- Over all 16 seeds, 1,369 checks, the lowest normalized score was 0.449, against the
  0.20 threshold. No check went under 0.40.
- The earlier floors were 0.257 (Rome) and 0.273 (the opencode corpus).

*Forcing rewinds took a higher ratio than planned.* At ratio 0.35 nothing fired on
this reasoning, so the cost samples ran at 0.50-0.54 with the raw gate off. That is
above `clean` (0.40). The cut then walks back to the previous check, or to the floor
when the first check fires. Every rewind below was forced: they are cost samples, not
detections.

*The cost of a rewind: the prompt is restored, and only the kept text and the nudge
prefill.*

| context | rewinds | kept before the cut | continuation prefill | admitted in | gap the client feels (tag → next delta) |
|---|---:|---|---|---|---|
| B, 137,772 | 2 | 0 chars | 23 tokens, 1 chunk; 137,770 restored from the prompt checkpoint (device, 18 ms) | 99-100 ms | 0.099-0.100 s |
| A, 78,773 | 14 (chat 5, socket 6, HTTP 3) | 0-413 chars | 23-127 tokens, 1 chunk; 78,771 restored (18 ms) | 78-112 ms | 0.078-0.090 s |
| A, 78,809 | 1 | 17,944 chars (about 3,645 tokens) | 3,668 tokens past the prompt, 4 chunks | 1,283 ms | 1.276 s (a second lane was decoding) |

- For comparison, the same prompts prefilled cold:
  - 78,809 tokens in 13.09 s (80 chunks);
  - 137,742 tokens in 34.5 s (137 chunks).
- The checkpoint is the request's own. It is the one every request publishes at its
  generation opener (ADR 0029). Seed 2001 is the proof:
  - its prompt was cold (1,152 cached), and it prefilled 77.6k tokens itself;
  - its continuations still restored 78,771.
  So a continuation does not depend on other seeds sharing the prompt.
- The 10-20 s once estimated for 100k assumed a full re-prefill. The measured cost is a
  tenth of a second plus the kept text's own prefill. The kept-heavy case is n = 1.
- **The confirm run, read again.** Its continuations did not re-prefill their whole
  prompt, as this finding first said.
  - Their done lines say `reuse_source: device`, `reused_prompt_tokens: 1984`.
  - Their chunk counts (3-12) are (prompt − 1,984) / 1,024.
  - So the 0.4-1.7 s was the kept text and the nudge, 2.4k-11.9k tokens.
  - The misreading came from `prefilled_tokens` on the admitted line. It is the
    position reached, equal to `prompt_tokens` with or without reuse.
    `prefill_chunks_consumed` and the done line say what was computed.
- **Telemetry.** A request restored through a *retained prefix* (page-floored: 78,784 =
  1,231 × 64) has no `reuse_source` on its done line and adds nothing to
  `ignis_retained_state_hits_total`. Only `cached_tokens` and
  `ignis_retained_reused_tokens_total{kind="prefix"}` show it. The deep continuation
  above was one of these: 4 chunks, no reuse field.
  Fixed by #325 (spec
  `docs/specs/kv-reuse/03-a-prefix-restore-is-reported.md`): the done line now carries
  `reuse_source`, `reused_prompt_tokens` and `reuse_kind=prefix`, and the hits counter moves. A continuation cancelled by the next
  rewind has no done line at all.

*The next turn after a rewind.* The shape is the agentic loop: the assistant turn
(reasoning, text, tool calls) after the last user message, then a tool result.
`/v1/tokenize` shows that the reasoning is rendered in this shape: 78,868 tokens without
`reasoning_content`, 78,935 with it.

Each arm sent `max_tokens` 8. The arms ran right after their source response, whose
own last continuation had ended at 78,912 (chat), 80,149 (socket) and 80,351 (HTTP)
tokens.

| surface | arm | prompt | cached |
|---|---|---:|---:|
| chat | `reasoning_content` as received (good hash) | 78,935 | 78,771 |
| chat | same, hash corrupted | 79,520 | 78,771 |
| socket | `previous_response_id` (the stored model view) | 80,170 | 78,771 |
| socket | full input, good hash, sent right after | 80,170 | **80,168** |
| socket | full input, hash corrupted | 81,036 | 78,771 |
| HTTP | reasoning item with `encrypted_content` | 80,373 | 78,771 |
| HTTP | visible text only, good hash, sent right after | 80,373 | **80,371** |
| HTTP | visible text, hash corrupted | 81,163 | 78,771 |
| chat, no rewind (control) | B's next turn | 138,432 | 137,740 |
| chat, no rewind (control) | short prompt, plain next user turn, `preserve_thinking` | 278 | 87 |

- **Reuse never passes the first turn's generation opener** (prompt − 2), on any
  surface, rewound or not. ADR 0029 keeps no checkpoint after generation: the next turn
  re-renders the assistant message, so the state at the end of a generation never
  matches it. The two controls without a rewind stop at the same place.
  - A rewind therefore costs the next turn no reuse that an ordinary turn would have.
  - "Good hash reuses past the cut" cannot be observed in this build.
- **What the arms do separate is the prompt:**
  - Good hash, the blob and the socket's stored view render the same tokens. The
    proof is the checkpoint hits marked in bold:
    - HTTP: the visible-text arm restored the blob arm's checkpoint at prompt − 2;
    - socket: the full good-hash input restored the chained turn's.

    A checkpoint is matched by content, so the repaired client text is the model's
    view token for token.
  - A corrupted hash leaves the struck text and the tags in the prompt: +585 (chat),
    +866 (socket) and +790 (HTTP) tokens the model never saw.
  - A first socket run put the corrupted arm between these two arms, and its
    checkpoint displaced the chained one. That run's good arm got only the opener; the
    rerun shown here fixed the order.

*The transports.*

- **Playground-shaped socket.** The client spoke the socket as the Playground's
  `socket.ts` does: subprotocol `responses`, `response.create` built as
  `buildResponseCreate` builds it. It used the Rome body with forced rewinds.
  - A complete run:
    - streamed three tags (two nudges, then the close) as
      `response.reasoning_text.delta`, gaps 0.052 and 0.054 s;
    - ended `response.completed`, with `output_tokens_details.rewound_tokens` 716
      (235 + 227 + 254) and `output_tokens` 3,228 against `max_output_tokens` 3,000
      (rewound tokens count, by design).
  - The response object's reasoning item carries the streamed text with its tags.
    The stored view and `encrypted_content` carry the model's.
- **Cancel on the socket.** A `response.cancel` sent on the tag's delta (1.39 s) ended
  the response at once.
  - It ended `response.incomplete`, status `cancelled`.
  - `ignis_requests_cancelled_total` rose by 2.
  - The scheduler showed 0 running and 0 waiting at the first sample.
  - The continuation's admitted line is empty: it was cancelled before it ran.
- **Chat SSE hang-up.** A client shut its socket on the tag's line (1.45 s), on the
  Rome body over chat SSE.
  - `ignis_requests_cancelled_total` rose by 2 at the first sample.
  - The scheduler was idle.
  - The continuation was cancelled before admission.
  - Chat's own `CancelOnDrop` named the first request at `dbe03653`, so the guard's
    `tx.closed()` select is what covers it, and it does.
- **opencode** 1.18.35 (`@ai-sdk/openai-compatible`, no `interleaved` option) sends
  `reasoning_content` back on the assistant tool-call message of its loop. This was
  seen through a logging proxy on one `opencode run`.

Raw material is in the worktree's `.scratch/` (not tracked), archived after the merge in
the main clone's `.scratch/worktrees/loop-rewind-scratch/` (the paths below are relative
to it):
- `loop-rewind/NOTES.md`: design, calibration, every run's table;
- `loop-rewind/research/`: labels, variant and cut scripts, `REPORT.md`;
- `loop-rewind/parity/`;
- `loop-rewind/confirm/`: responses, SSE captures, server log, `labels.json`,
  `summary.json`;
- `loop-gpu/`: the agentic bodies, the run and check scripts, their outputs, the server
  log.

The private corpora and bodies stay there.

## Finding

Observed:
- On the tuning data, the normalized score with the gate fires falsely on 2 of the
  5,655 scored qwen-code parts, and on no opencode part and no clean Rome run. It
  catches counting and template loops, which the raw score could not separate from
  code.
- On the held-out run (new seeds, same prompt) it made 25 rewinds, none on legitimate
  reasoning.
- On agentic contexts of 79k and 138k tokens it made no rewind: 16 seeds, 1,369 checks,
  the lowest normalized score 0.449.
- The rewind and nudge work as an exit: every cut landed before the acute onset, and
  9 of 16 runs escaped for good after the first nudge. Harmful runs went from about a
  quarter to 10%, though on different runs.
- What is left is in the 0.21-0.31 band. It holds slow templates and numeric loops
  that compression cannot tell from numeric tables (run 124 sits among them at raw
  0.33).
- Every continuation restores the original prompt's checkpoint and prefills only the
  kept text and the nudge: the client waits 0.08-0.10 s at 79k-138k tokens with little
  kept, and 1.276 s with about 3,645 tokens kept. (First written here as a re-prefill of
  the whole prompt, a misreading of `prefilled_tokens`; corrected 2026-10-15.)
- With the sentence's leading line breaks dropped after a kept line break, the repaired
  reasoning re-tokenizes into the continuation's prompt at every cut measured (4,165 /
  4,165 line-break cuts, 18 / 18 real cuts); as written, 0 and 5 do.
- The next turn after a rewind renders the model's view token for token from the
  repaired visible text, from `encrypted_content` and from the socket's stored view;
  its reuse stops at the previous turn's generation opener, as any turn's does.
- Chat SSE, the Responses socket and its cancel carry a rewind and stop its
  continuation; opencode sends `reasoning_content` back.

Inferred:
- No threshold change helps recall any more. The lowest legitimate texts sit at
  0.257 (Rome) and 0.273 (opencode), and the gate's 0.30 cuts through the numeric-loop
  tail. A new signal would be needed. Entropy was not shown to be that signal; the
  reasoning above argues it shares the confusion.
- A rewind's cost does not grow with the prompt: the prompt is restored from the
  checkpoint the request publishes at its generation opener, and the kept text sets the
  prefill. (First inferred here as a re-prefill of the whole prompt on a 50-100k-token
  agentic prompt, extrapolated from 1.7 s at 14k tokens; measured false on 2026-10-15.)
- Cuts that keep a degenerate lead-in ("?"-hedging) relapsed more often: 5 of 7,
  against 2 of 9 clean cuts. That is suggestive only at this n.

## Implications

- The data supports `rewind` as the default, which ADR 0052 adopts: no false rewind on
  the held-out run nor on agentic contexts of 79k-138k tokens, and a rewind costs about
  a tenth of a second there plus the kept text's prefill. The tag and strike-through
  make every rewind visible.
- The two preconditions this finding first set for a default on agents are met: the
  agentic run above, and a decision on `usage.completion_tokens` (ADR 0052: it counts
  the rewound tokens, and `rewound_tokens` says how many).
- A loop left alone can burn the rest of a 14,336-token budget. When the detection is
  right, the rewind is the cheaper of the two in wall time.
- History repair matters to agents: opencode sends its reasoning back. Token equality
  needs the junction rule (`sentence_after`), not text equality alone.
- Loops in the answer are a separate problem of the same size here. A nudge cannot go
  into content, so the action there has to differ, for example truncating at the cut
  or resampling.
- A restore through a retained prefix is invisible to `reuse_source` and
  `ignis_retained_state_hits_total`: reading reuse from the done line alone undercounts
  it.

## Limits and unknowns

- One prompt (the Rome enumeration) and one labeller for the detection's numbers.
  Thresholds were chosen on the same 98 runs. The confirm run is new seeds, not a new
  prompt.
- The arms share long prefixes per seed and use different seeds, so the comparisons
  are unpaired. 4 harmful against 8 is n too small to give a rate beyond its direction.
- The negatives are agentic coding plus one prose prompt. Other languages and other
  long enumerations are untested.
- The gate's 0.30 was set after seeing 7 negatives.
- A loop inside an open code block is never scored.
- The Rome runs guarded chat completions only; `/v1/responses` was measured only on
  the agentic run, with forced rewinds. Image requests are not rewound, and answers are
  not watched.
- Streaming was checked on 8 runs.
- The junction rule changed what the model reads after a kept line break (one line
  break before the nudge, not three); its effect on the continuations was not
  re-measured on the GPU. The junction at the sentence's end, between its closing
  `\n\n` and the continuation's first token, was not measured.
- One debugging turn (8 seeds) and one infrastructure turn carry the agentic
  false-rewind count; run 1 ran at concurrency 4.
- No agentic rewind was a detection. What a rewind does to the quality of an agentic
  turn is unmeasured.
- The kept-heavy cost sample is n = 1, at concurrency 2.
- A continuation under memory pressure could find its prompt checkpoint evicted
  (retained state is the first victim); not exercised.

## Follow-ups

- The continuation reuses the cancelled request's prompt: why no `reuse_source`, and
  what makes it one. (Answered 2026-10-15: it already does, through the prompt
  checkpoint; see the agentic run's cost.)
- `usage.completion_tokens` under a rewind (answered by ADR 0052: it counts the
  rewound tokens, and `rewound_tokens` says how many).
- An ADR on rewinding output a client already received, and the `loop_guard` default:
  done, [ADR 0052](../adr/0052-a-reasoning-loop-is-taken-back-and-nudged.md)
  (accepted 2026-10-15), `rewind` by default.
- Loops in the answer: what action, and whether the same score applies.
- What a natural rewind does to an agentic turn's quality.
- The retained-prefix restore's missing `reuse_source` on the done line (outside the
  guard).
