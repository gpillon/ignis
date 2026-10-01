# A forced close that says "answer now" ends an agent's thinking-budget loop; greedy is what makes the deferral repeat

- Kind: experiment
- Status: current
- Observed: 2026-10-01
- Last verified: 2026-10-01
- Scope: server / thinking budget close text, default sampling, agent tool loops
- Related: [spec server/12](../specs/server/12-a-close-that-answers.md) (GitHub #297),
  [thinking budget default](2026-09-24-thinking-budget-default.md) (spec server/08),
  [batched decode width drift](2026-09-14-batched-decode-width-drift.md), #296
- Superseded by: none

## Question

Two agent sessions looped on the served 27B on 2026-10-01, every turn forced
by the 6,144-token budget. Why does a forced close lead to the same turn again,
and which lever ends it: the budget's size, the reasoning kept in the tool
loop's history, the sampling, or the close text?

## Evidence

**The incidents** (server request log, `ignis.request.done`):

- an opencode reviewer subagent, lane 5, 16:56-17:14Z: 32 forced turns in a
  row, 6,249-6,255 tokens each, `thinking_forced_at` 6,145-6,151, ~80 tokens
  after the close (a one-line stub and two `glob` calls), the prompt growing
  6,306 tokens a turn to 192K;
- another agent session, lane 2, 12:16-12:26Z: 15 forced turns, the prompt to
  337K; every retry after it failed with #296's banded-carry `bad_alloc`;
- over the day, 90 forced turns, 81 of them followed by fewer than 300 tokens.
  The four other reviewers of the lane-5 job were forced too and answered
  after 1-3 deferring turns.

The client sent `reasoning_effort: "medium"`, `enable_thinking: true`,
`max_tokens` 32,000 and no sampling field, so the server served greedy, seed 0.
opencode 1.18.34 re-sends every past `reasoning_content`, and the template
keeps an in-flight tool loop's reasoning (spec server/04).

**What the cut reasoning is doing.** The reviewer's task was to score 9 library
groups on an 8-criterion rubric. The reasoning does the whole scoring inside
the block and is still in group 1 at 6,144. After the forced close it has
nothing ready, so it defers ("let me verify …" + `grep`/`glob`/`todowrite`).
The next turn restarts the analysis ("Now I have all the evidence. Let me
organize it") and is cut in the same place.

**The repro.** A synthetic workspace (a Java inventory, per-module `.deps`
files, a rubric skill), opencode 1.18.34 isolated from the user's config, the
job's provider config and agent prompt, a capture proxy in front of ignis.
The reviewer was forced at three turns and deferred at two of them before
answering, the shape of the four reviewers that escaped. A greedy replay of
the first forced turn reproduced it bit for bit (6,266 tokens, forced at
6,148). Replaying a later turn moved its reuse boundary (18,428 vs 25,067
reused tokens) and the greedy text, but not the deferral.

**The levers**, on the captured turn where every piece of evidence has been
read and the answer is due. "Short defer" = a tool call with fewer than 1K
tokens after the close; "model card" = temperature 1.0, top_p 0.95, top_k 20.

| condition | greedy | model card, 8 seeds: short defer |
|---|---|---|
| as sent (close C0, history kept) | defer | 2/8 |
| forced turns' reasoning removed from history | defer | 2/5 |
| all past reasoning removed | answer | 2/6 |
| `thinking_budget` 16,384 | defer (forced at 16,386, 202 after) | - |
| no budget / `reasoning_effort: "max"` | reasons past 32,768 (medium) / ~30K then answer cut at 32,768 (max) | - |
| close C1 ("…do whatever analysis is left directly in [the answer], calling a tool only if … truly missing") | answer | 3/8 |
| **close C2 ("My thinking time is over. I must now write the complete final answer from what I already have, without calling any more tools.")** | **answer** | **0/8** |

Under C2, two of the 8 sampled runs wrote the full report (20K+ tokens) and
then a trailing `todowrite` or `skill` call; the other 6 ended `stop`. On an
earlier turn whose module READMEs were genuinely unread, C2 greedy answered
without them and one sampled run still globbed for them.

**End to end**, the reviewer task from a fresh session:

| close, sampling | forced turns before the report | wall |
|---|---:|---:|
| C0, greedy (the job's config) | 3, 2 of them deferring | ~5 min |
| C2, greedy | 1, the report | 1m19s |
| C2, model card | 1, the report | 1m42s |
| the #297 build: C2 + default sampling, the job's config (no sampling fields) | 1, the report | 1m51s |

opencode forwards `temperature` only when the model definition declares
`"temperature": true`. Without it, a configured `top_p`/`top_k` reaches the
server alone and ignis answered `400 invalid_sampling_parameter`, because the
absent temperature resolved to 0.

Raw material (bodies, replays, the workspace generator, the proxy):
`.scratch/loop-repro/` in the clone that ran it.

## Finding

**Observed.** The deferral is the model's response to the close, not to the
budget's size or the history: 16K defers like 6K, no budget never answers, and
removing past reasoning leaves the sampled answer rate where it was.

**Observed.** The close text moves it. C2 answers greedy on both captured turns
and leaves no short deferral in 8 sampled runs (C0: always greedy, 2/8
sampled), and end to end it writes the report at the first forced turn.

**Observed.** Greedy is what turns one deferral into a loop. When the deferring
tool call returns what it returned last time, the next greedy turn is the same
turn. Under the model card's sampling a forced turn answers about 7 times in 10
even with C0, so a run of identical deferrals stops being likely.

**Inference.** A loop like these grows the prompt by the whole forced turn
each time (the reasoning stays in the tool loop), so it is also a path into
long-context failure modes: the lane-2 loop is what took that session to
337K and into #296.

## Implications

- Spec server/12 ships C2 as the close and the model card's sampling as the
  default for unset fields, with a fresh seed per seedless request.
- A measurement that needs greedy must say `temperature: 0`: the bench does
  (`crates/bench/src/client.rs`), the hq canary GPU test now does.

## Limits and unknowns

- n is small: 8 seeds per close on one captured turn of one synthetic task;
  the direction held greedy, sampled and end to end.
- C2 tells the model not to call tools. When evidence is genuinely missing it
  may answer without it; one sampled run still made the call it needed.
- Measured at `reasoning_effort: "medium"` only.

## Follow-ups

- The server cannot tell what sampling a request ran with: the request log
  carries none of it (out of scope in spec server/12).
- A stream cancelled mid-decode left no terminal request event in the log.
