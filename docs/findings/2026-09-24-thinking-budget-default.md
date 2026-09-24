# A 6,144-token thinking budget is the default: at `xhigh` it passes 26 of 32 coding runs to 8K's 23, a sixth faster, and bites on 3% of real agent turns

- Kind: experiment
- Status: current
- Observed: 2026-09-24
- Last verified: 2026-09-24
- Scope: server / thinking budget default, answer reserve, forced-close cost
- Related: [spec server/08](../specs/server/08-thinking-budget-default.md) (GitHub
  #265), [spec server/04](../specs/server/04-enable-thinking.md) §"Thinking budget",
  [reasoning_effort on coding tasks](2026-09-24-reasoning-effort-on-coding-tasks.md)
  (the first budget measurement), [copy drafting on agent traces](2026-09-24-copy-drafting-on-agent-traces.md)
  (the agent-turn dataset reused here), [batched decode width drift](2026-09-14-batched-decode-width-drift.md)
- Superseded by: none

## Question

Spec server/08 ships a thinking budget on by default, so a client that knows nothing
of the extension still gets an answer at the template's `xhigh`. Which value, and
how large must the answer reserve be?

The acceptance rule is the spec's: the value with the best pass rate at no worse
median wall time than 8K.

The budget must also hold on agent-shaped traffic, not only on the 8-task sweep. What
do the forced rounds cost a full batch?

## Evidence

**Setup.**
- RTX 5090, exclusive (`gpu-lock.sh`, timing holds).
- Branch `thinking-budget-265`, `make config` production flags: hq-e8-2b, `--spec
  dflash2 --draft-tokens 7`, 262,144 context, prefill chunk 1024.
- GPU healthy throughout: SM clock 2.6–2.7 GHz under load, device memory at most
  31.4 of 32.6 GB.
- The answer reserve was 1,500 for the first two holds and 2,048 after.
  No measured request's `max_tokens` came close enough for either to clamp.

**The coding sweep.** The 8 checkable tasks of the first finding, at `xhigh` and the
model card's thinking sampling (T=1.0, top_p 0.95, top_k 20), `max_tokens` 16,384, 8
at once. The budget travels per request; `0` is off.

- Driver: a copy of `.scratch/effort-2026-09-24/effort.py` that sends the budget
  explicitly. The original sends none for 0, which now means the server default.
- It also reads `thinking_budget_forced_at` off the finish chunk.

| budget | runs | pass | forced (pass) | hit the 16K cap | median tokens | median wall | mean wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 16 | 9 | 0 | 7 | 11,138 | 127.3 s | 114.3 s |
| **6,144** | 32 | **26** | 18 (13) | 0 | 6,472 | **58.5 s** | 49.8 s |
| 8,192 | 32 | 23 | 16 (8) | 0 | 8,352 | 70.7 s | 62.2 s |
| 12,288 | 16 | 14 | 8 (7) | 0 | 8,230 | 85.8 s | 86.3 s |

- 6K and 8K ran seeds 1–4, the others seeds 1–2.
- Per 16 runs: 6K scored 13 and 13, 8K 13 and 10.
- The first finding's session measured off at 8/16 and a 125.4 s median wall, and
  8K at 15/16 and 69.8 s. Pooled with it, 8K is 38/48 (79%) against 6K's 26/32 (81%).
- Every failure off-budget is a run still reasoning at the cap. With a budget, no
  run hits the cap.
- Raw data: `.scratch/effort-265/results/` in the `thinking-budget-265` worktree.

**Where the close lands.**
- `thinking_budget_forced_at` is the budget +1 to +6: the leaf's one-round lag, plus
  a speculative round's overshoot.
- The close runs 25 tokens (`completion − forced_at − answer`, 25–30 across forced
  runs).
- The answer after it, counted as content deltas at 3.62 characters each, has a
  median of 398 tokens (568 after a forced close), p95 1,084 and a maximum of 1,643,
  over 80 answers.

**What the forced rounds cost the batch.**
- A speculative round commits `accepted + 1` tokens. A request's tokens beyond that
  came from plain rounds: its own forced close, and its share of every round run
  plain while another lane was forced.
- Read from the server's `ignis.request.done` lines:

| leg (16 runs, 8 lanes) | forced | tok / spec round | tokens from plain rounds |
|---|---:|---:|---:|
| off | 0 | 3.04 | 0.0% |
| 6,144 | 9 / 9 | 3.01 / 2.95 | 1.45% / 1.62% |
| 8,192 | 8 / 8 | 3.00 / 3.05 | 1.03% / 1.15% |
| 12,288 | 8 | 3.05 | 0.83% |

**One recorded agent session.**
- The trace: `bench/traces/g4-load-trace.jsonl`, recorded from
  `bench/sim/g4-gate-session.json` (#118/#65). One main agent and 10 subagents
  audit the repository, and each request carries whole source files and 4 tools.
- Replayed at its own arrival offsets (2 s apart, all overlapping), greedy, as
  recorded, with the server restarted between legs.
- The bodies went as recorded: messages, tools, `max_tokens` 16,000/12,000.
  `ignis-bench replay` would re-send each line's body as one flat user message and
  record no channel split, so a small replay script did this
  (`.scratch/replay-265/replay.py`).

| leg | turns without content or tool call | forced | reasoning deltas / turn (mean, max) | main turn | session wall | aggregate |
|---|---:|---:|---|---|---:|---:|
| off | 0 | 0 | 1,274, 9,070 | 9,776 tok, then 2 tool calls | 77.4 s | 208.5 tok/s |
| default (6,144) | 0 | 1 | 997, 6,016 | forced at 6,146, then 2 tool calls | 60.8 s | 210.4 tok/s |

- The 10 subagent turns are token-identical in both legs. Each reasons 200–1,150
  tokens and answers with tool calls.
- Only the main turn reaches the budget, after the subagents have finished. So its
  plain rounds ran alone: 0.11% of tokens.

**Real agent turns.** Per-turn thinking length over the copy-drafting dataset:
13,102 real turns, 12,936 from qwen-code sessions on the local qwen3.8-27b and 166
from the #191 opencode session (`.scratch/replay-265/turn_thinking.py`).

| budget | turns it would reach | reasoning past it, share of all output |
|---|---:|---:|
| 6,144 | 383 (2.92%) | 9.2% |
| 8,192 | 220 (1.68%) | 6.0% |
| 12,288 | 86 (0.66%) | 2.7% |

- Median thinking per turn is 248 tokens, p90 2,748, p99 10,775, and the maximum
  60,676.
- The opencode turns never pass 2,603.

## Finding

**Observed.** At `xhigh`, 6,144 has the best pass rate measured at no worse median
wall time than 8K: 26/32 against 23/32 in the same build (79% pooled with the first
session), and 17% less median wall. 12,288 passes 14/16 at 85.8 s, which the rule
excludes. Unbudgeted `xhigh` again leaves 7 of 16 runs thinking at the cap.

**Observed.** The pass-rate gap between 6K and 8K is within the run-to-run spread.
Three 16-run legs of 8K scored 15, 13 and 10 across two sessions. The wall-time gap is
not: every 6K leg is faster. What ships is a value at least as good and faster.

**Observed.** A forced close costs its own run little and the batch less.
- The close is 25 plain rounds.
- In the sweep, over half the runs were forced, all at 8-wide concurrency, yet 1.0–1.6%
  of all tokens came from plain rounds. That is ~2–3% more lane-rounds than
  all-speculative.
- On the agent session one turn in 11 was forced, and aggregate throughput did not move.

**Observed.** On real agent traffic the default bites rarely, and where the output is.
- 2.9% of turns think past 6,144.
- Their reasoning past the budget is 9.2% of all output: the throughput a default
  budget returns.

**Observed.** The longest answer after a forced close ran ~1,650 tokens. The answer
reserve (`ANSWER_RESERVE`) is therefore **2,048**, not the ~1,500 the spec estimated.

**Inference.** The forced close cuts where agent output is least useful. A turn past
6K tokens of reasoning is the tail that stalls an agent. In this build half the
forced 8K runs failed anyway: the tasks that need that much thinking are the hard ones.

## Implications

- `--thinking-budget` / `IGNIS_THINKING_BUDGET` ships at **6144**
  (`crates/server/src/config.rs::DEFAULT_THINKING_BUDGET`), and `ANSWER_RESERVE` at
  **2048** (`crates/core/src/thinking_budget.rs`).
- `reasoning_effort: "max"` and `thinking_budget: 0` remain the ways to reason
  unbounded.
- The first finding's "`xhigh` + 8K budget, 15/16" is one session's sample. This
  finding's pooled 8K rate is the one to quote.
- A seeded run is reproducible only while the batch it runs in is. Changing the
  budget changes which lanes run together and when, and the widths drift
  ([batched decode width drift](2026-09-14-batched-decode-width-drift.md)). So legs
  of different budgets are separate samples, not paired ones.

## Limits and unknowns

- 8 short, single-function tasks, graded by hidden asserts.
  - Agent-turn quality under a forced close is not graded.
  - The real-turn table says how often the budget bites, not what the cut costs the
    answer.
- The replayed session exercises the first turn of each agent only (the trace is
  open-loop). Its one forced turn ran alone, so the concurrent cost comes from the
  sweep.
- The real-turn dataset was generated without a budget, at the effort qwen-code
  sent: mostly the default `xhigh`.
- Answer lengths are content deltas, a lower bound on tokens where the decoder holds
  back a partial character or marker. On this code-heavy text the two agree to within
  the chars-per-token ratio.

## Follow-ups

- If agent turns past 6K turn out to need their thinking (a graded agent benchmark),
  raise the default. The rule and the tables above are the ones to re-run.
- Re-measure with any model or template change: the default is a property of this
  model's `xhigh`, not of the mechanism.
