# At `xhigh`, Qwen3.8 thinks twice as long as at `medium` and half its coding answers never arrive inside 16K tokens; an 8K thinking budget takes it to 15 of 16

- Kind: experiment
- Status: current
- Observed: 2026-09-24
- Last verified: 2026-09-24
- Scope: server / `reasoning_effort` resolution (spec `server/04`), thinking cost, DFlash2 acceptance under sampling
- Related: [spec server/04](../specs/server/04-enable-thinking.md) (amended the same day:
  an effort the template does not take rounds up), [thinking budget default](2026-09-24-thinking-budget-default.md)
  (the budget's shipped value, measured on more seeds), [copy drafting on agent traces](2026-09-24-copy-drafting-on-agent-traces.md)
  (thinking is 73.5% of real agent output), `.scratch/sota-research-2026-09-24/SINTESI.md`
- Superseded by: none

## Question

Qwen3.8's chat template takes three efforts: `low`, `medium`, and `xhigh`. `xhigh`
is its default. OpenAI-vocabulary clients send `high`, which ignis now resolves to
`xhigh` (commit `1e8e7ee`). What does each effort cost a coding task in tokens and
wall time, and does the extra thinking buy correct answers?

## Evidence

**Setup.**
- RTX 5090, exclusive, main `4039aa9` (`make config` flags: hq-e8-2b,
  `--spec dflash2 --draft-tokens 7`).
- 8 self-contained Python tasks with hidden asserts. The asserts run on the
  answer's code block:
  - ISO-8601 durations
  - interval merge
  - O(1) LRU
  - fixing a buggy `lower_bound`
  - toposort with an alphabetical tie-break and cycle detection
  - an expression parser without `eval`
  - lexicographically-smallest LCS
  - glob matching with escapes
- 2 seeds each, the model card's thinking sampling (T=1.0, top_p 0.95, top_k 20),
  `max_tokens` 16,384, 8 requests at once.
- Driver `.scratch/effort-2026-09-24/effort.py`, raw results `results/sweep1.json`
  (every reasoning and answer text), server log `server.log`.

| effort | pass | hit the 16K cap | pass among finished | median tokens | mean tokens | median wall | DFlash2 tok/round |
|---|---:|---:|---:|---:|---:|---:|---:|
| `low` | 13/16 | 1 | 13/15 | 2,884 | 4,937 | 22.6 s | 4.33 |
| `medium` | 13/16 | 2 | 13/14 | 2,998 | 5,166 | 21.4 s | 4.41 |
| `xhigh` | **8/16** | **7** | 8/9 | 11,138 | 9,350 | 125.4 s | **3.03** |

**With a thinking budget** (`thinking_budget: 8192`, the scheduler forcing the model
card's close; same seeds, same server build otherwise; `results/sweep_budget8k*.json`):

| effort | pass | budget forced | hit the 16K cap | median tokens | median wall |
|---|---:|---:|---:|---:|---:|
| `xhigh` + 8K budget | **15/16** | 8 (all 8 pass) | 0 | 7,201 | 69.8 s |
| `medium` + 8K budget | 13/16 | 3 | 0 | 2,998 | 20.0 s |

The runs the budget did not touch are the unbudgeted runs token for token: same seeds,
the same failure (`lru` seed 1).

**The capped runs are not loops.** Every one is still reasoning at 16K tokens,
with 0.81–1.0 unique lines in its last 6,000 characters. It enumerates edge cases
against tests that do not exist ("Potential problem: If hidden test expects
`P1DT4H30M` valid, yes."), or re-derives a solution it already wrote.

## Finding

**Observed.**
- `xhigh` spends a median 3.7× the tokens of `low`/`medium` and 5.5× the wall time
  per task.
- Seven of its sixteen runs produce no answer inside 16K tokens.
- Among runs that finish, all three efforts are equally right: 8/9, 13/15 and 13/14.
- `low` and `medium` are indistinguishable at this sample size.

**Observed.** Under the thinking sampling, DFlash2 commits 4.3–4.4 tokens per round at
`low`/`medium` and 3.0 at `xhigh`, against 5.43 greedy on the quick-wins coding
prompts. `xhigh` is both longer and slower per token.

**Observed.** A forced close does not cost the answer. All eight `xhigh` runs that
reached 8K tokens of reasoning answered correctly once the block was closed for them.
Seven of those eight were the runs that never answered without the budget.

**Inference.** Unbudgeted, `xhigh` buys no correctness on tasks of this size
and costs 4–5× the latency. It also risks a turn with no answer when the client caps
`max_tokens`. That is the template's default, and it is what `high` now resolves to.

## Implications

- The effort a coding agent should get by default is `medium` (or `low`), not the
  template's `xhigh`. This is a server-default decision (`IGNIS_REASONING_EFFORT` /
  `--reasoning-effort`, today unset, so `xhigh`). It is **not changed** by this
  finding and waits for the owner.
- A thinking budget recovers the capped runs, and is built:
  - `thinking_budget` on the request, `--thinking-budget` / `IGNIS_THINKING_BUDGET`
    as the server default;
  - spec `server/04` §"Thinking budget", `crates/core/src/thinking_budget.rs`.

  `xhigh` + 8K budget gave the best pass rate measured (15/16) at about half
  unbudgeted `xhigh`'s wall time. `medium` stays ~3.5× faster. Which pair is the
  default is the owner's call. Both are measured-better than today's unset default:
  `xhigh` with no budget.
- Since #265 the server ships a budget by default: 6,144, chosen on 32 runs per value,
  where this session's 15/16 at 8K pooled to 79%
  ([thinking budget default](2026-09-24-thinking-budget-default.md)). The effort
  default is unchanged.
- Thinking is 73.5% of real agent output (copy-drafting finding). Cutting it is a
  bigger throughput lever than any drafter change measured so far.

## Limits and unknowns

- 16 runs per effort on 8 short, single-function tasks. Repo-scale agent turns (read,
  plan, edit across files) may need more thinking than these.
- One `max_tokens` (16,384). Uncapped, `xhigh` may finish more of its runs, but at the
  wall-time cost measured here or worse.
- Pass/fail is hidden asserts only. Answer quality beyond them was not graded.
- The DFlash2 tok/round column pools every request of an effort. At `xhigh` more
  requests run long at full batch width, so the columns are not a controlled comparison
  of acceptance.
