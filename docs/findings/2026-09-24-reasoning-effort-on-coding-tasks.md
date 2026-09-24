# At `xhigh`, Qwen3.8 thinks twice as long as at `medium` and half its coding answers never arrive inside 16K tokens

- Kind: experiment
- Status: current
- Observed: 2026-09-24
- Last verified: 2026-09-24
- Scope: server / `reasoning_effort` resolution (spec `server/04`), thinking cost, DFlash2 acceptance under sampling
- Related: [spec server/04](../specs/server/04-enable-thinking.md) (amended the same day:
  an effort the template does not take rounds up), [copy drafting on agent traces](2026-09-24-copy-drafting-on-agent-traces.md)
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

**Inference.** For a coding agent, `xhigh` buys no correctness on tasks of this size
and costs 4–5× the latency. It also risks a turn with no answer when the client caps
`max_tokens`. That is the template's default, and it is what `high` now resolves to.

## Implications

- The effort a coding agent should get by default is `medium` (or `low`), not the
  template's `xhigh`. This is a server-default decision (`IGNIS_REASONING_EFFORT` /
  `--reasoning-effort`, today unset, so `xhigh`). It is **not changed** by this
  finding and waits for the owner.
- A thinking budget would recover the capped runs: each had a correct plan long
  before 16K. The Qwen way closes the block with a forced `</think>`. The constrained
  decode's permitted-set schedule (`crates/core/src/constrained.rs`) already forces
  single tokens, so a budget can be a schedule of singletons armed at N reasoning
  tokens. That was not built here.
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
