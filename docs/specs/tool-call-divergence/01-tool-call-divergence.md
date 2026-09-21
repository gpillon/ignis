# 01 — tool-call decisions diverge from the reference on token-identical prompts

GitHub: #173

## Observation (2026-09-15)

20 prompts at hq-e8-2b, DFlash2/7, greedy, thinking off, `max_tokens`
1024, one engine on the card at a time: the 11 G4 trace prompts and 9 built
from repo files, all carrying the trace's 4 tools and ending with "answer
directly, do not call any tools, write a long answer". After #172
(df40ef9) the prompt token counts are **equal on 20/20**.

| | stops at a tool call | runs to `max_tokens` |
|---|---:|---:|
| reference (ninfer-serve, pin a00648cb build) | 18/20 | 2/20 |
| ignis df40ef9 | 7/20 | 12/20 (+1 `stop`) |

Before #172, with a different prompt, ignis gave 7/20 (and 5/20 with
client-sent `strict`), so the split does not move with the prompt fix.
Without tools (prompts equal token for token), both engines run 19–20/20 to
`max_tokens`, and pooled tok/round is 0.965 of the reference.

Records: `.scratch/diag-160/twenty/` (`prompts.jsonl`, `drive.py`,
`run.sh`, `run-ignis-172.sh`, `ref-tools.json`, `ignis172-tools.json`,
server logs), and `.scratch/diag-160/five/` for the earlier 5-prompt runs.

## Why it matters

The agent lanes are tool-driven. An engine that decides "call a tool" vs
"answer" differently on the same prompt changes agent trajectories, and
every tools A/B (acceptance, decode) compares different generation lengths.

## Known context

- #160: ignis hq-e8-2b's greedy first token differs from the reference's on
  G5 depth prompts, while ignis bf16 matches the reference token for token.
- #161, #153: spec-on vs spec-off near-ties at hq; the vendored hq small-T
  tile's pick depends on masked columns.
- On the 5 real prompts without tools, the reference at hq vs its own bf16
  diverges within ~5–30 characters, so first-token divergence alone is not
  proof of a defect. A systematic 18 vs 7 split in one direction is.

## Questions to answer

1. Where does each engine's text first diverge on these 20 prompts, and is
   the divergence at the tool-call decision itself (`<tool_call>` vs prose)?
2. Does ignis at **bf16** make the reference's hq decisions, the reference's
   own bf16 decisions, or neither? (Size the reference's bf16 KV explicitly:
   `--kv-capacity auto --max-concurrency 8` spilled into shared GPU memory.)
3. Does ignis with speculation off make the same decisions as with it on?
4. At the first diverging position: the top-2 logit margin on each engine
   (in-process for ignis; neither server exposes logprobs).

## Acceptance

- A classification of the split: hq target-path numerics (same root as
  #160/#161), speculation, sampler/stop handling, or a remaining prompt
  difference. The evidence for that call is recorded.
- If it is a defect: a follow-up issue naming the layer, with a reproducer
  on one of the 20 prompts.
