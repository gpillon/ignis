# Reasoning loops on a long enumeration are the model's, not ignis's

- Kind: experiment
- Status: current
- Observed: 2026-10-12
- Last verified: 2026-10-12
- Scope: serving / sampled reasoning on qwen3.8-27b (thinking budget, Playground agents), engine options (YaRN, DFlash2, KV format)
- Topic: reasoning-and-context
- Related: [ADR 0048](../adr/0048-an-end-of-turn-inside-the-reasoning-closes-it.md), [spec server/12](../specs/server/12-a-close-that-answers.md), [hq-e8-2b KV capacity](2026-09-11-hq-e8-2b-kv-capacity.md)
- Superseded by: none
- Summary: About a third of first turns on the Playground's Rome prompt loop at its defaults, with YaRN, DFlash2 and the hq codec off and on ninfer alike: the NVFP4 27B's loop at temperature 1, no penalty, `xhigh`; `presence_penalty` 1.5 halves it.

## Question

In the Playground, the prompt "dimmi l'intera storia di roma in forma estesa. molto
lunga" (every tool but web, the Playground's default settings) produced reasoning
that degenerated into loops: verbatim repetition (`"the "Pio…," the "Pio…,"`), a
single token filling every noun slot (`the *caput* of the *caput*`), and counting
(`2026? "2026? " 2025? …` down to 1184). The turns hit the thinking budget, and
agents that hit it returned no answer. It happened on the abliterated artifact first,
then on the stock one. Is any of ignis's engine options, or ignis itself, the
cause?

## Evidence

**The incident (server log, 2026-10-12 01:32-02:16Z, lost to a later restart; read
before it was).** On the abliterated artifact, the interactive turn (1,986 prompt
tokens) had its thinking closed by force at 14,340 of a 14,336 budget: the
Playground sends `max_tokens` 16,384 and no budget, so the budget is
16,384 − 2,048. Agent turns did the same, and five of them then ended `length` with
"the output limit was reached before any answer". After a switch, the stock artifact
did the same on the same 1,986-token prompt.

**The request.** `body.ts` rebuilds the Playground's own request from `web/src`:
`toolExtras(ALL_TOOLS minus web)` and `DEFAULT_SETTINGS` (temperature 1, top_p 0.95,
max_tokens 16,384, effort `xhigh`, no budget, `top_k` 20 by the server's default).
It renders to 1,996 prompt tokens on ignis and 1,986 on ninfer. Every run is one
non-streaming chat completion with a fixed `seed`, up to 8 at once, so the measure is
the first turn only.

**What counts as a loop.** A count of repeated 6-word grams misses counting loops,
where every gram is new. `analyze.py` scores each run by its most compressible
3,000-character window of reasoning and of answer (zlib ratio, step 500). Normal
text scores 0.30-0.54. The clear loops score 0.004-0.10, and the counting loops sit
around 0.2. The final call was made by eye on each candidate's tail.

| arm | server | loops |
|---|---|---:|
| serving default: `yarn:2`, 524,288, `hq-e8-2b`, DFlash2 k=7 | ignis (stock, fresh start, no switch) | 5/16 (+1 borderline) |
| same + `presence_penalty` 1.5 | ignis | 2/8 (5/8 forced closes, against 4/8 without it) |
| no system prompt, no tools | ignis | 2/8 (+1 borderline) |
| `ROPE_SCALING=none`, 262,144 | ignis | 5/16 (+1 borderline) |
| `ROPE_SCALING=none`, 40,960, **KV `bf16`**, **`SPEC=off`** | ignis | 3/6 |
| owner's `qwen3.8-27b-nvfp4full-ram-dflash2-262k.bat` (same v2 artifact, `hq-e8-2b`, DFlash2 k=7, native 262,144, no scaling) | ninfer `a00648cb` | 4/12 |

The ignis server was the server-view preview build `c1c929e`, whose `kernel/`,
`crates/core` and `crates/runtime` are identical to main `efeb6cb`. The seeds were
101-116 on the 16-run arms and 101-108, 101-106 and 101-112 on the others. Loops
start 16-93% of the way into the reasoning, and some are in the answer instead (an
answer repeating "usurpatori, usurpatori, …"). Base seed 109 broke out of its own
loop: "I'm repeating. Let me stop." On ninfer, three of the four loops end `stop`
with no answer at all: the model's EOS inside its reasoning ends the turn there.

Raw material is in the main checkout's `.scratch/loops/` (not tracked): `body.ts`,
`body.json`, `body-ninfer.json`, `run.py`, `analyze.py`, the responses
`out*/<arm>/<seed>.json`, the three ignis server logs, `ninfer-serve.log`, and
`NOTES.md` with the incident's request ids.

## Finding

Observed:

- On this prompt, at the Playground's defaults, roughly a third of first turns loop.
  The rate is the same with YaRN off, and still present with speculation off, the
  KV codec replaced by BF16 and YaRN off together. On the reference engine with the
  same artifact it is the same again.
- Every observed loop shape (verbatim, single-token, counting, and a deferral loop
  of the form "I'll write the tool call now.") appears on both engines.
- `presence_penalty` 1.5 halves the loops (2/8 against 4/8 on the same seeds) and
  lengthens the reasoning (35-43k characters against 21-38k), so more turns reach
  the forced close.

Inferred:

- The cause is not ignis code, and not any engine option the arms turned off: YaRN,
  DFlash2, the hq-e8-2b codec. Nor is it abliteration (the stock artifact loops), the
  model-switch path (a fresh start loops), or the server-view build (ninfer loops).
- What every arm shared is the NVFP4 artifact and the vendored kernels and sampler.
  The loop belongs to Qwen3.8-27B as quantized here, sampled at temperature 1 with no
  penalty and `xhigh` effort, on a very long enumeration.
- The tool prompt is not shown to matter: 2/8 without it against 4/8 with it, on the
  same seeds, is noise at this n.

## Implications

- A loop report on a long enumeration is not an engine regression by default. Do not
  re-run these arms: compare a new report's rate against this table first.
- ignis's [ADR 0048](../adr/0048-an-end-of-turn-inside-the-reasoning-closes-it.md) redirect
  and the forced close turn the ending that leaves a ninfer turn with nothing (EOS
  inside the reasoning) into an answer. Inside a loop, that close is the only thing
  that ends the turn.
- After a forced close an agent has `max_tokens` − 14,336 = 2,048 tokens left, the same
  for any `max_tokens` up to 34,816. A tool call cut there is dropped whole
  (`crates/server/src/toolcall.rs`), and the parent sees "(The agent returned no
  text.)". An explicit `thinking_budget` widens that reserve, and a lower effort
  shortens the reasoning that loops.
- `presence_penalty` trades loops for longer reasoning. It is not a fix for agents.
  `frequency_penalty` grows with each token's count, which code legitimately repeats,
  so it is not a candidate for tool-calling agents (an inference from the sampler's
  formula; not measured).
- Unrelated to loops, but found on the way: the serving default `yarn:2` (Makefile
  default since `6c1541b`, 2026-09-29) has not been through the 99% gate, whose G5
  legs ran at `MAX_CONTEXT=262144 ROPE_SCALING=none` (`mk/config.mk:124`).

## Limits and unknowns

- One prompt and one prompt shape. The rate on other long enumerations, and on code,
  is not measured.
- Only first turns. Agent turns deep into a tool loop (contexts up to ~50K), where the
  incident's identical `run_js` calls happened, were not re-run.
- n is 6-16 per arm: enough to say the loop does not vanish, not to rank arms within
  a factor of two.
- The model itself is not separated from its NVFP4 quantization or from the vendored
  sampler, which ignis and ninfer share. Only a BF16 run on another stack (vLLM,
  Transformers) would do that.
- A lower temperature was not tried.
- Loop calls on borderline runs are by eye.

## Follow-ups

- If the model-versus-quantization question matters: the same body, 16 seeds, on the
  BF16 checkpoint in vLLM or Transformers.
- Whether `yarn:2` serving needs a G5 leg of its own.
- Whether an agent's reserve after a forced close should scale with its task, rather
  than stay at 2,048 tokens.
