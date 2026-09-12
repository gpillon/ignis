# GitHub #144 — the canary budget decision (2026-09-13)

`canary-default-flags.json` and the run below are `ignis-bench canary`
against an `ignis-server` started with **its own default flags** — nothing
passed but the artifact, the model id and the bind address, so
`enable_thinking` is on and the KV format is `hq-e8-2b`. Free RTX 5090.

```
canary rust-hello   sane=true deterministic=true answer=reached
canary rust-sort    sane=true deterministic=true answer=thinking-only
canary math-greedy  sane=true deterministic=true answer=thinking-only
canary explain-reverse sane=true deterministic=true answer=thinking-only
self-consistency: PASS
```

The three canaries this issue reported as `sane=false ... empty output` are
exactly the three the report now labels `answer=thinking-only`, and the one
that passed before is the one that still reaches an answer. Nothing about the
engine changed between those two readings — 64 tokens still go to the
reasoning channel on those three prompts. What changed is that the harness
reads the thinking channel (#137) and now *states* the distinction instead of
leaving it to be inferred from an empty field.

In the record each result carries both channels, so `first: ""` beside a
populated `first_reasoning` is the same fact in JSON form:

| canary | `first` | `first_reasoning` (head) |
|---|---|---|
| `rust-hello` | `It defines a Rust …` | `We need answer user's request: …` |
| `rust-sort` | *(empty)* | `We need answer user's question: "What does \`let v = vec![3,1…` |
| `math-greedy` | *(empty)* | `We need answer user's simple request. Need compute step by s…` |
| `explain-reverse` | *(empty)* | `We need answer user: "Explain in one sentence what \`x.revers…` |

## The budget stays at 64

`canary::CANARY_MAX_TOKENS` keeps its rationale on it. The canary's job
(ADR 0007) is catching a degenerate engine — empty, NUL-ridden, or
stuck-repeating generation — and that shows on either channel: 64 tokens of
reasoning prove it as well as 64 tokens of answer. Checking the *answer* is
the canary oracle's job, which does it properly (teacher-forced, token by
token, against a recorded fixture, thinking off on purpose — ADR 0014); a
bigger budget here would duplicate that badly, trading token agreement for
prose sanity, and every raise is paid at every gate.

Raise it only with an answer to "what does this catch that the oracle does
not?".
