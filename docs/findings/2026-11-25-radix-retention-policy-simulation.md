# A worth-ranked radix index keeps more reusable state than today's retention at the same slot budget, in simulation

- Kind: experiment
- Status: current
- Observed: 2026-11-25
- Last verified: 2026-11-25
- Scope: core / prefix reuse, retained state slots, `crates/core/src/radix.rs`
- Topic: kv-and-state
- Related: [GitHub #199](https://github.com/gpillon/ignis/issues/199), [ADR 0029](../adr/0029-cross-request-state-reuse.md), [prior art for decision prefixes](2026-09-26-prefix-reuse-prior-art-for-decisions.md)
- Superseded by: none
- Summary: On four synthetic agent traces at 4, 8 and 16 slots a worth-ranked radix index never reuses less than ADR 0029's retention (ties on a swarm) and up to 16 points more (77.3% to 93.5%, context fork, 16 slots); fork discovery adds nothing.

## Question

A hybrid model cannot resume at an arbitrary token: a GDN state exists only
where it was captured, and one costs about as much memory as 16K tokens of KV
(144 MiB against ~9 KiB per token; ADR 0029, README). So a SGLang-style radix
cache with state at every node is out, and the open question is the policy:
which states a fixed number of retained slots should hold. ADR 0029 answers
with fixed rules (the system block, one checkpoint at each generation opener,
at most two per conversation, checkpoints before prefixes, LRU within a kind).
Does a tree that ranks states by what they save do better at the same budget,
and can it lose?

## Evidence

`crates/core/tests/radix_sim.rs` replays four deterministic traces through a
model of ADR 0029's policy and through `RadixIndex` retention, counting the
prompt tokens each would prefill, with the same number of state slots. Run
`cargo test -p ignis-core --test radix_sim -- --nocapture --test-threads=1`.
Every trace shares one 13,000-token system head; a turn appends 200 reply and
1,500 tool tokens.

| Trace | Slots | Today reused | Radix reused |
|---|---|---|---|
| A swarm, 8 conversations x 8 turns (control) | 8 / 16 | 67.3% / 91.1% | 67.3% / 91.1% |
| B context fork: subagents handed the main's history from 3 turns back | 8 / 16 | 77.3% / 77.3% | 90.6% / 93.5% |
| C rewind: 4 conversations, 30% chance per turn from turn 5 of resuming an earlier turn | 8 / 16 | 87.4% / 87.4% | 88.2% / 89.4% |
| D mixed: A, B and two C interleaved | 8 / 16 | 76.7% / 84.9% | 76.8% / 88.9% |

The same table at 4 slots is in the test output; radix is not lower in any
cell.

Worth of a state is `(tokens - tokens of the nearest state below it on its
path) x (1 + times resumed from) x 0.5^(idle ticks / 64)`, and the newest
never-resumed states, half the budget, are spared. Two earlier versions failed
the same traces and are why each term is there:

- ranking by walks through a node, not resumes, evicted a conversation's newest
  opener first: its ancestors absorb its saving. The swarm control fell to
  74.3% reused against 91.1%.
- re-publishing the system block replaced its history, so a proven state
  looked new and used up the shield.

## Finding

Observed, in this model: retention by worth does not lose to the fixed rules at
equal slots, ties where every boundary is one the fixed rules predict (A), and
gains where reuse is off the opener (B, D at 16 slots). The gain is where a
state the fixed rules drop (a conversation's third-latest opener) is still the
nearest shared head of a later request.

Observed: tree-discovered forks (a prompt leaving a path the tree continues
down, with no state there) gained nothing on these traces, cost a prefill cut
each, and cost reuse: prefill rose 25.5% on C at 8 slots while forks took the
probation shield, and 2.4% on D at 8 slots once they were made speculative and
spared none. Every fork in the traces sits on an opener that is already
checkpointed.

Inferred, not shown: the gain on a real agent session depends on how often it
resumes from states older than the latest two. Fork discovery's value depends
on the shape of shared heads no boundary predicts, which these traces do not
contain.

## Implications

`RadixIndex` is the candidate for #199's retention score. Wiring it replaces
ADR 0029's "at most two per conversation, checkpoints before prefixes" with
one ranking, so it needs an amendment and the GPU profile. Fork discovery
stays off until a recorded trace shows a fork it would find.

## Limits and unknowns

- The traces are assumptions, not recorded sessions, and "today" is a model of
  the policy, not `ConcreteScheduler`.
- Not modelled: KV page pressure, requests in one prefill batch, the KV-RAM
  and disk tiers, class (Interactive/Agent) and TTL, and the ~19 ms an extra
  prefill cut costs on the 27B.
- Tick-based decay uses request count as time; wall time was not modelled.
- Worth ignores state size (all slots are one size here); a Flash-Next slot
  differs.

## Follow-ups

- Replay a recorded agent session (spec bench/01, trace replay) through both
  policies and calibrate `half_life` and the shield share.
- Wire `RadixIndex` behind a flag in `ConcreteScheduler` (#199).
