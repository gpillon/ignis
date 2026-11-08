---
status: accepted
amends: [ADR-0005, ADR-0006, ADR-0064, ADR-0070]
---

# ignis stands on its own

ignis was built to serve the agents that build it. ADR 0005 put its own kernels after ignis was "functional and
dogfooding". ADR 0006 made GPU testing exclusive because ninfer, the engine that served the developer's coding agents,
held the same RTX 5090. The 27B's catalog, the tracker and many measurements took those agents as their clients. On
2026-11-08 the owner ended that: ignis no longer serves the agents that build it, and ninfer no longer runs on the
development machine. It runs there again only for a test someone asks for by name. ignis stands on its own.

**What no longer holds, premise by premise, and what does:**
- **ADR 0005, the dogfood milestone.** Its kernel policy re-implements the ported kernels "after ignis is already
  functional and dogfooding". ignis is functional, and its own kernels do not wait on dogfooding: they come one
  measured family at a time under ADRs 0010, 0031 and 0051, as the sm_8x prefill (ADR 0080) and the GDN prefill (#514) did.
  "Guided by the north-star" now means ADR 0083's.
- **ADR 0006, the context.** It assumed that the agent building ignis was served from the card. Now that agent may
  run on the card or elsewhere. The decision stays: the development host has one RTX 5090 and one run fills it, so
  one process holds the card at a time, checked before every GPU run (`make gpu-status`, the preflight).
  - If the agent's own model is served from the card, the process holding the card is the agent itself, as for a
    fork's contributor with one consumer card, and the GPU step goes to the user: stop the serving engine, run,
    restart it. The old runbook, stop ninfer and restart it, is that recipe for one engine.
  - If the agent runs elsewhere, a process holding the card is someone else's: another session's, an experiment's,
    another tool's (the shared card of ADR 0049). The agent waits or asks, and never stops a process it did not
    start.

  The recipe is in `docs/agents/testing.md`, the rule in `AGENTS.md`.
- **ADR 0064, "Clients name `qwen3.8-27b`: the agents, opencode, the Playground".** The agents that build ignis are
  no longer among its clients. The premise holds for the clients ignis has, a user's agents and other
  OpenAI-compatible clients and the Playground, configured with the family's default id, so the alias stands for the
  reason its considered options give.
- **ADR 0070, "qwen-code orchestrates a campaign of issues".** That qwen-code was served by the local engine, and it
  no longer is. The tracker stands as 0070 sets it: whichever agent runs a campaign reads the same issues and
  sub-issues.

**What stays of ninfer:**
- **The vendor source of the kernels**: `kernel/vendor/` at its pin, under ADRs 0010 and 0051, and its `.ninfer`
  container, which the engine reads directly (ADR 0002; whether the extension changes is #463).
- **Its numbers, as a historical reference** read with their date: the September comparisons, the fixtures recorded
  on it (the canary oracle among them), and the evidence measured on agent sessions it or ignis served (ADRs 0029,
  0030, 0048, 0052, #267's recorded turns). They record what was measured then. No new one is taken, and its gates
  are retired (ADR 0082).
