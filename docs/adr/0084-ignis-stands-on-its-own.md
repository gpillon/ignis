---
status: accepted
---

# ignis stands on its own

ignis was built to serve the agents that build it, and it no longer does. ninfer, the engine that served the
developer's coding agents from the development machine's card, no longer runs there: it runs again only for a test
someone asks for by name. The agents that build ignis are not among its clients, which are a user's agents, other
OpenAI-compatible clients and the Playground.

## Consequences

- ninfer stays the vendor source of the kernels, `kernel/vendor/` at its pin (ADRs 0010 and 0051), and of the
  `.ninfer` container, which the engine reads directly (ADR 0002).
- ninfer's numbers stay as a historical reference, read with their date as what was measured then: the September
  comparisons, the fixtures recorded on it (the canary oracle among them), and the evidence measured on agent
  sessions it or ignis served (ADRs 0029, 0030, 0048, 0052, #267's recorded turns). No new one is taken, and its
  gates are retired (ADR 0082).
