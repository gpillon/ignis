---
status: accepted
---

# Exclusive GPU testing (no concurrent engines)

One process holds the development card at a time: one run fills the RTX 5090, the weights of two full-size engines
alone exceed its memory, so two cannot coexist on it, and the run that loses the card dies with no diagnostic.
Exclusivity is a test-harness concern, not a runtime feature of the engine: the engine has no multi-tenant story and
does not refuse to start because another process uses the GPU, and `ignis-server` run by hand takes no lock.

- **A preflight.** The guard is a preflight check in the `bench`/test harness, run before every GPU run
  (`make gpu-status`, the preflight): it detects that another process holds the GPU and refuses or waits, so
  running two engines by accident is explicit and never silent.
- **A GPU lock**, one per card, records who holds it, and every way onto the card through `make` takes it: the GPU
  profile for its whole run (or for the few tests it is given), a server started through `make`, the kernel leaf's
  op tests, and any command run through `make gpu-run`. A server start and the guard refuse at once while someone
  else holds it, naming the holder; a test, `make gpu-run` and the profile wait for it up to a limit
  (`GPU_LOCK_WAIT`), then fail naming it. A holder that has exited, or whose pid now names another process, holds
  nothing, so a run killed before it could clean up leaves no lock standing. Two runs therefore no longer race
  between the check and the start, and an agent can run its own GPU tests while others share the card.
- **The lock is added to the preflight, not put in its place**: the profile's pass marker and the process check
  stay, since the engine's test helper reads the marker and a worktree on an older commit knows only those two.
- **Who stops what** depends on where the agent that runs the test is served, on the card or elsewhere. If its own
  model is served from the card, the process holding the card is the agent itself, as for a fork's contributor with
  one consumer card, and the GPU step goes to the user: stop the serving engine, run, restart it. If the agent runs
  elsewhere, a process holding the card is someone else's, another session's, an experiment's or another tool's (the
  shared card of ADR 0049): the agent waits or asks, and never stops a process it did not start. The recipe is in
  `docs/agents/testing.md`.

## Considered Options

- **Two engines side by side**: possible only with reduced engine sizes, which is not a goal.
