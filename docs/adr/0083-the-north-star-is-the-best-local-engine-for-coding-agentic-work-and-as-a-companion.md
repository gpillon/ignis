---
status: accepted
amends: [ADR-0005]
---

# The north-star is the best local engine for coding, for agentic work and as a companion

ADR 0005 (2026-09-02) named the north-star that decides its performance trade-offs: "the best local coding engine:
max performance + agent parallelism saturating the GPU in prefill and decode". ignis then served one model, the 27B,
to coding agents. It now also serves Flash-Next, the image model, two speech models and a diarizer, and the Playground
has views for more than the chat. On 2026-11-08 the owner widened the north-star.

**ignis aims to be the best local engine for three uses, on one NVIDIA card in the user's own machine, from a 16 GB
consumer card up, from one binary:**
- **coding**: decode speed, DFlash2 speculation, long contexts, reasoning that finishes;
- **agentic work**: an agent and its subagents on lanes of their own at once, state reused across turns, tool calls,
  the Responses API over a WebSocket;
- **a companion**: the utilities a person uses directly, each a view of the Playground over the same `/v1` routes a
  client calls. **Transcribe** writes a conversation down with its speakers, from the microphone, a call or a file,
  and the text model keeps its notes. **Decide** answers many questions over one piece of evidence. **Image** makes
  and edits pictures. The chat takes pictures in its prompt.

The Monitor and Server views are how the engine is run, not one of its uses.

What else ADR 0005 decided stands: correctness is a floor never traded away, and above it performance is the
tie-breaker within each use. Saturating the GPU is no longer the whole objective, since a companion utility shares the
card with the text model or leaves it to it: the speech models run on the CPU (ADR 0068), the image model is a model
switch away (ADR 0050), and an idle model unloads (ADR 0049). A trade-off between two of the uses is the owner's to
make.
