---
status: accepted
---

# The north-star is the best local engine for coding, for agentic work and as a companion

ignis aims to be the best local engine for three uses, on one NVIDIA card in the user's own machine, from a 12 GB
consumer card up (one lane on the 12 GB class, ADR 0098), from one binary. It no longer serves one model to coding
agents alone: it also serves Flash-Next,
the image model, two speech models and a diarizer, and the Playground has views for more than the chat. The uses:

- **coding**: decode speed, DFlash2 speculation, long contexts, reasoning that finishes;
- **agentic work**: an agent and its subagents on lanes of their own at once, saturating the GPU in prefill and
  decode, state reused across turns, tool calls, the Responses API over a WebSocket;
- **a companion**: the utilities a person uses directly, each a view of the Playground over the same `/v1` routes a
  client calls. **Transcribe** writes a conversation down with its speakers, from the microphone, a call or a file,
  and the text model keeps its notes. **Decide** answers many questions over one piece of evidence. **Image** makes
  and edits pictures. The chat takes pictures in its prompt.

The Monitor and Server views are how the engine is run, not one of its uses.

## Consequences

- Correctness stays a floor never traded away, and within each use performance is the tie-breaker above it
  (ADR 0005). A trade-off between two of the uses is the owner's to make.
- Saturating the GPU is not the whole objective, since a companion utility shares the card with the text model or
  leaves it to it: the speech models run on the CPU (ADR 0068), the image model is a model switch away (ADR 0050),
  and an idle model unloads (ADR 0049).
