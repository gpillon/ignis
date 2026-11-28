---
status: accepted
---

# The speech model waits on the host and borrows free KV pages

Users dictate prompts into the Playground and into their own coding agents, through an OpenAI-compatible route, and
the audio never leaves the machine. The speech model, Qwen3-ASR, is a third model kind: it takes audio and gives
back its transcript, never converses, and is loaded beside whichever text or image model serves, never reached by a
switch and never resident in VRAM. The feature is public, so its placement must hold on every card class ignis ships
and on Linux, and on a 16 GB card the text model takes the whole card. The CPU was
measured before anything was decided ([finding](../findings/2026-10-24-qwen3-asr-on-cpu.md)): every variant ran
between 2 and 10 times real time, go only through incremental Realtime transcription by the bands proposed beforehand
and confirmed by the owner after the numbers (at least 10 times is go, below 2 is no-go).

**Loaded beside the other model.** The speech model keeps serving during a switch, beside the image model and while
the text model is idle-unloaded. ignis then holds a second loaded model: the text or image model on the device, the
speech model on the host. ADR 0050's "one model at a time" still holds between the text models and the image model.

**On the host.**
- The weights wait in their memory-mapped file, in the OS page cache. Pinned RAM is an opt-in config field.
- The load repacks weights into private memory (ADR 0068), so the model is loaded at the first signal, prefetched
  like the loan, and unloaded after a period with no transcription, a config field. Nothing is held while it is
  unused.
- The CPU executor is the baseline: it runs every transcription the GPU does not. Its minimum is AVX2. It is one
  compute worker: Realtime sessions go first, batch requests fill the gaps, and past a limit of open sessions or of
  waiting requests (config fields) a request gets `503` with `Retry-After`.

**The GPU is borrowed, and only from the KV pool's free pages.** The loan is decided, not built: phase 3, #356,
parked. Until then every transcription runs on the CPU and the `speech.loan_*` fields are not read. On a signal (a Realtime session opening, a mic
press, a batch request), the speech model asks the text model's KV pool for free pages. It reserves them as a
sequence would, and its weights, its own KV and its workspace live on them. The loan reserves within the planned
pool, so ADR 0030 holds: nothing is allocated on the device while serving. The weights it uploads are the same GGUF
the CPU reads, so a model is one download.
- The loan is held for a grace period after its last transcription, a config field. It is given back at once if the
  text model needs pages and no transcription is running; a running transcription finishes first, bounded by its
  deadline.
- A loan never takes a live reservation and never pushes retained state down. When the free pages are not enough,
  the transcription runs on the CPU.
- There is no loan without a text model's KV pool: with the image model loaded, during a switch, or with the text
  model unloaded, transcription runs on the CPU.
- The GPU bar is latency: a 60 s dictation under 1 s on the 5090, and the 27B's decode no more than 10% slower
  during a transcription. We optimise only if one of the two fails. Transcription has priority over text decode.

**Unavailable is a state with a reason.** The executor detects AVX2 at run time (ADR 0068). If the CPU executor
cannot run and the loan does not fit, a transcription waits a short while, then gets `503` with `Retry-After`; on a
card where the loan rarely fits, the speech model is shown unavailable. `speech.model: off` or a missing file is
unavailable too, with its own reason.

**The text model's lifecycle is unchanged, with three exceptions:**
- A switch waits for a running loan, bounded by the loan's deadline.
- Transcriptions do not reset the text model's idle clock: a transcription is not use of the text model.
- Opening a Realtime transcription session starts a background reload of an unloaded text model: the dictation is
  going to land in its prompt.

## Considered Options

- **Always resident on the GPU.** Rejected: it holds VRAM while nobody dictates (owner: "do not keep the speech
  model in VRAM"), and a 16 GB card has no room for a second model.
- **Reached by a switch, with the whole card, as the image model is (ADR 0050).** Rejected: it costs a reload of the
  text model on each side of every dictation, and that dictation lands in the text model's prompt.
- **A separate process** (a second server beside ignis). Rejected: device memory belongs to a process, so a second
  process cannot borrow pages of ignis's pool, and ADR 0030's budget is whole-process; and it would duplicate what
  the speech routes reuse: the API key, the deadlines, `/metrics` and the Playground's socket auth.

## Consequences

- ignis serves audio in. Audio out is still not served.
- Host RAM: 1.4-3.1 GB resident while the model is in use, its KV at a 4,096-token context included, as measured on
  llama.cpp. Unloading frees it all, and the first dictation after an unload waits for the load, 1.4-1.8 s for the
  larger model, also measured on llama.cpp.
- A CPU transcription uses all but one physical core by default (`speech.threads`). The larger model's decode is
  bound by memory bandwidth, so it competes with whatever else uses host memory. That default is accepted only if
  text decode slows by less than 10% during a CPU transcription, measured on the GPU.
- A text admission can wait behind a running loan, for up to that transcription's deadline.
- A 16 GB card whose CPU lacks AVX2 may have no speech model at all.
- The speech model is a catalog entry, fetched by the `model download` command, never downloaded by itself.
- The CPU serves alone first, then with `server_vad`; the loan is the last phase.
