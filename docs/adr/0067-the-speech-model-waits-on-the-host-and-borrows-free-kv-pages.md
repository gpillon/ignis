---
status: accepted
amends: [ADR-0049, ADR-0050]
---

# ADR 0067 — the speech model waits on the host and borrows free KV pages

## Status

Accepted (2026-10-24, owner): the grilling session on speech recognition of
2026-10-23/24, decisions Q1-Q52 and the validated Playground mockup (Q17);
this ADR records Q8-Q9, Q14, Q26, Q28-Q35, Q39-Q40, Q44, Q46, Q53 and
Q56. Specs
`docs/specs/speech/`.
- **Amends** ADR 0050's "one model at a time", for the speech model only: it
  is loaded beside the text or image model. The image model is still never
  loaded beside a text model.
- **Amends ADR 0049**: a transcription is not use of the text model, and
  opening a Realtime transcription session loads an unloaded text model.
- **Amends** `GLOSSARY.md`'s *ignis* and *Model kind*.
- **Contradicts, on purpose,** the OpenAPI description and the README, which
  list "audio in or out" and "more than one loaded model" as not served.
  Spec speech/02 amends both, and the test that pins them.

Sources: the session's decision log, kept untracked in
`.scratch/speech-grill-2026-10-23/` (local to the owner's clone); the
measurements in `H:\speech-study\results\`; and the finding
[Qwen3-ASR on the CPU](../findings/2026-10-24-qwen3-asr-on-cpu.md).

## Context

Users want to dictate prompts into the Playground and into their own coding
agents (Q7). That needs an OpenAI-compatible route on ignis, and the audio
never leaves the machine. The model is **Qwen3-ASR** (Apache-2.0), in two
sizes, 0.6B and 1.7B. Each is an audio encoder and a Qwen3 decoder; it emits
13 audio tokens a second, holds 112 KiB of KV per token, and takes up to
20 minutes of audio.

The feature is public (owner, 2026-10-23), so its placement must hold on
every card class ignis ships (32 GB, 16 GB, 96 GB) and on Linux:
- On a 16 GB card the text model takes the whole card. No second model fits
  there (owner direction).
- ADR 0050 reached the image model by a switch, with the whole card. A
  switch costs a 13-33 s reload on each side. A dictation lands in the text
  model's prompt, so it cannot pay that.
- ADR 0030: serving allocates nothing on the device. The KV pool takes what
  the budget leaves (ADR 0045's resident policy), or is reserved first
  (offloaded).

The CPU was measured before anything was decided (the finding). llama.cpp,
warm, 10 threads on the owner's i9-10900K:
- 0.6B Q8_0 runs at 6.7x real time;
- 1.7B Q4_K_M runs at 4.8x;
- 1.7B Q4_K_M runs at 3.0x on a 4-core container.

On the owner's Italian dictations 1.7B Q4_K_M has 3.7% WER, and 2.4% with
context. The go bands were proposed before the numbers, and the owner
confirmed them after seeing them (Q34): at least 10x is go, 2-10x is go only
through incremental Realtime transcription, below 2x is no-go. Every
variant fell in 2-10x.

## Decision

**A third model kind, loaded beside the other.** A speech model takes audio
and gives back its transcript. It never converses (Q8, Q9). It is not
reached by a switch: it is loaded beside whichever text or image model
serves. It keeps serving during a switch, beside the image model and while
the text model is idle-unloaded (Q40). This is the first time ignis holds a
second loaded model: the text or image model on the device, and the speech
model on the host.

**Never resident in VRAM.**
- The weights wait in their memory-mapped file, in the OS page cache.
  Pinned RAM is an opt-in config field (Q31).
- **Loaded at first use, unloaded when idle** (Q53). The Q4_K repack copies
  weights into private memory, so the model is loaded and repacked at the
  first signal, prefetched like the loan (1.4-1.8 s for 1.7B, measured on
  llama.cpp). It
  is unloaded after a period with no transcription (a config field, 300 s by
  default, as `model.idle_unload`). Nothing is held while it is unused.
- The **CPU executor** is the baseline: it runs every transcription the GPU
  does not (Q28/Q29). Its minimum is AVX2 (Q28).
- The executor is one compute worker. Realtime sessions go first, window by
  window, and batch requests fill the gaps. Past 4 open sessions or 4
  waiting requests (config fields), a request gets `503` with `Retry-After`
  (Q44).

**The GPU is borrowed, and only from the KV pool's free pages** (Q30, owner
direction). On a signal (a Realtime session opening, a mic press, a batch
request), the speech model asks the text model's KV pool for free pages. It
reserves them as a sequence would, and its weights, its own KV and its
workspace live on them. The loan reserves within the planned pool, so ADR
0030 holds: nothing is allocated on the device while serving. The weights it
uploads are the same GGUF the CPU reads, so a model is one download (Q46).
- The **loan** is held for a grace period after its last transcription: a
  config field, 30 s by default.
- It is given back at once if the text model needs pages and no
  transcription is running.
- A running transcription finishes first, bounded by its deadline (Q32).
- A loan never takes a live reservation and never pushes retained state
  down. When the free pages are not enough, the transcription runs on the
  CPU (Q30).
- There is no loan without a text model's KV pool. With the image model
  loaded, during a switch, or with the text model unloaded, transcription
  runs on the CPU (Q40).
- The GPU bar is latency: a 60 s dictation under 1 s on the 5090, and the
  27B's decode no more than 10% slower during a transcription. We optimise
  only if one of the two fails (Q26). Transcription has priority over text
  decode (Q14).

**Unavailable is a state with a reason.** The executor detects AVX2 at run
time and picks its kernels by what it finds (ADR 0068). If the CPU executor
cannot run (no AVX2) and the loan does not fit, a transcription waits a
short while, then gets `503` with `Retry-After`. On a card where the loan
rarely fits, the speech model is shown unavailable (Q35).
`speech.model: off` or a missing file is unavailable too, with its own
reason (Q39).

**The text model's lifecycle is unchanged, with three exceptions** (Q40):
- A switch waits for a running loan, bounded by the loan's deadline.
- Transcriptions do not reset the text model's idle clock.
- Opening a Realtime transcription session starts a background reload of an
  unloaded text model. The dictation is going to land in its prompt.

## Consequences

- ignis serves audio in, and holds two loaded models. The OpenAPI document's
  "not served" list, its pinning test, the README and spec server/09 change
  (spec speech/02). Audio *out* is still not served.
- Host RAM: 1.4-3.1 GB resident while the model is in use, its KV at a
  4,096-token context included. Unloading frees it all (Q53). The first
  dictation after an unload waits 1.4-1.8 s more for 1.7B. Both figures were
  measured on llama.cpp; spec speech/01 times the executor's own load.
- A CPU transcription uses all but one physical core (`speech.threads`, Q38).
  1.7B decode is bound by memory bandwidth, so it competes with whatever else
  uses host memory. Q38 accepts the default only if text decode slows by
  less than 10% during a CPU transcription, measured on the GPU.
- A text admission can wait behind a running loan, for up to that
  transcription's deadline (Q32).
- A 16 GB card whose CPU lacks AVX2 may have no speech model at all (Q35).
- The speech model is a catalog entry, fetched by `ignis model download`,
  never downloaded by itself (Q39).
- Phase 1 serves on the CPU alone, and phase 2 adds `server_vad`. The loan
  is phase 3 (Q33, Q56).

## Considered Options

- **Always resident on the GPU.** Rejected:
  - it holds VRAM while nobody dictates (owner: "do not keep the speech model
    in VRAM");
  - a 16 GB card has no room for it.
- **Reached by a switch, as the image model is (ADR 0050).** Rejected: it
  costs a reload of the text model on each side of every dictation, and that
  dictation lands in the text model's prompt.
- **A separate process** (llama-server or similar beside ignis). Rejected:
  - device memory belongs to a process, so a second process cannot borrow
    pages of ignis's pool, and ADR 0030's budget is whole-process;
  - it would duplicate what the speech routes reuse: the API key, the
    deadlines, `/metrics` and the Playground's socket auth.
