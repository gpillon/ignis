---
status: accepted
---

# The speakers come from Nemotron 3 Diarization, beside Qwen3-ASR

The Playground transcribes a whole conversation with "Speaker 1, 2…" labels, renamable live, and Qwen3-ASR, ignis's
speech model, gives no speakers: its card has no diarization, its tokenizer no speaker tokens, its output no word
timestamps. So a second model tells the speakers apart: Qwen3-ASR stays the speech model and Nemotron 3 Diarization
runs beside it. Measured on the CPU ([finding](../findings/2026-10-25-nemotron-on-the-cpu.md)), the diarizer tells
Italian speakers apart although Italian is not named in its training data, while the ASR it is paired with loses
Italian to Qwen3-ASR.

- **The diarizer is a model of its own kind**, the fourth beside text, image and speech: it takes audio and gives
  back who spoke when, and never transcribes. It has its own catalog entry, of kind `diarizer`, downloaded with the
  `model download` command and chosen by `diarizer.model` (`auto`, `off` or an id). It is no entry of `/v1/models`'
  `data`, since no request names it; the `speech` object's `diarizer` member there says whether it is available, and
  the Server view shows it beside the speech model.
- **Ours, in Rust, on the CPU**, with no vendored runtime (ADR 0068's rule). Its graph and streaming state are
  ported from the checkpoint's Transformers code, which restates NeMo's Apache-2.0 source, as an adapted op under
  ADR 0051, as the VAD's Silero port is, with
  provenance in the file header and in `NOTICE`. The weights keep their OpenMDW-1.1 licence beside them. Their
  stored format is chosen by measurement in the port, since FP32 beat Q8_0 in the spike: each step is one large
  GEMM. NeMo-Speech.cpp is an out-of-tree reference.
- **Never resident in VRAM**, and loaded beside the text or image model, like the speech model (ADR 0067). It is
  loaded at the first session that asks for speakers, and unloaded with the speech model at `speech.idle_unload`.
- **It runs on a thread and a pool of its own** (`diarizer.threads`), beside the speech model's worker, on the
  condition that it hurt neither the commit's latency, nor the live text, nor the other sessions. A step never waits
  for the speech model's units and never stops them, and no partial is skipped for one. The speech model's pool is
  not shared: its dispatch takes one lock per op, and a decode token is about 200 small ops beside a diarizer op of
  2-5 ms.
- **Its configuration is a group and a family of its own** (ADR 0046): the `diarizer` group, as the image model has
  `image` and the speech model `speech`. Every field that is a concept of Sortformer applies to Nemotron 3
  Diarization's family only, so a diarizer of another family refuses it rather than applying it in silence. Every
  geometry parameter is a field with a measured default, by the owner's rule for thresholds: live, a long chunk,
  which cost no DER against the card's low-latency geometry at much less CPU; the offline geometry for a session
  fed a file. The speaker cache's constants are fields too, defaulting to `auto`, the checkpoint's values.
- **One speaker per utterance.** A speaker change inside an item makes the server end the item there, as the VAD
  does on a silence. A new speaker's label appears only after `diarizer.min_speaker_seconds` of their speech (0 is
  off).
- **An ignis extension of Realtime transcription**, by the names rule (`CODING_STANDARDS.md`, "Extensions get ignis
  names"): a session opts in with a `session.ignis` field, and each item's speaker arrives in an `ignis.` event
  before the item's `completed`. OpenAI's events do not change, and a client that does not opt in sees nothing new.

## Considered Options

- **NVIDIA's pairing, Nemotron 3.5 ASR Streaming with the diarizer**, which tags speakers per word: truly
  streaming and cheap, its final text far sooner than Qwen3-ASR's Realtime commit. Rejected on the measured Italian
  WER, about four times Qwen3-ASR's on the owner's dictations, and on the loss of context biasing, which the
  Transcribe view's context field relies on. On 4 cores the joint stream also fell behind real time at the
  low-latency geometry.
- **A small per-utterance speaker embedding** (CAM++ class), clustered online. It avoids the diarizer's recurring
  re-encode by construction, but was not measured; it cannot cut an utterance at a change with no pause, and its
  Italian quality is unknown. Kept as the fallback if the diarizer's port fails its gates.
- **Labels from the source only** ("Me" on the microphone, "Others" on a shared call). Kept for calls, where the
  microphone stream is "Me" and only the call stream is diarized. It cannot tell apart two people in one room or on
  one call.
- **The diarizer inside Qwen3-ASR's catalog entries**, as one more file. Rejected: it would be repeated on the
  three variants, and it has another licence.
- **Vendor NeMo-Speech.cpp.** Rejected by ADR 0068's reasoning: foreign C++ and ggml in the tree, a CMake build.
  Its standalone diarizer is also pinned at 4 threads.
- **The diarizer's steps on the speech model's worker**, ranked below the windows and commits and above the
  partials: the first decision. Replaced: a step that fell due stopped the refresh under way, and on a long
  monologue the live text stood still for the whole item. On its own thread the refreshes with speakers match those
  without, and the commits are no slower.

## Consequences

- A user who never asks for speakers pays nothing: no download, no load, no CPU. With speakers on, live
  transcription costs more CPU on top of Qwen3-ASR's partials: the diarizer's threads compute beside the speech
  model's, and on a machine with few cores they share them.
- A label never changes the transcript. Corrections (rename, move, merge, split) are mappings of labels in the
  Playground, and nothing goes back to the model: Sortformer has no enrolment input that could take them.
- At most 8 speakers, the model's channels. A ninth voice lands on one of the eight.
- The cut at a speaker change makes items shorter than the VAD alone would. Each part is transcribed with
  Qwen3-ASR's context and leading silence, as any item is.
- A batch `diarized_json` response becomes cheap with the offline geometry. It stays a parked idea.
- The quality numbers are on clean synthetic conversations; real meetings are harder.
