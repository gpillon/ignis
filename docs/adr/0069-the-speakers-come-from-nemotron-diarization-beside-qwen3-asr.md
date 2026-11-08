---
status: accepted
---

# ADR 0069 — the speakers come from Nemotron 3 Diarization, beside Qwen3-ASR

## Status

Accepted (2026-10-25, owner): the Transcribe grill of 2026-10-25, Q104 (option (i)), Q106-Q118, after the spike
Q91 asked for. Specs `docs/specs/speech/07-transcribe.md` (the Playground view) and
`docs/specs/speech/08-diarization.md` (the diarizer).
- **Holds ADR 0067**: the diarizer, like the speech model, is loaded beside the text or image model and never
  resident in VRAM.
- **Holds ADR 0068**: the diarizer's forward pass is ours, in Rust, on the CPU, with no vendored runtime.
- **Applies ADR 0051**: the Sortformer graph restated from NeMo (Apache-2.0) is an adapted op, as the VAD's Silero
  port is.

Amended 2026-10-26 (#361, the owner's choice on a condition, confirmed by the measurements): decision 5's shared worker
gives way to a thread and a pool of the diarizer's own (`diarizer.threads`), on the condition that it hurt neither the
commit's latency, nor the live text, nor the other sessions. On the shared worker a step that fell due stopped the
refresh under way, and on a long monologue the live text stood still for the whole item; on its own thread the
refreshes with speakers match those without, and the commits are no slower (spec speech/08, As built).

Sources: the finding [Nemotron on the CPU](../findings/2026-10-25-nemotron-on-the-cpu.md), with the spike's scripts
and raw data on `H:\speech-study\nemotron-spike\`; the finding
[Qwen3-ASR on the CPU](../findings/2026-10-24-qwen3-asr-on-cpu.md); the session's decision log, kept untracked in
`.scratch/speech-grill-2026-10-23/` (Q81 onwards).

## Context

The owner wants the Playground to transcribe a whole conversation, with "Speaker 1, 2…" labels renamable live
(spec speech/07). Qwen3-ASR, ignis's speech model, gives no speakers: its card has no diarization, its tokenizer no
speaker tokens, its output no word timestamps. So the speakers need a second model.

Three ways were weighed:
- **(i)** Qwen3-ASR stays, and a diarizer runs beside it;
- **(ii)** NVIDIA's pairing, Nemotron 3.5 ASR Streaming with Nemotron 3 Diarization, which tags speakers per word;
- **(iii)** a small per-utterance speaker embedding such as CAM++, clustered online.

The spike (the first finding) measured (i) and (ii) on the CPU, on the owner's dictations and on twenty synthetic
Italian conversations built from VoxPopuli:
- **Nemotron's ASR loses Italian:** a best strict WER of 13.2% on the eleven dictations against Qwen3-ASR's 3.2-3.4%
  with no context, and 4.7% against 2.7% even without the written-form spans. What remains is the English jargon of
  a developer's Italian. It also has no context biasing, which the Transcribe view's context field relies on.
- **Nemotron's ASR wins on latency and cost:** final text 0.18-0.30 s after the last audio on the i9, at 8% of the
  cores; Qwen3-ASR's Realtime commit takes 2.4-4.4 s.
- **Nemotron 3 Diarization tells Italian speakers apart,** although Italian is not named in its training data:
  DER 3.6-4.0% at every geometry, about 98% of turns right, the returning speaker kept in 19-20 of 20. It fails with
  a ghost speaker of a few seconds in 20-30% of conversations, and by merging two same-gender voices that change with
  no pause.
- **The diarizer's cost depends on its geometry.** Each step re-encodes its speaker cache, FIFO and chunk, about 540
  frames through 31 layers:
  - the card's low geometry takes half of a 4-core machine;
  - a 2.32 s chunk costs 2.6x less at no DER cost;
  - the offline geometry costs 25-47x less than real time.
- On the diarizer, PyTorch FP32 beat ggml Q8_0 (508 against 419 GFLOP/s): each step is one large GEMM.

## Decision

1. **Qwen3-ASR stays the speech model; Nemotron 3 Diarization tells the speakers apart beside it** (Q104). Nemotron's
   ASR is rejected on the measured Italian WER. CAM++-style embeddings, (iii), are the fallback if the port of (i)
   fails its gates.
2. **The diarizer is a model of its own kind**, the fourth beside text, image and speech: it takes audio and gives
   back who spoke when, and never transcribes (Q118). It has its own catalog entry, of kind `diarizer`, downloaded
   with `ignis-server model download` and chosen by `diarizer.model` (`auto`, `off` or an id). It is not in
   `/v1/models`, since no request names it; the Server view shows it beside the speech model.
3. **Ours, in Rust, on the CPU** (ADR 0068's rule; Q104). The 31-layer RoPE Transformer, the Sortformer head, the
   arrival-order speaker cache and the FIFO are restated from NeMo's Apache-2.0 source as an adapted op under
   ADR 0051, with provenance in the file header and in `NOTICE`. The weights keep their OpenMDW-1.1 licence beside
   them. Their stored format (F32, BF16 or Q8_0) is chosen by measurement in the port, since FP32 beat Q8_0 in the
   spike. NeMo-Speech.cpp and the checkpoint's Transformers code are out-of-tree references and oracles only.
4. **Never resident in VRAM** (ADR 0067; Q104). It is loaded at the first session that asks for speakers, and
   unloaded with the speech model at `speech.idle_unload` (Q116). It costs about 0.2-0.4 GB of host RAM, depending on
   the stored format.
5. **It runs on a thread and a pool of its own, beside the speech model's worker** (the owner, 2026-10-26; first
   decided: it shares the speech model's worker, its steps ranked below the windows and commits and above the
   partials, Q113). The thread serves every session's speakers' units in turns; the pool has `diarizer.threads`
   threads, made when the diarizer loads and dropped when it unloads, and computes with the speech executor's ISA. A
   step never waits for the speech model's units and never stops them, and no partial is skipped for one. The speech
   model's pool is not shared: its dispatch takes one lock per op, and a decode token is about 200 small ops beside a
   diarizer op of 2-5 ms.
6. **Its configuration is a group and a family of its own** (owner, 2026-10-25; ADR 0046):
   - the `diarizer` group, as the image model has `image` and the speech model `speech`;
   - `ModelFamily` gains `Nemotron3Diarization`, and every field that is a concept of Sortformer is
     `applicable_to: Only([Nemotron3Diarization])`, so a diarizer of another family refuses it rather than applying
     it in silence;
   - the geometry: the live default is a long chunk, about 2.3 s, which cost no DER against the card's low geometry
     at 2.6x less CPU, and a session fed a file asks for the offline geometry (Q106). Every geometry parameter is a
     field with a measured default, by the owner's rule for thresholds;
   - the speaker cache's constants (score threshold, boosts, minimum positive rate, silence frames per speaker) are
     fields too, defaulting to `auto`: the checkpoint's values, carried in the GGUF, overridable.
7. **One speaker per utterance.** A speaker change inside an item makes the server end the item there, as the VAD
   does on a silence (Q107). A new speaker's label appears only after `diarizer.min_speaker_seconds` of their speech
   (0 = off; Q108, Q116).
8. **An ignis extension of Realtime transcription** (the names rule, Q57): opt in with `session.ignis.diarization`;
   each item's speaker arrives in `ignis.input_audio_transcription.speaker` before its `completed` (Q111). OpenAI's
   events do not change, and a client that does not opt in sees nothing new.

## Consequences

- A user who never asks for speakers pays nothing: no download, no load, no CPU.
- With speakers on, live transcription costs more CPU on top of Qwen3-ASR's partials: the diarizer's threads compute
  beside the speech model's, and on a machine with few cores they share them.
- A label never changes the transcript. Corrections (rename, move, merge, split) are mappings of labels in the
  Playground, and nothing goes back to the model (Q109). Sortformer has no enrolment input that could take them.
- At most 8 speakers, the model's channels (Q110). A ninth voice lands on one of the eight.
- The cut at a speaker change makes items shorter than the VAD alone would. Each part is transcribed with
  Qwen3-ASR's context and leading silence, as any item is (spec speech/08).
- A batch `diarized_json` response on `/v1/audio/transcriptions` becomes cheap with the offline geometry. It stays
  a parked idea (Q94).
- The quality numbers are on clean synthetic conversations. Real meetings are harder: spec speech/08 reports a small
  AMI set and the owner's live check.

## Considered Options

- **(ii) Nemotron 3.5 ASR with the diarizer.** Truly streaming, cheap, with 98.1% of words on the right speaker.
  Rejected on the Italian WER, 13.2% against 3.2% on the owner's dictations, and on the loss of the context field.
  On 4 cores the joint stream also fell behind real time at the low geometry.
- **(iii) A per-utterance speaker embedding** (CAM++ class). It avoids the diarizer's recurring re-encode by
  construction, but was not measured. It cannot cut an utterance at a change with no pause, and its Italian quality
  is unknown. Kept as the fallback.
- **Labels from the source only** ("Me" on the microphone, "Others" on a shared call). Kept for calls, where the
  microphone stream is "Me" and only the call stream is diarized (Q112). It cannot tell apart two people in one room
  or on one call.
- **The diarizer inside Qwen3-ASR's catalog entries,** as one more file. Rejected: it would be repeated on the three
  variants, and it has another licence.
- **Vendor NeMo-Speech.cpp.** Rejected by ADR 0068's reasoning: foreign C++ and ggml in the tree, a CMake build. Its
  standalone diarizer is also pinned at 4 threads.
