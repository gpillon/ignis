---
title: "Speech"
description: "The speech model: which one, its compat names, its limits and its queue."
sidebar:
  order: 12
---

The `speech` group configures the [speech model](../api/transcriptions.md)
(Qwen3-ASR, on the CPU, beside the model on the card). Every field is read
live: `PATCH /v1/config` changes it with no reload, and `threads` and
`pinned_weights` hold from the model's next load. The generated list, with
every default, is [the configuration reference](../reference/configuration.md).

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--speech-model` | `IGNIS_SPEECH_MODEL` | `auto` | `auto` is `qwen3-asr-1.7b`, the 1.7B model with a Q4_K_M decoder; a catalog id picks another entry; `off` turns the speech model off. It is on when its files are on disk (`ignis-server model download <id>`); nothing downloads by itself. A catalog entry that is not a speech model (a text or image model's id) is refused, at start and by a `PATCH`. |
| `--speech-compat-names` | `IGNIS_SPEECH_COMPAT_NAMES` | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-transcribe`, `gpt-live-transcribe` | The `model` names OpenAI clients send that the speech model answers to (Aider hard-codes `whisper-1`). `;`- or `,`-separated. Not catalog aliases and not listed by `GET /v1/models`. A name equal to a served model id, or to an id or alias of the catalog in use (an operator's `download.catalog` among it), is refused at start and by a `PATCH`. |
| `--speech-threads` | `IGNIS_SPEECH_THREADS` | `auto` | The CPU threads the model computes with; `auto` is the physical cores less one, counted within the CPUs the server may run on (a container's cpuset or CPU quota). Each load logs the count and whether `auto` chose it (`ignis.speech.threads`). |
| `--speech-max-clip-seconds` | `IGNIS_SPEECH_MAX_CLIP_SECONDS` | `1200` | The longest clip one transcription takes (1..1200, the model's limit). A longer one is a `400`; nothing is split. |
| `--speech-context-max-tokens` | `IGNIS_SPEECH_CONTEXT_MAX_TOKENS` | `512` | The most tokens of `prompt` and `keywords[]` a transcription reads; past it the last are kept. 0 reads none. |
| `--speech-context-copy-words` | `IGNIS_SPEECH_CONTEXT_COPY_WORDS` | `3` | On a pause the model can write a line of the context in place of what it heard ([the copy guard](../api/transcriptions.md#the-request)): a transcript that only repeats this many words in a row of one line, or a whole line of two words or more, is taken for no speech and comes back empty. One word alone never is, since a name the context lists is said alone too. 0 turns the guard off. Read when a session opens, so a change holds from the next session. |
| `--speech-context-copy-edits` | `IGNIS_SPEECH_CONTEXT_COPY_EDITS` | `0` | The words, 0 to 3, such a copy may change, add or drop from a whole line, or from a piece of one between commas or sentence marks, and still be emptied, while it keeps `context_copy_words` of its words. The model writes such copies now and then, an article put before the topic or a word changed; people say a line that way too, a title after "So," or a term with a word inflected, and on [the measurement](../../findings/2026-10-31-asr-context-near-copy.md) every value above 0 emptied some of what they said, so 0 keeps the exact rule. Read when a session opens, so a change holds from the next session. |
| `--speech-pinned-weights` | `IGNIS_SPEECH_PINNED_WEIGHTS` | off | Pin the weights in host memory instead of mapping the file: the model's tensors are copied into the server's own memory at load (and locked in RAM where the OS allows it, on Linux), so the files are not held open. |
| `--speech-idle-unload` | `IGNIS_SPEECH_IDLE_UNLOAD` | `300` | Seconds with no transcription, and no live session open, after which the model is unloaded; the next one loads it again. 0 never unloads. |
| `--speech-max-queue` | `IGNIS_SPEECH_MAX_QUEUE` | `4` | The transcriptions that may wait behind the one running; one more is `503 engine_full` with `Retry-After`. 0 runs one at a time. |
| `--speech-max-sessions` | `IGNIS_SPEECH_MAX_SESSIONS` | `4` | The most [live transcription](../api/realtime-transcription.md) sessions open at once; one more is refused at the upgrade, `503 engine_full` with `Retry-After`. |
| `--speech-partial-interval-ms` | `IGNIS_SPEECH_PARTIAL_INTERVAL_MS` | `2000` | How often at most a live session's transcript refreshes while audio arrives: each refresh sends the new stable text, and the provisional tail to a session that asked for it. Best effort: a refresh is skipped while the session's audio is still being computed. Shorter is livelier text and more CPU per session. |
| `--speech-partial-cost-factor` | `IGNIS_SPEECH_PARTIAL_COST_FACTOR` | `1` | How long the next refresh of a live transcript waits at least, as a multiple of the last refresh's compute time; 1 only waits for the last refresh to finish ([the text](../api/realtime-transcription.md#the-text)). |
| `--speech-rollback-tokens` | `IGNIS_SPEECH_ROLLBACK_TOKENS` | `10` | The tokens of a live transcript held back as provisional, the rest being stable: the [provisional tail](../api/realtime-transcription.md#the-text) a session that asks for it gets. Read by the speech executor when a session opens, so a change holds from the next session. |
| `--speech-unfixed-partials` | `IGNIS_SPEECH_UNFIXED_PARTIALS` | `2` | The first refreshes of a live session's transcript that make no text stable, whatever `speech.rollback_tokens` says: the language and the opening words are decided on more than one interval of audio. 0 fixes text from the first refresh. Read when a session opens, so a change holds from the next session. |
| `--speech-vad-threshold`, `--speech-vad-prefix-padding-ms`, `--speech-vad-silence-duration-ms` | `IGNIS_SPEECH_VAD_*` | `0.5`, `300`, `500` | The defaults of a live session's [`server_vad`](../api/realtime-transcription.md#turn-detection), OpenAI's: the speech probability that counts as speech, the audio an utterance keeps from before it, and the silence after it at which the server commits it. A session reads them when it opens; one that sends its own values uses those. |
| `--speech-vad-leading-silence-ms` | `IGNIS_SPEECH_VAD_LEADING_SILENCE_MS` | `500` | The digital silence the speech model reads before each `server_vad` utterance, never shown to the client; it counts toward `speech.max_clip_seconds` ([turn detection](../api/realtime-transcription.md#turn-detection)). 0 feeds the utterance as heard. |
| `--speech-max-utterance-seconds` | `IGNIS_SPEECH_MAX_UTTERANCE_SECONDS` | `120` | The [length guard](../api/realtime-transcription.md#long-utterances): once a `server_vad` utterance holds this many seconds, the server ends it at the next short pause, or `utterance_cut_search_ms` later where the audio is, and the speech goes on in a new item, so a room that never falls silent loses nothing to `max_clip_seconds`. A session's `ignis.max_utterance_seconds` overrides it; 0 never cuts. At 240 s an utterance's commit took four times as long as at 120 s, at no gain in word error rate. |
| `--speech-utterance-cut-search-ms` | `IGNIS_SPEECH_UTTERANCE_CUT_SEARCH_MS` | `10000` | How long past the limit the length guard looks for a pause before it cuts where the audio is. It looks forward only, so no text already sent is past the cut. In fluent speech the pause comes within 10 s 95 times in 100: between words the VAD's speech probability seldom drops. |
| `--speech-max-queued-audio-seconds` | `IGNIS_SPEECH_MAX_QUEUED_AUDIO_SECONDS` | `120` | The audio a live session may hold that is not computed yet; an append past it is refused, `ignis_audio_queue_full`, and the session stays open ([sending audio](../api/realtime-transcription.md#sending-audio)). Above the Playground's 60 s pace for a file; provisional. |
| `--speech-loan-grace-seconds`, `--speech-loan-wait-seconds` | `IGNIS_SPEECH_LOAN_*` | `30`, `5` | The loan of free KV pages to the speech model. Not served yet. |

The fields marked "not served yet" are declared so that a configuration file
written today keeps its meaning: the loan that reads them comes with its own
ticket ([ADR 0067](../../adr/0067-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md)).

## The diarizer

The `diarizer` group configures the diarizer, Nemotron 3 Diarization, which
gives a live session that asks for them its [speakers](../api/realtime-transcription.md#speakers).
It runs on the CPU, on a thread and CPU threads of its own beside the speech
model's, so its steps never hold up the speech model's work. Every field but
`diarizer.threads` is read live, when a session's speakers start;
`diarizer.threads` is read by the next load. The diarizer is a
[catalog](../guides/models.md#the-catalog) entry of its own, `kind: diarizer`
(one GGUF, no sidecar): the built-in `nemotron-3-diarization`, fetched with
`ignis-server model download nemotron-3-diarization` like any entry, or one of
[an operator's catalog](../guides/operator-catalog.md). It is
loaded by the first session that asks for speakers, about 0.2 GB of RAM for
its F16 weights, never VRAM, and unloaded with the speech model, after
`speech.idle_unload` seconds with no session open.

What it costs, measured at the live geometry: on a 10-core i9-10900K a 2 s
step takes about 0.63 s on its 3 threads alone (0.88 s on 2, 0.56 s on 4), and
on a 297 s conversation with `qwen3-asr-1.7b` transcribing the same session at
real time, 0.73 s on average, the speakers staying within 3.9 s of the audio,
the commits no slower than with the diarizer sharing the speech model's
threads and the live text as frequent as without speakers; on 4
cores of a Linux container, the diarizer alone took 23 to 24 % of the four
cores (3.2 to 4.1 times faster than real time), where the speech model's
refreshes alone already fall behind. A file, at the file geometry, costs 25 to
47 times less than its length. `GET /v1/models` says whether it can serve and
whether it is loaded ([the `speech.diarizer` member](../api/transcriptions.md#turning-it-on)).
The geometries and the speaker
cache's constants are Sortformer's, Nemotron 3 Diarization's own: a diarizer
of another family would refuse them.

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--diarizer-model` | `IGNIS_DIARIZER_MODEL` | `auto` | `auto` is the catalog's diarizer entry; a catalog id picks another; `off` turns speakers off. A text, image or speech model's id is refused, at start and by a `PATCH /v1/config`. A live change takes the next session's load. A session open with speakers keeps the diarizer it opened with until it ends: meanwhile `GET /v1/models` reads the new one, and the old one's file is not protected from `model remove` (on Windows its removal fails while the session holds it). Nothing downloads by itself. |
| `--diarizer-threads` | `IGNIS_DIARIZER_THREADS` | `auto` | The CPU threads the diarizer computes with, its own beside the speech model's; `auto` is a third of the physical cores, at least 2 (3 on a 10-core machine), counted within the CPUs the server may run on. The load reads it, so a change holds from the next one, and logs the count (`ignis.diarizer.load`). |
| `--diarizer-min-speaker-seconds` | `IGNIS_DIARIZER_MIN_SPEAKER_SECONDS` | `2` | The seconds of speech a voice needs to be a speaker of its own; until then its speech counts for the speaker it sounds most like, so a short mistaken voice splits no utterance. 0 makes every voice a speaker at once. At 2 s a mistaken voice of a few seconds in test conversations went, and no participant of real meetings who spoke little was lost (at 5 s some were). The price: a new voice's first seconds go to someone else (a colleague who at first only says "yes, exactly" is labelled as another speaker until they have spoken this long; the run of speech that makes them a speaker becomes theirs), and when the voice changes with no pause, the utterance before the change ends only once the new voice is a speaker. |
| `--diarizer-live-speaker-cache-frames`, `-live-fifo-frames`, `-live-chunk-frames`, `-live-right-context-frames`, `-live-update-period-frames` | `IGNIS_DIARIZER_LIVE_*` | `264`, `264`, `25`, `4`, `222` | The live geometry (`ignis.diarization: "live"`), in 80 ms frames: the speaker cache, the FIFO, the chunk a step scores, the look-ahead it waits for, and the frames an overflowing FIFO gives the cache. The chunk and the look-ahead are the latency, 2.32 s; this geometry measured a better DER than the card's low-latency one at 2.6 times less CPU. |
| `--diarizer-file-speaker-cache-frames`, ... | `IGNIS_DIARIZER_FILE_*` | `264`, `40`, `340`, `40`, `300` | The file geometry (`"file"`), the card's offline one, for audio fed faster than real time: 30.4 s of latency, at 25 to 47 times less CPU than real time. |
| `--diarizer-prediction-score-threshold`, `-latest-frames-score-boost`, `-strong-boost-rate`, `-weak-boost-rate`, `-min-positive-scores-rate`, `-speaker-cache-silence-frames-per-speaker` | `IGNIS_DIARIZER_*` | `auto` | The speaker cache's constants, by the checkpoint's own names: how it scores and keeps the frames of each speaker. `auto` is the checkpoint's value (0.25, 0.05, 0.75, 1.5, 0.5 and 1), which its weights carry; an override is bounded, and the values in force when the diarizer loads are logged (`ignis.diarizer.load`). |
