---
title: "Speech to text"
description: "Transcribing audio over POST /v1/audio/transcriptions: the speech model, what it takes, what it answers, how long it waits."
sidebar:
  order: 5
---

The server transcribes speech with **Qwen3-ASR**
([ADR 0067](../../adr/0067-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md)),
on OpenAI's route `POST /v1/audio/transcriptions`. It is a model of its own
kind, loaded **beside** the text or image model and never instead of it: it
runs on the CPU, takes no VRAM, and is not switched to. A transcription is
served through a [model switch](model-switching.md), with the text model
[unloaded for being idle](idle-unload.md), and with the image model loaded; it
does not count as a use of the text model, so it never keeps it loaded.

To transcribe while the person speaks, the faster path on the CPU, open a
[live transcription](realtime-transcription.md) session instead.

## Turning it on

The speech model is **on when its files are on disk**, and nothing downloads
by itself:

```bash
ignis-server model download qwen3-asr-1.7b
```

`speech.model` picks it: `auto` (the default) is `qwen3-asr-1.7b` (1.5 GiB), the 1.7B
model with a Q4_K_M decoder; a catalog id picks another entry (`qwen3-asr-1.7b-q8`,
2.3 GiB, or `qwen3-asr-0.6b`, 0.9 GiB); `off` turns it
off. It is loaded by the first request, about a second or two of delay, and
unloaded after `speech.idle_unload` seconds (300) without a transcription
or a [live session](realtime-transcription.md) open.
`GET /v1/models` says whether it can serve:

```json
{ "data": [ {"id": "qwen3.8-27b", "active": true, "kind": "text"},
            {"id": "qwen3-asr-1.7b", "active": true, "kind": "speech", "available": true} ],
  "status": "serving",
  "speech": { "available": true,
              "diarizer": { "id": "acme-diarizer", "loaded": false, "available": true } } }
```

The speech entry is listed right after the model on the card, and only while
it is available, loaded or not. The top-level `speech` object is always there,
and while the model cannot serve its `reason` says why, so a page can disable
its microphone before it is pressed:

| `reason` | What to do |
|---|---|
| the speech model is turned off (`speech.model` is `off`) | set `speech.model` to `auto` or a catalog id |
| no speech model files for `qwen3-asr-1.7b` under download.path; fetch them with `ignis-server model download qwen3-asr-1.7b` | run that command |
| the CPU lacks AVX2, which the speech model needs | a CPU from 2013 (Haswell) or later is required |
| `speech.model` names an id that is not in the model catalog | fix the id, or add the entry to `download.catalog` |

The `speech` object's `diarizer` member says the same of the
[diarizer](realtime-transcription.md#speakers) that gives a live session its
speakers: `id` is the model `diarizer.model` names (absent when it names none),
`loaded` whether its weights are in memory now, and `available` and `reason` as
above. It is no entry of `data`: no request names it. Its `reason` is one of:

| `reason` | What to do |
|---|---|
| the diarizer is turned off (`diarizer.model` is `off`) | set `diarizer.model` to `auto` or a catalog id |
| `diarizer.model` is `auto` and the model catalog has no diarizer entry | name a `kind: diarizer` entry of your [catalog](../guides/operator-catalog.md), or set `diarizer.model` to its id |
| `diarizer.model` names an id that is not in the model catalog, or one that is not a diarizer | fix the id |
| the CPU lacks AVX2, which the speech model needs | the diarizer computes on the speech model's CPU, so it needs the same one |
| no diarizer files for `acme-diarizer` under download.path; fetch them with `ignis-server model download acme-diarizer` | run that command |

A transcription asked of a model that is unavailable is `503
speech_unavailable`, with the same reason and **no `Retry-After`**: waiting
changes nothing.

**Requirements.** A CPU with AVX2, and 1.4 GB (0.6B) to 3.1 GB (1.7B Q8_0) of
RAM while the model is in use, nothing once it is unloaded. On a 4-core
container sharing a desktop CPU's memory the 1.7B Q4_K_M model ran at 3.0 times
real time; a budget 4-core with slower RAM can fall under 2. The fallback for a
weak CPU is `speech.model: qwen3-asr-0.6b`.

## The request

`multipart/form-data`, as OpenAI's:

| Field | |
|---|---|
| `file` | **Required.** WAV, FLAC, MP3 or AAC (ADTS, or in an MP4/M4A), or an Ogg or Matroska/WebM container holding FLAC. Anything else is a `400` naming that set. Opus is not decoded yet. The audio is decoded in memory and never written to disk. |
| `model` | Absent, the speech model's id, or one of `speech.compat_names`: `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-transcribe`, `gpt-live-transcribe`. A text or image model's id is `409 model_kind_mismatch`, naming both kinds; any other name is `404`. |
| `prompt` | Context: names and terms the model should spell right. It is the model's system turn. |
| `keywords[]` | Words to spell right, joined after the prompt as one line. A keyword holding `<`, `>` or a line break is a `400`. |
| `language` | Forces the language: one of the model's 30, by ISO code (`it`) or English name (`Italian`). **Leave it off unless the model guesses wrong**: forcing Italian on a sentence with English jargon Italianises the jargon. |
| `languages[]` | One entry forces it. Several are each checked (one outside the 30 is a `400`), but the model then detects over all its 30 languages: it cannot be held to a list. Not together with `language`. |
| `response_format` | `json` (default), `text` or `verbose_json`. `srt`, `vtt` and `diarized_json` are refused. |
| `timestamp_granularities[]` | `segment`, with `verbose_json`. `word` is refused: there is no aligner. |
| `stream` | `true` for Server-Sent Events. |
| `temperature` | Absent or `0`: the model decodes greedily. Anything else is a `400`. |
| `chunking_strategy` | Absent or `null`. `auto` and `server_vad` are refused: a clip is one block. |
| `include[]`, `known_speaker_names[]`, `known_speaker_references[]` | Refused (`logprobs`; no diarization). |

Every field is honoured, inert, or refused with a `400` naming it in `param`;
none is dropped in silence.

`prompt` and `keywords[]` together are read up to `speech.context_max_tokens`
(512) tokens; past that the **last** ones are kept.

On a clip with little or no speech, a pause's room noise or a short "yes", the
model can write a line of the context instead of what it heard ([the
measurement](../../findings/2026-10-27-asr-context-echo.md)), now and then
with a word or two changed ([the second](../../findings/2026-10-31-asr-context-near-copy.md)).
A transcript that only repeats a line of the context, the whole line or three
or more of its words in a row (`speech.context_copy_words`; 0 turns this off),
is taken for no speech: the text comes back empty, with no language unless one
was forced. One word is kept, since a name the context lists is said alone too.
A copy with a word changed, added or dropped is kept by default, since people
say a line that way too: `la revisione settimanale del rilascio`, under the
topic `revisione settimanale del rilascio`, is a speaker's as often as the
model's. With `speech.context_copy_edits` at 1 to 3 (0 by default), a
transcript that many words or fewer from a whole line, or from a piece of one
between commas or sentence marks, is no speech either while it keeps three or
more of its words, and what people say that way is emptied with it. A clip longer than
`speech.max_clip_seconds` (1200, the model's limit) is a `400` naming the
limit, and is never split. The body limit is 384 MiB, enough for 20 minutes of
48 kHz stereo WAV.

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@meeting.flac -F model=whisper-1 \
  -F prompt='A talk about GPUs.' -F keywords[]=ignis -F keywords[]=Qwen3
```

Aider needs only `OPENAI_API_BASE=http://127.0.0.1:8000/v1`: it sends
`whisper-1`.

## The answer

- **`json`**: `{"text": "...", "usage": {"type": "duration", "seconds": 12.4}}`,
  and `"languages": [{"code": "it"}]` for the language the model heard, left
  out when it labelled the clip `None` (no speech).
- **`text`**: the transcript as the body.
- **`verbose_json`**: `{"task", "language", "duration", "text", "segments",
  "usage"}`. `language` is the model's own label (`Italian`), left out for
  `None`. `segments` is **one segment spanning the clip**, with the whole text
  and its token ids; the fields the engine does not compute (`avg_logprob`,
  `no_speech_prob`, ...) are 0.
- **`stream: true`**: `text/event-stream`. `transcript.text.delta` events as the
  text becomes stable, then one `transcript.text.done` with the whole `text`
  (the deltas, joined, unless the clip is taken for a copy of the context:
  then it is empty, whatever the deltas said) and `usage` in audio seconds,
  then `data: [DONE]`.

## Waiting and refusing

One compute worker serves the speech model. [Live transcription](realtime-transcription.md)
sessions take precedence; transcription requests are served first in, first out,
in the gaps. The wait is proportional
to the audio: on a 10-thread desktop CPU (4.1 to 4.7 times real time with the default model, the
longer the clip the nearer 4.7) a minute of speech takes some thirteen to fifteen seconds,
and a 20-minute clip four to five minutes. Up to `speech.max_queue` (4) requests
wait behind the one running; the next is `503 engine_full` with a `Retry-After`
of one to fifteen seconds, an estimate of when room frees.
It is answered before the audio is decoded, so a body that is not audio gets
the `503`, not a `400`, while the queue is full.
A long clip's transcript is decoded a little at a time (16 tokens a step, the
text already fixed prefilled 104 tokens a step), so live sessions are served
between the steps rather than after the whole clip.

A request is held to the [deadline](deadlines.md) from its admission, queue
included: `server.request_timeout`, or `server.stream_timeout` when it
streams. Past it the answer is `504 request_timeout`, or an `error` event after
the `200` of a stream, whatever `server.timeout_reply` says: a transcript has
no answer that says it is incomplete. A client that goes away cancels its
transcription at the next window.

## What is logged and counted

Neither the audio nor the transcript is ever written to a log, and the audio is
never written to disk. Each transcription writes one `ignis.speech.transcription`
line: audio seconds, executor (`cpu`), variant (the served id), x real time, the
wait from the last byte received, the detected language and the outcome. One
the speech model failed is the same line at WARN, with `reason`, the executor's
own words (a load that failed names the file). The
[metrics](../reference/metrics.md) are `ignis_speech_transcriptions_total{executor,outcome}`,
`ignis_speech_audio_seconds_total`, `ignis_speech_commit_wait_seconds` and the
gauges `ignis_speech_sessions_open` and `ignis_speech_queue_length`, shown in
the Playground's [Monitor](../playground/monitor.md). The fields are in
[the speech configuration](../configuration/speech.md).

Each load of the model writes two lines: `ignis.speech.threads`, the threads it
computes with and whether `auto` chose them, and `ignis.speech.load`, how long
it took and how many bytes it repacked into its own memory (the Q4_K weights,
rearranged for the CPU's kernels) and read from the files, and whether the
weights are pinned. An unload after `speech.idle_unload` seconds unused writes
`ignis.speech.unloaded`, with `idle_seconds`.

`POST /v1/audio/translations` is not served (`404`): the model transcribes the
language it hears.
