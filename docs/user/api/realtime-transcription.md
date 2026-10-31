---
title: "Live transcription"
description: "Dictation over a WebSocket: OpenAI's Realtime transcription session on /v1/realtime, with the text as it settles."
sidebar:
  order: 6
---

`GET /v1/realtime?intent=transcription` opens **OpenAI's Realtime
transcription session** on the [speech model](transcriptions.md): the client
streams audio while the person speaks, and the server sends the text as it
settles. On the CPU this is the fast path: the model encodes and prefills the
audio while it arrives, so a commit leaves only the last few seconds to
decode, where [`POST /v1/audio/transcriptions`](transcriptions.md) starts on
the whole clip once it is in
([ADR 0067](../../adr/0067-the-speech-model-waits-on-the-host-and-borrows-free-kv-pages.md)).

It is the GA protocol's transcription subset: a client written for OpenAI's
`wss://api.openai.com/v1/realtime?intent=transcription` connects with the
base URL changed. Conversation sessions (a `model` in the URL, voices,
responses) are not served.

## Opening a session

```text
GET /v1/realtime?intent=transcription
Authorization: Bearer <key>
```

From a browser, which cannot set that header, offer the key as a subprotocol:
`new WebSocket(url, ["realtime", "openai-insecure-api-key.<key>"])`. The server
selects `realtime` and never echoes the key. Without a key configured, no
header is needed.

The upgrade is refused before the socket opens:

| Answer | When |
|---|---|
| `400`, `param: "intent"` | the URL has no `intent=transcription`, or names a `model`: that is a conversation session |
| `401 invalid_api_key` | a key is set and the upgrade carried neither the bearer token nor the subprotocol |
| `503 speech_unavailable`, no `Retry-After` | the speech model cannot serve; the message says why, as `GET /v1/models`' `speech` object does |
| `503 engine_full`, with `Retry-After` | `speech.max_sessions` (4) sessions are open; one more waits for one to close |

A plain `GET` of the route, with no upgrade, is answered what an upgrade would
be, and claims no session: the same `400` for a wrong `intent`, the same `503`s,
and, when the socket could open, `400 upgrade_required`. A page that cannot read
a refused upgrade's status asks this way why it was refused.

On open the server loads the speech model if it is not loaded, prefills the
context, and, if the text model was [unloaded for being idle](idle-unload.md),
starts loading it in the background: a person dictating a prompt sends it
next. The session is served on the CPU meanwhile, and through a
[model switch](model-switching.md). It is no use of the text model and never
keeps it loaded.

The first event is `session.created`, with the configuration in effect:

```json
{ "type": "session.created", "event_id": "event_…",
  "session": { "type": "transcription", "id": "sess_…",
    "audio": { "input": {
      "format": { "type": "audio/pcm", "rate": 24000 },
      "transcription": { "model": "qwen3-asr-1.7b", "prompt": "", "language": null },
      "noise_reduction": null,
      "turn_detection": { "type": "server_vad", "threshold": 0.5,
        "prefix_padding_ms": 300, "silence_duration_ms": 500, "idle_timeout_ms": null } } },
    "include": [],
    "ignis": { "provisional_tail": false, "diarization": null, "max_utterance_seconds": null,
      "item_language": false } } }
```

A session that sends no `turn_detection` runs **`server_vad`**, as OpenAI's
do: the server finds the speech and commits each utterance itself
([turn detection](#turn-detection)). A client that commits on its own sends
`turn_detection: null`, as the Playground does.

## Configuring it

`session.update` changes what it names and is answered `session.updated` with
the whole configuration in effect. A field left out keeps its value, `null`
puts it back to its default, and one refused field refuses the whole update
with an `error`, the session keeping the configuration it had.

```json
{ "type": "session.update", "session": { "type": "transcription",
  "audio": { "input": {
    "format": { "type": "audio/pcm", "rate": 16000 },
    "transcription": { "prompt": "A talk about GPUs.", "keywords": ["ignis", "Qwen3"] } } },
  "ignis": { "provisional_tail": true } } }
```

| Field | |
|---|---|
| `audio.input.format` | `{type: "audio/pcm", rate: 24000}`, OpenAI's, resampled to 16 kHz; or `rate: 16000`, the model's own rate, an ignis extension that skips the resampling. `audio/pcmu` and `audio/pcma` (G.711) are refused. |
| `audio.input.transcription.model` | Absent, the speech model's id, or one of `speech.compat_names` (`gpt-4o-transcribe`, `whisper-1`, ...): the session reports the speech model's id. A text or image model's id is an `error` with `model_kind_mismatch`; any other name `model_not_found`. |
| `audio.input.transcription.prompt`, `.keywords` | The context, as for [a transcription](transcriptions.md#the-request): names and terms to spell right. It is **prefilled when the update arrives**, so it costs nothing at the commit. A change while an item holds audio applies from the next item. An item whose transcript only repeats a line of the context, or with `speech.context_copy_edits` above 0 is that few words off a line or a piece of one, is no speech, as for [a transcription](transcriptions.md#the-request): it completes empty. |
| `audio.input.transcription.language`, `.languages` | As for a transcription: one language forces it; several are each checked (one outside the 30 is an `error`), but the model then detects over all its 30 languages, since it cannot be held to a list; both together are refused. Leave it off unless the model guesses wrong. |
| `audio.input.transcription.delay` | Accepted and inert: the live text refreshes at most every `speech.partial_interval_ms` (2000). |
| `audio.input.turn_detection` | `{type: "server_vad", ...}`, the default, or `null` for manual commit: [turn detection](#turn-detection). `semantic_vad` is refused. |
| `audio.input.noise_reduction` | `null`. `near_field` and `far_field` are refused. |
| `include` | Empty. `item.input_audio_transcription.logprobs` is refused. |
| `ignis.provisional_tail` | `true` to receive the provisional tail, below. |
| `ignis.diarization` | `"live"` or `"file"` to label each item with its [speaker](#speakers), `null` (the default) for none. Only under `server_vad`: on a manual-commit session it is an `error` naming `turn_detection`, and so is a switch to manual commit while it is on. With the diarizer unavailable it is an `error`, `ignis_diarizer_unavailable`, with the reason, and the session goes on without speakers. A change while an item holds audio holds from the next item. |
| `ignis.max_utterance_seconds` | The [length guard](#long-utterances)'s limit for this session: `null` (the default) is the server's `speech.max_utterance_seconds`, `0` turns it off, otherwise whole seconds up to `speech.max_clip_seconds`. Declared as set. |
| `ignis.item_language` | `true` to receive each item's language, below; `false` or `null` (the default) for none. |

`ignis` takes those four keys; any other is an `error` naming it and the
four.

`transcription_session.update`, the retired beta's event, is an `error` naming
`session.update`, and every conversation event (`response.create`,
`conversation.item.create`, ...) is an `error`. No `error` closes the socket.

## Sending audio

```json
{ "type": "input_audio_buffer.append", "audio": "<base64>" }
```

`audio` is base64 of 16-bit little-endian mono PCM at the session's rate, up
to 15 MiB of base64 an event. Nothing is sent back. Under manual commit the
audio between two commits is one **item**, which gets its `item_id` with its
first append; under `server_vad` an item is one utterance the server found.

A session holds at most `speech.max_queued_audio_seconds` (120) of audio that
is not computed yet: received, and still waiting for the speech model's
windows or the diarizer's steps, a second that both wait for counted once. An append that would take it past is
refused with an `error`, `ignis_audio_queue_full`, before any of it is read,
and the session stays open: send it again once the session catches up. A
client fed at real time never meets it; one that sends a file flat out paces
itself by it, or sends the file nearer real time.

- `input_audio_buffer.commit` ends the item: `input_audio_buffer.committed`
  `{item_id, previous_item_id}` at once, then its transcription. A commit with
  no audio since the last one is an `error`, `input_audio_buffer_commit_empty`.
- `input_audio_buffer.clear` discards the item, its audio and its partial text,
  answered `input_audio_buffer.cleared`.

## Turn detection

Under `server_vad` the server runs a voice activity detector on the CPU,
[Silero VAD](https://github.com/snakers4/silero-vad)'s model ported to ignis's
own code, on every 32 ms of the session's audio: each gets a speech
probability. That costs about a hundredth of one core, beside the audio
itself.

- The first 32 ms at or over `threshold` start an item:
  `input_audio_buffer.speech_started` `{audio_start_ms, item_id}`. The item
  keeps `prefix_padding_ms` of the audio before the speech, so a soft first
  syllable is not cut; `audio_start_ms` is where that padding begins. The
  padding takes the silence before the speech, even the silence that ended
  the item before, and never an earlier item's speech: when speech resumes
  right after an item ends, the two items' times overlap by that silence.
- Once the speech has been under `threshold` for `silence_duration_ms`,
  `input_audio_buffer.speech_stopped` `{audio_end_ms, item_id}`, and the server
  commits the item with that silence in it: `input_audio_buffer.committed`,
  then its transcription, as for a commit of the client's.
- Audio outside an item goes nowhere and is not transcribed.
- With `idle_timeout_ms` set, that much audio with no speech, since the
  session opened, the last item ended or the field was set, is committed as
  an item of its own after `input_audio_buffer.timeout_triggered`
  `{audio_start_ms, audio_end_ms, item_id}`. It costs a transcription of the
  silence, whose transcript is usually empty.

Every time is in milliseconds of the audio the session has received, from its
first sample, and falls on a 32 ms step. The client may still commit: a commit
during an utterance ends its item at once, with no `speech_stopped`. A commit
when the server has heard no speech since the last item makes an item, as
OpenAI's does, of the audio appended since that item ended (at most
`speech.max_clip_seconds` of it, the leading silence below included): a quiet
speaker's dictation is transcribed, not dropped. With nothing appended since,
it is an `error`, `input_audio_buffer_commit_empty`. A clear discards the
utterance in progress, the padding the server holds and that audio.

The speech model reads each of these items after `speech.vad_leading_silence_ms`
(500) of digital silence, which the server adds as it feeds the item to the
model: Qwen3-ASR's transcript depends on how much audio comes before the
speech, and an item starts only `prefix_padding_ms` before it. The client sees
none of it: the times, `usage` and the events are the audio's. On eleven
Italian dictations with the default model, the items joined score 4.7% WER
with it, 7.6% without it, against 4.0% for one manual commit of each whole
dictation. The silence counts toward `speech.max_clip_seconds`, since the
model reads it: an utterance that reaches the limit with it fails, and the
error names the silence. A manual commit's item is fed as sent.

| `server_vad` field | |
|---|---|
| `threshold` | 0 to 1; default `speech.vad_threshold` (0.5). Higher needs louder speech, for a noisy room. |
| `prefix_padding_ms` | default `speech.vad_prefix_padding_ms` (300). |
| `silence_duration_ms` | default `speech.vad_silence_duration_ms` (500). Shorter commits sooner, but cuts at a short pause. |
| `idle_timeout_ms` | default `null`, no idle timeout. |
| `create_response`, `interrupt_response` | Accepted and inert: they belong to conversation sessions. Echoed when set. |

A `session.update` with `turn_detection` `{type: "server_vad", ...}` changes
the fields it names; a field left out keeps its value, and `null` is the
server's default. The three lengths take at most `speech.max_clip_seconds`,
since an item holds no more. The fields hold at once; a switch between `null`
and `server_vad` while an item holds audio holds from the next item, as a
changed context does.

`turn_detection: null` is **manual commit**: every sample goes to the current
item and the client commits it. The Playground's dictation commits this way, at
**Done**, when its silence setting is Off or the shortcut is held; by default it
asks for `server_vad` with a 3 s `silence_duration_ms` and treats the first
commit as Done ([dictation](../playground/chat.md#dictating-a-prompt)), with
the length guard off (`ignis.max_utterance_seconds: 0`), so a long dictation
with short pauses is not ended by it.

### Long utterances

Under `server_vad` an item ends only at a silence, so a room that never falls
silent would hold one open until `speech.max_clip_seconds` fails all of it.
**The length guard** ends it sooner: once an item holds
`speech.max_utterance_seconds` (120) of audio, the server ends it at the next
32 ms the VAD hears under `threshold`, a pause too short to end the item, or
`speech.utterance_cut_search_ms` (10000) later at the latest, where the audio
is. The search looks forward only, so no text already sent is past the cut.
The events are those of an utterance that ended and one that began at once:
`speech_stopped` with `audio_end_ms` at the cut, `committed`, then
`speech_started` for a new item with `audio_start_ms` at the cut, which takes
no prefix padding (it would be the end of the item before) and is fed after
the leading silence as every item is. The speech goes on in the new item,
with the same speaker. A session sets its own limit with
`ignis.max_utterance_seconds`, `0` for none.

## The text

While audio arrives the server refreshes the item's transcript at most every
`speech.partial_interval_ms`, and once the item is committed it decodes the
rest. Refreshes are best effort: each one decodes the item so far again, so
while the session's audio is still being computed a refresh is skipped rather
than queued behind it. A CPU that cannot keep up sends fewer deltas, never a
later transcript: the audio and the commit go first. The text after the last
refresh is decoded at the commit, and a commit (or a clear) that arrives
while a refresh is running stops it at its next check, between the steps of
its work and before every token it decodes: a stopped refresh sends nothing,
and the commit decodes from the text already sent. Measured on a 67 s dictation
with the default model, before a commit stopped the refresh, the commit took
2.4 to 4.4 s to `completed` on a 10-core desktop CPU (2 to 2.5 s of its own,
the rest the refresh it waited for; 1.7 to 2.2 s for the median of eleven
dictations) and about 6 s on 4 cores; the time with the refresh stopped is
not measured yet. A busier or slower
machine adds to it: an earlier run on the same desktop, its windows computing
60% slower, gave 6.7 s. A longer `speech.partial_interval_ms` (4000) gave
2.8 s.

A refresh is not queued before `speech.partial_cost_factor` times the last
refresh's compute time has passed since that one was queued, so refreshes take
at most about one part in that many of the speech model's CPU. The default, 1,
only waits for the last refresh to finish: 2 and 3 measured no faster on either CPU, and send fewer
deltas.

- `conversation.item.input_audio_transcription.delta` `{item_id,
  content_index: 0, delta}` carries **stable text only**, text the model will not
  revise, so deltas only ever append.
- `conversation.item.input_audio_transcription.completed` `{item_id,
  content_index, transcript, usage: {type: "duration", seconds}}`. Its
  `transcript` is the item's deltas joined, unless the item is taken for a
  copy of the context: then it is empty, whatever its deltas said.
- `conversation.item.input_audio_transcription.failed` `{item_id,
  content_index, error}`: the item held more than `speech.max_clip_seconds`
  (1200) of audio, it ran past `server.stream_timeout` from its first audio
  (`request_timeout`, [deadlines](deadlines.md)), or the speech model failed on
  it. It comes at the commit, or at once if the item is already committed; an
  item that fails before its commit also gets an `error` at once, with the same
  reason, so the client can stop sending its audio. The session goes on with the
  next item.

**The provisional tail** is ignis's. The last few tokens of each refresh
are not stable yet: the next refresh may revise them (how many is
`speech.rollback_tokens`, 10, which the speech executor reads when a session
opens, so a change holds from the next session; the first `speech.unfixed_partials`
(2) refreshes fix no text at all, so the language and the opening words are
decided on more than one interval of audio). A session that set `ignis.provisional_tail: true` receives them as

```json
{ "type": "ignis.input_audio_transcription.provisional", "item_id": "item_…", "content_index": 0, "text": " per il" }
```

after each refresh that changed them: the whole current tail, which replaces
the previous one, and `""` when the item ends. Show the deltas, then the tail
greyed after them. A session that does not opt in never receives the event, so
a client that rejects unknown events is safe.

**The item's language** is ignis's too. Qwen3-ASR writes the language it heard
before each transcript, and a session that set `ignis.item_language: true`
receives it for each item, as its ISO 639 code:

```json
{ "type": "ignis.input_audio_transcription.language", "item_id": "item_…", "content_index": 0, "language": "it" }
```

It comes right before the item's `completed`, after its
[speaker](#speakers) in a session with speakers. Each item's language is
guessed from its own audio, so the items of a conversation show the languages
it is held in, and a short item the model mislabels shows up as a language
nobody spoke. Under a forced language the event carries that one. An item the
model labelled `None` (no speech) gets no event, and neither does one whose
label is not one of the model's 30 languages. A session that does not opt in
never receives it.

## Speakers

A session that sets `ignis.diarization` gets each item's **speaker**: the
diarizer, Nemotron 3 Diarization on the CPU beside the speech model, reads the
session's audio as the VAD does, from where the speakers start, and tells up
to 8 voices apart, numbered in the order they first spoke. It never
transcribes. `"live"` is its geometry for audio at real time, which scores
2 s of audio a step and waits for 0.32 s more: a frame's speaker is known
about 2.3 s after it is heard. `"file"` is the card's offline geometry, for a
file fed faster than real time: 27.2 s a step and 30.4 s of wait, at a
fraction of the CPU. The `diarizer` group sets both ([speech
configuration](../configuration/speech.md#the-diarizer)).

```json
{ "type": "ignis.input_audio_transcription.speaker", "event_id": "event_…", "item_id": "item_…", "content_index": 0, "speaker": 1 }
```

It comes after the item's last delta and before its `completed`; a session
that does not opt in never receives it. An item's speaker is the voice that
holds most of its speech; an item with no speech in it (an idle timeout's
silence) gets no speaker event. A voice counts as a speaker of its own only
once it has spoken `diarizer.min_speaker_seconds`; until then its speech
counts for the speaker it sounds most like, so a short mistaken voice splits
nothing. The session's first voice is a speaker at once. A ninth voice lands
on one of the eight.

- **One speaker per item.** When the speaker changes inside an item, the
  server ends the item at the change, as it does at a silence:
  `speech_stopped` with `audio_end_ms` at the first 80 ms of the new speaker,
  `committed`, then `speech_started` for a new item with `audio_start_ms` there,
  with no prefix padding and the leading silence. The change is found when the
  diarizer's frames show it, after the audio was heard: the item ended is
  decoded again from its audio up to the change, and the audio from the change
  on is the new item's. A change found only once the item has ended does not
  split it: the item takes the speaker that holds most of it.
- **The text comes at the end.** A change can move an item's end after its
  text was decoded, and the model gives no word times, so in a session with
  speakers no text is fixed before the item ends: the refreshes still run and
  the provisional tail carries all of the item's text, and at the end one
  `delta` carries the whole transcript, so the deltas joined still equal
  `completed.transcript`. A session with speakers that does not opt in to the
  provisional tail sees an item's text only at its end.
- **`completed` waits for the speaker**: until the diarizer's frames cover the
  item's end. While the audio keeps coming, the diarizer's next step covers
  it, with its look-ahead. When it stops (a file's end, a pause in what the
  client sends, a commit with no audio after it), the server waits as long as
  that step's missing audio takes at real time, plus the client's last piece
  again (what a client at real-time pace takes to send it), then scores what
  the diarizer has heard past its last step as if the audio ended there. That
  wait is at most the geometry's latency (2.32 s live, 30.4 s for a file) and
  the last piece.
- A diarizer that fails is an `error`, `ignis_diarizer_failed`, saying why, and
  the session goes on without speakers.

## Several sessions, and the queue

One compute worker serves the speech model. Up to `speech.max_sessions` (4)
sessions are open at once; they take turns a unit of work at a time (a window
of audio, a refresh, or a step of a commit: a window, 104 tokens of the
decode's prefill or 16 tokens decoded) and every unit goes ahead of the
[transcription requests](transcriptions.md#waiting-and-refusing), which run in
the gaps. The diarizer has a thread of its own beside the worker, with CPU
threads of its own (`diarizer.threads`, [speech
configuration](../configuration/speech.md#the-diarizer)): the sessions with
speakers take turns there a step at a time, and a step never waits for the
speech model's work nor stops it. A refresh is skipped only while its own
session's speech work is in the worker, never for a step, so with speakers on
the live text keeps the pace it has without them.

A session with no committed item still being transcribed and no frame from its
client for `server.socket_idle_close` seconds (300; 0 never) is closed with a
normal close frame (1000) whose reason says so, as the
[Responses socket](responses.md) is, and the server logs `ignis.ws.idle_closed`
with `route: "realtime"`. Open a new session to go on.

While a session is open the speech model stays loaded, however quiet the
session; `speech.idle_unload` counts from the end of the last one.

## What is logged and counted

As for a transcription: neither the audio nor the text is logged. Each
committed item writes one `ignis.speech.transcription` line, its wait counted
from the commit (at WARN with a `reason` when the speech model failed it), and counts in `ignis_speech_transcriptions_total` and
`ignis_speech_commit_wait_seconds`; `ignis_speech_sessions_open` is the open
sessions ([metrics](../reference/metrics.md)).

A session with [speakers](#speakers) writes one more line when it ends,
`ignis.speech.diarization`: its `speakers` (how many voices its items were given),
its `items` (how many had one), its `steps`, the diarizer's step time on its
own threads, `step_mean_ms` and `step_max_ms`, its `tails` (the times the audio
stopped with an item waiting for its speaker, and the diarizer scored what it
had heard past its last step), and `lag_max_ms`, the most a stretch
of audio waited from its arrival for the diarizer's view of it: on a machine
that keeps up, the geometry's latency (2.32 s live) plus a step or two. The diarizer's load is `ignis.diarizer.load`, with its
`threads` and the speaker cache's constants in force (`auto` resolved to the checkpoint's values, or the
override); a failed one is `ignis.speech.diarizer_failed`.
