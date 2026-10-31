# 07 - the Transcribe view: a conversation transcribed in the Playground

GitHub: #362 (master #350). Track: Transcribe (spec 08 is the diarizer). Feature: `spec.md`. Needs 05
(`server_vad`); the speakers and every server change it relies on are spec 08's. Web only.

ADRs: 0067 (the speech model), 0069 (the speakers).

Decided 2026-10-25 in the Transcribe grill (Q81-Q118, the log in `.scratch/speech-grill-2026-10-23/`), with the
spike Q91 asked for: the finding [Nemotron on the CPU](../../findings/2026-10-25-nemotron-on-the-cpu.md).

## Problem Statement

- The Playground takes speech only as **dictation** (spec 04): one session per dictation, ended by Done or by its
  first server commit (Q80), its transcript landing in a composer. Nothing transcribes a whole conversation: a
  meeting in a room, a call on the PC, a recording.
- The owner wants a section that does: it hears a conversation, writes it down utterance by utterance with who spoke,
  lets the labels be renamed and corrected live, and keeps the result (the grill's ask).

## Solution

**The implementer designs the view with the `frontend-design` skill** (Q103). No mockup was made; the owner trusts
the skill. The view follows the Playground's existing style and components.

### The view (Q81, Q84)

- **A new header tab, Transcribe**, beside chat, Decide, Image, Monitor and Server (`web/src/app/Header.tsx`'s
  `View`). It is mounted from its first visit on, as Decide is, so a listening goes on while the user looks at
  another view. The header shows that a listening is under way, from any view.
- **What it holds:** a **Transcript**, the record of one conversation heard there. Its **utterances** come in order,
  each with its times, its **speaker** and its text, and never its audio (GLOSSARY.md).

### Sources (the grill's first round, Q93, Q94)

- **Microphone:** a conversation in the room. One Realtime session.
- **Call:** the microphone plus one `getDisplayMedia` pick: a tab (its audio), the whole screen with system audio
  (on Windows every application's sound, mixed), or a window where the browser offers its audio (Chrome 141+
  `windowAudio`). **Two sessions, never mixed**: the microphone's voice is "Me", the shared audio's are the others
  (Q93, Q112).
  - Before the call starts the view says to use headphones: with speakers, the far voices come back into the
    microphone and are transcribed twice (Q93).
  - The two sessions' times are aligned on their wall-clock starts, and their utterances are merged in time order.
- **File:** a recording the user picks. It is decoded in the browser (`decodeAudioData` into an
  `OfflineAudioContext` at 16 kHz mono: wav, mp3, ogg, m4a, webm, an mp4's audio) and fed over Realtime faster than
  real time (Q94):
  - **The client paces it** (Q99): it sends while the audio not yet transcribed stays under N seconds. That audio is
    the audio past the end of the last utterance whose `completed` came. N is a view setting, per viewer, 60 s by
    default. The server's own limit is spec 08's.
  - A progress bar and an estimate of the time left.
  - The session asks for the file geometry (spec 08).

### The session (Q83; not a dictation's)

A listening opens its session(s) at Start and closes them at Stop, which commits the last item. It is **not** the
dictation's session. `web/src/speech/realtime.ts`'s `openTranscription` today configures a dictation, and the view
must not inherit three of its behaviours. It gets a sibling, or a parameter, that:
- asks for `server_vad` at the server's defaults (`turn_detection: {type: "server_vad"}` with no lengths, so the
  `speech.vad_*` fields hold), never the viewer's auto-stop silence (Q80);
- treats **every** `input_audio_buffer.committed` as one utterance, and never ends at the first;
- opts in to `session.ignis.provisional_tail: true` (Q82) and, when Speakers is on, `session.ignis.diarization:
  "live"`, or `"file"` for a file (spec 08). It leaves `session.ignis.max_utterance_seconds` unset, so the server's
  length guard holds (spec 08).

The shared pieces stay shared: the capture (`capture.ts`, `pcm.ts`), the refusal and availability handling
(`availability.ts`), the busy wait. One listening takes one of `speech.max_sessions`, or two on a call.

### Live text (Q82)

- The utterance under way is drawn as the dictation strip draws it: its fixed text, then the provisional tail drawn
  light. It settles on `completed`.
- With Speakers on, the server fixes no text before an item ends (spec 08): the whole utterance under way is the
  provisional tail until `completed`.
- **Following** (Q102): the view follows new utterances. Scrolling up stops following and shows "Jump to live".

### Context and language (Q87, Q88)

- A free **context** field (topic, names, jargon) is sent as the session's `prompt` at open. Changed mid-listening,
  it goes out in a `session.update` and holds from the next utterance. Earlier utterances are never sent as context.
- No `language` is ever sent: the model detects it per utterance.

### Utterances and speakers (Q92, Q98, Q100, Q109, Q112)

- **Times** are relative to the listening's start, `mm:ss` (`h:mm:ss` past an hour): an utterance starts at its
  `speech_started` and ends at its `speech_stopped` (spec 08 gives both times to every item). A live listening also
  keeps its wall-clock start, shown in the Transcript's header. A paused and resumed listening continues its clock.
- **Speakers.** "Speaker 1, 2…" are numbered in the order they first speak in the Transcript, from spec 08's
  `ignis.input_audio_transcription.speaker`. On a call the microphone's utterances are "Me", and the shared audio's
  are diarized into Speaker n, or "Others" with Speakers off.
- **A per-listening Speakers toggle**, on by default (Q112). Off, no label is shown, or only Me/Others on a call. With
  the diarizer unavailable (`/v1/models`' `speech.diarizer`, spec 08) the toggle is struck, with its reason in the
  way spec 04 states an unavailable mic.
- **Corrections** are mappings in the page, and nothing goes back to the server (Q109):
  - **rename** a speaker: every one of their utterances, past and future;
  - **move** an utterance to another speaker or a new one;
  - **merge** two speakers: one label, and the future utterances of either take it;
  - **split** an utterance at the cursor: two utterances, the second to a chosen or new speaker. No audio is kept, so
    the split acts on the text alone. The split time is proportional to the characters and marked approximate (`~`)
    (Q98).
- **The text is editable inline**, even while listening, on every completed utterance. The edited text is what export
  and Open in chat carry (Q96).

### The Transcript (Q86, Q97, Q101)

- **Saved** in the page's store (`web/src/store/`, IndexedDB, #171) as a new kind of session, held to the store's
  contract tests. Each utterance is saved as it completes. Text, times, speakers and the corrections are saved; the
  audio never is.
- **Title:** automatic, "date time · source" (for example "25 Oct 15:42 · Call"), renamable. It is listed in the
  view's sessions panel, as the chat's and Decide's are.
- **Pause** stops the capture. **Resume** goes on, opening a new session when the server closed the old one
  (`server.socket_idle_close`), with the times continuing from the offset.
- **Continue** on a saved Transcript opens a new listening that appends to it.
- **A reload** keeps every completed utterance and ends the listening.

### While listening (Q85)

The mics of the other views are struck, saying "Listening in Transcribe", and open no second session. A dictation
under way when a listening starts ends as a dictation does when another mic starts (spec 04's "one dictation per
page").

### Out of the view (Q89, Q95)

- **Export**, computed in the page from the utterances:
  - the clipboard and `.txt`: one line per utterance, `[mm:ss] Speaker: text`;
  - `.md`: the title and start in a header, then `**Speaker** (mm:ss): text` per utterance;
  - `.srt` and `.vtt`: one cue per utterance from its start to its end, the speaker in front of its text (in `.vtt`
    as a `<v Speaker>` voice tag). An approximate split time is written as it is.
- **Open in chat:** a new chat, its composer seeded with the Transcript as Markdown, never sent by itself. No summary
  is built into the view.

## Acceptance

1. The Transcribe tab is in the header, mounted from its first visit; a listening survives a look at another view,
   and the header shows it from every view (Q84).
2. A listening's session asks for `server_vad` with no lengths, `session.ignis.provisional_tail: true`, and
   `session.ignis.diarization` as the Speakers toggle and the source say. It treats every commit as an utterance and
   sends no `session.ignis.max_utterance_seconds`. A test against a fake socket speaking spec 08's events pins it
   (Q83, Q82, spec 08).
3. The three sources work: the microphone (one session), a call (two sessions, "Me" and the shared audio, merged in
   time order, the headphones sentence before it starts), a file (decoded at 16 kHz mono, paced under N seconds not
   yet transcribed, N a viewer setting defaulting to 60, with progress and an estimate) (Q93, Q94, Q99).
4. Live text: fixed text and the light provisional tail, settling on `completed`; with Speakers on, the whole
   utterance stays provisional until `completed` (Q82, spec 08).
5. The view follows new utterances; scrolling up stops it and shows "Jump to live" (Q102).
6. The context field goes out as `prompt` at open and by `session.update` when changed; no `language` is ever sent
   (Q87, Q88).
7. Times are relative to the start, from each item's `speech_started` and `speech_stopped`; the wall-clock start of a
   live listening is in the header; a resumed listening continues its clock (Q100, Q97).
8. Speakers are numbered by first appearance from the speaker event; a call's microphone is "Me"; the Speakers
   toggle is on by default, and struck with its reason when the diarizer is unavailable (Q112, spec 08).
9. Rename, move, merge and split behave as specified. A merge holds for future utterances, and a split's time is
   proportional and marked `~` (Q92, Q98, Q109).
10. Completed utterances' text is editable while listening; export and Open in chat carry the edit (Q96).
11. The Transcript is saved in the store as each utterance completes, never with audio, under an automatic, renamable
    title, and listed in the sessions panel; a reload keeps the completed utterances (Q86, Q101, Q97).
12. Pause, Resume (with a new session after an idle close) and Continue behave as specified (Q97).
13. While listening, the other views' mics are struck with "Listening in Transcribe" and open no session (Q85).
14. The five exports produce the formats above from the edited Transcript; Open in chat seeds a new chat's composer
    and sends nothing (Q95, Q89).
15. A refused upgrade, a busy server and an unavailable speech model show spec 04's states (the shared
    `availability.ts`); a lost connection keeps every completed utterance.
16. `docs/user/playground/transcribe.md` describes the view and the Playground section of the site lists it.
    `docs/user/playground/sessions.md` names the new kind of session. The page's screenshot comes from
    `make docs-screenshots`, as the other Playground pages' do (AGENTS.md, `docs/agents/user-docs.md`).
17. The tests use synthetic tones and public audio only (FLEURS, VoxPopuli); no real conversation and no name from
    one reaches a commit or an issue (the privacy rule, Q90).
18. The web tests pass, and `cargo test` passes workspace-wide (AGENTS.md).

## Out of Scope

- The diarizer, the speaker event, the cuts, the length guard, the server's queue limit: spec 08.
- A summary inside the view (Q89).
- Keeping the audio, re-transcribing an utterance (Q86).
- A language picker (Q88).
- More than one `getDisplayMedia` pick on a call (Q93).
- `diarized_json` for files on the batch route: a parked idea (Q94).
- Corrections that teach the diarizer (Q109).

## Notes from #361 (2026-10-26, ideas, not ACs)

The owner's third scenario, two people working together for a whole day, raised three things for this view's build:
- **Speakers across sessions.** A quiet socket closes after `server.socket_idle_close` (300 s); Resume opens a new
  session, and a new diarizer stream numbers its speakers afresh (the speaker event carries the stream's channel). The
  view maps channels to its own Speaker n per stream, so after a Resume the same person can come back as a new
  Speaker until the user merges them: worth a sentence in the view, or a merge suggestion.
- **The cost of a microphone left open.** With silence still flowing the socket never idles, and the diarizer steps on
  the silence all day: about 0.33 s of nine threads every 2 s on the i9 (some 20 %), more on four cores. Pause is how
  a user stops it.
- **Hours are not measured.** Nothing has run for hours; the owner's live check of a long session is the test of its
  memory and stability.

## As built

Built 2026-10-26 on branch `speech-transcribe` (#362), web only, in `web/src/transcribe/` (the view, its model and
its listening) and `web/src/speech/` (the session, the capture, the strike of the other mics). The guide's page is
`docs/user/playground/transcribe.md`; its picture is the scene `transcribe` of `docs/site/scripts/scenes.mjs`.

### The pieces

- **The session** is `openListening` in `speech/realtime.ts`, the dictation's sibling (AC 2): `server_vad` with no
  lengths, the provisional tail, `diarization` `live` or `file` when Speakers is on (omitted otherwise, and always on
  a call's microphone), the context as `prompt`, no `language`, no `max_utterance_seconds`. Its tests drive a fake
  socket with spec 08's events.
- **The listening** is `ListeningController` (`transcribe/listening.ts`), one lane per source, each lane a session; a
  lane's session is a stream of its own for the speakers. The clock starts at the Transcript's (a Continue) and
  stands still while paused; a lane's origin on it is its first audio. Its tests drive fake sessions, captures, a
  fake decoder and fake timers.
- **The Transcript** (`transcribe/transcript.ts`) maps a stream's channel to the page's label the first time it
  speaks; rename, move, merge and split are functions of it; it is stored as `transcribe-sessions`
  (`transcribe/stored.ts`), round-tripped on the memory and IndexedDB adapters.
- **The state** is the page's (`transcribe/useTranscribe.ts`), as the Image view's is: the header's badge and the view
  each subscribe to the listening themselves, so its level meters redraw them and not the page.

### Decisions taken while building

- **The file's pace** (Q99). The audio not yet transcribed runs from the start of the oldest utterance whose text has
  not come to the audio sent, and is none when no utterance is under way or waiting. The spec's words ("past the end
  of the last utterance whose completed came") stall on a silence in a file longer than N (nothing completes, nothing
  more is sent) and on an utterance under way longer than N (only more audio ends it). Past N the page still sends an
  utterance under way at real time, and stops only while utterances wait for their text. With the file geometry
  `completed` lags an item's end by up to 30.4 s of audio, so items over about 30 s go stop-and-go at N = 60. An
  append refused for the queue limit (`ignis_audio_queue_full`, its `event_id` echoed) is sent again from there after
  1 s, after the file's end too. One look sends up to 5 s of appends, so if the server refuses one and takes a later
  one of the same look, that later audio goes twice: the pace keeps the backlog under N, and N under the server's
  default limit, so a refusal needs a server limit set lower. N's choices are 45, 60 and 90 s: 120 would sit at the
  server's default limit, and below about 45 s a file with speakers runs near real time.
- **A file's Pause** stops the sending and commits nothing: a commit would cut the utterance under way in two. A live
  source's Pause commits it, as Stop commits the last item (the spec says only that Pause "stops the capture"), and
  only when one is under way: a commit with no speech since the last item would make an item of the silence.
- **An end flushes first.** Stop and a file's end send each session an empty `session.update` and commit the
  utterance under way only once it is answered: the server reads events in order and runs its VAD as it reads each
  append, so by then the VAD's events for the last audio sent have come (the empty update changes no context, so it
  prefills nothing), and an utterance whose start was still on its way is not lost. The session closes
  once every utterance's text has come; a session that closes while one still waits ends the listening as a lost
  connection.
- **A Continue's clock** starts at the larger of the clock the last listening left and the last utterance's end: a
  reload in the middle of a listening keeps the utterances but never hands the clock back.
- **A call's Pause** lets the microphone go and keeps the shared tab, window or screen picked (not sent), so Resume
  asks for no new pick; the browser's own Stop sharing stops the listening as Stop does.
- **A file's title** is its name ("25 Oct 15:42 · standup.m4a"), not "· File": the list tells files apart.
- **The page-wide flag** that strikes the other mics (AC 13) is up while the listening captures, not while it is
  paused: a paused listening holds no microphone, and a dictation then works; Resume ends it as Start does.
- **An unavailable diarizer refuses the whole `session.update`** that asks for speakers (spec 08's server), so the
  session's audio would wait for a configuration that never comes: `openListening` sends the configuration again
  without `diarization`, and the view turns Speakers off with the server's reason. A server whose `/v1/models` says
  nothing of a diarizer is taken as one without: it would refuse the field.
- **Corrections use native menus** (the utterance's speaker, Split at the cursor, Merge into), which open over a
  clipped `.cut` box and on a phone.

### The owner's live check (2026-10-27)

The first try on a call (grill Q121-Q123):
- The server's log showed items of 1 to 2.5 s, each in a language of its own (Thai, Chinese, Portuguese, Hindi;
  the 3.7 and 5.5 s ones Italian), and the far side cut often with one speaker heard: the server's VAD silence,
  500 ms, not speaker changes.
- **Q121: a call holds the mic while the far side speaks.** The page reads the shared audio's level (`HOLD_LEVEL`,
  about -34 dBFS) and sends the microphone's session silence of the same length while it is loud and for
  `HOLD_AFTER_MS` (500 ms) after, so its times stay the clock's and the voices that come back through the speakers
  are not transcribed as Me. On by default, with a switch in the deck (headphones need none); speech over the far
  side is lost while it holds.
- **Q122: the view asks for its own silence** (`silence_duration_ms`), a per-viewer setting of 0.5, 1, 1.5 or 2 s,
  1.5 s by default; it amends "server_vad at the server's defaults" above. A change holds at once.
- **Q123, the language, is open.** Q88 stands: the owner first tries the Context field (an Italian context biases
  the model's guess) before choosing among a language field, a language held after the first long utterance (a
  server change), or nothing more. `transcription.languages` with several entries is no help: the server detects
  among all thirty then.

### Notes (2026-10-27, Q124-Q127)

The owner asked, after the first live check, for the view to sum a conversation up as it goes: a kind of meeting
chosen, the model estimating what that kind wants (a work call's topics, people, actions; a game's games and wins),
in a tab of the right panel as the chat's General and Tools are, kept up during a call, each feature behind a
switch. Built as a draft for the owner to try, beyond Q81-Q123 (transcribe/notes.ts, notesWriter.ts, Notes.tsx):
- **Q124: a Notes tab** beside Tools, open by default (the header's right drawer is "Notes"). Four kinds, each its
  sections: General (Summary, Key points, Next steps), Work call (Summary, Topics, People, Decisions, Action items,
  Open questions), Game session (Games, Players, Results, Highlights), Lesson (Topics, Key concepts, Questions,
  To review). Write notes / Update notes / Stop, Copy notes. The notes and their kind live on the Transcript
  (`notes`, under version 1: a field the page cannot read is dropped on restore, not the Transcript); a new one
  starts at the kind picked last (`ignis.transcribeNoteKind`). Nothing goes to the model before a press or the
  switch below.
- **Q125: the request keeps the prefix.** The system prompt is the kind's alone and constant; the user turn is the
  title and the transcript's `.txt`, which a listening only appends to. The notes written before are not sent back:
  each writing starts from the transcript, so a correction holds and an error does not compound. The headings stay
  as the prompt writes them and the text is in the transcript's language: probed on the 27B (2026-10-27), asking for
  the headings in the conversation's language turned the whole notes English, while "keep each heading as given,
  the text in the transcript's language, a transcript in Italian gets notes in Italian" wrote Italian notes under
  the English headings. A kind that does not fit writes a dash in every section (a Game session on a work call).
- **Q126: Keep updating while listening**, a per-viewer switch, off by default (`ignis.transcribeNotesLive`). A
  look every 5 s: the first notes once 3 utterances are in, then at least 60 s from the last writing's start while
  something new was said, a writing at once when another kind is picked, and one more when the listening ends,
  after the writing under way. A writing that failed waits the 60 s before the next, whatever the cadence says.
- **Q127: the model.** The model loaded, never named, so no request switches (the image model loaded, or none
  serving, strikes the tab with its reason); `class: "agent"`, `reasoning_effort: "none"`, temperature 0.3, top_p
  0.9, at most 1,500 tokens; a full engine is waited out a second at a time, up to 60 times (api/busy.ts, now
  shared with the agents and the safety check). A transcript past the model's context is sent from where its tail
  fits (3 characters a token, 2,500 tokens kept for the prompt and the notes), and the line says from when. The
  probe on the 27B: 6 utterances, 517 prompt tokens, 355 written, 2 s.
- Open for the owner: kinds or sections of their own; the notes in the exports and in Open in chat; whether this
  stays in #362 or becomes a ticket of its own.

### The call's context, its languages, suggestions and corrections (2026-10-27, Q128-Q135)

The owner's second round, approved whole (grill Q128-Q135), built in six slices, one commit each:
- **S2, the panels (Q129).** The right panel's tabs are Call (the context and the speakers), Notes, and Settings
  (Utterance ends after, File pace); it opens on Call, and the header's drawer button says Call. The title, the
  start and the exports leave the scroller: a heading above the sheet with Copy, a Download menu (.txt, .md, .srt,
  .vtt) and Open in chat, so they stay in sight in a long transcript. Hold my mic stays in the deck: it is the
  listening's.
- **S1, the context (Q128).** A Transcript keeps what its transcription is told, `context` under version 1
  (topic, terms, anything else, the languages, `appliedAt`), read back like the notes. The Call tab shows it
  applied, as a record with its time and its share of `speech.context_max_tokens` (512, about three characters a
  token, the server keeping the last), or a draft framed in the ember after Edit: nothing goes out as it is typed,
  where every keystroke was a `session.update` and a prefill before. Apply sends one update (prompt, keywords,
  language; `null` clears what is no longer told), which holds from the next utterance; a Start applies a draft
  left open. The topic and anything else are the `prompt`, the terms and the speakers' names the user gave (never
  "Speaker n", Me or Others) the `keywords`, which the server refuses with `<`, `>` or a line break, so the field
  does too. A rename during a listening is sent as it is made. The deck's first line says what is told, with Edit.
  After the review: an Apply (or a Start) merges the draft onto what the Transcript holds then, changing only what
  the draft changed since Edit (`mergeDraft`), so a suggestion accepted or a fix's term added meanwhile is kept;
  the limit shown is the running server's `speech.context_max_tokens` (`GET /v1/config`), 512 until it is read.
- **S3, the languages (Q130).** The context's Languages: Detect (the default) lists the expected ones, which go
  into the prompt as `Languages: Italian, English.` and nothing goes to `language`; Force one sends the first as the
  session's `language` and leaves it out of the prompt. Measured first (docs/findings/2026-10-27-asr-language-hint.md):
  on 240 pieces of 1 to 2.5 s cut from Italian FLEURS, none heard 96.7 % as Italian (91.0 % under 1.5 s), the hint
  `Languages: Italian.` 100 %, `Languages: Italian, English.` 99.6 %, forced 100 %; worded in Italian (`Lingua:
  italiano.`) it did harm (the word echoed as a transcript, a label `italiano`). On 30 English clips: no hint 5.9 %
  WER, the Italian hint 5.7 %, the two-language hint 9.6 % (one clip translated), forced Italian 24.4 % (seven
  translated). So Detect with the hint in the model's English names, first and with its period as measured, is the
  default (`LANGUAGE_HINT`); Force one is for one language that still flips. Caveats: FLEURS is clean read speech,
  the pieces are cut, not a VAD's, and only the file route was measured (the Realtime context goes through the same
  `route::context_of`); under a hint an English utterance can be labelled Italian (4 to 7 of 30), so the language
  suggestion's count leans to the languages already listed. The guide's `languages[]` sentence ("detect among
  these") was wrong and is fixed in S4 (several entries detect over all 30).
- **S4, each utterance's language (Q131).** The server sends it on request: `session.ignis.item_language: true`,
  then `ignis.input_audio_transcription.language {item_id, content_index, language}` (the ISO code the model labelled
  the item with) after the speaker and before `completed`; none for an item labelled `None`. Every listening asks
  for it; a server older than it refuses the key with `param: session.ignis.item_language`, and the page configures
  the session again without it, saying nothing. An utterance keeps its `language`; the language suggestion (S5)
  reads them.
- **S5, the suggestions (Q132, Q134).** A Transcript keeps its `suggestions` (each open, accepted or dismissed, and
  the kinds accepted always), read back like the notes. The languages come from the utterances' `language` once six
  are in, no model: Detect the top language, as a hint, when it holds 70 % and the others came only on utterances
  under 2.5 s (the owner's first call), else Detect those with 15 % and two utterances; never Force (amended after
  the review: S3 measured the hint as good on those utterances and forcing as translating other speech, and Force
  stays the user's, in the form). The topic, a title (only over the
  listening's own) and the terms come from a look by the text model: first at six utterances, then 90 s from the
  last look's start with three new, once more at the end (assist.ts's Follower, shared with the notes), a failed look
  waiting the gap; Suggest now on demand. An offer made before, whatever became of it, is not made again; a term
  taken out is dismissed alone. Always applies every later one of its kind in that Transcript at once. Accepting
  tells the context (applied, so sent) or renames. The bell beside the tabs counts what waits. Suggest while
  listening, a per-viewer switch in Settings, on by default (they only propose).
  - **Q134 measured, and amended.** The engine reuses an earlier prompt only whole (the checkpoint match is
    all-or-nothing at its length, `crates/core/src/checkpoint.rs`): on the 27B a request whose prompt began with
    the last one's reused 559 of 563 tokens, two tasks after one transcript 64 (the system prompt's first page),
    the transcript in its own message 64 too. So a task after the transcript, or a task shared, is never reused;
    the task in the system prompt with the transcript after it, which a listening only appends to, reused 448 of
    565 on the next request. Each task keeps its own chain: assist.ts's preamble and the task as the system
    prompt, the title and the `.txt` as the user turn; the notes and the suggestions share the shape, not a
    prefix. Measured over HTTP, and again over the Responses socket on the branch's server: the notes' request with
    the transcript grown by two lines reused 448 of 556 tokens on a new stream (`cached_tokens`, and
    `reused_prompt_tokens` in the server's log).
  - **The look's task asks for the misheard names first** (`FIX: [mm:ss] heard => meant`), then TERM, TOPIC,
    TITLE. Asked for the terms alone, the 27B with thinking off copied a misheard product name as heard; asked to
    check them first, it fixed it and spelled the term right, twice out of two at temperature 0.2 (greedy split
    the misheard word in two instead). With the task's examples open-source names, a call where two of them were
    misheard got both fixed three times out of three. Every name in the view's copy, examples and tests is generic
    or open source: no closed product is named (the owner, 2026-10-27). The FIX lines are S6's; S5 reads the rest leniently (bullets, bold, NONE).
- **S6, the fixes (Q133).** A FIX line (`[mm:ss] heard => meant`, `→` and `->` too) becomes a fix suggestion when
  both sides are one to four words the server could take as a keyword, differ beyond case, and `heard` stands as
  words of its own (no letter or digit around it, any case) in at least one utterance: the fix covers every
  utterance that holds it then. Fixes stand side by side, one card each. Accepted, each of those utterances gets the
  replacement, keeps `heard` (its text before its first fix) and `fixed` (the words put in, marked dotted in the
  ember), and the right spelling joins the terms, so the speech model spells it right from the next utterance.
  Undo fix puts the text back as heard, an edit since with it; a split leaves neither half a way back. The exports,
  the notes and Open in chat read the fixed text. Always applies the later fixes as they come; only names and terms,
  never a rephrasing (the task says so, and a side over four words is dropped). A fix offered again (by a look, or
  by `followFixes`, which follows each accepted fix into the utterances heard since, every few seconds and with no
  model) is offered only for the utterances no fix like it covered, gathered into its open card or, accepted
  always, applied; dismissed, never again; an utterance undone keeps its text as heard. Found in answering the
  owner: the first version offered a fix once, so the same misheard word in a later utterance stayed unfixed even
  with Always.

### Search, delete, rewrite, ask (2026-10-27, Q136-Q141)

The owner's third round, after trying the second (grill Q136-Q141), one commit each:
- **Delete (Q137).** An utterance's editor has Delete; the utterance leaves the Transcript and an Undo stands at the
  sheet's foot for 8 s (`UNDO_DELETE_MS`), putting it back where its time places it. Nothing else changes: its
  speaker stays, a fix that names it does nothing to it, and the notes' "more since" never goes below zero.
- **Search (Q136).** search.ts folds case and accents one character for one, so a hit's offsets are the text's.
  The heading's field lights every hit in the open Transcript (the one it stands on in the ember, the others a
  wash of it, over a fixed word's dotted line where both fall), counts "3 of 12", steps with Enter, Shift+Enter
  and the arrows, clears with Escape, and brings the current hit into view, no longer following the newest
  words; Ctrl+F (Cmd+F) focuses it while the view shows. The list's field keeps the Transcripts whose utterances
  or title hold the query, each row saying its matches in the ember; opening one carries the query.
- **Rewrite (Q138).** rewrite.ts: the task in the system prompt; the user turn the context told (told(): languages,
  topic, anything else, then the keywords as Terms), the conversation around the utterance (20 before, 5 after) and
  the utterance named last. `reasoning_effort: "low"` on the interactive lane, the owner's choice after a probe on
  their own misheard line: thinking off rebuilt the product's name 2 times in 3 (once also changing a dialect word),
  thinking low 3 in 3, the dialect kept, about 2 s. The answer is tidied (a label, a time, the speaker's name,
  quotes the model added), shown as a word diff (longest common subsequence) with Accept and Discard; accepted,
  the utterance keeps its text as heard and marks the words put in, so Undo fix and the marks are the fixes'. One
  rewrite at a time; the button is off while the text model cannot be asked.
- **Checked to here (Q140).** A look hands back up to where it read (the end of the last utterance it had), which
  the Transcript keeps as `checkedUntilMs`, a malformed one dropped on restore. The sheet draws a dashed ember line
  with a diamond on the spine between the last utterance read and the next, labelled Checked to here and, while a
  listening keeps up with suggestions, "next check in about 45 s" (the last look's start and the 90 s gap, redrawn
  every 5 s). A rewrite of an utterance below it says the automatic check has not read it yet. The line is not drawn
  before any look, nor once everything was read.
- **Ask, and the moments as links (Q139, Q141).** ask.ts: the task in the system prompt, the transcript and the
  question last, on the interactive lane, thinking off, 800 tokens; the answer streams, and lands on the Transcript
  asked about (`asked`, latest first, read back like the rest; a cross forgets one). In the notes and the answers,
  each `[mm:ss]` that is not already a link's text becomes a link `#t=mm:ss` (jump.ts), styled as a time; a click
  is a jump, which the sheet answers by bringing into view the last utterance that starts within that second, lit
  for 1.6 s, and the right drawer, below `lg`, closes.
- **The review of this round** found, and the next commit fixed: the search brought its hit back into view at every
  change of the Transcript (a new utterance, a keystroke in an editor), and a query carried into a Transcript just
  opened landed at its foot: now only the user's search or step (`seek`) brings it, after the following. The rewrite's
  "not read yet" note left with the proposal and ignored the switch: it stays, and only with Suggest while listening
  on. The look counted the utterances it had read, so a delete hid new ones from the last look: it counts by time.
  Ask and Rewrite are one at a time for the page, and in another Transcript they now say so instead of dropping the
  click and the question. The line also stands at the foot once everything is read; its next check says "once more
  is said" while the look waits for three new utterances. Rows are keyed fragments, so a delete or a split remounts
  no other row; the line is a separator, not an utterance, to a screen reader; Undo takes the focus and waits while
  pointed at or focused; the editor's buttons keep its focus, so Safari's click lands. The answers' settling is one
  helper (`settle`).

### A pause written as the context (2026-10-27)

The owner, on a live call with a context applied: utterances, often, were the keyword line whole. The speech model,
given a VAD cut with little or no speech, writes a line of its system turn instead of what it heard. A probe
(docs/findings/2026-10-27-asr-context-echo.md) reproduced it on AMI's silent gaps and short turns: no copy with no
context or with the language hint alone, 3 to 10 in 50 with a topic and terms; digital silence and white noise never.
The fix is the server's, so every client has it: `copies_context` (crates/speech/src/text.rs) takes a transcript whose
words are a whole line of the context as the model read it (after the token cap) of two words or more, or three or
more of one line's words in a row, a leading `Label:` read with and without, for no speech: empty, labelled only by
a forced language, for partials and commits alike. One word is never a copy. Applied to the probe's answers it
empties 15 of the 17 copies (the two left are `migrazione della`, two words) and nothing else. The parser also drops
a second `<asr_text>` marker, which came back as text 4 times in 50 with no context. A partial is emptied only when
the whole of it is a copy, so a live row can show the first two words of one (`migrazione della`) before its commit
empties it and the row goes. After the fourth review: the run is `speech.context_copy_words` (3; 0 turns the guard
off), the owner's rule that limits are config fields; a line's label is read off only when it is one word of letters
before the first colon, never at a colon further on (`10:30`); and an item emptied as a copy after its stable text
went out as deltas completes empty without the `ignis.speech.stable_text_revised` warning, the API pages saying a
copy is the one case where the deltas joined are not the transcript. Not done: dropping the topic or
rewording the context to copy less is the owner's to decide (Q128); a copied pause that held a real "sì" loses it.

### Ask in the flow (2026-10-27)

The owner, trying Ask in the Notes tab during a call, did not like it, and of three directions picked this one: the
question asked from a bar under the sheet, the answer standing in the sheet where it was asked. The Notes tab keeps
only the notes. Ask.tsx: `AskBar` between the sheet and the deck, there once something is heard, with three quick
questions (What did I miss?, Decisions, Who does what) and the field; while no text model serves, the reason stands in
place of the quick questions. `AnswerAside` on the spine: the end of the last utterance the answer read
(`Answered.untilMs`, kept with it; `answerAt` places it after the last utterance that had ended by then, so on a
call an utterance of the other lane begun before the question and ended after it stands above it, unread; an answer
kept before at the foot) in the time column in the ember, a hollow diamond on the rule, the question and the answer
in a clipped surface, Forget beside the question; while it streams the rule burns, and the sheet follows its foot,
where a new question's answer stands. The quick questions are shown in English and sent with "Answer in the language
of the conversation.", which a probe on :8000 answered in Italian on an Italian call (`.scratch/362/ask_preset_lang.mjs`).
What did I miss? names the first utterance that ended in the last five minutes by its start (`[05:40]`): asked from
the end less five minutes, a time that shares the last line's seconds, the model took `[00:52]` for `[05:52]`, said
nothing came since, and answered in English. Answers are neither searched nor exported, as before; whether they
should be is the owner's (open item c).

### One footer (2026-10-28)

The owner found the two bands at the foot, the ask bar and the deck, each with its rule, not pretty, and saw the
deck's switches as settings left from before the side panel. The ask bar lost its rule: it is the sheet's foot, a
fade above it, and from `sm` its field stands in the sheet's text column, where its answer will, the quick questions
beside it on one row (the bar and the sheet keep the same scrollbar gutter, so the columns line up). The deck is
Start alone: Microphone, Call and File moved into a menu beside it (`StartButton`, this browser's pick under
`ignis.transcribeSource`), the line of what is told went (the Call tab shows it), and Speakers and Hold my mic moved
to the Settings tab. Speakers is now this browser's too (`ignis.transcribeSpeakers`, on at first), where Q112 had it
per listening: a switch in Settings that a reload turns back on reads as one that never took.

### The first Start asks (2026-10-28)

With the deck's line of what is told gone, the owner asked for the reminder where it counts: the first Start of a
Transcript whose context was neither applied nor opened (`asksFirst`: never started, nothing applied, no draft) does
not start; over Start the deck says what the call would be told (`ContextCheck`, `contextQuestion`), with Use the
context of the latest Transcript told something else, which opens it as a draft (`copyContext`), and Edit the
context, the Call tab shows the context framed in the ember, and Start reads Start anyway. The question stays in the
deck and the Call tab is picked, never the drawer, which below `lg` would cover the deck. A second press within
500 ms is the first one's double click and starts nothing. A Start applies what it tells (`contextAtStart`), so a
Continue never asks. Save as default, the owner's addition, keeps a context in this browser
(`ignis.transcribeContext`, defaultContext.ts); a Transcript never started shows it and is told it, and the first
Start asks all the same, the owner's word: it costs a click and spares a call told another's context. Clear the
default, in an empty draft, drops it.

### What did I miss? from a moment (2026-10-28)

The owner asked for other spans than five minutes, without more buttons, and the field, Ask and the quick questions
at one height (all 32 px now). What did I miss? reads from a point (missed.ts, `MissedFrom`): the last five minutes by
default; after the page was hidden or its window left for a minute or more while listening (`AwayWatch`, on the
page's state, so another view counts as present), from the first utterance that ended after what the user had seen;
or from an utterance picked by its time on the spine, the time a button now while a text model serves. The button says
since when (`missedSince`: "since 12:40 · 8 min"); a second press on the time picked, or Escape outside a field (a
drawer out takes the first Escape), puts back the five minutes, and so does asking it. The spine and the times from that utterance on are in the ember while the point
stands or the button is pointed at. Decisions and Who does what read the whole call, as before.

### A pause written as the context, a word or two changed (2026-10-31, #442)

On a long call the owner saw about twenty utterances of one four-word phrase, other voices, through the call, some a
word's inflection apart, none emptied. A probe on AMI's short items (docs/findings/2026-10-31-asr-context-near-copy.md:
three meetings, the items a session with speakers cuts, thirteen contexts, 2,660 answers) found the copies as before,
and as often with a short topic (`riunione settimanale del team`, the head of its line, 24 times from 30 s to 1048 s
of one meeting): all 396 exact ones were emptied, none missed for another reason, and 6 more were a line with a word
or two changed or added (`migrate la pila software verso GitOps.`, the topic as an imperative; `La revisione
settimanale del rilascio`), all on three short items where two speakers overlap. `copies_context` can now also take
a transcript at most `speech.context_copy_edits` words (0 to 3) from a whole line, read without its label, or from a
whole piece of one between commas or sentence marks, while it keeps `speech.context_copy_words` of its words: at 1
edit it empties 5 of the 6, at 2 and 3 all 6, and nothing else in those answers; the 2026-10-27 probe's 15 of 250 are
emptied at every value. After the review the field defaults to 0, the exact rule: on a same-language control (English
contexts of each meeting's own agenda and terms, in its speakers' words, over 744 recorded utterances) every value
above 0 emptied real speech the exact rule keeps, 4 utterances at 1 edit and 7 at 2 and 3 (`Production costs at
twelve fifty.`, a slide's line read out with a word inflected), and Italian said a word off a four-word line
(`Allora, revisione settimanale del rilascio.`, `Nella pipeline di integrazione continua.`, `la revisione
settimanale del rilascio`, word for word a copy the model wrote) is one edit off it. Emptying what people say is
worse than keeping a copy, so the near rule is an operator's choice, with that cost. Not done: a head of a line with
a word changed is not matched (none was seen, and a window of the keyword line a word off is real speech: "Jenkins e
Helm Chart"); a narrower near rule (no rule on the text tells `la revisione settimanale del rilascio` said from
written) and a rule on a phrase repeated across the call, which the issue allowed, were not built.

### Not done here

- The view was built against fakes and the dev mock, which has no Realtime socket. The owner used it live on calls
  on 2026-10-27 and 2026-10-28 (an hour and more, four voices, Ask during the call), which found the context echo;
  a room and a file were not reported checked live.
- The picture of the guide shows a made-up call the scene puts in the page's store; no recording reaches the guide.
