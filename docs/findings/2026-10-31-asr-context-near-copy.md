# Given a pause, the speech model writes a short topic again and again through a meeting, now and then with a word or two changed

- Kind: experiment
- Status: current
- Observed: 2026-10-31
- Last verified: 2026-10-31
- Scope: speech / Qwen3-ASR-1.7B (the `auto` entry, Q4_K_M decoder) on the CPU through the speech crate: the text
  of three meetings' short items, cut as a Realtime session with speakers cuts them, under thirteen contexts of a
  topic and many multi-word terms; the guard `copies_context`, its near-copy rule (`speech.context_copy_edits`) and
  a same-language control of what that rule takes of real speech
- Related: [#442](https://github.com/gpillon/ignis/issues/442), [the first measurement](2026-10-27-asr-context-echo.md),
  [#428](https://github.com/gpillon/ignis/issues/428), [spec speech/07](../specs/speech/07-transcribe.md),
  [Speech to text](../user/api/transcriptions.md)
- Superseded by: none

## Question

On a long call in the Transcribe view, with a context of many terms applied, the transcript held about twenty
utterances whose whole text was one short phrase of four words, attributed to other voices and spaced through the
call; some differed from the others by one word's inflection, and the guard of 2026-10-27 (`copies_context`) emptied
none. The owner's audio and context are not available. On public recordings, under contexts of that kind: is such a
phrase a copy the guard misses for another reason, or a near copy its exact-words rule misses (a word inflected, a
term turned into an imperative)? Does it fall on pauses, backchannels or overlaps? And what would a rule that also
empties near copies take of what people actually say?

## Evidence

**The probe.** `crates/speech/examples/context_probe.rs`, built in the dev profile, 6 threads, greedy, one run at a
time, 2026-10-31. It cuts each meeting into the items a Realtime session with speakers gives (spec speech/08): the
reference turns (RTTM), a speaker's joined while no one else starts between and the silence between is under 1.5 s
(the Transcribe view's default utterance end), each from 300 ms before (the prefix padding) to 1.5 s after or to the
next turn's start, after the server's 500 ms of leading silence. It keeps the items of 3.5 s or less, where copies
fall, and transcribes each under each context with the guard off, so every row is the model's own text. The items
run in one push (`--batch`): an item of 3.5 s or less gets at most one refresh at `speech.partial_interval_ms` (2000)
and `speech.unfixed_partials` (2) fixes no text from it, so its commit decodes from no stable text, as one push does.
`--rescore` applies the guard of a build to recorded rows, with `speech.context_copy_edits` at each of 0 (the rule
of 2026-10-27) to 3, and with `--control` applies other contexts to the same rows; every count below is its output.
Two runs of the same items gave the same text, 165 of 165. The rows, the contexts and the scripts that built them
are kept on the study disk, outside the repo.

**The meetings** (AMI Meeting Corpus, Mix-Headset, CC BY 4.0, English): ES2004a (17.5 min, 156 short items of 249),
IS1009a (14.0 min, 105 of 184), TS3003a (14.2 min, 53 of 97).

**The contexts** (each `Languages: …` then `Topic: …` as `prompt`, the terms as `keywords[]`, which the server joins
as one line). Thirteen: Italian contexts on English audio, where the 2026-10-27 probe found its copies, and two
English ones of the meeting's own talk:

| context | topic | terms |
|---|---|---|
| `page16` | `migrazione della pila software verso GitOps.` | the 2026-10-27 probe's 16 |
| `it-phrases` | the same | 16, half of them phrases of 2 to 5 words (`chiusura del ticket aperto`, `aggiornamento della documentazione tecnica`, `pipeline di integrazione continua`) |
| `it-topic-many40` | the same | 40: those 16 and 24 more, most of them phrases |
| `it-many40` | none | the 40 |
| `it-other` | the same, and a line of free text: `Si parla del rilascio in produzione e della chiusura dei ticket aperti.` | the 16 phrases |
| `it-topic4` | `revisione settimanale del rilascio` | the 16 phrases |
| `it-topic4-verb` | `aggiornare la documentazione tecnica` | the 16 phrases |
| `it-topic-team` | `riunione settimanale del team di vendita` | the 16 phrases |
| `it-topic-abbr` | `revisione del budget Q3 2026` | the 16 phrases |
| `it-topic-hyphen` | `on-boarding dei nuovi clienti B2B` | the 16 phrases |
| `it-topic-app` | `nuove funzionalità dell'app mobile` | the 16 phrases |
| `en-phrases` | `kickoff of the remote control design project` | 16 English phrases of the meeting's own talk (`selling price of twenty-five euros`, `functional design phase`) |
| `en-topic4` | `remote control design kickoff` | 16 shorter ones |

**The short items' answers.** *Exact*: a copy the rule of 2026-10-27 empties (`speech.context_copy_edits` 0);
*near*: one only the near rule empties, at 1 to 3 edits.

| meeting | context | items | exact | near | the exact copies were |
|---|---|---|---|---|---|
| ES2004a | `page16` | 156 | 24 | 0 | the topic, 24 |
| | `it-phrases` | 156 | 37 | 1 | the topic, 37 |
| | `it-topic-many40` | 156 | 28 | 1 | the topic, 28 |
| | `it-many40` | 156 | 53 | 0 | the keyword line's head, 52; a run of it from `Marta` on, 1 |
| | `it-other` | 156 | 26 | 0 | the topic, 26 |
| | `it-topic4` | 156 | 23 | 1 | the topic, 22; the keyword line's tail `Marta, Paolo, Gen Two`, 1 |
| | `it-topic4-verb` | 156 | 34 | 0 | the topic, 33; the keyword line's tail, 1 |
| | `it-topic-team` | 156 | 28 | 0 | the topic's head `riunione settimanale del team`, 24; the topic, 4 |
| | `it-topic-abbr` | 156 | 27 | 0 | the topic with `Q3 2026`, 18; the keyword line, 6; the topic's head, 3 |
| | `it-topic-hyphen` | 156 | 15 | 0 | the topic's head with `on-boarding`, 14; the keyword line's tail, 1 |
| | `it-topic-app` | 156 | 23 | 0 | the topic with `funzionalità dell'app`, 21; the keyword line, 2 |
| | `en-phrases` | 156 | 2 | 0 | the topic, 1; the keyword line, 1 |
| | `en-topic4` | 156 | 1 | 0 | the keyword line, 1 |
| IS1009a | `page16` | 105 | 5 | 1 | the topic, 5 |
| | `it-phrases` | 105 | 6 | 1 | the topic, 6 |
| | `it-topic4` | 105 | 6 | 1 | the topic, 6 |
| | `it-many40` | 105 | 21 | 0 | the keyword line's head, 19; the keyword line, 2 |
| TS3003a | `page16` | 53 | 8 | 0 | the topic, 8 |
| | `it-phrases` | 53 | 7 | 0 | the topic, 7 |
| | `it-topic4` | 53 | 9 | 0 | the topic, 9 |
| | `it-many40` | 53 | 13 | 0 | the keyword line's head, 12; the keyword line, 1 |
| **all** | | **2,660** | **396** | **6** | |

A short topic, or a short head of a longer one, comes back the same way through a meeting: in ES2004a `riunione
settimanale del team` 24 times from 30 s to 1,048 s (median gap 48 s), `revisione settimanale del rilascio` 22 times
from 287 s to 1,048 s, `aggiornare la documentazione tecnica` 33 times from 18 s to 1,048 s.

**The near copies**, every one in every row recorded:

| meeting, item | context | the model wrote | the line | edits |
|---|---|---|---|---|
| ES2004a 603.6 s | `it-phrases` | `migrate della pila software verso GitOps` | the topic, `migrazione della pila software verso GitOps.` | 1, a word changed |
| ES2004a 603.6 s | `it-topic-many40` | `migrate la pila software verso GitOps.` | the same topic, as an imperative | 2, two words changed |
| ES2004a 519.6 s | `it-topic4` | `la revisione settimanale del rilascio` | the topic, `revisione settimanale del rilascio` | 1, a word added |
| IS1009a 358.3 s | `it-topic4` | `La revisione settimanale del rilascio` | the same | 1, a word added |
| IS1009a 358.3 s | `it-phrases` | `La pila software, Argo CD, …, Gen Two, OKD.` | the keyword line, whole | 1, a word added |
| IS1009a 358.3 s | `page16` | `La pila software, Argo CD, Forgejo Enterprise, …` | the keyword line, whole | 1, a word added |

The same audio under other contexts: at 603.6 s `Right. Yeah.` (`page16`, `en-phrases`, `en-topic4`, `it-topic-app`);
at 519.6 s `Look at it go.` (`en-phrases`) and the topic whole (`page16`); at 358.3 s `I hope the.` (`it-many40`).
Under the reference turns all three are short items where two speakers overlap: at 603.6 s two backchannels of
0.71 s and 0.34 s, one inside the other; at 519.6 s a 1.08 s turn the next speaker starts over 0.4 s in; at 358.3 s a
1.87 s turn the next speaker starts over 1.4 s in.

**The near rule on the same rows.** At 1 edit it empties 5 of the 6 near copies (not `migrate la …`, two words
changed), at 2 and at 3 edits all 6, and no other row of the 2,660. A near copy is read off a line without its
label: read with it, `Topic` counted as a word, and a three-word topic took a four-word transcript one word off it.

**The same-language control.** What a near-copy rule costs is what people say a word or two off a line of their own
context, and Italian contexts on English audio cannot show it: no English utterance comes within reach of an Italian
line. So English contexts were written for each meeting as its organiser would write them, its agenda and its
terms, in the words its speakers say them (the recorded transcripts), and applied to every utterance recorded of the
meeting: an item's text under any recording context, counted once, less the 336 that copy something of the context
they were recorded under. That is 744 utterances: ES2004a's 509 (all 249 items, the short ones under thirteen
contexts and every one under `en-phrases` and `it-phrases`), IS1009a's 167 and TS3003a's 68 (their short items
under four contexts). Each context starts `Languages: English.`:

| context | meeting | the rest of the prompt | terms |
|---|---|---|---|
| `es-agenda` | ES2004a | `Topic: our first meeting, developing a remote control.` `Agenda: get to know each other, tool training, the project plan, discuss our own ideas.` | 18 terms of the kick-off's slides as the chair reads them: `functional design`, `selling price`, `twenty-five euros`, `production cost`, `twelve fifty`, `profit aim`, `market range`, `target group`, … |
| `es-slides` | ES2004a | `Topic: kick-off meeting of the remote control design project.` `We want it to be original, trendy and user friendly.` | 11 of the slides' lines as the chair reads them: `a selling price at twenty-five euros`, `production cost at twelve fifty`, `profit aim is fifty million euros`, `draw our favourite animal on the whiteboard`, … |
| `es-terms` | ES2004a | none | 15 terms of the meeting's talk: `LCD display`, `older generation`, `flip top remote control`, `cost implications`, `big target group`, … |
| `en-phrases`, `en-topic4` | ES2004a | as above | as above |
| `is-agenda`, `ts-agenda` | IS1009a, TS3003a | the same kick-off's topic and an agenda (`opening, tool training, project plan, discussion, closing`; TS3003a's with its `this is our agenda` first) | the slides' 18 terms |
| `is-terms` | IS1009a | `Topic: what the new remote control should do.` | 10 of its speakers' terms: `light sensor`, `a sound or a signal`, `sufficient features`, `my role is to design`, … |
| `ts-terms` | TS3003a | `Topic: tool training on the whiteboard.` | 8 of its speakers' terms: `draw an animal`, `open it from the folder`, `write your name`, … |

Under them (3,015 utterances under a context), and against the near copies of the rows above:

| `speech.context_copy_edits` | real utterances emptied | of them, kept by the exact rule | near copies emptied, of 6 |
|---|---|---|---|
| 0 | 3 | 0 | 0 |
| 1 | 7 | 4 | 5 |
| 2 | 10 | 7 | 6 |
| 3 | 10 | 7 | 6 |

Every one of them, all said for real (the same item gave the same words under every recording context, and none was
a backchannel elsewhere):

| utterance | meeting, item | context | its line or piece | edits |
|---|---|---|---|---|
| `Production cost at twelve fifty.` | ES2004a 613.8 s | `es-slides` | `production cost at twelve fifty` | 0 |
| `Big target group.` | ES2004a 653.1 s | `es-terms` | `big target group` | 0 |
| `This is our agenda.` | TS3003a 140.3 s | `ts-agenda` | `this is our agenda` | 0 |
| `Production costs at twelve fifty.` | ES2004a 613.8 s | `es-slides` | the same, a word inflected | 1 |
| `A sound or a signal, yeah.` | IS1009a 571.7 s | `is-terms` | `a sound or a signal`, a word added | 1 |
| `And my role is to design.` | IS1009a 75.8 s | `is-terms` | `my role is to design`, a word added | 1 |
| `Just open it from the folder.` | TS3003a 435.0 s | `ts-terms` | `open it from the folder`, a word added | 1 |
| `Yeah. Um, production cost at twelve fifty.` | ES2004a 613.8 s | `es-slides` | two words added | 2 |
| `Yes, and my role is to design`, with and without its period | IS1009a 75.8 s | `is-terms` | two words added | 2 |

Italian, as people say it, is in the crate's unit tests (`copies_context`), against the contexts above: a chair
opening the item (`Allora, revisione settimanale del rilascio.` under `it-topic4`), a term after its preposition
(`Nella pipeline di integrazione continua.`), the topic with its article (`la revisione settimanale del rilascio`)
and a verb conjugated (`Aggiorniamo la documentazione tecnica.` under the topic `aggiornare la documentazione
tecnica`). Each is one edit off a line or piece of four words and keeps three: the exact rule keeps it and the near
rule empties it from 1 edit. The third is word for word the near copy the model wrote at IS1009a 358.3 s.

**The 2026-10-27 probe's 250 answers** (`run1.jsonl`, rescored): 15 emptied at every edit count from 0 to 3, nothing
else; the two copies it leaves are `migrazione della`, two words, as before.

Run again through this probe (`--dir`, one push each, as the route takes a clip), that probe's 50 inputs under its
five contexts gave the same text 246 times in 250; the other 4 were the `<asr_text>` marker under no context, which
the parser has dropped since that measurement. The rescore of the new answers: 15 emptied at every edit count from 0
to 3.

**A long call.** ES2004a whole under `it-phrases`, all 249 items (22 min of items, 17.5 of meeting): the topic as the
whole transcript of 39 items, 2 of them over 3.5 s, all emptied by the exact rule, and the one near copy above.

## Finding

Observed:

1. No exact copy escaped the guard: 396 of 396, through abbreviations (`Q3 2026`), a hyphen (`on-boarding`), accents
   and apostrophes (`funzionalità dell'app`), labels, the keyword line's head and a run of it.
2. A copy can have a word or two changed or added: 6 of the 402 copies (1.5%), the topic or the keyword line with an
   article put before it (`La …`, 4), its first word changed (`migrate`, 1) or turned into an imperative with its
   article (`migrate la …`, 1). The guard of 2026-10-27 kept all 6 as speech.
3. Every near copy fell on a short item where two speakers overlap; under other contexts the same audio was a
   backchannel (`Right. Yeah.`) or a short remark.
4. A short topic, or the head of a long one, is written as the whole transcript of one short item in six or seven
   (`riunione settimanale del team`, 24 of 156), spaced through the meeting: the issue's shape, a four-word phrase
   twenty times through the call, from the other voices' short items.
5. People say a line of their context a word or two off it, and a rule on the text cannot tell them from the model:
   on the same-language control every edit count above 0 empties real speech the exact rule keeps (4 utterances at
   1 edit, 7 at 2 and at 3, among them a slide's line read out with a word inflected), while it gains 5 or 6 near
   copies; and one Italian near copy is word for word what a speaker says.

Inferred:

- The owner's phrase was most likely a short topic or the head of one, written on the other voices' backchannels and
  overlaps, as here. That none of theirs was emptied while every exact copy here was suggests their forms were not
  exact: a word or two off the line, as the 6 here.
- A kept near copy is a phantom line a reader, or the notes, may take for speech; an emptied utterance is speech lost,
  and nothing shows it was there. On this evidence no edit count above 0 trades the one for the other without loss.

## Implications

- `speech.context_copy_edits` defaults to 0, the rule of 2026-10-27, and takes 0 to 3, since emptying what people
  say is worse than keeping a copy and every value above 0 emptied real speech on the control: at 1 edit the rule
  already takes the slide's line read out with a word inflected, and the Italian above (a title after `Allora,`, a
  term after its preposition, the topic with its article, a verb conjugated); the crate's tests hold them as speech
  at the default.
- Above 0 (an operator who takes that cost), `copies_context` (crates/speech/src/text.rs) also takes for a copy a
  transcript at most that many words (changed, added or dropped) from a whole line of the context, read without its
  label, or from a whole piece of one between commas or sentence marks (a keyword, or a clause of a prose line: every
  line is split), while it keeps `speech.context_copy_words` (3) of its words, those edits taken off the shorter of
  the two. A line or piece of `run` words or fewer is thus matched only exactly, and one of `run` + k words with k
  edits at most: a four-word line takes one.
- The rule matches whole lines and pieces, never any window of a line: a window of the keyword line a word off is
  real speech (`Jenkins e Helm Chart`).

## Limits and unknowns

- The owner's context, audio and copies are not available; their forms are inferred, not seen. On AMI the repeated
  short phrase came back exact, and the guard already empties it.
- Items are cut from the reference turns, not by the diarizer, and run in one push; English audio with Italian
  contexts, as in the first measurement. The rate on a live Italian call is not measured.
- The control's utterances are the model's transcripts, not AMI's reference transcripts, and its contexts were
  written for it from those transcripts, with phrases the speakers then say: a context of other phrases may cost
  less. It is thin on IS1009a and TS3003a, whose short items alone were transcribed (167 and 68 utterances), and its
  Italian side is four unit tests, not a recording.
- A head of a line with a word changed was not seen in 2,660 answers, and is not matched.
- Three meetings, thirteen contexts, 6 near copies: the share of near copies (1.5%) is rough.

## Follow-ups

- If a copy is reported again that the guard keeps, its text and the context's line it echoes are what decides
  between a narrower near rule (one that would keep a line said with a word in front, which the 6 here and the
  control's losses do not separate), a rule on a phrase repeated across the call, which #442 allowed, or keeping the
  phrase out of what the speech model is told.
