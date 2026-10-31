# Given a pause, the speech model writes a short topic again and again through a meeting, now and then with a word or two changed

- Kind: experiment
- Status: current
- Observed: 2026-10-31
- Last verified: 2026-10-31
- Scope: speech / Qwen3-ASR-1.7B (the `auto` entry, Q4_K_M decoder) on the CPU through the speech crate: the text
  of three meetings' short items, cut as a Realtime session with speakers cuts them, under fourteen contexts of a
  topic and many multi-word terms; the guard `copies_context` and its near-copy rule (`speech.context_copy_edits`)
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
term turned into an imperative)? Does it fall on pauses, backchannels or overlaps?

## Evidence

**The probe.** `crates/speech/examples/context_probe.rs`, built in the dev profile, 6 threads, greedy, one run at a
time, 2026-10-31. It cuts each meeting into the items a Realtime session with speakers gives (spec speech/08): the
reference turns (RTTM), a speaker's joined while no one else starts between and the silence between is under 1.5 s
(the Transcribe view's default utterance end), each from 300 ms before (the prefix padding) to 1.5 s after or to the
next turn's start, after the server's 500 ms of leading silence. It keeps the items of 3.5 s or less, where copies
fall, and transcribes each under each context with the guard off, so every row is the model's own text. The items
run in one push (`--batch`): an item of 3.5 s or less gets at most one refresh at `speech.partial_interval_ms` (2000)
and `speech.unfixed_partials` (2) fixes no text from it, so its commit decodes from no stable text, as one push does.
`--rescore` applies the guard of a build to recorded rows, with `speech.context_copy_edits` at 0 (the rule of
2026-10-27) and at its default; every count below is its output. Two runs of the same items gave the same text, 165
of 165. Rows, contexts and the analysis scripts: `H:\speech-study\context-phrase-442\`.

**The meetings** (AMI Meeting Corpus, Mix-Headset, CC BY 4.0, English): ES2004a (17.5 min, 156 short items of 249),
IS1009a (14.0 min, 105 of 184), TS3003a (14.2 min, 53 of 97).

**The contexts** (each `Languages: …` then `Topic: …` as `prompt`, the terms as `keywords[]`, which the server joins
as one line). Italian contexts on English audio, where the 2026-10-27 probe found its copies, and two English ones of
the meeting's own talk:

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

**The short items' answers.** *Exact*: a copy today's guard empties (`speech.context_copy_edits` 0); *near*: one only
the near rule empties (at its default, 2).

| meeting | context | items | exact | near | the exact copies were |
|---|---|---|---|---|---|
| ES2004a | `page16` | 156 | 24 | 0 | the topic, 24 |
| | `it-phrases` | 156 | 37 | 1 | the topic, 37 |
| | `it-topic-many40` | 156 | 28 | 1 | the topic, 28 |
| | `it-many40` | 156 | 53 | 0 | the keyword line's head, 53 |
| | `it-other` | 156 | 26 | 0 | the topic, 26 |
| | `it-topic4` | 156 | 23 | 1 | the topic, 22 |
| | `it-topic4-verb` | 156 | 34 | 0 | the topic, 33 |
| | `it-topic-team` | 156 | 28 | 0 | the topic's head `riunione settimanale del team`, 24; the topic, 4 |
| | `it-topic-abbr` | 156 | 27 | 0 | the topic with `Q3 2026`, 18; the keyword line, 6; the topic's head, 3 |
| | `it-topic-hyphen` | 156 | 15 | 0 | the topic's head with `on-boarding`, 14 |
| | `it-topic-app` | 156 | 23 | 0 | the topic with `funzionalità dell'app`, 21; the keyword line, 2 |
| | `en-phrases` | 156 | 2 | 0 | the topic, 1; the keyword line, 1 |
| | `en-topic4` | 156 | 1 | 0 | the keyword line, 1 |
| IS1009a | `page16` | 105 | 5 | 1 | the topic, 5 |
| | `it-phrases` | 105 | 6 | 1 | the topic, 6 |
| | `it-topic4` | 105 | 6 | 1 | the topic, 6 |
| | `it-many40` | 105 | 21 | 0 | the keyword line's head, 21 |
| TS3003a | `page16` | 53 | 8 | 0 | the topic, 8 |
| | `it-phrases` | 53 | 7 | 0 | the topic, 7 |
| | `it-topic4` | 53 | 9 | 0 | the topic, 9 |
| | `it-many40` | 53 | 13 | 0 | the keyword line's head, 13 |
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

**What the near rule takes that is not a copy.** Applied to every row above it empties the 6 near copies and nothing
else, at 1, 2 or 3 edits. On ES2004a's whole meeting under `en-phrases` (249 items; the context's phrases are the
speakers' own, and 10 real utterances name one: `remote control`, `draw our favourite animal`), it empties the 2 copies
today's guard empties and no other item: of its other transcripts of three words or more, the nearest to a line or a
keyword keeps none of its words.

**The 2026-10-27 probe's 250 answers** (`run1.jsonl`, rescored): 15 emptied with edits 0 and the same 15 with edits
2, nothing else; the two copies it leaves are `migrazione della`, two words, as before.

Run again through this probe (`--dir`, one push each, as the route takes a clip), that probe's 50 inputs under its
five contexts gave the same text 246 times in 250; the other 4 were the `<asr_text>` marker under no context, which
the parser has dropped since that measurement. The rescore of the new answers: 15 emptied with edits 0 and with edits 2.

**A long call.** ES2004a whole under `it-phrases`, all 249 items (22 min of items, 17.5 of meeting): the topic as the
whole transcript of 39 items, 2 of them over 3.5 s, all emptied by today's guard, and the one near copy above.

## Finding

Observed:

1. No exact copy escaped the guard: 396 of 396, through abbreviations (`Q3 2026`), a hyphen (`on-boarding`), accents
   and apostrophes (`funzionalità dell'app`), labels and the keyword line's head.
2. A copy can have a word or two changed or added: 6 of the 402 copies (1.5%), the topic or the keyword line with an
   article put before it (`La …`, 4), its first word changed (`migrate`, 1) or turned into an imperative with its
   article (`migrate la …`, 1). The guard of 2026-10-27 kept all 6 as speech.
3. Every near copy fell on a short item where two speakers overlap; under other contexts the same audio was a
   backchannel (`Right. Yeah.`) or a short remark.
4. A short topic, or the head of a long one, is written as the whole transcript of one short item in six or seven
   (`riunione settimanale del team`, 24 of 156), spaced through the meeting: the issue's shape, a four-word phrase
   twenty times through the call, from the other voices' short items.

Inferred:

- The owner's phrase was most likely a short topic or the head of one, written on the other voices' backchannels and
  overlaps, as here. That none of theirs was emptied while every exact copy here was suggests their forms were not
  exact: a word or two off the line, as the 6 here, which the near rule empties. A form three words off, or a head of a
  line with a word changed, it would still keep.

## Implications

- `copies_context` (crates/speech/src/text.rs) also takes for a copy a transcript at most `speech.context_copy_edits`
  words (2; changed, added or dropped) from a whole line of the context or from a whole piece of one between commas
  or sentence marks (a keyword), while it keeps `speech.context_copy_words` (3) of its words, those edits taken off
  the shorter of the two. 0 keeps the rule of 2026-10-27. A four-word line or keyword is thus matched with one edit
  at most, whatever the field.
- What it costs: a real utterance that is a whole line or keyword of four words or more with a word or two changed,
  and nothing else, is lost, as an exact one was already. None was seen.
- The rule matches whole lines and pieces, never any window of a line: a window of the keyword line a word off is
  real speech (`Jenkins e Helm Chart`).

## Limits and unknowns

- The owner's context, audio and copies are not available; their forms are inferred, not seen. On AMI the repeated
  short phrase came back exact, and today's guard already emptied it.
- Items are cut from the reference turns, not by the diarizer, and run in one push; English audio with Italian
  contexts, as in the first measurement. The rate on a live Italian call is not measured.
- A head of a line with a word changed was not seen in 2,660 answers, and is not matched.
- Three meetings, fourteen contexts, 6 near copies: the share of near copies (1.5%) is rough.

## Follow-ups

- If a copy is reported again that the guard keeps, its text and the context's line it echoes are what decides
  between a wider match (heads) and a rule on a phrase repeated across the call, which #442 allowed and which no copy
  seen here needed.
