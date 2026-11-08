# Given a pause, the speech model writes its context as the transcript

- Kind: experiment
- Status: current
- Observed: 2026-10-27
- Last verified: 2026-10-27
- Scope: speech / Qwen3-ASR-1.7B (the `auto` entry, Q4_K_M decoder) on the CPU through
  `POST /v1/audio/transcriptions`: the text on silence, white noise, a meeting's silent gaps and short turns and the
  heads of Italian clips, under five contexts; the guard `copies_context` that answers it
- Topic: speech
- Related: [#362](https://github.com/gpillon/ignis/issues/362), [spec speech/07](../specs/speech/07-transcribe.md),
  [Naming the language in the context](2026-10-27-asr-language-hint.md), [Speech to text](../user/api/transcriptions.md)
- Superseded by: none
- Summary: Given a topic and terms in its context, Qwen3-ASR wrote the topic line as the transcript of 4 to 10 of 50 pauses and short turns, never without context, and a guard emptying a transcript that repeats a context line takes 15 of 17 copies.

## Question

On the owner's live call in the Transcribe view, with a context applied (languages, a topic, sixteen terms), some
utterances, often, were the keyword line whole: the terms joined by commas, as the server writes them after the
prompt. What the call's audio held under those utterances is not known. Does the model write its context as the
transcript, on what audio, and under which contexts?

## Evidence

Run on 2026-10-27 against the server at `127.0.0.1:8000` (`qwen3-asr-1.7b`, CPU), one request at a time, greedy,
`response_format=verbose_json`; 250 requests, every one `200`. The script, `probe.py`, the port of the guard,
`rule.py`, and the raw `run1.jsonl` are in `H:\speech-study\context-echo-362\`.

**Inputs** (50, made by the script, seed 362):

- digital silence, 0.5, 1 and 2 s (3);
- white noise at −50, −40 and −30 dBFS, 0.5, 1 and 2 s (9);
- AMI ES2004a (headset mix, CC BY 4.0): 1 s from the middle of 15 gaps where no one speaks (room noise, breath,
  paper), and 15 turns under 0.8 s whole, with 0.1 s either side (backchannels: "okay", "yes", "I mean");
- the first 0.4 s of 8 Italian FLEURS pieces from the language-hint study.

**Contexts** (the server joins `keywords[]` as one line after `prompt`):

- `none`: no context;
- `hint`: `Languages: Italian.`;
- `page16`: `Languages: Italian.`, `Topic: migrazione della pila software verso GitOps.`, and 16 terms as `keywords[]`:
  the owner's context's shape, with open or generic names in place of theirs;
- `page5`: the same with 5 of the terms;
- `terms-line16`: the 16 terms in the prompt as a third line, `Terms: ….`, and no `keywords[]`.

**Transcripts that are a context line or part of one** (two words or more of it in a row, and nothing else):

| context | silence, noise (12) | AMI gaps (15) | AMI short turns (15) | Italian heads (8) | all (50) |
|---|---|---|---|---|---|
| `none` | 0 | 0 | 0 | 0 | 0 |
| `hint` | 0 | 0 | 0 | 0 | 0 |
| `page16` | 0 | 0 | 3 | 1 | 4 |
| `page5` | 0 | 6 | 3 | 1 | 10 |
| `terms-line16` | 0 | 2 | 1 | 0 | 3 |

Every copy was the topic line: `migrazione della pila software verso GitOps.` whole, with or without its period, 15
times, and its first two words twice. None was the keyword line, which is what the owner saw. Under the other
contexts the same short turns came back as words: `sì.`, `okay`, `be useful.`, `hm`, `massa`.

Separately, `none` came back as the literal text `<asr_text>` 4 times (the −30 dBFS noise of 1 and 2 s, two AMI
gaps): the model wrote its marker twice and the parser kept the second.

## Finding

Observed:

1. The model writes a line of its context as the transcript of audio with little or no speech in it: a meeting's
   pause, a backchannel, the first syllables of a word. Digital silence and white noise did not set it off at any
   level: it took a real room.
2. Only free text did. The language hint alone was never copied, in 50 inputs; with a topic and terms 3 to 10 of the
   50 were.
3. The copy is a whole line or the head of one, stripped of its `Topic:` label.

Inferred:

- The owner's keyword line and this probe's topic line are the same failure: the model copies a sentence-like line of
  its system turn, and which line depends on the context. Their sixteen terms included Italian words (`pila
  software`), which may make that line read as speech; this probe's terms are mostly names.
- The fewer terms, the more copies here (10 against 4), but 50 inputs do not settle it.

## Implications

- The server takes a transcript that repeats its context for no speech (`copies_context`, crates/speech/src/text.rs):
  the words of a whole line of the context as the model read it, of two words or more, or three or more of one line's
  words in a row, a leading `Label:` read with and without; one word never. The text is empty, labelled only by a
  forced language, so no client sees the copy and the Playground leaves no utterance. Applied to this probe's 250
  answers, it empties 15 of the 17 copies (the two left are `migrazione della`) and nothing else.
- What the guard costs: a pause that held a real short word under the copy loses it, as a pause with nothing in it
  should. A real utterance that is exactly a line of the context, or three of its words in a row and nothing else,
  is lost too.
- The parser drops a second `<asr_text>` marker.

## Limits and unknowns

- 50 inputs, one meeting, one topic. The rate on a live call is not measured; the owner saw it "often".
- The owner's audio under the copied utterances was not available, so that their pauses are like AMI's is inferred.
- Whether a context worded otherwise (no topic, terms as a sentence) copies less is not measured beyond the one
  `terms-line16` condition, and the context's shape is the owner's to decide.
