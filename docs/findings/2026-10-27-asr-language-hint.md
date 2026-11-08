# Naming the language in the context fixes a short clip's language as forcing does, and forcing translates English

- Kind: experiment
- Status: current
- Observed: 2026-10-27
- Last verified: 2026-10-27
- Scope: speech / Qwen3-ASR-1.7B (the `auto` entry, Q4_K_M decoder) on the CPU through
  `POST /v1/audio/transcriptions`: the language label and the text on Italian FLEURS clips, on 1.0-2.5 s pieces cut
  from them and on English FLEURS clips, with no context, with a language hint in `prompt` and with `language=it`;
  the default of the Playground Transcribe view's languages setting
- Topic: speech
- Related: [#362](https://github.com/gpillon/ignis/issues/362), [spec speech/07](../specs/speech/07-transcribe.md),
  [Qwen3-ASR on the CPU](2026-10-24-qwen3-asr-on-cpu.md), [Speech to text](../user/api/transcriptions.md)
- Superseded by: none
- Summary: With no context Qwen3-ASR mislabels 8 of 240 Italian pieces, all 1.5 s or shorter, and `Languages: Italian.` in `prompt` fixes all 8 as `language=it` does, while forcing Italian translated 7 of 30 English clips, so the hint is the default.

## Question

On the owner's first live check of the Transcribe view on an Italian call (2026-10-27), items of 1-2.5 s were often
labelled Thai, Chinese, Portuguese or Hindi, and items of 3.7-5.5 s came out Italian. The view is getting a
languages setting, and its default is one of two:

- **Free**: the model detects the language, and the expected languages are only written into the context (`prompt`,
  the model's system turn) as a hint, e.g. `Languages: Italian.`;
- **Force**: one `language` is sent and the server forces it.

Does a hint in the context fix a short clip's language at all, and as well as forcing does? What does forcing cost
when the speech is not the forced language? Does the hint's wording matter?

## Evidence

Run on 2026-10-27 against the server at `127.0.0.1:8000` (`qwen3-asr-1.7b`, CPU), one request at a time, greedy,
`response_format=verbose_json`, so that the label is the model's own (`language`; the json format's
`languages[0].code` is the same label mapped to its code). There were 1,560 requests and every one returned `200`. Scripts, clips,
raw JSONL and tables are in `H:\speech-study\language-hint-362\`: `scripts\` (`prep.py`, `run.py`, `analyze.py`,
`en_detail.py`, `details.py`), `raw\results.jsonl`, `results\tables.md`, `results\en-detail.txt` and
`results\details.txt`. The decision rule, written before any result was read, is `results\preregistration.md`.

**Items** (seed 362, FLEURS test split, one recording per sentence, clips of 4-25 s):

- 60 Italian clips, whole (6.7-23.2 s, median 14.2);
- 240 pieces cut from them, 4 per clip, of a uniform length in 1.0-2.5 s (median 1.84) at a uniform offset inside the
  clip's speech region. They start and end mid-word, in the length regime that failed live;
- 30 English clips, whole (4.1-15.0 s, median 8.2). Forcing Italian on English speech stands in for forcing Italian
  on the English stretches of an Italian call.

**Conditions:**

- `none`: no prompt and no language;
- `hint-it`: `prompt` = `Languages: Italian.`;
- `hint-it-ita`: `prompt` = `Lingua: italiano.`, on the pieces only;
- `hint-it-en`: `prompt` = `Languages: Italian, English.`, what a mixed call would send;
- `force-it`: `language=it`.

Every item ran under every condition of its set, so each contrast with `none` is paired: items gained and items lost,
with an exact two-sided McNemar test. Rates carry Wilson 95% intervals. WER pools the clips with the earlier study's
normaliser and has a 95% clip bootstrap. A piece has no reference of its own, so its *word agreement* is the share of
its words found in its clip's reference.

**Pieces (240), label Italian:**

| condition | Italian (95%) | vs `none` | other labels | 1.0-1.5 s (78 pieces) | word agreement |
|---|---|---|---|---|---|
| `none` | 232 = 96.7% [93.6, 98.3] | | Portuguese 3, English 2, Arabic 1, German 1, None 1 | 71 = 91.0% [82.6, 95.6] | 85.8% |
| `hint-it` | 240 = 100% [98.4, 100] | +8 / −0, p = 0.0078 | | 78 = 100% | 85.9% |
| `hint-it-ita` | 234 = 97.5% [94.7, 98.8] | +8 / −6, p = 0.79 | `italiano` 3, None 3 | 77 = 98.7% | 84.3% |
| `hint-it-en` | 239 = 99.6% [97.7, 99.9] | +7 / −0, p = 0.016 | None 1 | 78 = 100% | 85.7% |
| `force-it` | 240 (forced) | +8 / −0, p = 0.0078 | | 78 (forced) | 86.5% |

All eight of `none`'s misses are pieces of 1.50 s or less. At 1.5-2.0 s `none` was 76 of 77 (the miss is that
1.50 s piece) and at 2.0-2.5 s 85 of 85. Seven of the misses have text, and the text is in the language named:

| `none` | `Languages: Italian.` |
|---|---|
| `die Linie` (German) | `di linea.` |
| `reportagem` (Portuguese) | `reportage` |
| `يعلم.` (Arabic) | `gli elementi` |
| `está do outro.` (Portuguese) | `è stato poi tra` |
| `The Bennett School.` (English) | `la Bennett School` |

Every hint and `force-it` labelled those seven Italian. The eighth is empty under `none` and under every hint, and
`force-it` wrote `quattro` for it. No hinted piece came back in a non-Latin script. Under `none` two did: the Arabic
one, and an Italian one with one stray Devanagari sign, which `force-it` kept.

A hint is not inert: `Languages: Italian.` changed the text of 112 of 240 pieces and 11 of 60 clips, while word
agreement and WER stayed where they were. `force-it` changed the text only of the 8 pieces that `none` did not label
Italian: it was identical on the other 232 and on all 60 clips.

**Italian whole clips (60):** labelled Italian 60 of 60 under every condition.

| condition | WER (95%) | clips worse / better than `none` |
|---|---|---|
| `none` | 2.9% [1.9, 4.1] | |
| `hint-it` | 3.1% [2.0, 4.4] | 4 / 4 |
| `hint-it-en` | 2.8% [1.7, 4.0] | 3 / 5 |
| `force-it` | 2.9% [1.9, 4.1] | 0 / 0 (identical text) |

**English clips (30):**

| condition | label English (95%) | WER (95%) | clips worse / better than `none` | text translated into Italian |
|---|---|---|---|---|
| `none` | 30 = 100% [88.6, 100] | 5.9% [2.8, 9.4] | | 0 |
| `hint-it` | 26 = 86.7% [70.3, 94.7], Italian 4 | 5.7% [2.7, 9.2] | 0 / 1 | 0 |
| `hint-it-en` | 23 = 76.7% [59.1, 88.2], Italian 7 | 9.6% [3.6, 18.0] | 2 / 1 | 1 |
| `force-it` | 0, Italian 30 (forced) | 24.4% [12.2, 39.3] | 11 / 1 | 7 whole, 1 in part |

Under `force-it`, `The 25 Dunlap broadsides still known to exist are the oldest surviving copies of the document.`
came back as `Le 25 Dunlap Broadside, ancora conosciute, sono le copie più antiche del documento.`, and
`The 802.11n standard operates on both the 2.4Ghz and 5.0Ghz frequencies.` as
`Il 802.11n standard opera su entrambe le frequenze 2.4 GHz e 5.0 GHz.`. One clip came back half translated:
`Lakha Singh presentò il Chopan Bhajan, as well. Singer Raju Kandelwal was accompanying him.` Three more clips moved
by one word. The hints' Italian labels on English (4 and 7) left the text in English with one exception: under
`hint-it-en`, `MS is a disease that affects the central nervous system...` came back as
`MS è una malattia che colpisce il sistema nervoso centrale...`.

**The Italian-worded hint wrote itself into the transcript.** Under `Lingua: italiano.`, three pieces came back as
the single word `lingua` with the label None. For the same pieces, `none` and `hint-it` heard `dominano ripide`,
`libero leggings` and `via di comunicazione fa sì che la diffusione`. Three more pieces came back with the right text
and the label `italiano`, which is not one of the model's 30 labels. `code_of_label` passes such a label through
unchanged, so the json route would answer `languages: [{"code": "italiano"}]`. No word of either English-worded hint
appeared in the transcript of any piece.

## Finding

Observed:

1. On clean read Italian the short-clip failure is real but small. `none` mislabels 8 of 240 pieces (3.3%), all of
   them 1.50 s or shorter: 7 of the 78 pieces under 1.5 s (9.0%), and 1 of the 162 from 1.5 s up, at exactly
   1.50 s. The model transcribes into the language it names, so a wrong label means a wrong transcript, not just a
   wrong tag.
2. `Languages: Italian.` in the context fixes all eight and loses none (+8 / −0, p = 0.0078). These are the same eight
   that forcing fixes. Italian WER stays where it was (2.9% to 3.1%, 4 clips worse and 4 better).
   `Languages: Italian, English.` does the same on Italian (+7 / −0, p = 0.016).
3. Forcing Italian costs nothing on Italian speech: the text is identical wherever the model already said Italian.
   It is costly on English speech: 7 of 30 English clips were translated into Italian whole and 1 in part, and WER
   went from 5.9% to 24.4%.
4. `Languages: Italian.` translated no English clip (0 of 30, WER 5.7%). `Languages: Italian, English.` translated
   one (WER 9.6%, an interval overlapping `none`'s). Both hints changed some English clips' *label* to Italian (4 and
   7 of 30) while the text stayed in English.
5. The wording matters. `Lingua: italiano.` gains the same eight pieces but loses six (+8 / −6, p = 0.79): the model
   wrote the prompt's own word as the transcript three times and labelled `italiano` three times.

Inferred:

- **Free with the hint should be the default, not Force.** On the failing regime that this measurement reaches, the
  hint fixes everything that forcing fixes. It does not translate English speech, which forcing did in roughly a
  quarter of the English clips. The hint is worth writing: it is the difference between 91% and 100% on the
  1.0-1.5 s pieces.
- The pre-registered rule asked that `hint-it-en` keep English WER within about a point of `none`'s. It missed:
  9.6% against 5.9%, +3.7 points, with intervals that overlap. The one clip it translated accounts for all 23 net
  extra edits; its other two changes cancel out. The single-language
  hint met the clause (5.7%). Forcing costs +18.5 points and translates 7 clips, so the decision holds. That a
  two-language hint costs nothing on English speech is not established.
- The hint names the languages by their English names, which are the model's own labels (`Languages: Italian.`,
  `Languages: Italian, English.`), and never in the call's language.

## Implications

- The Transcribe view's languages setting defaults to Free. It writes `Languages: <English names, comma-separated>.`
  into the context, and Force is an opt-in for monolingual speech that the model still mislabels.
- Under a hint the label does not reliably name the text's language: a hint labelled English text Italian on 4 to 7
  of 30 clips. Anything that shows or acts on the label (an item's language, a per-language rule) has to allow for
  this, as the earlier study already concluded for the label without a hint
  ([Qwen3-ASR on the CPU](2026-10-24-qwen3-asr-on-cpu.md): the label is never trusted to force anything).
- A label outside the model's 30 (`italiano`) reaches `languages[].code` unmapped.

## Limits and unknowns

- FLEURS is clean read speech. `none` failed on 3.3% of the pieces, far less often than on the live call, and none of
  the live call's scripts (Thai, Chinese, Devanagari) appeared here, apart from one stray sign. The live failures come
  from call audio: far voices compressed by the call, echo, spontaneous speech, VAD cuts. That the hint fixes those as
  it fixes these is inferred, not measured. The owner's next live call is the check.
- The pieces are cut at arbitrary points, mid-word. They are not utterances cut by a VAD at a speech onset, and the
  bias could go either way.
- Only the file route was measured. The Realtime session builds its context with the same `route::context_of`, so the
  hint reaches the model as the same system turn, but a live item is decoded as it arrives, and that path was not
  measured.
- The cost arm forces Italian on monolingual English, as a proxy for English jargon inside Italian speech. In the
  earlier dictation study, forcing Italian on one dictation took its WER from 2.7% to 10.8%: "route" became "root"
  three times.
- With 30 English clips the hints' WER differences sit inside the intervals, and the one translation under
  `hint-it-en` is a single clip. Whether listing Italian first is what turns English labels to Italian is untested.
- Also untested: the hint combined with the user's own Context text, the order of the languages, other wordings,
  other languages, and the 0.6B and Q8_0 entries. Each condition ran once (greedy decoding).

## Follow-ups

- [#362](https://github.com/gpillon/ignis/issues/362): the Transcribe view's languages default (Free with the hint),
  to be confirmed on the owner's next live call.
