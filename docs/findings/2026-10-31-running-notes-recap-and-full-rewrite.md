# Running notes, a recap and today's full rewrite on three public conversations: the thread keeps 87-88% of its lines word for word and costs less than the rewrite, the recap carries the decisions, and a refused rewrite must count as an idle round

- Kind: experiment
- Status: current
- Observed: 2026-10-31
- Last verified: 2026-10-31
- Scope: Playground / the Transcribe view's notes: today's full rewrite, the running notes kept by a notes thread with tools (#428's shape), the recap; the 27B (`qwen3.8-27b`, NVFP4, `rtx5090` profile) on the RTX 5090; transcripts made by ignis (Qwen3-ASR 1.7B and the diarizer, on the CPU)
- Related: [ADR 0076](../adr/0076-the-running-notes-send-the-notes-back-and-the-recap-does-not.md), [spec speech/07, Q125](../specs/speech/07-transcribe.md), https://github.com/gpillon/ignis/issues/428, https://github.com/gpillon/ignis/issues/429
- Superseded by: none

## Question

Spec #428 splits the Transcribe view's notes into **running notes**, kept during
a conversation by an append-only thread with the text model, and a **recap**,
one writing over the whole transcript at the end, and gives every text-model
task the transcript's context. On conversations of different kinds, never
one, what does each way of keeping the notes cost and give, against today's
full rewrite at every update?

- **Running notes:** the time and decode per update, how often the round cap
  is hit, restarts, the thread's size, the lines kept word for word, how
  People grows, whether Asked of me fills, duplicate people, malformed or
  refused calls, prompt reuse.
- **Recap and full rewrite:** time and tokens, and a reading of Summary,
  Decisions, Action items and People against the conversation.

## Evidence

**The conversations.** Each was transcribed by ignis 0.9.4 in the Playground's
Transcribe view (source File, Speakers on, Suggest while listening off) and
exported as `.md`. The audio stays out of the repo, under `H:\speech-study\`.

| Label | Recording | Licence | Kind | Kept | Utterance end | Utterances | Speakers |
|---|---|---|---|---|---|---|---|
| `ami-ES2004a` | AMI Meeting Corpus ES2004a, Mix-Headset (a project kickoff, four people) | CC BY 4.0 | Work call, Me = Speaker 1 (the project manager) | 17.2 min, whole | 1.5 s | 193 | 4 |
| `mit-hst512-l1` | MIT OpenCourseWare HST.512 *Genomic Medicine* (2004), Lecture 1, archive.org `MITHST.512S04`, `Lecture1-16k.mp3` | CC BY-NC-SA 3.0 | Lesson, no Me | first 27.6 min | 1 s | 70 | 4 (66 utterances one) |
| `tdc-ep49` | *The Djekian Crusaders* ep. 49, an actual-play D&D podcast, archive.org `TdcEp49`, converted to 16 kHz mono | CC BY-SA 4.0 | Game session, no Me | first 7.1 min | 1 s | 92 | 6 |

**The replay.** `scripts/notes-replay/replay.py` replays a transcript against
the live server at the view's cadence (the first update at three utterances,
then at most once a minute of transcript time, one more at the end) and runs
the look between updates as the view does (from the sixth utterance, then
every 90 s while three new utterances came). Every request carries #440's
context block (here: write in English, nothing else told, "Me" is the user, a
hypothetical is not a fact). Three ways, each its own pass so they never share
the server's prompt cache:

- **R, the running notes:** the kind's live sections with their caps (Work
  call: Now 1, People 20, Figures & dates 12, Asked of me 6, Commitments heard
  10; Lesson: Now 1, Concepts 12, Terms 20, Questions 8, Homework & dates 8;
  Game session: Now playing 1, Players & sides 12, Score 8, Moments 10, Rules
  in question 6), four tools, one-line results, the page's guards (ids
  stripped, duplicates and lines over 240 characters refused), times in total
  minutes, at most 6 rounds an update, a restart from a snapshot past 64k
  prompt tokens;
- **A, today's full rewrite** at the same triggers, with the prompts as of
  2026-10-31 (`scripts/notes-replay/prompts.json`) and #440's language rule;
- **C, the recap:** one writing over the whole transcript at the end, in the
  kind's sections plus Figures & dates (Work call) or Terms and Homework &
  dates (Lesson), at most 3,000 tokens.

`scripts/notes-replay/summary.py` sums each run up. The 27B is
`qwen3_8_27b_nvfp4full-v2.ninfer` under the `rtx5090` profile (DFlash2, 7 draft
tokens, 8 decode lanes), `make start` from `main` at e600a7d9, thinking off,
temperature 0.3, one request at a time, nothing else on the card. Raw output:
`.scratch/notes-replay/<label>/` in the main checkout (untracked): the rows,
the running notes' change log and snapshots, the last full rewrite, the recap.

**The first run of R on the lecture found a loop.** Concepts filled its 12
lines at the fifth update. From the seventh on, the model called
`rewrite_section` with 13 lines (the 12 plus the new concept), was refused
("13 lines; Concepts holds 12 at most."), and sent the same 13 lines again
until the round cap: 63 refusals, the cap hit on 19 of 20 updates, the notes
frozen (94% of the lines kept, only because nothing changed), 36,599 decode
tokens against the full rewrite's 15,883, a thread of 47k tokens. Two general
changes fixed it, and every R figure below is from runs with both (labels
`-v2`; the game session was replayed only after the fix, as `tdc-ep49`):

1. the refusal says what to do: "Send 12 or fewer: merge lines that belong
   together, or drop the least useful.";
2. a round whose calls all changed nothing, unchanged **or refused**, is idle,
   and two idle rounds in a row end the update (the spec's "two rounds in a
   row change nothing"; the first run counted only unchanged calls).

**Cost per update.** Mean (p95) seconds and decode tokens per update; the
thread's largest prompt; first-round prompt reuse (cached over prompt tokens,
updates after the first).

| | R s/update | R decode/update | R decode total | A s/update | A decode total | C s | C decode | R thread max | R reuse |
|---|---|---|---|---|---|---|---|---|---|
| `ami-ES2004a` (16 updates) | 1.65 (3.1) | 281 | 4,497 | 2.87 (4.5) | 8,919 | 5.1 | 878 | 12.9k | 0.87 |
| `mit-hst512-l1` (20 updates) | 4.25 (7.9) | 742 | 14,842 | 4.02 (6.0) | 15,883 | 8.5 | 1,500 | 25.2k | 0.87 |
| `tdc-ep49` (7 updates) | 6.37 (23.4) | 421 | 2,948 | 2.23 (3.3) | 2,706 | 4.1 | 505 | 8.1k | 0.77 |

**Stability and the calls.**

| | R kept word for word (mean of previous lines) | A kept | Rounds/update | Round cap hit | Calls | Unchanged | Refused | Not JSON | Text instead of calls | Ids stripped |
|---|---|---|---|---|---|---|---|---|---|---|
| `ami-ES2004a` | 0.88 | 0.00 | 3.2 | 1 / 16 | 67 | 30 | 0 | 0 | 1 | 0 |
| `mit-hst512-l1` | 0.88 | 0.01 | 5.5 | 13 / 20 | 108 | 4 | 20 | 0 | 1 | 0 |
| `tdc-ep49` | 0.87 | 0.02 | 5.7 | 5 / 7 | 38 | 0 | 7 | 0 | 2 | 0 |

The lecture's 20 refusals were adds to a full section (12), ids that a rewrite
had replaced (7) and one rewrite of a section not yet full.

**Restarts.** No run reached the 64k threshold (the largest thread was the
lecture's 25.2k). A run of the lecture with the threshold at 8k restarted 7
times in 20 updates: the first round's reuse fell to 0.84, decode was 13,346
(close to the default run's), and after each restart the model copied the
snapshot's ids into its texts 20 times, all stripped by the guard. A section
rewritten after a restart came back consolidated (Terms 20 to 7 lines once in
the first, unfixed run).

**The sections.** Lines per section at each update:

- `ami-ES2004a`: People 2 to 4 by the sixth update and never more (the four
  speakers, no duplicate); Figures & dates 0 to 9; Commitments heard 1 to 4;
  **Asked of me never filled** (the project manager asks, she is rarely
  asked).
- `mit-hst512-l1`: Concepts filled to 12 and then consolidated between 9 and
  12; Terms grew to its 20 and stayed full; Questions 3; **Homework & dates
  stayed empty** in the default run (the lecture says once that the slides and
  video go on the course's website, which the 8k run noted there).
- `tdc-ep49`: Players & sides 5 from the first update (the dungeon master and
  four players with their characters, no duplicate); Moments 0 to 10, full at
  the end; Rules in question 5; **Score stayed empty** (a role-playing
  session keeps no score). Its first update took 23 s for 293 decode tokens
  (the server's first request of the pass), the others 2.3-5.3 s.

**A reading of the notes against the conversations** (mine; the owner judges
where the owner can):

- **AMI, the last full rewrite (A)** lists a decision that was not taken
  ("the product will use lightweight plastic rather than metal": a suggestion
  at 15:59, never agreed) and gives the next presentation to Me, where the
  transcript's line at 16:12 is labelled Speaker 4. **The recap (C)** has two
  decisions, both taken (the 30-minute break, an international market), the
  same four people, five real figures, and the open questions the call left.
  The running notes vary from run to run. In every run Figures & dates took
  a few lines that are not figures (the design phases, the market), and
  Commitments heard took lines that are not commitments (the first run noted
  animals drawn in the warm-up; the `-v2` run, "lights will stay on"). People
  is the weakest: in the `-v2` run each person's line stayed what the first
  minutes said of them ("asking whether they should stand up front"), where
  an earlier run had completed them with what each proposed later.
- **The lecture:** the running notes' Concepts and Terms read as a student's
  notes, each with its time, and the recap's Topics follow the lecture's own
  outline by weight; both are in English as the context said. The full
  rewrite's sections carry the same content in other words at every update.
- **The game session:** the running notes named every player with their
  character and the dungeon master from the first update, and Moments
  followed the fight. Rules in question mixed real rulings (the unarmored
  armor class, the monsters' armor class) with table talk (who may post in
  the rolls channel, what counts as a joke). The recap's Players agree with
  the running notes; its Results has no result, since the fight is still on
  at the cut, and its Highlights give the off-game banter the weight the
  table gave it.

## Finding

- **The running notes keep the notes still.** 87-88% of the lines a running
  update found were still there word for word after it, against 0-2% for
  today's full rewrite, which rewords every line at every update.
- **They cost less than the full rewrite, once a refused call counts as an
  idle round.** Decode was 50% of the full rewrite's on the meeting (50-74%
  over its three runs: 4,497, 6,145 and 6,629 tokens against 8,919 and 9,036)
  and 93% on the lecture, at a few seconds an update; prompt reuse held at
  0.77-0.87 of the thread's prompt. On the 7-minute game sample, too short
  for the thread to pay off, it was 109%. Without the idle rule, one section
  the model would not cut below its cap cost 2.3 times the full rewrite and
  froze the notes.
- **The recap is where decisions and action items belong.** One writing over
  the whole transcript, at 5-9 s and under 1,500 decode tokens, carried the
  decisions that were taken and none that were not, where the last full
  rewrite invented one.
- **A line written at a person's first mention tends to stay as it was.**
  "Complete their line as more is learned" is followed in some runs and not
  in others; the recap, which reads the whole call, is where people are
  described well.
- **Some running sections stay empty when the conversation has nothing for
  them** (Asked of me with a Me who asks, Homework & dates in a lecture's
  first half, Score in a role-playing game). That is correct, not a fault,
  but a kind's live sections should be ones most conversations of that kind
  fill.

## Implications

- The ADR's split stands: the running notes send the notes back, the recap
  does not.
- The page must count a refused or unchanged call alike toward the two idle
  rounds that end an update, and its over-cap refusal must say what to do.
  #432's first build counts only unchanged calls: a follow-up to #432 in #428's
  review fixes it, citing this finding.
- The restart from a snapshot works, at a small cost in reuse; the id guard is
  needed after every restart.

## Limits and unknowns

- Three conversations, one of each kind, all in English; the language rule
  (#440) is not stressed. One run per way per conversation, at temperature
  0.3, except the meeting's: R three times (two before the idle fix, which
  changes nothing on a run with no refusal, and the `-v2` run in the tables)
  and A twice, with the same picture of cost and a different one of People.
- One model, the 27B; the Flash-Next model and a 16 GB card are not measured.
- No edits during the replay: no corrections, renames or user lines reach the
  thread (#434, #436), and no context image (#437).
- The lecture is its first 27.6 minutes and the game session its first 7.1
  minutes, to bound the CPU transcription (other work shared the CPU, and the
  game's file ran at about half of real time); no run reached the default
  restart threshold.
- The reading of the notes is the coordinator's, not yet the owner's.
- A lecture with few pauses made utterances of ~50 s at a 1.5 s utterance end,
  and the CPU transcription slowed to about a third of real time with other
  CPU work on the machine; the lecture was redone at 1 s. That is the speech
  model's, not the notes', and is noted for #442's area.

## Follow-ups

- #432: count refused calls toward the idle rounds and say what to do in the
  over-cap refusal (in #428's review fixes).
- A kind's live sections: revisit Asked of me and Homework & dates on more
  conversations before changing them.
