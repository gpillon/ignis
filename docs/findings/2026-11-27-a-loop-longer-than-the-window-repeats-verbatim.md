# A loop longer than the loop guard's window repeats verbatim: the repeat signal catches it at its second copy

- Kind: experiment
- Status: current
- Observed: 2026-11-27
- Last verified: 2026-11-27
- Scope: server / reasoning-loop guard, its repeat signal (`crates/server/src/loop_guard.rs`, config v2 `model.loop_guard_repeat_*`); qwen3.8-27b reasoning: the Rome prompt's 98 labelled runs and the owner's qwen-code and opencode sessions
- Topic: reasoning-and-context
- Related: https://github.com/gpillon/ignis/issues/727, [ADR 0053](../adr/0053-a-loop-is-acted-on-once-it-persists.md), [ADR 0052](../adr/0052-a-reasoning-loop-is-taken-back-and-nudged.md), [A loop persists](2026-10-16-a-loop-persists.md), [Reasoning-loop guard](2026-10-14-reasoning-loop-guard.md)
- Superseded by: none
- Summary: Sentences repeated back to back catch verbatim loops of 250 to 8,000 characters at their second copy, with 0 fires on agent reasoning once 21 parts were read as loops; 7 of 8 incident turns caught, Rome 25/25 as before.

## Question

The loop guard scores the zlib ratio of the last 1,000 scored characters of the reasoning, and
nothing before them. A loop whose repeated block is longer than that window never shows two copies
in it, and scores like prose (#727). Can a signal that compares the reasoning with itself catch such
a loop by the time its third copy starts? Can it keep the first copy, without new false rewinds on
agent reasoning that repeats legitimately: a log transcribed, a plan re-drafted, a checklist?

## Evidence

**Data.** All offline, CPU only. The private texts and every script stay in `.scratch/727/` of the
main checkout: `measure/` holds the scores and `diag/a`, `diag/b` the two diagnoses. Nothing below
quotes them.
- The 98 labelled Rome runs of the 2026-10-14 finding: 25 harmful, 47 with no loop and 9 micro,
  after merging aliases.
- The qwen-code snapshot of the 2026-10-16 finding: 12,936 unique reasoning parts, 9,398 of them
  300 characters or longer.
- The opencode database as of 2026-11-27: 6,174 unique Qwen reasoning parts, 3,623 of them 300
  characters or longer. No part carries a rewind tag.

**The triage scan, reproduced.**
- The scan of #727 ran again unchanged and gives the same counts.
- In qwen-code it finds 13 parts that repeat a block of 400 or more scored characters back to back:
  - 8 consecutive turns of one session (2026-09-03), with periods of 1,796 to 6,471;
  - 5 more parts, in three other sessions and in the incident's own (`qwen:1386`, 32 minutes after
    the 8 turns).
- Today's guard, the compression score alone, acts on none of the 13. Their lowest score is 0.227 to
  0.344, and the longest run of loop scores is 3 checks, against the 4 the guard needs.

**The signal** (#727, `Normalized::repeat`).
- **Sentences.** The scored text is cut into sentences: at a line break, at a full-width stop, and at
  white space after `.`, `!` or `?` past any closing quote or bracket. So `engine.rs` and `0.5` split
  nothing.
- **Prose and material.** A sentence that ends in a stop is prose. A line that ends in none is
  material: a list item, a log line, a table row.
- **Comparison.** Two sentences are the same when the raw text they span, trimmed, is identical.
- **Periods tried.** The periods tried are the distances back to an earlier copy of one of the last
  two sentences, shortest first.
- **The walk.** At each period, the walk compares each sentence with the one a period before it. A
  copy may differ in one sentence: `loop_guard_repeat_varying`, 1.
- **A repeat.** A repeat is 2 copies back to back (`loop_guard_repeat_copies`) whose copy repeats at
  least 150 characters of prose verbatim (`loop_guard_repeat_min`), with a period up to 16,000 scored
  characters (`loop_guard_repeat_max`).
- **The shortest period decides.** A block shorter than 150 that repeats over 150 characters in all
  is the compression score's, and returns none. Two short sentences in a row skip their period and
  hide nothing.
- **The cut.** A rewind cuts at the start of the second copy.

**What the legitimate side showed while the signal was built.** Each rule was added for a fire on
legitimate text, before any held-out score.
- **The first form, scored text compared and every sentence counted, fired on 15 agent parts.**
  Read by eye:
  - 10 were the same paragraph written twice;
  - 3 were tables or lists whose rows differ only in their numbers. The digit fold made them equal.
  - 1 was a passage of a file quoted four times while the model went in circles;
  - 1 was two agents' identical file listings, transcribed to compare them.
- **Comparing raw text** took the 3 tables out.
- **Counting only prose towards the 150** took out the listings and the quoted passage, whose lines
  end in no stop. Two duplicated paragraphs whose last sentence ends the reasoning also left the
  set: that sentence never ends.

**What fires at the defaults.**
- **8 agent parts:** 6 in qwen-code and 2 in opencode. Each is a paragraph written two or three
  times back to back, verbatim or with one sentence changed, which the model then leaves by itself.
  One (`oc:1197`) is a whole file written twice.
- **All 8 are read as loops,** with the 13 of the scan: 21 parts in all. Of the 19 in qwen-code, the
  2026-10-16 finding labelled 4 as loops and counted 15 among its legitimate texts. The 2 in
  opencode are in today's snapshot.
- **After that relabelling, the guard fires on no legitimate agent part.**
- **Rome:**
  - 25 of 25 harmful loops are caught, as before;
  - the repeat signal acts first on 3 of them;
  - no run with no loop fires, and no micro run.

**How the limits move the legitimate side** (qwen-code and opencode, fires on parts not read as
loops):

| run | fires on legitimate parts |
|---|---|
| defaults (min 150, max 16,000, copies 2, varying 1) | 0 |
| min 100 | 0 (and the same 8 loop parts) |
| min 200 or 250 | 0 (7 of the loop parts) |
| min 300 | 0 (5) |
| varying 0 | 0 (4) |
| varying 2 | 1: an edit's old and new strings, written out with one line break between them; plus one more loop part |
| copies 3 | 0 (1) |
| max 100,000 | 0 (8) |

- **The 150 comes from the mechanism.** A 250-character copy whose one varying sentence is up to 100
  characters long keeps 150 of prose verbatim. The legitimate side is flat from 100 to 250.

**Synthetic loops** (unit tests, `loop_guard.rs`):
- **Sources.** Three prose sources written for the tests: a coding agent in English, a history in
  English and a coding agent in Italian.
- **The grid.** Periods of 250, 300, 400, 500, 750, 1,000, 1,500, 2,000, 3,000, 4,000, 6,000 and
  8,000 characters, verbatim and with each copy's middle sentence its own, opening the reasoning and
  after 3,000 characters of it: 144 cases.
- **Result.** Every one is a repeat by the third copy's first character. Every cut falls between the
  first copy's end and the second's start.
- **The layout.** The copies are paragraphs, and the middle sentence is the one that varies. When
  the sentence that varies opens each copy instead, the copies are read from the sentence after it.
  The repeat is then found when the third copy's opening sentence ends, and the cut keeps the second
  copy's opening sentence (a unit test holds this). With one sentence free to vary, that reading
  is as valid as the other, and it never cuts into what came before the loop.
- **The stress set.** A pasted log transcribed twice to compare two runs, a plan re-drafted after a
  review, a per-item checklist and a listing quoted twice. Neither signal fires on any.

**The incident, scored once** after the limits were fixed in code
(`.scratch/727/measure/limits-chosen.txt`):

| turn | long block (scan) | first action | where its cut lands |
|---|---|---|---|
| 1355 | 3,103 × 7.6 | period 3,559, a copy late (below) | keeps 24.2K of the loop |
| 1356 | 5,847 × 6.7 | period 199, inside copy 1 | keeps 0.35 copy |
| 1357 | 6,471 × 2.0, earlier 3,491 × 3.0 | period 3,491, end of copy 2 | keeps 1 copy of the 3,491 cycle |
| 1358 | 4,117 × 7.5 | period 197, inside copy 1 | keeps 0.90 copy |
| 1360 | 4,117 × 14.5, earlier 3,491 × 2.6 | period 3,491, end of copy 2 | keeps 1 copy of the 3,491 cycle |
| 1361, 1362 | 4,117 × 4.5 and × 8.5 | period 197, inside copy 1 | keeps 0.90 copy |
| 1364 | 1,796 × 25.7 | period 1,796, the third copy's first character | keeps 1.00 copy |

- **One cycle.** All 8 turns run one deliberation cycle in several forms, and each turn opens
  inside it. So no cut can keep less than about one copy.
- **The compression score never acts** on any of them, with or without the repeat signal.
- **The 197-period repeat** (1356, 1358, 1361, 1362) is the cycle's re-entry stutter: two sentences,
  173 characters of prose, said twice inside every copy.
  - It fires one copy before the long block would, and keeps about what that catch keeps.
  - Its log line names a period of 197 for a cycle of about 4,200.
  - At `loop_guard_repeat_min` 200 these turns are caught at their own period, one copy later.

**What the scan got wrong, and 1355.**
- **The scan's copy counts are artifacts.** It sums runs at one sentence offset from anywhere in the
  text, so 1355's "7.6 copies" are four separate runs. Its third-copy offsets are a + 2d, with no
  third copy behind them.
- **1355 cut at its own repeated sentence:**
  - its first four copies are still being rewritten: two thirds of what differs is blocks inserted
    or dropped;
  - from its fourth copy on, the loop is verbatim, 58 sentences;
  - two of the verbatim copies insert a 13- or 14-sentence digression, which the next copy drops.
- **The signal catches the verbatim loop** once its copies realign past the first digression: about
  4,700 characters after the verbatim loop's third copy starts, a copy late.

**Loops still missed.** `qwen:328` and `qwen:1612`'s 2,197-character block reword or rebuild their
copies: 7 to 31 sentences differ between consecutive copies, mostly in whole blocks. Three looser
tolerances were simulated on them (`diag/a`):
- **one sentence inserted or dropped per copy:** changes nothing;
- **a near-identical sentence counted as the same:** catches only 328's 12-sentence frame, where
  varying 2 does;
- **near-identical sentences also counted towards the 150:** fires before the loops of 1355 and
  1364, a false-fire exposure.

## Finding

Observed:
- On these corpora, a loop whose block repeats verbatim back to back is caught at its second copy's
  end, whatever its period from 250 to 8,000 scored characters, with a cut that keeps one copy.
- After 21 agent parts are read as loops, it fires on no legitimate agent reasoning, Rome's runs with
  no loop, or the stress set.
- The 2026-10-16 finding's qwen-code negatives held 15 verbatim loops it counted as legitimate. 6
  are one incident's turns, which ran until the reasoning closed. Most of the rest are a paragraph
  written twice.
- Comparing as written, and leaving lines of material out of the count, are what keep tables,
  listings and quoted passages out.
- 7 of the incident's 8 turns are caught by the time their third copy starts, 4 of them on a
  stutter inside the cycle rather than on the cycle. The eighth is caught a copy late.

Inferred:
- Two copies of a verbatim paragraph are rare in legitimate reasoning. Rewriting, quoting and
  transcribing are what repeat legitimately, and they change sentences or are line-shaped. Thin
  evidence: one reader, one model.
- The loops the signal still misses change whole blocks between copies. A tolerance loose enough to
  reach them fires on ordinary near-repetition. A detector for them would need persistence, more
  copies, which the "before the third copy" goal rules out.

## Implications

- **The guard gains the repeat signal beside the compression score** (ADR 0053, edited in place):
  - four live config fields, `loop_guard_repeat_min` 150, `_max` 16,000, `_copies` 2 and
    `_varying` 1;
  - its cut keeps the first copy;
  - its log lines carry `signal`, `period_chars` and `copies`.
- **Two-copy duplicates the model would leave by itself are now rewound.** The compression score's
  persistence rule leaves such loops alone; this signal, by the issue's goal, does not.
- The 2026-10-16 finding is corrected: agent reasoning does hold persistent verbatim loops, most
  of them too long for the window to hold two copies, the rest stopping at two copies, too few for
  the persistence rule.

## Limits and unknowns

- **Labels.** Loop or legitimate is one reader's call. The signal's own fires were read and
  relabelled, so the "0 legitimate fires" rests on that reading.
- **What the stress set does not cover.** Any block of prose written twice in a row is a
  repeat, whatever it is for. Two variants of the stress set fire, both written by the review
  after the limits were fixed:
  - a checklist whose items each have one sentence of their own and share 150 characters of
    prose;
  - a log quoted twice verbatim whose lines end in a full stop.
- **Margins.**
  - The stutter clears the 150 floor by 23 characters.
  - One fire in `qwen:1612` rests on a single repeated 171-character sentence, because varying 1
    lets half of a two-sentence copy differ.
- **The incident.** It was held out until the limits were fixed, then scored once. Two diagnoses
  read it afterwards, so it is no longer a held-out test of any later change.
- **What was not measured.**
  - The live quality of a rewind after a two-copy duplicate on an agent turn.
  - Whether the nudge's "move on with what I already have" helps there.
  - Any model other than qwen3.8-27b.

## Follow-ups

- A copy that repeats the one before it with a digression inserted (1355): a tolerance for one
  inserted block per copy, measured afresh on the false-fire side.
- Loops that reword their copies (328, 1612) remain with the indecision loops of the 2026-10-16
  finding.
