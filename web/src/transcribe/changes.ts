import { minutesClock, toText } from "./export.ts";
import type { NoteLine, Notes } from "./notes.ts";
import type { Transcript, Utterance } from "./transcript.ts";

// What changed since a notes thread's last turn (spec #428, "Each update"): the
// utterances said since and what the user changed in the transcript, by a diff
// of the transcript against what the thread told of it (#434), and the notes'
// lines the user edited, added or removed by hand, by a diff of the notes
// against the lines the thread knows (#436). The thread (notesWriter.ts) tells
// both in its next turn (running.ts's laterTurn), under "Changed since:".

/** What the model is told of utterances heard: each one's line, from its start. */
export type Said = { startMs: number; line: string };

/**
 * What a thread told the model of an utterance: its start, its speaker and the name it had, the text, and its line
 * without the line break, "" while the model knows of no line (its text emptied, or the utterance deleted).
 */
export type Told = { startMs: number; speaker: number | null; who: string | null; text: string; line: string };

/** An utterance of `t`'s line as the thread tells it, in total minutes: "" for an emptied one. */
const lineOf = (t: Transcript, u: Utterance): string => toText({ ...t, utterances: [u] }, minutesClock);

/**
 * Whether `u` is the utterance the thread told as `was`: the same id from the same start; not an id the page gave again
 * once the last utterance was deleted.
 */
const isTold = (u: Utterance | undefined, was: Told | undefined): boolean => u !== undefined && was !== undefined && was.startMs === u.startMs;

/** `t`'s utterances by id. */
const byIdOf = (t: Transcript) => new Map(t.utterances.map((u) => [u.id, u]));

/**
 * What a thread tells the model of `t`'s utterances, by id, starting from it or appending a turn of it: each utterance
 * as it stands, and, from what it told `before`, each one since deleted as no line, so a deletion is told once and an
 * Undo after it as put back.
 */
export function toldOf(t: Transcript, before?: Map<number, Told>): Map<number, Told> {
  const names = new Map(t.speakers.map((s) => [s.id, s.name]));
  const byId = byIdOf(t);
  const told = new Map<number, Told>();
  for (const [id, was] of before ?? []) if (!isTold(byId.get(id), was)) told.set(id, { ...was, line: "" });
  for (const u of t.utterances) {
    const who = u.speaker === null ? null : (names.get(u.speaker) ?? null);
    told.set(u.id, { startMs: u.startMs, speaker: u.speaker, who, text: u.text.trim(), line: lineOf(t, u).trimEnd() });
  }
  return told;
}

/**
 * What a later turn tells of `t`, against what the thread `told` before: the utterances said since, and one event a
 * change the user made meanwhile (#434), each with its time and its new text: first a speaker renamed or merged into
 * another, by the name the model knows them by; then, in the order of their times, an utterance corrected (an edit, an
 * accepted fix, a Rewrite), moved to another speaker, split, deleted or put back. A rename is the model's to carry into
 * the notes' lines: the page never puts a name in one.
 */
export function changesSince(told: Map<number, Told>, t: Transcript): { said: Said[]; events: string[] } {
  const byId = byIdOf(t);
  // The speakers the model knows, by the name it knows and the first time it heard them; one gone from the speakers
  // was merged into the one its utterances are now.
  const known = new Map<number, { who: string; at: number; now: Map<number, number> }>();
  for (const [id, was] of told) {
    if (was.line === "" || was.speaker === null || was.who === null) continue;
    const speaker = known.get(was.speaker) ?? { who: was.who, at: was.startMs, now: new Map<number, number>() };
    speaker.at = Math.min(speaker.at, was.startMs);
    const u = byId.get(id);
    if (u?.speaker != null && isTold(u, was)) speaker.now.set(u.speaker, (speaker.now.get(u.speaker) ?? 0) + 1);
    known.set(was.speaker, speaker);
  }
  const names = new Map(t.speakers.map((s) => [s.id, s.name]));
  const events: string[] = [];
  const mergedInto = new Map<number, number>();
  for (const [id, { who, at, now }] of known) {
    const name = names.get(id);
    const heard = `${who}, heard from [${minutesClock(at)}],`;
    if (name !== undefined) {
      if (name !== who) events.push(`Renamed: ${heard} is ${name}.`);
      continue;
    }
    const into = [...now].sort((a, b) => b[1] - a[1])[0]?.[0];
    if (into === undefined || !names.has(into)) continue;
    mergedInto.set(id, into);
    events.push(`Merged: ${heard} is ${names.get(into)}.`);
  }
  // A split keeps its first half's id and puts the rest right after it, under new ids, each start guessed (`~`): the
  // utterances so placed after one told are its halves, not something said.
  const halves = new Map<number, Utterance[]>();
  const halved = new Set<Utterance>();
  let split: Utterance | null = null;
  for (const u of t.utterances) {
    if (isTold(u, told.get(u.id))) split = u;
    else if (split && u.approxStart) {
      halves.set(split.id, [...(halves.get(split.id) ?? []), u]);
      halved.add(u);
    } else split = null;
  }
  const said: Said[] = [];
  const changes: { at: number; event: string }[] = [];
  const change = (at: number, event: string) => changes.push({ at, event });
  for (const u of t.utterances) {
    const was = told.get(u.id);
    const line = lineOf(t, u);
    if (was === undefined || !isTold(u, was)) {
      if (line !== "" && !halved.has(u)) said.push({ startMs: u.startMs, line });
      continue;
    }
    const rest = halves.get(u.id) ?? [];
    if (rest.length > 0) {
      const lines = [u, ...rest].map((x) => lineOf(t, x).trimEnd()).filter((l) => l !== "");
      change(u.startMs, `Split: ${lines.join(" | ")}`);
      continue;
    }
    const now = line.trimEnd();
    if (was.line === "" || now === "") {
      // Emptied is deleted, as far as the model knows: it is told no line of it.
      if (now !== was.line) change(u.startMs, now === "" ? `Deleted: ${was.line}` : `Put back: ${now}`);
      continue;
    }
    const moved = u.speaker !== was.speaker && (was.speaker === null || mergedInto.get(was.speaker) !== u.speaker);
    if (u.text.trim() !== was.text) change(u.startMs, `Corrected: ${now} It was: ${moved && was.who !== null ? `${was.who}: ` : ""}${was.text}`);
    else if (moved) change(u.startMs, `Moved: ${now}${was.who === null ? "" : ` It was ${was.who}'s.`}`);
  }
  for (const [id, was] of told) if (was.line !== "" && !isTold(byId.get(id), was)) change(was.startMs, `Deleted: ${was.line}`);
  changes.sort((a, b) => a.at - b.at);
  return { said, events: [...events, ...changes.map((c) => c.event)] };
}

/** The notes' lines as a thread told them, by id: a snapshot's or a turn's, then what each call of the model did. */
export const linesOf = (notes: Notes): Map<string, NoteLine> => new Map(notes.lines.map((l) => [l.id, l]));

/**
 * The lines a thread knows once a call of the model took the notes from `before` to `after`: the ones it added, changed
 * or removed. An edit the user made meanwhile is not among them, so the next turn tells it.
 */
export function afterCall(lines: Map<string, NoteLine>, before: Notes, after: Notes): Map<string, NoteLine> {
  const known = new Map(lines);
  const was = new Map(before.lines.map((l) => [l.id, l]));
  const now = new Set(after.lines.map((l) => l.id));
  for (const l of before.lines) if (!now.has(l.id)) known.delete(l.id);
  for (const l of after.lines) {
    const old = was.get(l.id);
    if (!old || old.text !== l.text || old.section !== l.section || old.author !== l.author) known.set(l.id, l);
  }
  return known;
}

/**
 * What the user did to the notes by hand since the thread told it the `lines` (#436), one event a line, as `notes` stand
 * now: a line they edited, now theirs, with what it was; one they added; one they removed, never to be written again. A
 * line they added and removed meanwhile is not told.
 */
export function editsSince(lines: Map<string, NoteLine>, notes: Notes): string[] {
  const events: string[] = [];
  for (const l of notes.lines) {
    const was = lines.get(l.id);
    if (l.author !== "user") continue;
    if (!was) events.push(`Added by the user to ${l.section}: ${l.id} (the user's) ${l.text}`);
    else if (was.author !== "user" || was.text !== l.text || was.section !== l.section) {
      events.push(`Edited by the user in ${l.section}: ${l.id} (the user's) ${l.text}${was.text === l.text ? "" : ` It was: ${was.text}`}`);
    }
  }
  for (const l of notes.removed) if (lines.has(l.id)) events.push(`Removed by the user from ${l.section}, never to be written again: ${l.id} ${l.text}`);
  return events;
}
