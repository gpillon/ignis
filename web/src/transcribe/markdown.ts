import { type Answered, answerAt } from "./ask.ts";
import { imageAt, readImages, readingLine } from "./contextImages.ts";
import { startedLine, startOf } from "./export.ts";
import { notesMarkdown } from "./notes.ts";
import { recapMarkdown } from "./recap.ts";
import type { Transcript } from "./transcript.ts";

// The `.md` export (spec #428, "Exports"). It lives apart from export.ts, which
// the notes, Ask and the assist lane import: it reads theirs back.

/** An answer as a quoted block, its question the title; every line of it quoted, a blank one too. */
function answerBlock(a: Answered): string {
  return [`**${a.question.trim()}**`, "", ...a.answer.trim().split("\n")].map((line) => (line === "" ? ">" : `> ${line}`)).join("\n");
}

/**
 * `.md`, and what Open in chat seeds the composer with: the title and the start, then the notes (the recap when
 * there is one, else the running notes) in their sections, then `**Speaker** (mm:ss): text` with each of Ask's
 * answers as a quoted block, titled by its question, after the utterances it read (an answer with no place, at the
 * foot), and each context image read as a line `[mm:ss] (image) ...` with what was read, at its moment, ahead of an
 * answer at the same place. With neither notes, answers nor images read it is the title, the start and the lines.
 */
export function toMarkdown(t: Transcript): string {
  const header = [`# ${t.title}`, ...(t.startedAt === null ? [] : [startedLine(t.startedAt)])];
  const notes = recapMarkdown(t.recap) || notesMarkdown(t.notes);
  const answers = [...(t.asked ?? [])].sort((x, y) => x.id - y.id).filter((a) => a.answer.trim() !== "");
  const names = new Map(t.speakers.map((s) => [s.id, s.name]));
  const body: string[] = [];
  const images = readImages(t);
  const asidesAt = (place: number) => {
    for (const image of images) if (imageAt(t.utterances, image.atMs) === place) body.push(readingLine(image));
    for (const a of answers) if (answerAt(t.utterances, a.untilMs) === place) body.push(answerBlock(a));
  };
  t.utterances.forEach((u, i) => {
    asidesAt(i);
    const text = u.text.trim();
    const speaker = u.speaker === null ? null : (names.get(u.speaker) ?? null);
    if (text !== "") body.push(`${speaker === null ? "" : `**${speaker}** `}(${startOf(u)}): ${text}`);
  });
  asidesAt(t.utterances.length);
  return `${[...header, ...(notes === "" ? [] : [notes, ...(body.length > 0 ? ["## Transcript"] : [])]), ...body].join("\n\n")}\n`;
}
