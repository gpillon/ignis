import { type Answered, answerAt, answersKept } from "./ask.ts";
import { imageAt, readImages, readingLine } from "./contextImages.ts";
import { speakerOf, startedLine, startOf } from "./export.ts";
import { notesForAfter } from "./recap.ts";
import { placedAmong, type Transcript } from "./transcript.ts";

// The `.md` export (spec #428, "Exports"). It lives apart from export.ts, which
// the notes, Ask and the assist lane import: it reads theirs back.

/** An answer as a quoted block, its question the title; every line of it quoted, a blank one too. */
function answerBlock(a: Answered): string {
  return [`**${a.question.trim()}**`, "", ...a.answer.trim().split("\n")].map((line) => (line === "" ? ">" : `> ${line}`)).join("\n");
}

/**
 * `.md`: the title and the start, then the notes (the recap when there is one, else the running notes) in their
 * sections, then `**Speaker** (mm:ss): text` with each of Ask's
 * answers as a quoted block, titled by its question, after the utterances it read (an answer with no place, at the
 * foot), and each context image read as a line `[mm:ss] (image) ...` with what was read, at its moment, ahead of an
 * answer at the same place. With neither notes, answers nor images read it is the title, the start and the lines.
 */
export function toMarkdown(t: Transcript): string {
  const header = [`# ${t.title}`, ...(t.startedAt === null ? [] : [startedLine(t.startedAt)])];
  const notes = notesForAfter(t)?.markdown ?? "";
  const nameOf = speakerOf(t);
  const body: string[] = [];
  const images = placedAmong(readImages(t), (i) => imageAt(t.utterances, i.atMs));
  const answers = placedAmong(answersKept(t), (a) => answerAt(t.utterances, a.untilMs));
  const asidesAt = (place: number) => {
    for (const image of images.get(place) ?? []) body.push(readingLine(image));
    for (const a of answers.get(place) ?? []) body.push(answerBlock(a));
  };
  t.utterances.forEach((u, i) => {
    asidesAt(i);
    const text = u.text.trim();
    const speaker = nameOf(u);
    if (text !== "") body.push(`${speaker === null ? "" : `**${speaker}** `}(${startOf(u)}): ${text}`);
  });
  asidesAt(t.utterances.length);
  return `${[...header, ...(notes === "" ? [] : [notes, ...(body.length > 0 ? ["## Transcript"] : [])]), ...body].join("\n\n")}\n`;
}

/**
 * The transcript alone as Markdown, what Open in chat seeds the composer with and Send to workspace files: the `.md`
 * of the title, the start and the lines, without the notes, Ask's answers or the context images' lines, which stay
 * out of a chat (spec #428, Out of Scope).
 */
export const transcriptMarkdown = (t: Transcript): string => toMarkdown({ ...t, notes: undefined, recap: undefined, asked: undefined, images: undefined });
