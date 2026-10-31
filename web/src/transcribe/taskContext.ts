import { isSpeechLanguage, languageName } from "../speech/languages.ts";
import { CHARS_PER_TOKEN, transcriptInput } from "./assist.ts";
import { EMPTY_CONTEXT, isBlank, namedSpeakers } from "./context.ts";
import type { Transcript } from "./transcript.ts";

// The context every text-model task is told (spec #428, Q142-Q178): the
// languages with the one to write in named, the terms, the topic and the rest,
// who "Me" is and the speakers by their names. It is the first lines of the
// user turn, ahead of the transcript, never in the system prompt: each task's
// system block stays the same from one request to the next, and the engine
// reuses a prompt only whole (assist.ts).

/** The language the tasks write in when nothing says: the model's own. */
const FALLBACK_LANGUAGE = "English";

const mostHeard = (t: Transcript): string | null => {
  const counts = new Map<string, number>();
  for (const u of t.utterances) if (u.language !== undefined && isSpeechLanguage(u.language)) counts.set(u.language, (counts.get(u.language) ?? 0) + 1);
  let best: string | null = null;
  for (const [code, n] of counts) if (best === null || n > (counts.get(best) ?? 0)) best = code;
  return best;
};

/**
 * The language a task writes in: the first under Force, otherwise the one most heard in the transcript, otherwise the
 * first expected, English when nothing is known.
 */
export function writeIn(t: Transcript): string {
  const c = t.context ?? EMPTY_CONTEXT;
  const expected = c.languages.codes.filter(isSpeechLanguage);
  if (c.languages.mode === "force" && expected.length > 0) return languageName(expected[0]);
  const heard = mostHeard(t);
  if (heard !== null) return languageName(heard);
  return expected.length > 0 ? languageName(expected[0]) : FALLBACK_LANGUAGE;
}

/** The context as a block of the user turn, ending with a blank line; the same for every task. */
export function contextBlock(t: Transcript): string {
  const c = t.context ?? EMPTY_CONTEXT;
  const lines = ["Context of the conversation, told by the user:", `- Write in: ${writeIn(t)}.`];
  if (isBlank(t.context)) lines.push("- Nothing else was told: no languages, terms or topic.");
  else {
    const codes = c.languages.codes.filter(isSpeechLanguage);
    if (codes.length > 0) lines.push(c.languages.mode === "force" ? `- Languages: ${languageName(codes[0])} only.` : `- Languages: ${codes.map(languageName).join(", ")}.`);
    if (c.topic.trim() !== "") lines.push(`- Topic: ${c.topic.trim()}`);
    if (c.terms.length > 0) lines.push(`- Terms, spelled so: ${c.terms.join(", ")}.`);
    if (c.other.trim() !== "") lines.push(`- More: ${c.other.trim()}`);
  }
  const named = namedSpeakers(t.speakers);
  lines.push(`- "Me" is the user, the person you work for. ${named.length > 0 ? `The user named the speakers ${named.join(", ")}: they are people, not terms.` : "The other speakers are not named."}`);
  lines.push("- A sentence said as a hypothetical or an example is not a fact about the speakers.");
  return `${lines.join("\n")}\n\n`;
}

/**
 * The user turn of a task: the context block, then the title and the transcript (`transcriptInput`'s budget less the
 * block, `time` writing each moment).
 */
export function taskInput(t: Transcript, contextLimit: number | null, answerTokens: number, time?: (ms: number) => string): { content: string; fromMs: number | null } {
  const block = contextBlock(t);
  const { content, fromMs } = transcriptInput(t, contextLimit, answerTokens + Math.ceil(block.length / CHARS_PER_TOKEN), time);
  return { content: block + content, fromMs };
}
