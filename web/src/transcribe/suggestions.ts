import type { Settings } from "../api/request.ts";
import { isSpeechLanguage, languageName } from "../speech/languages.ts";
import { addTerms, type CallContext, EMPTY_CONTEXT, isStrings, termProblem } from "./context.ts";
import { clock } from "./export.ts";
import { dayAndTime, fixUtterances, titleFor, type Transcript, wordsOf } from "./transcript.ts";

// What the Transcribe view suggests for a Transcript (spec speech/07, Q132):
// the languages, from what the server says of each utterance (no model);
// the topic, a title, the terms to spell right and the names it misheard
// (Q133), from the text model, which
// reads the transcript about every minute and a half while listening, or on
// Suggest now. A suggestion waits in the Call tab until the user accepts it,
// dismisses it, or accepts it always: then every later one of its kind, in
// this Transcript, is applied as it comes. Accepting changes what the
// Transcript is told (context.ts), its title, or the words a fix replaces in
// the utterances that hold them, each keeping the text it was heard as.

export type SuggestionKind = "languages" | "topic" | "title" | "terms" | "fix";

type Status = { id: number; status: "open" | "accepted" | "dismissed"; auto?: true };

export type Offer =
  | { kind: "languages"; languages: CallContext["languages"]; why: string }
  | { kind: "topic"; topic: string }
  | { kind: "title"; title: string }
  | { kind: "terms"; terms: string[] }
  /** `from`, misheard, is `to` in each of `utterances`, where the model found it (Q133). */
  | { kind: "fix"; from: string; to: string; utterances: number[] };

export type Suggestion = Status & Offer;

/** A Transcript's suggestions, as its document keeps them, and the kinds accepted always. */
export type Suggestions = { items: Suggestion[]; always: SuggestionKind[] };

export const NO_SUGGESTIONS: Suggestions = { items: [], always: [] };

const suggestionsOf = (t: Transcript): Suggestions => t.suggestions ?? NO_SUGGESTIONS;

export const openSuggestions = (t: Transcript): Suggestion[] => suggestionsOf(t).items.filter((s) => s.status === "open");

const sameCodes = (a: string[], b: string[]) => a.length === b.length && a.every((code, i) => code === b[i]);

/** Whether two offers propose the same thing. */
function same(a: Offer, b: Offer): boolean {
  switch (a.kind) {
    case "languages":
      return b.kind === "languages" && a.languages.mode === b.languages.mode && sameCodes(a.languages.codes, b.languages.codes);
    case "topic":
      return b.kind === "topic" && a.topic.toLocaleLowerCase() === b.topic.toLocaleLowerCase();
    case "title":
      return b.kind === "title" && a.title === b.title;
    case "terms":
      return b.kind === "terms" && sameCodes(a.terms, b.terms);
    case "fix":
      return b.kind === "fix" && a.from.toLocaleLowerCase() === b.from.toLocaleLowerCase() && a.to.toLocaleLowerCase() === b.to.toLocaleLowerCase();
  }
}

/** What accepting `offer` does to the Transcript. */
function applied(t: Transcript, offer: Offer, now: number): Transcript {
  const context = t.context ?? EMPTY_CONTEXT;
  switch (offer.kind) {
    case "languages":
      return { ...t, context: { ...context, languages: offer.languages, appliedAt: now } };
    case "topic":
      return { ...t, context: { ...context, topic: offer.topic, appliedAt: now } };
    case "terms":
      return { ...t, context: { ...context, terms: addTerms(context.terms, offer.terms.join(",")), appliedAt: now } };
    case "title":
      return { ...t, title: offer.title };
    case "fix": {
      // The right spelling is a term from now on: the speech model spells it so in what comes next.
      const fixed = fixUtterances(t, offer.from, offer.to, offer.utterances);
      return { ...fixed, context: { ...context, terms: addTerms(context.terms, offer.to), appliedAt: now } };
    }
  }
}

const nextId = (items: Suggestion[]) => Math.max(0, ...items.map((s) => s.id)) + 1;

/**
 * `offer`, among `t`'s suggestions: nothing when one like it was offered before, whatever became of it; in place of
 * an open one of its kind otherwise (the terms gather into one); applied at once when its kind is accepted always.
 * A fix offered again is offered for the utterances no fix like it covered yet, gathered into its open card; never
 * once dismissed.
 */
export function suggest(t: Transcript, offer: Offer, now: number): Transcript {
  const { items, always } = suggestionsOf(t);
  let fresh = offer;
  if (offer.kind === "terms") {
    // A term in the context, or offered before, is not offered again.
    const known = new Set([...(t.context?.terms ?? []), ...items.flatMap((s) => (s.kind === "terms" ? s.terms : []))].map((term) => term.toLocaleLowerCase()));
    const terms = offer.terms.filter((term) => !known.has(term.toLocaleLowerCase()));
    if (terms.length === 0) return t;
    const open = items.find((s): s is Suggestion & { kind: "terms" } => s.kind === "terms" && s.status === "open");
    fresh = { kind: "terms", terms: [...(open && !always.includes("terms") ? open.terms : []), ...terms] };
  } else if (offer.kind === "fix") {
    const alike = items.filter((s) => same(s, offer));
    if (alike.some((s) => s.status === "dismissed")) return t;
    // An utterance a fix like it covered, undone since or not, is not offered again.
    const covered = new Set(alike.flatMap((s) => (s.kind === "fix" ? s.utterances : [])));
    const uncovered = offer.utterances.filter((id) => !covered.has(id));
    if (uncovered.length === 0) return t;
    const open = alike.find((s) => s.status === "open");
    if (open && open.kind === "fix") {
      const gathered: Suggestion = { ...open, utterances: [...open.utterances, ...uncovered] };
      return { ...t, suggestions: { items: items.map((s) => (s.id === open.id ? gathered : s)), always } };
    }
    fresh = { ...offer, utterances: uncovered };
  } else if (items.some((s) => same(s, offer))) {
    return t;
  }
  // Fixes stand side by side, one card each; of every other kind the new one takes the open one's place.
  const kept = offer.kind === "fix" ? items : items.filter((s) => !(s.status === "open" && s.kind === offer.kind));
  if (always.includes(offer.kind)) {
    const done: Suggestion = { ...fresh, id: nextId(items), status: "accepted", auto: true };
    return { ...applied(t, fresh, now), suggestions: { items: [...items, done], always } };
  }
  return { ...t, suggestions: { items: [...kept, { ...fresh, id: nextId(items), status: "open" }], always } };
}

/**
 * The fixes accepted, followed into the utterances heard since that hold the same misheard words: offered for them
 * (or, accepted always, applied), with no model; called as utterances come.
 */
export function followFixes(t: Transcript, now: number): Transcript {
  let next = t;
  for (const s of suggestionsOf(t).items) {
    if (s.kind !== "fix" || s.status !== "accepted") continue;
    const holding = next.utterances.filter((u) => wordsOf(s.from).test(u.text)).map((u) => u.id);
    if (holding.length > 0) next = suggest(next, { kind: "fix", from: s.from, to: s.to, utterances: holding }, now);
  }
  return next;
}

/** Accepts a suggestion; `always`, and every later one of its kind in this Transcript. */
export function accept(t: Transcript, id: number, now: number, { always = false } = {}): Transcript {
  const { items, always: kinds } = suggestionsOf(t);
  const chosen = items.find((s) => s.id === id && s.status === "open");
  if (!chosen) return t;
  const next = applied(t, chosen, now);
  return {
    ...next,
    suggestions: {
      items: items.map((s) => (s.id === id ? { ...s, status: "accepted" } : s)),
      always: always && !kinds.includes(chosen.kind) ? [...kinds, chosen.kind] : kinds,
    },
  };
}

export function dismiss(t: Transcript, id: number): Transcript {
  const { items, always } = suggestionsOf(t);
  return { ...t, suggestions: { items: items.map((s) => (s.id === id && s.status === "open" ? { ...s, status: "dismissed" } : s)), always } };
}

/** Takes one term out of an open terms suggestion: dismissed alone, never offered again. */
export function dropTerm(t: Transcript, id: number, term: string): Transcript {
  const { items, always } = suggestionsOf(t);
  const open = items.find((s) => s.id === id && s.status === "open" && s.kind === "terms");
  if (!open || open.kind !== "terms") return t;
  const rest = open.terms.filter((x) => x !== term);
  const dropped: Suggestion = { kind: "terms", terms: [term], id: nextId(items), status: "dismissed" };
  const updated = items.flatMap((s): Suggestion[] => (s.id !== id ? [s] : rest.length > 0 ? [{ ...open, terms: rest }] : []));
  return { ...t, suggestions: { items: [...updated, dropped], always } };
}

// --- The languages, from the server's word on each utterance.

/** Before this many utterances with a language, none is suggested. */
export const LANGUAGE_EVIDENCE = 6;
/** A language of at least this share, with two utterances, is one the conversation is in. */
const SPOKEN_SHARE = 0.15;
/** A language this dominant, the others heard only on short utterances, is one to expect. */
const DOMINANT_SHARE = 0.7;
/** An utterance this short is where the model's guess of the language goes astray (2026-10-27). */
export const SHORT_UTTERANCE_MS = 2_500;

const names = (codes: string[]) => codes.map(languageName).join(" and ");

/**
 * What the utterances' languages suggest, or `null`: the one language spoken, as a hint, when the others came only
 * on short utterances (the model's guess going astray, as on the owner's first call), else the languages spoken.
 * Never Force: measured, the hint fixes the short utterances as forcing does, and forcing translates whatever is
 * said in another language (docs/findings/2026-10-27-asr-language-hint.md); Force stays the user's, in the form.
 */
export function languageOffer(t: Transcript): Offer | null {
  const heard = t.utterances.filter((u) => u.language !== undefined && isSpeechLanguage(u.language));
  if (heard.length < LANGUAGE_EVIDENCE) return null;
  const counts = new Map<string, number>();
  for (const u of heard) counts.set(u.language!, (counts.get(u.language!) ?? 0) + 1);
  const ranked = [...counts].sort((a, b) => b[1] - a[1]);
  const [top, topCount] = ranked[0];
  const current = t.context?.languages ?? EMPTY_CONTEXT.languages;
  const others = heard.filter((u) => u.language !== top);
  if (current.mode === "force") return null;
  if (topCount / heard.length >= DOMINANT_SHARE && others.length >= 2 && others.every((u) => u.endMs - u.startMs < SHORT_UTTERANCE_MS)) {
    if (current.codes.includes(top)) return null;
    const strays = [...new Set(others.map((u) => u.language!))];
    return {
      kind: "languages",
      languages: { mode: "free", codes: [top] },
      why: `${others.length} short utterances came out as ${names(strays)}, among ${topCount} in ${languageName(top)}.`,
    };
  }
  const spoken = ranked.filter(([, n]) => n >= 2 && n / heard.length >= SPOKEN_SHARE).map(([code]) => code);
  if (spoken.length === 0 || sameCodes([...spoken].sort(), [...current.codes].sort())) return null;
  return {
    kind: "languages",
    languages: { mode: "free", codes: spoken },
    why: `Heard ${spoken.map((code) => `${languageName(code)} (${counts.get(code)})`).join(" and ")}.`,
  };
}

// --- The topic, a title and the terms, from the text model. The task asks for
// the misheard names first (FIX lines): without them the 27B, thinking off,
// copies a misheard name into the terms as it was heard; asked to
// check them first, it fixes it and spells the term right (probed 2026-10-27,
// at temperature 0.2; greedy did worse). Each FIX becomes a fix suggestion.

export const ANALYSIS_TASK = [
  "Task: read the transcript and write what it suggests, one item a line, in exactly these forms, in this order, and nothing else:",
  'FIX: [mm:ss] heard => meant — first, check every name, product and technical term: where speech recognition misheard one as other words or a wrong spelling ("kuber netis" for Kubernetes, "libre ofis" for LibreOffice), write the utterance\'s time, the words exactly as the transcript has them, and the real name. Only names, products and terms, never a rephrasing; at most ten.',
  "TERM: each name, product or technical term the conversation mentions, spelled correctly, as fixed above (one a line, at most ten)",
  "TOPIC: what the conversation is about, in a few words, in the language the context says to write in",
  "TITLE: a short title for it, at most six words, in the language the context says to write in",
  "Write a line only where the transcript shows it. With nothing to write, write NONE.",
].join("\n");

/** The tokens a look may take. */
export const ANALYSIS_MAX_TOKENS = 400;

export const ANALYSIS_SETTINGS: Omit<Settings, "systemPrompt"> = {
  temperature: 0.2,
  topP: 0.9,
  maxTokens: ANALYSIS_MAX_TOKENS,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "agent",
};

/** A misheard name the model found: at `at` (mm:ss), `from` as the transcript has it, `to` as it was said. */
export type FoundFix = { at: string; from: string; to: string };

/** What a look found. */
export type Analysis = { topic: string | null; title: string | null; terms: string[]; fixes: FoundFix[] };

/** The model's lines, read leniently: a bullet or bold around them is fine, a line in no form is dropped. */
export function readAnalysis(text: string): Analysis {
  const found: Analysis = { topic: null, title: null, terms: [], fixes: [] };
  for (const raw of text.split("\n")) {
    const line = raw.replace(/^[\s*>-]+/, "").replace(/\*\*/g, "").trim();
    const fix = /^FIX\s*:\s*\[?~?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*(.+?)\s*(?:=>|→|->)\s*(.+)$/i.exec(line);
    if (fix) {
      const [from, to] = [fix[2], fix[3]].map((side) => side.trim().replace(/^["“']|["”']$/g, "").replace(/\.$/, "").trim());
      const plain = (side: string) => side !== "" && side.length <= 60 && side.split(/\s+/).length <= 4 && termProblem(side) === null;
      if (plain(from) && plain(to) && from.toLocaleLowerCase() !== to.toLocaleLowerCase() && found.fixes.length < 10) found.fixes.push({ at: fix[1], from, to });
      continue;
    }
    const m = /^(TOPIC|TITLE|TERM)\s*:\s*(.+)$/i.exec(line);
    if (!m) continue;
    const value = m[2].trim().replace(/^["“']|["”']$/g, "").replace(/\.$/, "").trim();
    if (value === "" || /^none$/i.test(value)) continue;
    const label = m[1].toUpperCase();
    if (label === "TOPIC") found.topic ??= value.slice(0, 120);
    else if (label === "TITLE") found.title ??= value.slice(0, 80);
    else if (termProblem(value) === null && value.split(/\s+/).length <= 4 && found.terms.length < 10) found.terms.push(value);
  }
  return found;
}

/** The title a listening gave it, unchanged since: one a suggestion may replace. */
export function autoTitled(t: Transcript): boolean {
  return t.source !== null && t.source !== "file" && t.startedAt !== null && t.title === titleFor(t.source, new Date(t.startedAt));
}

/** The offers a look's findings make: the topic while none is told, a title while the listening's stands, the terms. */
export function analysisOffers(t: Transcript, found: Analysis): Offer[] {
  const offers: Offer[] = [];
  if (found.topic && (t.context?.topic.trim() ?? "") === "") offers.push({ kind: "topic", topic: found.topic });
  if (found.title && autoTitled(t)) {
    const { day, time } = dayAndTime(new Date(t.startedAt!));
    offers.push({ kind: "title", title: `${day} ${time} · ${found.title}` });
  }
  if (found.terms.length > 0) offers.push({ kind: "terms", terms: found.terms });
  for (const fix of found.fixes) {
    // Every utterance that holds the words as words of their own, the one at its time or not; none, no fix.
    const holding = t.utterances.filter((u) => wordsOf(fix.from).test(u.text)).map((u) => u.id);
    if (holding.length > 0) offers.push({ kind: "fix", from: fix.from, to: fix.to, utterances: holding });
  }
  return offers;
}

/** Where a fix applies, as a card says it: "at 03:12", or "in 3 utterances". */
export function fixWhere(t: Transcript, s: Offer & { kind: "fix" }): string {
  const ids = s.utterances.filter((id) => t.utterances.some((u) => u.id === id));
  if (ids.length !== 1) return `in ${ids.length} utterances`;
  return `at ${clock(t.utterances.find((u) => u.id === ids[0])!.startMs)}`;
}

/** Looking for suggestions: the first once this many utterances are in. */
export const FIRST_ANALYSIS_UTTERANCES = 6;
/** Looking for suggestions: the least time from one look's start to the next, and what has to be new by then. */
export const ANALYSIS_GAP_MS = 90_000;
export const ANALYSIS_FRESH = 3;

/** Whether a listening looks for suggestions now; `analyzed`, the utterances the last look read. */
export function dueForAnalysis({ utterances, analyzed, sinceLastMs, final = false }: { utterances: number; analyzed: number; sinceLastMs: number | null; final?: boolean }): boolean {
  const fresh = utterances - analyzed;
  if (fresh <= 0) return false;
  if (final) return true;
  if (analyzed === 0) return utterances >= FIRST_ANALYSIS_UTTERANCES;
  return fresh >= ANALYSIS_FRESH && (sinceLastMs === null || sinceLastMs >= ANALYSIS_GAP_MS);
}

const KINDS: SuggestionKind[] = ["languages", "topic", "title", "terms", "fix"];

function readOne(value: unknown): Suggestion | null {
  if (typeof value !== "object" || value === null) return null;
  const s = value as Record<string, unknown>;
  if (typeof s.id !== "number" || !(s.status === "open" || s.status === "accepted" || s.status === "dismissed")) return null;
  switch (s.kind) {
    case "languages": {
      const l = s.languages as Record<string, unknown> | undefined;
      return l && (l.mode === "free" || l.mode === "force") && isStrings(l.codes) && typeof s.why === "string" ? (s as Suggestion) : null;
    }
    case "topic":
      return typeof s.topic === "string" ? (s as Suggestion) : null;
    case "title":
      return typeof s.title === "string" ? (s as Suggestion) : null;
    case "terms":
      return isStrings(s.terms) ? (s as Suggestion) : null;
    case "fix":
      return typeof s.from === "string" && typeof s.to === "string" && Array.isArray(s.utterances) && s.utterances.every((id) => typeof id === "number")
        ? (s as Suggestion)
        : null;
    default:
      return null;
  }
}

/** A stored Transcript's suggestions, those this page can read; `undefined` with none. */
export function readSuggestions(value: unknown): Suggestions | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const v = value as Record<string, unknown>;
  if (!Array.isArray(v.items) || !Array.isArray(v.always)) return undefined;
  const items = v.items.map(readOne).filter((s): s is Suggestion => s !== null);
  const always = v.always.filter((k): k is SuggestionKind => KINDS.includes(k as SuggestionKind));
  return { items, always };
}
