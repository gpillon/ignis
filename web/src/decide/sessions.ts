// Decisions being worked on, one per session (GitHub #247).
//
// A new module rather than the chat's (`sessions/sessions.ts`): a conversation
// carries messages, a figures log, attached files, tool runs and the moment it
// opened; a decision carries a draft, the evidence shapes set aside, and one
// answered run. The two share `id` and a name, so a shared type would be an
// abstraction over nothing. What *is* borrowed is the mechanics — the same
// rules for opening, removing and the active one — spelled the same way, so a
// reader who knows one knows the other.
//
// They live in memory only; a reload starts over. Every function returns a new
// list and leaves the one it was given alone.

import { asText } from "./json.ts";
import { type Draft, EMPTY_DRAFT, EMPTY_SPARE, type Spare } from "./model.ts";
import type { Run } from "./request.ts";

export type Decision = {
  id: number;
  draft: Draft;
  /** The evidence shapes not in use, so switching mode and back is not a loss. */
  spare: Spare;
  /**
   * The last run and **the draft it answered**. The draft is kept beside it
   * because the panels read the answers in the order that request declared,
   * and the live draft may have been edited since.
   */
  answer: { run: Run; draft: Draft } | null;
  /** Why the last send produced no answers. */
  refusal: string | null;
  /** A send is in flight for this decision. */
  running: boolean;
};

export type DecisionList = { decisions: Decision[]; activeId: number };

export const UNTITLED = "New decision";

const TITLE_LENGTH = 40;

export function createDecision(id: number): Decision {
  return { id, draft: EMPTY_DRAFT, spare: EMPTY_SPARE, answer: null, refusal: null, running: false };
}

export function startList(id = 1): DecisionList {
  return { decisions: [createDecision(id)], activeId: id };
}

/** An id nothing in the list is using. Derived, so it cannot drift from the list. */
export const freeDecisionId = (list: DecisionList): number => Math.max(0, ...list.decisions.map((d) => d.id)) + 1;

export const activeOf = (list: DecisionList): Decision =>
  list.decisions.find((d) => d.id === list.activeId) ?? list.decisions[0];

/** Nothing has been written in it and nothing has been asked of it. */
export const isUntouched = (decision: Decision): boolean =>
  decision.draft.questions.length === 0 && evidenceIsEmpty(decision.draft) && decision.answer === null;

function evidenceIsEmpty(draft: Draft): boolean {
  const { evidence } = draft;
  return evidence.mode === "image" ? evidence.images.length === 0 && evidence.text.trim() === "" : evidence.text.trim() === "";
}

/**
 * A decision to work in. An untouched active one is reused, so repeated
 * clicks do not pile up empty sessions; otherwise `newId` opens at the top.
 */
export function openDecision(list: DecisionList, newId: number): DecisionList {
  const active = activeOf(list);
  if (active && isUntouched(active)) return list;
  return { decisions: [createDecision(newId), ...list.decisions], activeId: newId };
}

/**
 * Drops a decision. Removing the active one moves to whichever took its place
 * in the list; removing the last one leaves a fresh `newId`.
 */
export function removeDecision(list: DecisionList, id: number, newId: number): DecisionList {
  const index = list.decisions.findIndex((d) => d.id === id);
  if (index === -1) return list;
  const decisions = list.decisions.filter((d) => d.id !== id);
  if (decisions.length === 0) return startList(newId);
  if (id !== list.activeId) return { decisions, activeId: list.activeId };
  return { decisions, activeId: decisions[Math.min(index, decisions.length - 1)].id };
}

export function updateDecision(list: DecisionList, id: number, change: (decision: Decision) => Decision): DecisionList {
  return { ...list, decisions: list.decisions.map((d) => (d.id === id ? change(d) : d)) };
}

/**
 * What to call a decision in the list: what its first question asks, else the
 * evidence it is about.
 *
 * Derived rather than stored, unlike a conversation's title. A prompt is fixed
 * the moment it is sent, so the chat can name a session once; a draft is edited
 * until it is sent and then edited again, so a name written once would be
 * describing a decision that no longer exists.
 */
export function titleOf(decision: Decision): string {
  const first = decision.draft.questions[0];
  const asked = first ? shorten(asText(first.instructions)) : "";
  if (asked) return asked;
  const { evidence } = decision.draft;
  const about = shorten(evidence.mode === "image" ? (evidence.images[0]?.name ?? evidence.text) : evidence.text);
  return about || UNTITLED;
}

function shorten(text: string): string {
  const line = text.split("\n").find((l) => l.trim() !== "")?.trim().replace(/\s+/g, " ") ?? "";
  if (line === "") return "";
  return line.length > TITLE_LENGTH ? `${line.slice(0, TITLE_LENGTH - 1).trimEnd()}…` : line;
}

/** How many questions a decision asks, and how many it has answers for. */
export function counts(decision: Decision): { asked: number; answered: number } {
  const answered = decision.answer
    ? Object.values(decision.answer.run.response.answers).filter((a) => a.type !== "error").length
    : 0;
  return { asked: decision.draft.questions.length, answered };
}
