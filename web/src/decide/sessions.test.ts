import { describe, expect, it } from "vitest";
import { jsonString } from "./json.ts";
import { type Draft, EMPTY_DRAFT, EMPTY_SPARE, newQuestion } from "./model.ts";
import type { DecideResponse, Run } from "./request.ts";
import {
  activeOf,
  counts,
  createDecision,
  type Decision,
  freeDecisionId,
  isUntouched,
  openDecision,
  removeDecision,
  startList,
  titleOf,
  UNTITLED,
  updateDecision,
} from "./sessions.ts";

// Decisions as sessions (GitHub #247): the same rules the chat's sessions
// follow, over a different thing.

const asking = (text: string): Draft => ({
  ...EMPTY_DRAFT,
  questions: [{ ...newQuestion("noul", "a"), instructions: jsonString(text) }],
});

const about = (text: string): Draft => ({ ...EMPTY_DRAFT, evidence: { mode: "text", text } });

function ran(answers: DecideResponse["answers"]): Run {
  const response: DecideResponse = { model: "m", answers, usage: { input_tokens: 1, output_tokens: 0 } };
  return { response, elapsedMs: 1, raw: JSON.stringify(response) };
}

const withDraft = (decision: Decision, draft: Draft): Decision => ({ ...decision, draft });

describe("startList", () => {
  it("opens on one empty decision, which is the active one", () => {
    const list = startList();
    expect(list.decisions).toHaveLength(1);
    expect(activeOf(list)).toBe(list.decisions[0]);
    expect(isUntouched(activeOf(list))).toBe(true);
  });
});

describe("freeDecisionId", () => {
  it("is past every id in the list, so it cannot drift from it", () => {
    expect(freeDecisionId(startList())).toBe(2);
    expect(freeDecisionId({ decisions: [createDecision(4), createDecision(9)], activeId: 4 })).toBe(10);
  });
});

describe("openDecision", () => {
  it("reuses an untouched active decision, so repeated clicks do not pile up empty ones", () => {
    const list = startList();
    expect(openDecision(list, 2)).toBe(list);
  });

  it("opens a new one at the top once anything has been written", () => {
    const written = updateDecision(startList(), 1, (d) => withDraft(d, asking("Urgent?")));
    const opened = openDecision(written, 2);
    expect(opened.decisions.map((d) => d.id)).toEqual([2, 1]);
    expect(opened.activeId).toBe(2);
  });

  it("counts evidence with no questions as written, and an answer too", () => {
    const evidence = updateDecision(startList(), 1, (d) => withDraft(d, about("Help!")));
    expect(openDecision(evidence, 2).decisions).toHaveLength(2);
    const answered = updateDecision(startList(), 1, (d) => ({ ...d, answer: { run: ran({}), draft: EMPTY_DRAFT } }));
    expect(openDecision(answered, 2).decisions).toHaveLength(2);
  });
});

describe("removeDecision", () => {
  const three = () => ({ decisions: [createDecision(3), createDecision(2), createDecision(1)], activeId: 2 });

  it("leaves a fresh one when the last goes, so there is always somewhere to write", () => {
    const list = removeDecision(startList(), 1, 7);
    expect(list.decisions.map((d) => d.id)).toEqual([7]);
    expect(list.activeId).toBe(7);
  });

  it("keeps the active one when another is removed", () => {
    expect(removeDecision(three(), 3, 9)).toMatchObject({ activeId: 2 });
  });

  it("moves to whichever took the active one's place", () => {
    expect(removeDecision(three(), 2, 9).activeId).toBe(1);
  });

  it("moves back when the removed one was last in the list", () => {
    expect(removeDecision({ ...three(), activeId: 1 }, 1, 9).activeId).toBe(2);
  });

  it("ignores an id that is not there", () => {
    const list = three();
    expect(removeDecision(list, 42, 9)).toBe(list);
  });
});

describe("titleOf", () => {
  it("is what the first question asks", () => {
    expect(titleOf(withDraft(createDecision(1), asking("Does this convey urgency?")))).toBe("Does this convey urgency?");
  });

  it("falls back to the evidence when nothing has been asked yet", () => {
    expect(titleOf(withDraft(createDecision(1), about("Help! My payouts are failing.")))).toBe("Help! My payouts are failing.");
  });

  it("names an image evidence by the file, since its text may be empty", () => {
    const draft: Draft = { ...EMPTY_DRAFT, evidence: { mode: "image", images: [{ name: "receipt.jpg", url: "data:,", width: 1, height: 1 }], text: "" } };
    expect(titleOf(withDraft(createDecision(1), draft))).toBe("receipt.jpg");
  });

  it("is the placeholder while a decision is empty", () => {
    expect(titleOf(createDecision(1))).toBe(UNTITLED);
  });

  it("takes the first non-blank line, collapses runs of space, and shortens a long one", () => {
    expect(titleOf(withDraft(createDecision(1), asking("\n\n  which   team?  \nand more")))).toBe("which team?");
    const long = titleOf(withDraft(createDecision(1), asking("a".repeat(80))));
    expect(long).toHaveLength(40);
    expect(long.endsWith("…")).toBe(true);
  });
});

describe("counts", () => {
  it("counts the questions asked and the answers that are not errors", () => {
    const decision = {
      ...withDraft(createDecision(1), { ...EMPTY_DRAFT, questions: [newQuestion("noul", "a"), newQuestion("noul", "b")] }),
      answer: { run: ran({ a: { type: "noul", noul: 0.5 }, b: { type: "error", code: "x", message: "y" } }), draft: EMPTY_DRAFT },
    };
    expect(counts(decision)).toEqual({ asked: 2, answered: 1 });
  });

  it("counts nothing answered before a run", () => {
    expect(counts(withDraft(createDecision(1), asking("?")))).toEqual({ asked: 1, answered: 0 });
  });
});

describe("updateDecision", () => {
  it("changes one decision and leaves the list it was given alone", () => {
    const list = startList();
    const next = updateDecision(list, 1, (d) => ({ ...d, spare: { ...EMPTY_SPARE, text: "kept" } }));
    expect(next.decisions[0].spare.text).toBe("kept");
    expect(list.decisions[0].spare.text).toBe("");
  });
});
