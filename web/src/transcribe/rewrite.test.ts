import { describe, expect, it } from "vitest";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { ASSIST_SYSTEM } from "./assist.ts";
import { acceptRewrite, REWRITE_TASK, Rewriter, rewriteRequest, tidyRewrite, wordDiff } from "./rewrite.ts";
import { addUtterance, beginListening, createTranscript, type Transcript } from "./transcript.ts";

// Rewrite (spec speech/07, Q138): the text model rebuilds one misheard
// utterance from the context, the conversation around it and the misheard
// words; a proposal, the words it changes shown; accepted, the text as heard
// is kept, as a fix's.

function call(): Transcript {
  let t = beginListening(createTranscript(1), { source: "call", at: new Date(2026, 9, 27, 14, 30) });
  t = { ...t, context: { languages: { mode: "free", codes: ["it"] }, topic: "Supporto informatico", terms: ["VPN"], other: "", appliedAt: 1 } };
  const lines: [string, string][] = [
    ["others", "Il portatile non si collega alla stampante."],
    ["me", "Mi collego io da remoto."],
    ["me", "aspetta che me riattacco a Postgress"],
    ["others", "Vedi lo schermo adesso?"],
  ];
  lines.forEach(([who, text], i) => (t = addUtterance(t, { stream: "s", startMs: i * 5_000, endMs: i * 5_000 + 3_000, text, voice: who === "me" ? { kind: "me" } : { kind: "others" } })));
  return t;
}

describe("a rewrite's request", () => {
  it("tells the task, the context, the conversation around the utterance and the utterance last, thinking a little, on the interactive lane", () => {
    const request = rewriteRequest(call(), 3)!;
    expect(request.settings).toMatchObject({ systemPrompt: `${ASSIST_SYSTEM}\n\n${REWRITE_TASK}`, reasoningEffort: "low", laneTag: "interactive" });
    const content = request.turns[0].content;
    expect(content.startsWith("Context of the conversation, told by the user:\n- Write in: Italian.")).toBe(true);
    expect(content).toContain("- Topic: Supporto informatico\n- Terms, spelled so: VPN.");
    expect(content).toContain("\n\nTranscript, around the utterance:\n[00:00] Others: Il portatile");
    expect(content).toContain("[00:15] Others: Vedi lo schermo adesso?");
    expect(content.endsWith("Misheard utterance: [00:10] Me: aspetta che me riattacco a Postgress")).toBe(true);
    expect(rewriteRequest(call(), 99)).toBeNull();
  });
});

describe("a rewrite's answer", () => {
  it("is the text alone: no label, time, speaker or quotes the model may have added", () => {
    expect(tidyRewrite('Corrected utterance: [00:10] Me: "aspetta che me riattacco a PostgreSQL"', "Me")).toBe("aspetta che me riattacco a PostgreSQL");
    expect(tidyRewrite("  aspetta che me riattacco a PostgreSQL  ", null)).toBe("aspetta che me riattacco a PostgreSQL");
  });

  it("is shown as the words it keeps, adds and drops", () => {
    expect(wordDiff("aspetta che me riattacco a Postgress", "aspetta che mi riattacco a PostgreSQL")).toEqual([
      { text: "aspetta", kind: "same" },
      { text: "che", kind: "same" },
      { text: "mi", kind: "added" },
      { text: "me", kind: "removed" },
      { text: "riattacco", kind: "same" },
      { text: "a", kind: "same" },
      { text: "PostgreSQL", kind: "added" },
      { text: "Postgress", kind: "removed" },
    ]);
  });

  it("accepted, is the utterance's text, the text as heard kept and the words it put in marked", () => {
    const t = acceptRewrite(call(), 3, "aspetta che me riattacco a PostgreSQL.");
    expect(t.utterances[2]).toMatchObject({ text: "aspetta che me riattacco a PostgreSQL.", heard: "aspetta che me riattacco a Postgress", fixed: ["PostgreSQL"] });
    expect(acceptRewrite(t, 2, t.utterances[1].text)).toEqual(t);
  });
});

describe("the rewriter", () => {
  const answering = (text: string | Error) => async (options: StreamOptions): Promise<StreamResult> => {
    const timeline = { sentAt: 0, stopped: false };
    if (text instanceof Error) return { ok: false, message: text.message, timeline };
    options.onEvent({ kind: "content", text });
    return { ok: true, timeline };
  };

  it("proposes the model's text for the utterance asked, or says why it could not", async () => {
    const t = call();
    const r = new Rewriter({ stream: answering("Me: aspetta che me riattacco a PostgreSQL") });
    await r.rewrite(t, t.utterances[2]);
    expect(r.getState()).toEqual({ transcript: 1, utterance: 3, writing: false, proposal: "aspetta che me riattacco a PostgreSQL", failed: null });
    r.close();
    expect(r.getState().utterance).toBeNull();
    const failing = new Rewriter({ stream: answering(new Error("ignis answered 500")) });
    await failing.rewrite(t, t.utterances[2]);
    expect(failing.getState()).toMatchObject({ writing: false, proposal: null, failed: "ignis answered 500" });
  });
});
