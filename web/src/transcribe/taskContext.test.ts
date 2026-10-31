import { describe, expect, it } from "vitest";
import type { ConversationRequest } from "../api/request.ts";
import type { StreamOptions, StreamResult } from "../api/stream.ts";
import { Analyst } from "./analyst.ts";
import { ASK_TASK, Asker, askRequest } from "./ask.ts";
import { NotesWriter } from "./notesWriter.ts";
import { Rewriter, rewriteRequest } from "./rewrite.ts";
import { contextBlock, taskInput, writeIn } from "./taskContext.ts";
import { addUtterance, createTranscript, renameSpeaker, type Transcript } from "./transcript.ts";
import { EMPTY_CONTEXT, type CallContext } from "./context.ts";

// The context every text-model task is told (spec #428, ticket #440): in the
// user turn ahead of the transcript, the language to write in named, the
// system block untouched by it.

function call(language?: string[]): Transcript {
  let t = { ...createTranscript(1, { source: "call", at: new Date(2026, 9, 27, 15, 42) }), title: "Call" };
  (language ?? ["it", "it", "en"]).forEach((l, i) => {
    t = addUtterance(t, { stream: "s", startMs: i * 10_000, endMs: i * 10_000 + 4_000, text: `Frase ${i}.`, voice: i % 2 === 0 ? { kind: "me" } : { kind: "others" }, language: l });
  });
  return t;
}

const told = (over: Partial<CallContext>): CallContext => ({ ...EMPTY_CONTEXT, ...over });
const asContext = (t: Transcript, c: CallContext): Transcript => ({ ...t, context: c });

describe("the language to write in", () => {
  it("is the first under Force, the most heard otherwise, English when nothing is known", () => {
    const t = call();
    expect(writeIn(asContext(t, told({ languages: { mode: "force", codes: ["de"] } })))).toBe("German");
    expect(writeIn(asContext(t, told({ languages: { mode: "free", codes: ["de"] } })))).toBe("Italian");
    expect(writeIn(t)).toBe("Italian");
    const bare = { ...t, utterances: t.utterances.map(({ language: _, ...u }) => u) };
    expect(writeIn(bare)).toBe("English");
  });
});

describe("the context block", () => {
  it("says what was told, that Me is the user, the speakers by name and the hypotheticals", () => {
    const t = renameSpeaker(asContext(call(), told({ topic: "Support", terms: ["VPN", "Postgres"], other: "Internal call", languages: { mode: "free", codes: ["it", "en"] } })), 2, "Anna");
    const block = contextBlock(t);
    expect(block).toContain("- Write in: Italian.");
    expect(block).toContain("- Languages: Italian, English.");
    expect(block).toContain("- Topic: Support");
    expect(block).toContain("- Terms, spelled so: VPN, Postgres.");
    expect(block).toContain("- More: Internal call");
    expect(block).toContain('"Me" is the user');
    expect(block).toContain("Anna");
    expect(block).toContain("hypothetical or an example is not a fact");
  });

  it("says nothing was told for a transcript with no context, and still names the language", () => {
    const block = contextBlock(call());
    expect(block).toContain("Nothing else was told");
    expect(block).toContain("- Write in: Italian.");
  });

  it("goes ahead of the transcript, which the budget keeps whole when it fits", () => {
    const t = call();
    const { content } = taskInput(t, null, 1_500);
    expect(content.startsWith(contextBlock(t))).toBe(true);
    expect(content).toContain("Call\n\n[00:00] Me: Frase 0.");
  });
});

describe("every task's request", () => {
  const t = renameSpeaker(asContext(call(), told({ languages: { mode: "force", codes: ["fr"] }, terms: ["Zebrafish"] })), 2, "Anna");
  const block = contextBlock(t);

  it("carries the block ahead of the transcript, in the notes, the look, Ask and Rewrite", async () => {
    const seen: ConversationRequest[] = [];
    const stream = async (o: StreamOptions): Promise<StreamResult> => {
      seen.push(o.request);
      return { ok: true } as StreamResult;
    };
    await new Analyst({ stream, onFound: () => {} }).analyze(t, null);
    await new Asker({ stream, onAnswered: () => {} }).ask(t, "Why?", null);
    await new Rewriter({ stream }).rewrite(t, t.utterances[1]);
    await new NotesWriter({ stream, onNotes: () => {}, onRecap: () => {} }).write(t, "general", null);
    expect(seen).toHaveLength(4);
    for (const r of seen) {
      expect(r.turns[0].content.startsWith(block)).toBe(true);
      expect(r.turns[0].content).toContain("- Write in: French.");
      expect(r.settings.systemPrompt).not.toContain("French");
      expect(r.settings.systemPrompt).not.toContain("Zebrafish");
      expect(r.settings.systemPrompt).not.toContain("German");
    }
  });

  it("keeps a task's system block byte-identical whatever the context, so the engine reuses the prompt", async () => {
    const other = asContext(call(["en"]), told({ topic: "Else", terms: ["x"] }));
    /** The first request of a notes thread on `x`. */
    const threadOf = async (x: Transcript) => {
      let sent: ConversationRequest | undefined;
      const stream = async (o: StreamOptions): Promise<StreamResult> => {
        sent ??= o.request;
        return { ok: true } as StreamResult;
      };
      await new NotesWriter({ stream, onNotes: () => {}, onRecap: () => {} }).write(x, "work", null);
      return sent!;
    };
    expect(askRequest(t, "a", null).settings.systemPrompt).toBe(askRequest(other, "b", null).settings.systemPrompt);
    const [mine, theirs] = [await threadOf(t), await threadOf(other)];
    expect(mine.settings.systemPrompt).toBe(theirs.settings.systemPrompt);
    expect(mine.extras).toEqual(theirs.extras);
    expect(rewriteRequest(t, 1)!.settings.systemPrompt).toBe(rewriteRequest(other, 1)!.settings.systemPrompt);
    expect(askRequest(t, "a", null).settings.systemPrompt).toContain(ASK_TASK);
  });
});
