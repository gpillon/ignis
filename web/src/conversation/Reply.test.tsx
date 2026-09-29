import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Figures } from "../metrics/figures.ts";
import type { Message } from "../sessions/sessions.ts";
import { Reply } from "./Reply.tsx";

// A reply whose reasoning the thinking budget closed says so on its
// reasoning block (spec playground/04); one that closed it itself does not.

const figures: Figures = {
  ttftMs: 100,
  decodeTokensPerSec: 50,
  durationMs: 1000,
  promptTokens: 10,
  completionTokens: 20,
  finishReason: "stop",
  partial: false,
};

const reply = (change: Partial<Message> = {}): Message => ({
  id: 1,
  role: "assistant",
  content: "The answer.",
  reasoning: "Thinking about it.",
  streaming: false,
  figures,
  ...change,
});

const render = (message: Message) =>
  renderToStaticMarkup(
    <Reply
      message={message}
      markdown={false}
      last
      actions={{ canRerun: true, onSave: () => {}, onFork: () => {} }}
      onRegenerate={() => {}}
      openCallId={null}
      onOpenAgent={() => {}}
      onAnswer={() => {}}
    />,
  );

/** The reasoning block's summary line, where the marker sits. */
const summary = (html: string) => /<summary[^>]*>(.*?)<\/summary>/s.exec(html)?.[1] ?? "";

describe("Reply and the thinking budget", () => {
  it("marks the reasoning block when the budget forced the close, with where it did", () => {
    const html = render(reply({ figures: { ...figures, thinkingForcedAt: 4096 } }));
    expect(summary(html)).toContain("Budget reached");
    expect(html).toContain("4096 reasoning tokens");
  });

  it("carries no marker when the reply closed its reasoning itself", () => {
    expect(render(reply())).not.toContain("Budget reached");
  });

  it("carries no marker while the reply streams, before it has any figures", () => {
    expect(render(reply({ streaming: true, content: "", figures: undefined }))).not.toContain("Budget reached");
  });
});

// A reply the engine queued says so while it waits, and its figures keep the
// wait apart from TTFT (GitHub #283).

describe("Reply and the engine's queue", () => {
  it("says the reply is queued while it waits for a lane", () => {
    expect(render(reply({ streaming: true, queued: true, content: "", reasoning: "", figures: undefined }))).toContain("Queued: the engine is full");
    expect(render(reply({ streaming: true, content: "", reasoning: "", figures: undefined }))).not.toContain("Queued");
  });

  it("shows the queue time beside TTFT once the reply is done, and nothing for a reply admitted at once", () => {
    const html = render(reply({ figures: { ...figures, queueMs: 1200 } }));
    expect(html).toMatch(/Queued<\/dt><dd[^>]*>1.20 s<\/dd>/);
    expect(html.indexOf("Queued")).toBeLessThan(html.indexOf("TTFT"));
    expect(render(reply())).not.toContain("Queued");
  });
});
