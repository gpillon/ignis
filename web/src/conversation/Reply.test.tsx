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
