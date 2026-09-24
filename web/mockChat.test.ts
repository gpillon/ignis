import { describe, expect, it } from "vitest";
import { completionBody, FORCED_CLOSE_TEXT, finishFields, MOCK_DEFAULT_THINKING_BUDGET, mockThinking, readMockBudget } from "./mockChat.ts";

// The dev mock's thinking budget (spec playground/04): it takes the field as
// server/08 does, echoes what it made of it, and "/budget" in a prompt makes
// it think past the budget so the Playground's forced-close display can be
// worked on without the GPU.

describe("readMockBudget", () => {
  it("falls back to the mock's server default when the field is absent or null", () => {
    for (const body of [{}, { thinking_budget: null }]) {
      expect(readMockBudget(body)).toMatchObject({ ok: true, thinking: true, budget: MOCK_DEFAULT_THINKING_BUDGET });
    }
  });

  it("runs a count as that budget and 0 as none", () => {
    expect(readMockBudget({ thinking_budget: 4096 })).toMatchObject({ ok: true, budget: 4096, label: "4096" });
    expect(readMockBudget({ thinking_budget: 0 })).toMatchObject({ ok: true, budget: null, label: "off" });
  });

  it("runs max with no budget, whatever the request sent", () => {
    for (const thinking_budget of [undefined, 0, 4096]) {
      expect(readMockBudget({ reasoning_effort: "max", thinking_budget })).toMatchObject({ ok: true, budget: null, label: "none (max)" });
    }
  });

  it("leaves the budget inert when thinking is off", () => {
    expect(readMockBudget({ reasoning_effort: "none", thinking_budget: 4096 })).toMatchObject({ ok: true, thinking: false, budget: null });
    expect(readMockBudget({ enable_thinking: false })).toMatchObject({ ok: true, thinking: false });
  });

  it("refuses what ignis refuses, naming the field", () => {
    for (const thinking_budget of [-1, 1.5, "4096", true, {}, 4294967296]) {
      const read = readMockBudget({ thinking_budget });
      expect(read.ok).toBe(false);
      expect(!read.ok && read.error.param).toBe("thinking_budget");
    }
  });
});

describe("mockThinking", () => {
  const read = (body: Record<string, unknown>) => {
    const budget = readMockBudget(body);
    if (!budget.ok) throw new Error("refused");
    return budget;
  };

  it("echoes the budget it ran with in its reasoning, and forces nothing", () => {
    const thinking = mockThinking(read({ thinking_budget: 2048 }), "hi");
    expect(thinking.reasoning.join("")).toContain("Thinking budget: 2048.");
    expect(thinking.forcedAt).toBeUndefined();
  });

  it("thinks past its budget on /budget, and closes with the budget spent", () => {
    const thinking = mockThinking(read({ thinking_budget: 2048 }), "hard one /budget");
    expect(thinking.forcedAt).toBe(2048);
    expect(thinking.reasoning.at(-1)).toBe(FORCED_CLOSE_TEXT);
  });

  it("is never forced when there is no budget to spend", () => {
    for (const body of [{ thinking_budget: 0 }, { reasoning_effort: "max", thinking_budget: 2048 }]) {
      expect(mockThinking(read(body), "/budget").forcedAt).toBeUndefined();
    }
    expect(mockThinking(read({ reasoning_effort: "none" }), "/budget")).toEqual({ reasoning: [] });
  });
});

describe("the forced close on the wire", () => {
  it("rides the finishing choice only when the close was forced", () => {
    expect(finishFields(4096)).toEqual({ thinking_budget_forced_at: 4096 });
    expect(finishFields(undefined)).toEqual({});
  });

  it("is on the choice of a non-streaming completion, beside finish_reason", () => {
    const forced = completionBody({ id: "c", reasoning: ["hm"], content: ["Hi"], calls: [], forcedAt: 4096, usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 } });
    expect(forced.choices[0]).toEqual({
      index: 0,
      message: { role: "assistant", content: "Hi", reasoning_content: "hm" },
      finish_reason: "stop",
      thinking_budget_forced_at: 4096,
    });
    const free = completionBody({ id: "c", reasoning: [], content: ["Hi"], calls: [], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } });
    expect("thinking_budget_forced_at" in free.choices[0]).toBe(false);
  });

  it("carries tool calls whole, finishing on tool_calls", () => {
    const body = completionBody({ id: "c", reasoning: [], content: [], calls: [{ id: "call_0", tool: "agent", args: { prompt: "x" } }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } });
    expect(body.choices[0].finish_reason).toBe("tool_calls");
    expect(body.choices[0].message.tool_calls).toEqual([{ id: "call_0", type: "function", function: { name: "agent", arguments: '{"prompt":"x"}' } }]);
  });
});
