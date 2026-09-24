// The parts of the dev mock's /v1/chat/completions worth a test (mock.ts is
// the glue): the thinking budget as spec server/08 has ignis read it, the
// reasoning the mock streams under it, and where a forced close goes on the
// wire — the finishing choice of a stream, or of a non-streaming completion.
// Development only, like the rest of the mock.

/** The mock's `--thinking-budget`, the default ignis ships. */
export const MOCK_DEFAULT_THINKING_BUDGET = 8192;

const U32_MAX = 4_294_967_295;

/** What ignis writes into the reasoning when the budget closes it (thinking.rs), less the `</think>` the stream never shows. */
export const FORCED_CLOSE_TEXT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.";

export type MockBudget =
  | { ok: false; error: { message: string; type: "invalid_request_error"; param: "thinking_budget"; code: null } }
  | {
      ok: true;
      /** The request thinks at all. */
      thinking: boolean;
      /** The budget it runs with, or null for none. */
      budget: number | null;
      /** What the mock made of the field, echoed in the reasoning. */
      label: string;
    };

/**
 * A request's thinking budget, as server/08 resolves it: absent or null is
 * the server default, 0 is none, a count is that budget, `max` has none
 * whatever was sent, and with thinking off the field is inert. Anything but
 * a whole count from 0 to u32::MAX is a 400 naming the field.
 */
export function readMockBudget(body: Record<string, unknown>): MockBudget {
  const value = body.thinking_budget;
  const count = typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= U32_MAX;
  if (value !== undefined && value !== null && !count) {
    return {
      ok: false,
      error: {
        message: `\`thinking_budget\` must be a whole number of tokens, got ${JSON.stringify(value)}`,
        type: "invalid_request_error",
        param: "thinking_budget",
        code: null,
      },
    };
  }
  if (body.reasoning_effort === "none" || body.enable_thinking === false) {
    return { ok: true, thinking: false, budget: null, label: "inert (thinking off)" };
  }
  if (body.reasoning_effort === "max") return { ok: true, thinking: true, budget: null, label: "none (max)" };
  if (value === 0) return { ok: true, thinking: true, budget: null, label: "off" };
  if (typeof value === "number") return { ok: true, thinking: true, budget: value, label: String(value) };
  return { ok: true, thinking: true, budget: MOCK_DEFAULT_THINKING_BUDGET, label: `server default (${MOCK_DEFAULT_THINKING_BUDGET})` };
}

/**
 * The reasoning the mock streams, and — when the budget closed it — the
 * reasoning tokens spent. "/budget" in the prompt makes the mock think past
 * any budget it has, which it then spends whole, as ignis reports it.
 */
export function mockThinking(read: Extract<MockBudget, { ok: true }>, prompt: string): { reasoning: string[]; forcedAt?: number } {
  if (!read.thinking) return { reasoning: [] };
  if (!prompt.includes("/budget")) return { reasoning: ["Thinking ", "about ", "it. ", `(Thinking budget: ${read.label}.)`] };
  const long = Array.from({ length: 12 }, (_, i) => `Weighing option ${i + 1}, which needs a closer look. `);
  if (read.budget === null) return { reasoning: [...long, `(No budget to spend: ${read.label}.)`] };
  return { reasoning: [...long, FORCED_CLOSE_TEXT], forcedAt: read.budget };
}

/** What the finishing choice adds for a forced close: ignis's field, absent (not null) otherwise. */
export function finishFields(forcedAt: number | undefined): { thinking_budget_forced_at?: number } {
  return forcedAt === undefined ? {} : { thinking_budget_forced_at: forcedAt };
}

type Usage = { prompt_tokens: number; completion_tokens: number; total_tokens: number };

/** A `stream: false` answer: the whole reply as one `chat.completion`. */
export function completionBody(reply: {
  id: string;
  reasoning: string[];
  content: string[];
  calls: { id: string; tool: string; args: object }[];
  forcedAt?: number;
  usage: Usage;
}) {
  const reasoning = reply.reasoning.join("");
  return {
    id: reply.id,
    object: "chat.completion",
    model: "mock-model",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: reply.content.join(""),
          ...(reasoning ? { reasoning_content: reasoning } : {}),
          ...(reply.calls.length
            ? {
                tool_calls: reply.calls.map((c) => ({
                  id: c.id,
                  type: "function",
                  function: { name: c.tool, arguments: JSON.stringify(c.args) },
                })),
              }
            : {}),
        },
        finish_reason: reply.calls.length ? "tool_calls" : "stop",
        ...finishFields(reply.forcedAt),
      },
    ],
    usage: reply.usage,
  };
}
