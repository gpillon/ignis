import type { ToolDefinition } from "../../api/request.ts";
import type { ToolCall } from "../../api/sse.ts";

// The `ask_user` tool: the model stops to ask the user a question, with
// optional answers to pick from; the turn waits, and the answer goes back as
// the call's result. Only the conversation asks — agents never get this tool.

export const ASK_USER_TOOL_NAME = "ask_user";

export const ASK_USER_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: ASK_USER_TOOL_NAME,
    description:
      "Ask the user a question and wait for the answer. Use it whenever the user's intent, preferences or a missing detail would change what you do.",
    parameters: {
      type: "object",
      properties: {
        question: { type: "string", description: "One clear question, as the user will read it." },
        options: {
          type: "array",
          items: { type: "string" },
          description: "Optional short answers the user can pick; they can always write their own instead.",
        },
      },
      required: ["question"],
    },
  },
};

/** What the ask tool adds to the ignis system prompt of the conversation. */
export const ASK_USER_IGNIS_PROMPT = `# Asking the user
You can ask the user questions with the \`ask_user\` tool, and you should: a quick question beats a long answer built on a wrong guess.
- Ask before you start when the request is ambiguous or underspecified, when the answer depends on the user's preferences, situation or goals, or when there are several reasonable ways to go and the choice is theirs.
- Ask before long or costly work — many agents, many searches — whenever a detail would change what you do.
- Ask one clear question per call; several calls in the same reply ask several questions at once. Pass \`options\` when the likely answers are few; the user can always write their own.
- The turn waits for the answers, which come back as the calls' results.
- Do not ask for facts you can find out with your other tools.`;

export type QuestionStatus = "waiting" | "answered" | "skipped" | "invalid";

/** One question as the Playground shows it and the model is told about it. */
export type Question = {
  callId: string;
  question: string;
  options: string[];
  status: QuestionStatus;
  answer?: string;
  /** Why an invalid call was not asked. */
  error?: string;
};

/** A call the model made, read as a question waiting for the user, or as an invalid one. */
export function parseAskCall(call: ToolCall): Question {
  const invalid = (error: string): Question => ({ callId: call.id, question: call.arguments, options: [], status: "invalid", error });
  let args: unknown;
  try {
    args = JSON.parse(call.arguments);
  } catch {
    return invalid("The call's arguments are not valid JSON.");
  }
  const { question, options } = (typeof args === "object" && args !== null ? args : {}) as { question?: unknown; options?: unknown };
  if (typeof question !== "string" || question.trim() === "") return invalid('The call needs a non-empty "question".');
  const choices = Array.isArray(options)
    ? options.filter((o): o is string => typeof o === "string" && o.trim() !== "").map((o) => o.trim())
    : [];
  return { callId: call.id, question: question.trim(), options: choices, status: "waiting" };
}

/** The tool result the model receives for a question once it is settled. */
export function askToolResult(question: Question): string {
  if (question.status === "answered") return `The user answered: ${question.answer ?? ""}`;
  if (question.status === "invalid") return `The question was not asked: ${question.error ?? "invalid call"}`;
  return "The user did not answer: the turn was stopped.";
}
