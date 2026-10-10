import { type ConversationRequest, type ReasoningEffort, systemPromptOf, thinkingBudgetOf, type Turn } from "./request.ts";
import type { ChunkEvent } from "./sse.ts";

// The Playground's request as a Responses `response.create`, and the
// Responses events back as the chunk events the turn loop already reads
// (GitHub #283, spec 01 for the wire). The page writes the same conversation
// it would send as chat completions: the system prompt becomes
// `instructions`, each turn one or more input items, the tools the flat
// function shape, and the settings the same fields — the ignis extensions
// under the names chat completions gives them.
//
// One difference: an assistant turn's reasoning goes back as a `reasoning`
// item. ignis keeps each stream's latest response with its reasoning to
// continue from, so a full history has to carry it too, or a continued turn
// and a resent one would not read the same prompt. The template decides what
// of it the model sees.

export type InputContent =
  | { type: "input_text"; text: string }
  | { type: "input_image"; image_url: string }
  | { type: "output_text"; text: string };

export type InputItem =
  | { type: "message"; role: "user" | "developer" | "assistant"; content: InputContent[] }
  | { type: "function_call"; call_id: string; name: string; arguments: string }
  | { type: "function_call_output"; call_id: string; output: string }
  | { type: "reasoning"; content: { type: "reasoning_text"; text: string }[]; summary: [] };

/** A function tool in the Responses flat shape. */
export type ResponseTool = { type: "function"; name: string; description: string; parameters: object };

export type ResponseCreate = {
  type: "response.create";
  stream_id: string;
  instructions?: string;
  input: InputItem[];
  previous_response_id?: string;
  tools?: ResponseTool[];
  temperature: number;
  top_p: number;
  max_output_tokens?: number;
  reasoning_effort: ReasoningEffort;
  thinking_budget?: number;
  class: ConversationRequest["settings"]["laneTag"];
};

const text = (text: string): InputContent[] => [{ type: "input_text", text }];

/** One turn's items, in the order a reply is generated: reasoning, text, calls. */
function turnItems(turn: Turn): InputItem[] {
  const moment: InputItem[] = turn.dateTime ? [{ type: "message", role: "developer", content: text(turn.dateTime) }] : [];
  if (turn.role === "tool") return [...moment, { type: "function_call_output", call_id: turn.toolCallId ?? "", output: turn.content }];
  if (turn.role === "user") {
    // Images first and the text behind them, as the chat request orders its parts; no words, no text part.
    const images = (turn.images ?? []).map((image): InputContent => ({ type: "input_image", image_url: image.url }));
    const content = images.length > 0 && turn.content === "" ? images : [...images, ...text(turn.content)];
    return [...moment, { type: "message", role: "user", content }];
  }
  return [
    ...moment,
    ...(turn.reasoning ? [{ type: "reasoning" as const, content: [{ type: "reasoning_text" as const, text: turn.reasoning }], summary: [] as [] }] : []),
    ...(turn.content ? [{ type: "message" as const, role: "assistant" as const, content: [{ type: "output_text" as const, text: turn.content }] }] : []),
    ...(turn.toolCalls ?? []).map((c): InputItem => ({ type: "function_call", call_id: c.id, name: c.name, arguments: c.arguments })),
  ];
}

/** The conversation as input items: every turn's, in order. */
export function responseItems(turns: Turn[]): InputItem[] {
  return turns.flatMap(turnItems);
}

/**
 * What a response added to its stream, as ignis keeps it to continue from:
 * its output items in their order, in the shape the page sends them. A reply
 * with one message is exactly the items its turn becomes; one whose text
 * continues after a call is not (the transcript joins the text into one
 * turn), so the next request carries the whole history rather than continue.
 */
export function outputItems(response: ResponseObject): InputItem[] {
  return (response.output ?? []).flatMap((item): InputItem[] => {
    const texts = (item.content ?? []).map((part) => ({ text: part.text ?? "" }));
    if (item.type === "reasoning") return [{ type: "reasoning", content: texts.map(({ text }) => ({ type: "reasoning_text", text })), summary: [] }];
    if (item.type === "message") return [{ type: "message", role: "assistant", content: texts.map(({ text }) => ({ type: "output_text", text })) }];
    if (item.type === "function_call") return [{ type: "function_call", call_id: item.call_id ?? "", name: item.name ?? "", arguments: item.arguments ?? "" }];
    return [];
  });
}

/**
 * The `response.create` for `request` on `streamId`, carrying `input`: the
 * whole history, or only what is new after `previousResponseId`.
 * `instructions` and `tools` go on every request: a response never inherits
 * them from the one it continues.
 */
export function buildResponseCreate(
  request: ConversationRequest,
  streamId: string,
  input: InputItem[],
  previousResponseId?: string,
): ResponseCreate {
  const { settings, extras = {} } = request;
  const instructions = systemPromptOf(settings, extras);
  const thinkingBudget = thinkingBudgetOf(settings);
  return {
    type: "response.create",
    stream_id: streamId,
    ...(instructions ? { instructions } : {}),
    input,
    ...(previousResponseId !== undefined ? { previous_response_id: previousResponseId } : {}),
    ...(extras.tools?.length
      ? { tools: extras.tools.map((t): ResponseTool => ({ type: "function", name: t.function.name, description: t.function.description, parameters: t.function.parameters })) }
      : {}),
    temperature: settings.temperature,
    top_p: settings.topP,
    ...(settings.maxTokens !== null ? { max_output_tokens: settings.maxTokens } : {}),
    reasoning_effort: settings.reasoningEffort,
    ...(thinkingBudget !== undefined ? { thinking_budget: thinkingBudget } : {}),
    class: settings.laneTag,
  };
}

/** An item of a response's output: only what the page reads of it. */
export type OutputItem = { type: string; content?: { type?: string; text?: string }[]; call_id?: string; name?: string; arguments?: string };

/** The response object the lifecycle events carry: only what the page reads of it. */
export type ResponseObject = {
  id: string;
  status?: string;
  output?: OutputItem[];
  usage?: { input_tokens: number; output_tokens: number; total_tokens: number } | null;
  incomplete_details?: { reason?: string } | null;
  error?: { code?: string | null; message?: string } | null;
  /** ignis's forced-close extension (spec server/08), as on chat completions. */
  thinking_budget_forced_at?: unknown;
};

/** An event ignis sends on the socket, as far as the page reads it. */
export type ServerEvent = {
  type: string;
  stream_id?: string;
  response?: ResponseObject;
  delta?: string;
  item?: { type: string; id?: string; call_id?: string; name?: string; arguments?: string; status?: string };
  /** On an `error` event: the HTTP status the same failure has over HTTP, and OpenAI's error body. */
  status?: number;
  error?: { type?: string; code?: string | null; message?: string; param?: string | null };
};

/** The events that carry generated tokens, for the timeline: a call's arguments count as output. */
export const TOKEN_EVENTS: ReadonlySet<string> = new Set([
  "response.output_text.delta",
  "response.reasoning_text.delta",
  "response.function_call_arguments.delta",
]);

/**
 * What one Responses event means to the turn loop, in the chunk events a
 * chat stream would have produced: text and reasoning deltas, each tool
 * call whole once its item is done, and at the end the finish reason, the
 * usage and the end. A cancelled response has no finish: the page stopped it.
 */
export function parseResponseEvent(event: ServerEvent): ChunkEvent[] {
  switch (event.type) {
    case "response.reasoning_text.delta":
      return event.delta ? [{ kind: "reasoning", text: event.delta }] : [];
    case "response.output_text.delta":
      return event.delta ? [{ kind: "content", text: event.delta }] : [];
    case "response.output_item.done": {
      const item = event.item;
      if (item?.type !== "function_call" || !item.call_id || !item.name) return [];
      return [{ kind: "tool_call", call: { id: item.call_id, name: item.name, arguments: item.arguments ?? "{}" } }];
    }
    case "response.completed":
    case "response.incomplete": {
      const response = event.response;
      if (!response || response.status === "cancelled") return [];
      const reason =
        event.type === "response.completed"
          ? (response.output ?? []).some((item) => item.type === "function_call")
            ? "tool_calls"
            : "stop"
          : response.incomplete_details?.reason === "max_output_tokens" || !response.incomplete_details?.reason
            ? "length"
            : response.incomplete_details.reason;
      const forcedAt = response.thinking_budget_forced_at;
      const forced = typeof forcedAt === "number" && Number.isInteger(forcedAt) && forcedAt >= 0;
      const usage = response.usage;
      return [
        { kind: "finish", reason, ...(forced ? { thinkingForcedAt: forcedAt } : {}) },
        ...(usage
          ? [{ kind: "usage" as const, usage: { prompt_tokens: usage.input_tokens, completion_tokens: usage.output_tokens, total_tokens: usage.total_tokens } }]
          : []),
        { kind: "done" },
      ];
    }
    default:
      return [];
  }
}
