// The chat request the Playground sends (GitHub #164): ignis's own
// `POST /v1/chat/completions`, streaming with the usage chunk, plus the
// thinking effort and the ignis lane tag extension `class`. With tools on,
// it also declares them, and the system message is the ignis prompt (what
// the tools add, read-only) followed by the owner's own.

import type { PromptImage } from "../conversation/images.ts";
import type { ToolCall } from "./sse.ts";

/** Where every chat request goes; the connection budget reads these back off the timeline. */
export const CHAT_PATH = "/v1/chat/completions";

export type LaneTag = "interactive" | "agent";

/**
 * The efforts the loaded Qwen 3.8 template accepts (`high` is refused).
 * `none` turns thinking off, so the request never also sends
 * `enable_thinking` — ignis rejects the two when they disagree.
 */
export const REASONING_EFFORTS = ["none", "low", "medium", "xhigh"] as const;
export type ReasoningEffort = (typeof REASONING_EFFORTS)[number];

export type Settings = {
  model: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  /** `null` leaves the cap to the engine. */
  maxTokens: number | null;
  reasoningEffort: ReasoningEffort;
  laneTag: LaneTag;
};

/** A turn of the conversation: an assistant turn may call tools, a tool turn answers one call. */
export type Turn = {
  role: "user" | "assistant" | "tool";
  content: string;
  /** Images sent with this turn; they make its content a list of parts. */
  images?: PromptImage[];
  toolCalls?: ToolCall[];
  toolCallId?: string;
  /**
   * What the date and time tool wrote for the moment this turn was sent, when
   * it updates every prompt: a developer message goes in ahead of the turn.
   * Each turn keeps the one it was sent with, so a later request repeats them
   * unchanged and only appends — which is what leaves the engine a prefix to
   * reuse. Rewriting them to the present moment would cost the whole prefill.
   */
  dateTime?: string;
};

/** An OpenAI function tool, as `tools[]` declares it. */
export type ToolDefinition = {
  type: "function";
  function: { name: string; description: string; parameters: object };
};

/**
 * A message's content when it carries images (GitHub #174): the OpenAI
 * content-part list ignis's vision path reads. The images come first and the
 * text after, the order the reference's own requests use.
 */
export type ContentPart = { type: "text"; text: string } | { type: "image_url"; image_url: { url: string } };

type WireMessage = {
  role: "system" | "developer" | "user" | "assistant" | "tool";
  /** A plain string unless the turn carries images: a turn without them must tokenize exactly as it always did. */
  content: string | ContentPart[];
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
  tool_call_id?: string;
};

export type ChatRequest = {
  model: string;
  messages: WireMessage[];
  tools?: ToolDefinition[];
  stream: true;
  stream_options: { include_usage: true };
  temperature: number;
  top_p: number;
  max_tokens?: number;
  reasoning_effort: ReasoningEffort;
  class: LaneTag;
};

/** One message already in the Playground's conversation. */
export type Exchange = Turn & { failed: boolean };

/**
 * What of the conversation goes back to the engine: every exchange except
 * the ones that produced nothing — a prompt whose reply failed or came back
 * empty is dropped together with that reply, so the history never holds two
 * user turns in a row. A reply stopped part-way keeps the text it got.
 * Reasoning is never resent.
 */
export function conversationTurns(entries: Exchange[]): Turn[] {
  const turns: Turn[] = [];
  // A reply that only called tools is not empty: its calls are the turn.
  const empty = (e: Exchange | undefined) =>
    e !== undefined && e.role === "assistant" && (e.failed || (!e.content && !e.toolCalls?.length));
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    if (entry.role === "user" && empty(entries[i + 1])) {
      i++;
      continue;
    }
    if (empty(entry)) continue;
    turns.push({
      role: entry.role,
      content: entry.content,
      ...(entry.images?.length ? { images: entry.images } : {}),
      ...(entry.toolCalls?.length ? { toolCalls: entry.toolCalls } : {}),
      ...(entry.toolCallId ? { toolCallId: entry.toolCallId } : {}),
      ...(entry.dateTime ? { dateTime: entry.dateTime } : {}),
    });
  }
  return turns;
}

/**
 * A turn's content: its text, or — when it carries images — the part list,
 * images first and the text behind them. A prompt sent with an image and no
 * words of its own contributes no text part at all.
 */
function wireContent(turn: Turn): string | ContentPart[] {
  if (!turn.images?.length) return turn.content;
  const images = turn.images.map((image): ContentPart => ({ type: "image_url", image_url: { url: image.url } }));
  return turn.content === "" ? images : [...images, { type: "text", text: turn.content }];
}

/** A turn on the wire, behind the developer message carrying its moment when it has one. */
function wireMessages(turn: Turn): WireMessage[] {
  const message: WireMessage = {
    role: turn.role,
    content: wireContent(turn),
    ...(turn.toolCalls?.length
      ? {
          tool_calls: turn.toolCalls.map((c) => ({
            id: c.id,
            type: "function" as const,
            function: { name: c.name, arguments: c.arguments },
          })),
        }
      : {}),
    ...(turn.toolCallId ? { tool_call_id: turn.toolCallId } : {}),
  };
  return turn.dateTime ? [{ role: "developer", content: turn.dateTime }, message] : [message];
}

/** What the enabled tools add to a request: their prompt, ahead of the owner's, and their definitions. */
export type ToolExtras = { ignisPrompt?: string; tools?: ToolDefinition[] };

export function buildChatRequest(settings: Settings, turns: Turn[], extras: ToolExtras = {}): ChatRequest {
  const system = [extras.ignisPrompt ?? "", settings.systemPrompt].filter((part) => part.trim() !== "").join("\n\n");
  return {
    model: settings.model,
    messages: [...(system ? [{ role: "system" as const, content: system }] : []), ...turns.flatMap(wireMessages)],
    ...(extras.tools?.length ? { tools: extras.tools } : {}),
    stream: true,
    stream_options: { include_usage: true },
    temperature: settings.temperature,
    top_p: settings.topP,
    ...(settings.maxTokens !== null ? { max_tokens: settings.maxTokens } : {}),
    reasoning_effort: settings.reasoningEffort,
    class: settings.laneTag,
  };
}
