import type { ToolCall } from "../api/sse.ts";

/** The error a call to a tool the request did not declare gets back. */
export function unknownToolError(name: string, available: string[]): string {
  if (available.length === 0) return `Unknown tool "${name}": no tools are available.`;
  return `Unknown tool "${name}": the tools available are ${available.map((n) => `"${n}"`).join(", ")}.`;
}

/** A call to a tool the request did not declare: never run, answered with `error`. */
export type UnknownCall = { callId: string; name: string; arguments: string; error: string };

export function unknownCall(call: ToolCall, available: string[]): UnknownCall {
  return { callId: call.id, name: call.name, arguments: call.arguments, error: unknownToolError(call.name, available) };
}
