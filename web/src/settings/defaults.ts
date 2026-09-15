import type { ReasoningEffort, Settings } from "../api/request.ts";

/** What the Settings panel sets: a request's settings, less the model ignis names. */
export type PlaygroundSettings = Omit<Settings, "model">;

export const DEFAULT_SETTINGS: PlaygroundSettings = {
  systemPrompt: "",
  temperature: 1,
  topP: 0.95,
  maxTokens: 16384,
  reasoningEffort: "xhigh",
  laneTag: "interactive",
};

export const EFFORT_LABELS: Record<ReasoningEffort, string> = { none: "Off", low: "Low", medium: "Medium", xhigh: "X-high" };
