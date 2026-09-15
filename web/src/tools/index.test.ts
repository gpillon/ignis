import { describe, expect, it } from "vitest";
import { AGENT_TOOL, AGENTS_IGNIS_PROMPT } from "./agents/agents.ts";
import { ignisPrompt, NO_TOOLS, routeCall, toolExtras } from "./index.ts";
import { WEB_FETCH_TOOL, WEB_IGNIS_PROMPT, WEB_SEARCH_TOOL } from "./web/web.ts";

describe("tools", () => {
  it("adds nothing while no tool is on", () => {
    expect(ignisPrompt(NO_TOOLS)).toBe("");
    expect(toolExtras(NO_TOOLS)).toEqual({ ignisPrompt: "", tools: [] });
  });

  it("declares agents and adds their part of the ignis prompt", () => {
    expect(toolExtras({ agents: true, web: false })).toEqual({ ignisPrompt: AGENTS_IGNIS_PROMPT, tools: [AGENT_TOOL] });
  });

  it("declares search and fetch for web, after agents", () => {
    expect(toolExtras({ agents: true, web: true })).toEqual({
      ignisPrompt: `${AGENTS_IGNIS_PROMPT}\n\n${WEB_IGNIS_PROMPT}`,
      tools: [AGENT_TOOL, WEB_SEARCH_TOOL, WEB_FETCH_TOOL],
    });
  });

  it("routes declared web calls to web, the rest to agents while they are on", () => {
    const both = ["agent", "web_search", "web_fetch"];
    expect(routeCall("web_search", both)).toBe("web");
    expect(routeCall("agent", both)).toBe("agent");
    expect(routeCall("read_file", both)).toBe("agent");
    expect(routeCall("web_fetch", ["agent"])).toBe("agent");
    expect(routeCall("agent", ["web_search", "web_fetch"])).toBe("web");
  });
});
