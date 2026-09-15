import { describe, expect, it } from "vitest";
import { AGENT_TOOL, AGENTS_IGNIS_PROMPT } from "./agents/agents.ts";
import { ASK_USER_IGNIS_PROMPT, ASK_USER_TOOL } from "./ask/ask.ts";
import { dateTimePrompt } from "./datetime/datetime.ts";
import { agentExtras, ALL_TOOLS, ignisPrompt, NO_TOOLS, routeCall, setAllTools, toolExtras } from "./index.ts";
import { WEB_FETCH_TOOL, WEB_IGNIS_PROMPT, WEB_SEARCH_TOOL } from "./web/web.ts";

const now = new Date("2026-09-15T01:52:00+02:00");
const all = { enabled: true, agents: true, web: true, askUser: true, dateTime: true };

describe("tools", () => {
  it("adds nothing while no tool is on", () => {
    expect(ignisPrompt(NO_TOOLS, now)).toBe("");
    expect(toolExtras(NO_TOOLS, now)).toEqual({ ignisPrompt: "", tools: [] });
  });

  it("declares agents and adds their part of the ignis prompt", () => {
    expect(toolExtras({ ...NO_TOOLS, agents: true }, now)).toEqual({ ignisPrompt: AGENTS_IGNIS_PROMPT, tools: [AGENT_TOOL] });
  });

  it("writes the date and time into the prompt without declaring a tool", () => {
    expect(toolExtras({ ...NO_TOOLS, dateTime: true }, now)).toEqual({ ignisPrompt: dateTimePrompt(now), tools: [] });
  });

  it("puts every tool in a fixed order, the date first", () => {
    expect(toolExtras(all, now)).toEqual({
      ignisPrompt: [dateTimePrompt(now), AGENTS_IGNIS_PROMPT, WEB_IGNIS_PROMPT, ASK_USER_IGNIS_PROMPT].join("\n\n"),
      tools: [AGENT_TOOL, WEB_SEARCH_TOOL, WEB_FETCH_TOOL, ASK_USER_TOOL],
    });
  });

  it("declares nothing while the switch for all tools is off", () => {
    expect(toolExtras({ ...all, enabled: false }, now)).toEqual({ ignisPrompt: "", tools: [] });
    expect(agentExtras({ ...all, enabled: false }, now)).toEqual({ ignisPrompt: "", tools: [] });
  });

  it("gives agents every tool but agent and ask_user", () => {
    expect(agentExtras(all, now)).toEqual({
      ignisPrompt: `${dateTimePrompt(now)}\n\n${WEB_IGNIS_PROMPT}`,
      tools: [WEB_SEARCH_TOOL, WEB_FETCH_TOOL],
    });
    expect(agentExtras({ ...NO_TOOLS, agents: true, askUser: true }, now)).toEqual({ ignisPrompt: "", tools: [] });
  });

  it("keeps each tool's choice under the switch for all, and turns all on when none was chosen", () => {
    const some = { ...NO_TOOLS, web: true };
    expect(setAllTools(some, false)).toEqual({ ...some, enabled: false });
    expect(setAllTools({ ...some, enabled: false }, true)).toEqual(some);
    expect(setAllTools({ ...NO_TOOLS, enabled: false }, true)).toEqual(all);
    expect(ALL_TOOLS).toEqual(all);
  });

  it("routes declared calls to their runner, and any undeclared tool to unknown", () => {
    const every = ["agent", "web_search", "web_fetch", "ask_user"];
    expect(routeCall("web_search", every)).toBe("web");
    expect(routeCall("agent", every)).toBe("agent");
    expect(routeCall("ask_user", every)).toBe("ask");
    expect(routeCall("read_file", every)).toBe("unknown");
    expect(routeCall("web_search", ["agent"])).toBe("unknown");
    expect(routeCall("ask_user", ["web_search", "web_fetch"])).toBe("unknown");
    expect(routeCall("agent", [])).toBe("unknown");
  });
});
