import { describe, expect, it } from "vitest";
import { AGENT_TOOL, AGENTS_IGNIS_PROMPT } from "./agents/agents.ts";
import { ASK_USER_IGNIS_PROMPT, ASK_USER_TOOL } from "./ask/ask.ts";
import { dateTimePrompt } from "./datetime/datetime.ts";
import { agentExtras, ALL_TOOLS, ignisPrompt, NO_TOOLS, routeCall, setAllTools, toolExtras, toolsInUse, turnDateTime } from "./index.ts";
import type { Attachment } from "./local/attachments.ts";
import {
  attachmentsPrompt,
  CREATE_FILE_TOOL,
  FILES_PROMPT,
  MEMORY_DELETE_TOOL,
  MEMORY_READ_TOOL,
  MEMORY_SAVE_TOOL,
  memoryPrompt,
  PLAN_PROMPT,
  READ_FILE_TOOL,
  RUN_JS_TOOL,
  runJsPrompt,
  UPDATE_PLAN_TOOL,
} from "./local/local.ts";
import { WEB_FETCH_TOOL, WEB_IGNIS_PROMPT, WEB_SEARCH_TOOL } from "./web/web.ts";

const now = new Date("2026-09-15T01:52:00+02:00");
const ctx = { now, notes: [] };
const doc: Attachment = { name: "a.txt", size: 2, kind: "text", text: "hi" };
const all = {
  enabled: true,
  agents: true,
  web: true,
  askUser: true,
  dateTime: true,
  dateTimeLive: false,
  runJs: true,
  jsSafetyCheck: true,
  plan: true,
  memory: true,
  files: true,
  readFiles: true,
};
const names = (extras: { tools?: { function: { name: string } }[] }) => (extras.tools ?? []).map((t) => t.function.name);

describe("tools", () => {
  it("adds nothing while no tool is on", () => {
    expect(ignisPrompt(NO_TOOLS, ctx)).toBe("");
    expect(toolExtras(NO_TOOLS, ctx)).toEqual({ ignisPrompt: "", tools: [] });
    expect(toolsInUse(NO_TOOLS)).toBe(0);
  });

  it("declares agents and adds their part of the ignis prompt", () => {
    expect(toolExtras({ ...NO_TOOLS, agents: true }, ctx)).toEqual({ ignisPrompt: AGENTS_IGNIS_PROMPT, tools: [AGENT_TOOL] });
  });

  it("writes the date and time into the prompt without declaring a tool", () => {
    expect(toolExtras({ ...NO_TOOLS, dateTime: true }, ctx)).toEqual({ ignisPrompt: dateTimePrompt(now), tools: [] });
    expect(turnDateTime({ ...NO_TOOLS, dateTime: true }, now)).toBeUndefined();
  });

  /**
   * Updating every prompt moves the moment out of the system prompt: the two
   * together would say two different times, and the point of the option is to
   * leave the prompt's head — where the system prompt is — still.
   */
  it("sends the moment as a turn's own developer message once it updates every prompt", () => {
    const live = { ...NO_TOOLS, dateTime: true, dateTimeLive: true };
    expect(ignisPrompt(live, ctx)).toBe("");
    expect(turnDateTime(live, now)).toBe(dateTimePrompt(now));
    expect(turnDateTime({ ...live, enabled: false }, now)).toBeUndefined();
    expect(turnDateTime({ ...NO_TOOLS, dateTimeLive: true }, now)).toBeUndefined();
  });

  /**
   * An agent answers one prompt and is gone: it has no history to put a
   * developer message after, so it keeps taking the session's moment in its
   * system prompt — which is also what has every agent of a session open with
   * the same tokens.
   */
  it("keeps the date in an agent's system prompt even while the conversation updates every prompt", () => {
    const extras = agentExtras({ ...NO_TOOLS, dateTime: true, dateTimeLive: true }, ctx);
    expect(extras.ignisPrompt).toBe(dateTimePrompt(now));
  });

  it("puts every tool in a fixed order, the date first", () => {
    expect(toolExtras(all, { ...ctx, attachments: [doc] })).toEqual({
      ignisPrompt: [
        dateTimePrompt(now),
        AGENTS_IGNIS_PROMPT,
        WEB_IGNIS_PROMPT,
        ASK_USER_IGNIS_PROMPT,
        runJsPrompt(true),
        PLAN_PROMPT,
        memoryPrompt([]),
        FILES_PROMPT,
        attachmentsPrompt([doc]),
      ].join("\n\n"),
      tools: [
        AGENT_TOOL,
        WEB_SEARCH_TOOL,
        WEB_FETCH_TOOL,
        ASK_USER_TOOL,
        RUN_JS_TOOL,
        UPDATE_PLAN_TOOL,
        MEMORY_SAVE_TOOL,
        MEMORY_READ_TOOL,
        MEMORY_DELETE_TOOL,
        CREATE_FILE_TOOL,
        READ_FILE_TOOL,
      ],
    });
    expect(toolsInUse(all)).toBe(9);
  });

  it("declares read_file only once a file is attached, and tells run_js about the safety check", () => {
    expect(toolExtras({ ...NO_TOOLS, readFiles: true }, ctx)).toEqual({ ignisPrompt: "", tools: [] });
    expect(ignisPrompt({ ...NO_TOOLS, runJs: true, jsSafetyCheck: false }, ctx)).toBe(runJsPrompt(false));
    expect(runJsPrompt(true)).toContain("reviewed before it runs");
    expect(runJsPrompt(false)).not.toContain("reviewed");
  });

  it("declares nothing while the switch for all tools is off", () => {
    expect(toolExtras({ ...all, enabled: false }, ctx)).toEqual({ ignisPrompt: "", tools: [] });
    expect(agentExtras({ ...all, enabled: false }, ctx)).toEqual({ ignisPrompt: "", tools: [] });
    expect(toolsInUse({ ...all, enabled: false })).toBe(0);
  });

  it("gives agents every tool but agent, ask_user and update_plan", () => {
    expect(names(agentExtras(all, { ...ctx, attachments: [doc] }))).toEqual([
      "web_search",
      "web_fetch",
      "run_js",
      "memory_save",
      "memory_read",
      "memory_delete",
      "create_file",
      "read_file",
    ]);
    expect(agentExtras({ ...NO_TOOLS, agents: true, askUser: true, plan: true }, ctx)).toEqual({ ignisPrompt: "", tools: [] });
  });

  it("keeps each tool's choice under the switch for all, and turns all on when none was chosen", () => {
    const some = { ...NO_TOOLS, web: true };
    expect(setAllTools(some, false)).toEqual({ ...some, enabled: false });
    expect(setAllTools({ ...some, enabled: false }, true)).toEqual(some);
    expect(setAllTools({ ...NO_TOOLS, enabled: false }, true)).toEqual(all);
    expect(ALL_TOOLS).toEqual(all);
  });

  it("routes declared calls to their runner, and any undeclared tool to unknown", () => {
    const every = names(toolExtras(all, { ...ctx, attachments: [doc] }));
    expect(routeCall("web_search", every)).toBe("web");
    expect(routeCall("agent", every)).toBe("agent");
    expect(routeCall("ask_user", every)).toBe("ask");
    for (const local of ["run_js", "update_plan", "memory_save", "memory_read", "memory_delete", "create_file", "read_file"]) {
      expect(routeCall(local, every)).toBe("local");
    }
    expect(routeCall("read_file_x", every)).toBe("unknown");
    expect(routeCall("run_js", ["agent"])).toBe("unknown");
    expect(routeCall("agent", [])).toBe("unknown");
  });
});
