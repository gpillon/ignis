import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { NO_TOOLS, type ToolsState } from "../tools/index.ts";
import { DEFAULT_SETTINGS, type PlaygroundSettings } from "./defaults.ts";
import { SettingsPanel, ToolsSetting } from "./SettingsPanel.tsx";
import { readBudgetInput } from "./ThinkingBudgetField.tsx";

// The rounds are a setting of the tools, so they live where the tools do:
// under the switch for all of them, and gone when it is off.

const render = (tools: ToolsState) =>
  renderToStaticMarkup(<ToolsSetting tools={tools} onChange={() => {}} onOpenMemory={() => {}} />);

describe("ToolsSetting", () => {
  it("puts the rounds under the switch for all tools", () => {
    const html = render({ ...NO_TOOLS, enabled: true });
    expect(html).toContain("Tool rounds");
    expect(html).toContain('value="16"');
    expect(html.indexOf("All tools")).toBeLessThan(html.indexOf("Tool rounds"));
  });

  it("takes the rounds away with the tools", () => {
    const html = render({ ...NO_TOOLS, enabled: false });
    expect(html).toContain("All tools");
    expect(html).not.toContain("Tool rounds");
  });
});

// The thinking budget sits beside the effort it caps (spec playground/04):
// the server's default, no budget, or a count, and nothing to set while the
// effort cannot use one.

const panel = (change: Partial<PlaygroundSettings> = {}) =>
  renderToStaticMarkup(
    <SettingsPanel
      open
      settings={{ ...DEFAULT_SETTINGS, ...change }}
      set={() => {}}
      tools={NO_TOOLS}
      onToolsChange={() => {}}
      markdown
      onMarkdownChange={() => {}}
      parallel={false}
      onParallelChange={() => {}}
      attachments={[]}
      onOpenMemory={() => {}}
    />,
  );

/** The budget control's part of the panel: from the fieldset its caption heads to the sampling sliders. */
const budgetRegion = (html: string) =>
  html.slice(html.lastIndexOf("<fieldset", html.indexOf("Thinking budget")), html.indexOf("Temperature"));

/** Each radio of a region as its label and whether it is checked and enabled. */
const radios = (region: string) =>
  [...region.matchAll(/<input([^>]*type="radio"[^>]*)\/><span[^>]*>([^<]*)<\/span>/g)].map((m) => ({
    label: m[2],
    checked: m[1].includes('checked=""'),
  }));

const checked = (region: string) => radios(region).filter((r) => r.checked).map((r) => r.label);

/** The typed-count box's attributes. */
const box = (region: string) => /<input([^>]*name="thinking-budget"[^>]*)\/>/.exec(region)?.[1] ?? "";

describe("SettingsPanel: the thinking budget", () => {
  it("offers max after X-high among the efforts, and the budget right after them", () => {
    const html = panel();
    const efforts = radios(html.slice(0, html.indexOf("Thinking budget"))).map((r) => r.label);
    expect(efforts.slice(-2)).toEqual(["X-high", "Max"]);
    expect(html.indexOf("Thinking budget")).toBeLessThan(html.indexOf("Temperature"));
  });

  it("offers the server default, off and the presets, and starts on the server default", () => {
    const region = budgetRegion(panel());
    expect(radios(region).map((r) => r.label)).toEqual(["Default", "Off", "2K", "4K", "8K", "16K"]);
    expect(checked(region)).toEqual(["Default"]);
    expect(box(region)).toContain('value=""');
  });

  it("shows off, a preset, and a typed count that is no preset, the count in the box", () => {
    expect(checked(budgetRegion(panel({ thinkingBudget: 0 })))).toEqual(["Off"]);
    const preset = budgetRegion(panel({ thinkingBudget: 4096 }));
    expect(checked(preset)).toEqual(["4K"]);
    expect(box(preset)).toContain('value="4096"');
    const typed = budgetRegion(panel({ thinkingBudget: 6000 }));
    expect(checked(typed)).toEqual([]);
    expect(box(typed)).toContain('value="6000"');
  });

  it("is disabled while thinking is off, and shows no budget", () => {
    const region = budgetRegion(panel({ reasoningEffort: "none", thinkingBudget: 4096 }));
    expect(region).toMatch(/<fieldset[^>]*disabled=""/);
    expect(box(region)).toContain('disabled=""');
    expect(checked(region)).toEqual([]);
    expect(box(region)).toContain('value=""');
  });

  it("is disabled under max and reads no budget (max), keeping the choice for the next effort", () => {
    const max = budgetRegion(panel({ reasoningEffort: "max", thinkingBudget: 4096 }));
    expect(max).toMatch(/<fieldset[^>]*disabled=""/);
    expect(box(max)).toContain('disabled=""');
    expect(box(max)).toContain('placeholder="no budget (max)"');
    expect(checked(max)).toEqual([]);
    // The same setting under another effort: the choice is back.
    expect(checked(budgetRegion(panel({ reasoningEffort: "medium", thinkingBudget: 4096 })))).toEqual(["4K"]);
  });

  it("warns when the budget is more than max tokens, and only then", () => {
    expect(budgetRegion(panel({ thinkingBudget: 16384, maxTokens: 8192 }))).toContain("More than max tokens");
    for (const change of [
      { thinkingBudget: 8192, maxTokens: 8192 },
      { thinkingBudget: 16384, maxTokens: null },
      { thinkingBudget: null, maxTokens: 1024 },
      { thinkingBudget: 0, maxTokens: 1024 },
      { thinkingBudget: 16384, maxTokens: 8192, reasoningEffort: "max" as const },
      { thinkingBudget: 16384, maxTokens: 8192, reasoningEffort: "none" as const },
    ]) {
      expect(budgetRegion(panel(change))).not.toContain("More than max tokens");
    }
  });
});

describe("readBudgetInput", () => {
  it("takes a whole number of tokens, spaces around it included", () => {
    expect(readBudgetInput("6000")).toEqual({ ok: true, tokens: 6000 });
    expect(readBudgetInput(" 300 ")).toEqual({ ok: true, tokens: 300 });
  });

  it("refuses 0, negatives, fractions and text, with a reason, so no request earns a 400", () => {
    for (const text of ["0", "-5", "1.5", "abc", "4k", "1e3"]) {
      const read = readBudgetInput(text);
      expect(read.ok).toBe(false);
      expect(!read.ok && read.error).toBeTruthy();
    }
    expect(readBudgetInput("0")).toMatchObject({ error: expect.stringMatching(/Off/) });
  });

  it("refuses a count past what ignis takes, a u32", () => {
    expect(readBudgetInput("4294967295")).toEqual({ ok: true, tokens: 4294967295 });
    expect(readBudgetInput("4294967296").ok).toBe(false);
  });

  it("says nothing of an empty box", () => {
    expect(readBudgetInput("  ")).toEqual({ ok: false, error: null });
  });
});
