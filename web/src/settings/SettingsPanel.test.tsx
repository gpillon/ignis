import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";
import { activeTransport, chooseTransport, fallBackToHttp } from "../api/transport.ts";
import { chooseSilence, dictationSilence } from "../speech/silence.ts";
import { ALL_TOOLS, NO_TOOLS, type ToolsState } from "../tools/index.ts";
import { DEFAULT_SETTINGS, type PlaygroundSettings } from "./defaults.ts";
import { type KeepView, KeepSessions, SettingsPanel, ToolsSetting } from "./SettingsPanel.tsx";
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

  it("has one Workspace switch, on by default, where Create files and Attachments were", () => {
    const html = render(ALL_TOOLS);
    expect(html).toMatch(/id="tool-workspace-label"[^>]*>Workspace</);
    expect(html).not.toContain("Create files");
    expect(html).not.toContain("Attachments");
    expect(ALL_TOOLS.workspace).toBe(true);
  });

  it("says the prompt box attaches every kind the workspace takes from it: text, PDF and audio files, and pictures", () => {
    expect(render(ALL_TOOLS)).toContain("the prompt box attaches text, PDF and audio files, and pictures, to it");
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
  it("offers max after Xhigh among the efforts, and the budget right after them", () => {
    const html = panel();
    const efforts = radios(html.slice(0, html.indexOf("Thinking budget"))).map((r) => r.label);
    expect(efforts.slice(-2)).toEqual(["Xhigh", "Max"]);
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

// The dictation's silence (Q80): a per-viewer setting in General, on at 3 s.

describe("SettingsPanel: the dictation's silence", () => {
  afterEach(() => chooseSilence(3_000));

  const region = (html: string) => html.slice(html.indexOf("Dictation stops after"), html.indexOf("Transport"));

  it("sits in General with Off, 2 s, 3 s and 5 s, on at 3 s", () => {
    const html = region(panel());
    expect(radios(html).map((r) => r.label)).toEqual(["Off", "2 s", "3 s", "5 s"]);
    expect(checked(html)).toEqual(["3 s"]);
  });

  it("shows the pick, Off included, and says what a held key does", () => {
    chooseSilence(null);
    expect(checked(region(panel()))).toEqual(["Off"]);
    chooseSilence(5_000);
    expect(checked(region(panel()))).toEqual(["5 s"]);
    expect(region(panel())).toContain("never stops by itself");
    expect(dictationSilence()).toBe(5_000);
  });
});

// The wire the conversation goes over (GitHub #283): WebSocket by default,
// HTTP when picked, and a word when the page fell back to HTTP on its own.

describe("SettingsPanel: the transport", () => {
  afterEach(() => chooseTransport("websocket"));

  /** The transport control's part of the panel. */
  const transportRegion = (html: string) => html.slice(html.indexOf("Transport"), html.indexOf("Markdown in replies"));

  it("sits in General and starts on the WebSocket", () => {
    const region = transportRegion(panel());
    expect(radios(region).map((r) => r.label)).toEqual(["WebSocket", "HTTP"]);
    expect(checked(region)).toEqual(["WebSocket"]);
  });

  it("forces HTTP when picked", () => {
    chooseTransport("http");
    expect(checked(transportRegion(panel()))).toEqual(["HTTP"]);
    expect(activeTransport()).toBe("http");
  });

  it("says when the page fell back to HTTP by itself, and tries the socket again once WebSocket is picked", () => {
    fallBackToHttp();
    const fellBack = transportRegion(panel());
    // The control shows the wire in use, so WebSocket is there to pick again.
    expect(checked(fellBack)).toEqual(["HTTP"]);
    expect(fellBack).toContain("did not open the WebSocket");
    expect(fellBack).toContain("Pick WebSocket to try it again");
    expect(fellBack).not.toContain("reloads");
    expect(activeTransport()).toBe("http");
    chooseTransport("websocket");
    expect(activeTransport()).toBe("websocket");
    expect(transportRegion(panel())).not.toContain("did not open the WebSocket");
  });
});

// The sessions section says whether the browser keeps them and offers the two
// clicks, the second only after a refusal and only while notifications can
// still be asked for.

const keepView = (over: Partial<KeepView>): string =>
  renderToStaticMarkup(
    <KeepSessions
      view={{ keeping: "best-effort", asked: null, via: null, step: "available", busy: false, ...over }}
      onAsk={() => {}}
      onNotify={() => {}}
    />,
  );

describe("KeepSessions", () => {
  it("says kept and offers nothing", () => {
    const html = keepView({ keeping: "kept" });
    expect(html).toContain("Kept: the browser will not delete");
    expect(html).not.toContain("<button");
  });

  it("says best-effort and offers the first ask only", () => {
    const html = keepView({});
    expect(html).toContain("Best-effort");
    expect(html).toContain("Ask the browser to keep them");
    expect(html).not.toContain("Allow notifications");
  });

  it("says there is no Storage API and offers nothing", () => {
    const html = keepView({ keeping: "unsupported" });
    expect(html).toContain("no Storage API");
    expect(html).not.toContain("<button");
  });

  it("offers the notifications step after a refusal, with its explanation", () => {
    const html = keepView({ asked: "refused" });
    expect(html).toContain("Allow notifications, then ask again");
    expect(html).toContain("Chromium-based browser");
    expect(html).not.toContain("Ask the browser to keep them");
  });

  it("hides the notifications step when denied, granted or missing, and says why", () => {
    for (const step of ["denied", "granted", "unsupported"] as const) {
      const html = keepView({ asked: "refused", step });
      expect(html).not.toContain("<button");
      expect(html.toLowerCase()).toContain("installing the page as an app");
    }
    expect(keepView({ asked: "refused", step: "denied" })).toContain("blocked");
  });

  it("points to installing the app when notifications did not help, and says when they did", () => {
    expect(keepView({ asked: "refused", via: { permission: "granted", persistence: "refused" } })).toContain("Installing the page as an app");
    expect(keepView({ asked: "refused", via: { permission: "denied", persistence: "refused" } })).not.toContain("<button");
    expect(keepView({ keeping: "kept", asked: "refused", via: { permission: "granted", persistence: "granted" } })).toContain(
      "once notifications were allowed",
    );
  });

  it("says what a first grant or a failed call came of", () => {
    expect(keepView({ keeping: "kept", asked: "granted" })).toContain("Kept. The browser will not delete");
    expect(keepView({ asked: "failed" })).toContain("failed");
  });

  it("sits in the General tab", () => {
    expect(panel()).toContain("Keeping the sessions");
  });
});
