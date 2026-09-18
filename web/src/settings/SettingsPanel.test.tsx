import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { NO_TOOLS, type ToolsState } from "../tools/index.ts";
import { ToolsSetting } from "./SettingsPanel.tsx";

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
