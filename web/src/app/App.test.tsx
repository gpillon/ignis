import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { forgetKey, saveKey } from "../api/auth.ts";
import { App } from "./App.tsx";

// The page is composed from components across the feature folders; this
// checks the first render still puts every region on it.

describe("App", () => {
  it("renders the header, both panels, the prompt and the session log", () => {
    const html = renderToStaticMarkup(<App />);
    expect(html).toContain("Playground");
    expect(html).toContain('aria-label="Sessions"');
    expect(html).toContain('aria-label="Settings"');
    expect(html).toContain('aria-label="Prompt"');
    expect(html).toContain('aria-label="This session"');
    expect(html).toContain("Loading model…");
    expect(html).toContain("Warming up");
  });

  it("gives every form field an id or a name, as the browser asks", () => {
    const fields = renderToStaticMarkup(<App />).match(/<(input|textarea|select)\b[^>]*>/g) ?? [];
    expect(fields.length).toBeGreaterThan(0);
    for (const field of fields) expect(field).toMatch(/\s(id|name)="[^"]+"/);
  });

  it("shows the key prompt in place of the page when ignis wants an API key", () => {
    forgetKey();
    const html = renderToStaticMarkup(<App />);
    expect(html).toContain("API key required");
    expect(html).toMatch(/<input[^>]*id="api-key"[^>]*type="password"/);
    expect(html).not.toContain('aria-label="Prompt"');

    saveKey("sk-test");
    const unlocked = renderToStaticMarkup(<App />);
    expect(unlocked).toContain('aria-label="Prompt"');
    expect(unlocked).toContain("Lock");
  });
});
