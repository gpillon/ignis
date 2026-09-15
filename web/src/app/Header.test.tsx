import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PulseChip } from "../monitor/MonitorPulse.tsx";
import { Header } from "./Header.tsx";

// The header's Playground/Monitor switch and the engine pulse (GitHub #165).

const base = { model: { state: "loading" } as const, busy: false, onOpen: () => {}, onOpenMemory: () => {} };

describe("Header", () => {
  it("shows the plain Playground label without metrics", () => {
    const html = renderToStaticMarkup(<Header {...base} />);
    expect(html).toContain("Playground");
    expect(html).not.toContain('aria-label="View"');
  });

  it("switches views when metrics answer, marking the current one and hiding the chat drawers in the Monitor", () => {
    const chat = renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} />);
    expect(chat).toMatch(/aria-current="page"[^>]*>Playground</);
    expect(chat).toContain('aria-label="Sessions"');

    const monitor = renderToStaticMarkup(<Header {...base} view="monitor" onView={() => {}} />);
    expect(monitor).toMatch(/aria-current="page"[^>]*>Monitor</);
    expect(monitor).not.toContain('aria-label="Sessions"');
    expect(monitor).not.toContain('aria-label="Settings"');
  });
});

describe("PulseChip", () => {
  it("shows the gauges, the queue only when there is one, and says tokens count at completion", () => {
    const busy = renderToStaticMarkup(<PulseChip running={3} waiting={2} tokensPerSec={96} paused={false} onOpen={() => {}} />);
    expect(busy).toMatch(/>3<\/b> running/);
    expect(busy).toMatch(/>2<\/b> waiting/);
    expect(busy).toMatch(/>96<\/b> tok\/s done/);
    expect(busy).toContain("counted as requests complete");
    expect(busy).toContain("pulse-dot");

    const quiet = renderToStaticMarkup(<PulseChip running={0} waiting={0} tokensPerSec={null} paused onOpen={() => {}} />);
    expect(quiet).not.toContain("waiting");
    expect(quiet).not.toContain("pulse-dot");
  });
});
