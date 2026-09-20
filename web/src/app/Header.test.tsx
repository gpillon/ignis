import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PulseChip } from "../monitor/MonitorPulse.tsx";
import { Header } from "./Header.tsx";

// The header's view switch (GitHub #165, #247) and the engine pulse.

const base = { model: { state: "loading" } as const, busy: false, onOpen: () => {}, onOpenMemory: () => {} };

describe("Header", () => {
  it("shows the plain Playground label with no switch at all", () => {
    const html = renderToStaticMarkup(<Header {...base} />);
    expect(html).toContain("Playground");
    expect(html).not.toContain('aria-label="View"');
  });

  it("offers Decide without metrics and adds the Monitor only when they answer", () => {
    const plain = renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} />);
    expect(plain).toContain(">Decide<");
    expect(plain).not.toContain(">Monitor<");
    // The pulse chip reads the metrics too, so it waits for them as well.
    expect(plain).not.toContain('aria-label="Monitor"');

    const withMetrics = renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} monitorAvailable />);
    expect(withMetrics).toContain(">Monitor<");
  });

  it("marks the current view and hides the chat drawers outside the chat", () => {
    const chat = renderToStaticMarkup(<Header {...base} view="chat" onView={() => {}} monitorAvailable />);
    expect(chat).toMatch(/aria-current="page"[^>]*>Playground</);
    expect(chat).toContain('aria-label="Sessions"');

    const monitor = renderToStaticMarkup(<Header {...base} view="monitor" onView={() => {}} monitorAvailable />);
    expect(monitor).toMatch(/aria-current="page"[^>]*>Monitor</);
    expect(monitor).not.toContain('aria-label="Sessions"');
    expect(monitor).not.toContain('aria-label="Settings"');

    const decide = renderToStaticMarkup(<Header {...base} view="decide" onView={() => {}} monitorAvailable />);
    expect(decide).toMatch(/aria-current="page"[^>]*>Decide</);
    expect(decide).not.toContain('aria-label="Sessions"');
    expect(decide).not.toContain('aria-label="Settings"');
  });
});

describe("PulseChip", () => {
  it("shows the gauges, the queue only when there is one, and a live decoded-token rate", () => {
    const busy = renderToStaticMarkup(<PulseChip running={3} waiting={2} tokensPerSec={96} live paused={false} onOpen={() => {}} />);
    expect(busy).toMatch(/>3<\/b> running/);
    expect(busy).toMatch(/>2<\/b> waiting/);
    expect(busy).toMatch(/>96<\/b> tok\/s</);
    expect(busy).toContain("Tokens decoded per second");
    expect(busy).toContain("pulse-dot");

    const quiet = renderToStaticMarkup(<PulseChip running={0} waiting={0} tokensPerSec={null} live paused onOpen={() => {}} />);
    expect(quiet).not.toContain("waiting");
    expect(quiet).not.toContain("pulse-dot");
  });

  it("says tokens count only at completion on a server without decoded tokens", () => {
    const html = renderToStaticMarkup(<PulseChip running={1} waiting={0} tokensPerSec={40} live={false} paused={false} onOpen={() => {}} />);
    expect(html).toMatch(/>40<\/b> tok\/s done/);
    expect(html).toContain("counted as requests complete");
  });
});
