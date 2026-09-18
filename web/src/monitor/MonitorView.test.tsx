import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { IGNIS_EXPOSITION } from "./fixture.ts";
import { MonitorView } from "./MonitorView.tsx";
import { applyScrape, initialMonitor } from "./scrape.ts";

// The Monitor's first render over two scrapes of the fixture: every region
// is on the page, the verdict is in words, and the latest figures show.

const later = IGNIS_EXPOSITION.replace("ignis_requests_accepted_total 42", "ignis_requests_accepted_total 50")
  .replace('reason="full"} 6', 'reason="full"} 9')
  .replace('ignis_request_ttft_seconds_bucket{le="+Inf"} 30', 'ignis_request_ttft_seconds_bucket{le="+Inf"} 34')
  .replace("ignis_request_ttft_seconds_count 30", "ignis_request_ttft_seconds_count 34");

describe("MonitorView", () => {
  it("renders the verdict, throughput, admission, latency, cache, scraper and series regions", () => {
    // Scraped just now, so the scrapes read as live rather than late.
    const now = Date.now();
    const state = applyScrape(applyScrape(initialMonitor(), { kind: "ok", text: IGNIS_EXPOSITION }, now - 5_000, 3), { kind: "ok", text: later }, now, 4);
    const html = renderToStaticMarkup(<MonitorView state={state} />);
    for (const region of [
      "Server health",
      "Throughput and load",
      "Scheduler load",
      "Request flow",
      "Time to first token",
      "Request duration",
      "VRAM plan",
      "In use now",
      "Retained state",
      "Prefix reuse",
      "Evictions",
      "Scraper",
    ]) {
      expect(html).toContain(`aria-label="${region}"`);
    }
    expect(html).toContain("Saturated");
    expect(html).toContain("3 requests turned away: engine full");
    expect(html).toContain("ignis 0.1.0");
    expect(html).toContain("All series");
    // GitHub #224: three tier rows, with disk present and visibly inert
    // rather than absent or showing a zero.
    for (const tier of ["VRAM", "RAM", "Disk"]) expect(html).toContain(tier);
    expect(html).toContain("not implemented");
    expect(html).toContain("Live");
  });

  it("reads the memory panel's live figures against the constants that bound them", () => {
    const now = Date.now();
    const state = applyScrape(applyScrape(initialMonitor(), { kind: "ok", text: IGNIS_EXPOSITION }, now - 5_000, 3), { kind: "ok", text: later }, now, 4);
    const html = renderToStaticMarkup(<MonitorView state={state} />);
    // 1,536 of 4,032 pages, 2 GiB of 8 GiB, 7 of 10 slots: both terms, never a bare number.
    expect(html).toContain("1,536");
    expect(html).toContain("4,032");
    expect(html).toContain("2 GiB");
    expect(html).toContain("8 GiB");
    expect(html).toContain("29 GiB budget");
    // 4,032 pages of 64 tokens: the room's tokens, beside the bytes.
    expect(html).toContain("258K tokens");
    expect(html).toContain("Weights");
    expect(html).toContain("KV pool");
    // The prefix miss series is zero by construction, and says so.
    expect(html).toContain("not measured");
    // 12 + 5 + 1 skips, beside the slots rather than in a corner of their own.
    expect(html).toContain("publishes and captures found no room since start");
    expect(html).toContain("A prefix found no slot to publish into");
  });

  it("renders a load that exported no plan at all, rather than an empty bar", () => {
    const bare = IGNIS_EXPOSITION.split("\n")
      .filter((line) => !/ignis_(vram|kv_pool|kv_page|kv_ram|retained)/.test(line))
      .join("\n");
    const now = Date.now();
    const state = applyScrape(initialMonitor(), { kind: "ok", text: bare }, now, 2);
    const html = renderToStaticMarkup(<MonitorView state={state} />);
    // What a scrape does not carry is said of the scrape, never of the load.
    expect(html).toContain("This scrape carries no plan");
    expect(html).toContain("this scrape carries no bound to read it against");
    expect(html).toContain("This scrape carries no skip counter");
    expect(html).not.toContain("Every publish and capture found room");
    expect(html).toContain('aria-label="Retained state"');
    expect(html).toContain('aria-label="Server health"');
  });

  it("tells a bound of zero apart from a bound the scrape does not carry", () => {
    // --prompt-reuse off hands out no slots (#215): the bound is exported, and
    // it is zero. That is a load with none to give, not a missing figure.
    const noSlots = IGNIS_EXPOSITION.replace('ignis_retained_slots{state="capacity"} 10', 'ignis_retained_slots{state="capacity"} 0').replace(
      'ignis_retained_slots{state="in_use"} 7',
      'ignis_retained_slots{state="in_use"} 0',
    );
    const html = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "ok", text: noSlots }, Date.now(), 2)} />);
    expect(html).toContain("this load has none of it to give");
    expect(html).toContain("this load hands out none");
    expect(html).not.toContain("this scrape carries no bound to read it against");
  });

  it("says when the key is wanted instead of drawing charts", () => {
    const html = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "unauthorized" }, 0, 1)} />);
    expect(html).toContain("ignis wants the API key for its metrics");
    expect(html).toContain("Locked");
    expect(html).not.toContain('aria-label="Server health"');
  });
});
