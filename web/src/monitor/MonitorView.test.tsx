import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { FLASH_NEXT_EXPOSITION, IGNIS_EXPOSITION } from "./fixture.ts";
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

  it("names the expert cache and the residency, and no longer says the rest is the KV pool's (GitHub #306)", () => {
    const text = IGNIS_EXPOSITION.replace('line="residency"} 0', 'line="residency"} 134217728').replace('ignis_vram_budget_bytes 31138512896', 'ignis_vram_budget_bytes 42949672960').replace('line="expert_cache"} 0', 'line="expert_cache"} 8589934592');
    const state = applyScrape(initialMonitor(), { kind: "ok", text }, Date.now(), 3);
    const html = renderToStaticMarkup(<MonitorView state={state} />);
    expect(html).toContain("Expert cache");
    expect(html).toContain("Residency");
    expect(html.replace(/<!-- -->/g, "")).toContain("sized by context and lanes");
    expect(html.replace(/<!-- -->/g, "")).not.toContain("left for the KV pool");
    // The 27B's plan, with no expert cache, keeps saying it.
    const plain = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "ok", text: IGNIS_EXPOSITION }, Date.now(), 3)} />);
    expect(plain.replace(/<!-- -->/g, "")).toContain("left for the KV pool");
    expect(plain).not.toContain("Expert cache");
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
    // GitHub #281: of the 10 slots, 8 on the host in 1.73 GiB pinned, 2 in VRAM.
    expect(html).toContain("8 on the host (1.73 GiB pinned), 2 in VRAM");
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

  it("draws a Flash-Next load's expert residency and n-gram rows, and a 27B load's neither", () => {
    const now = Date.now();
    // Ten seconds on: 900 decode hits and 100 misses, 50 tokens decoded.
    const later = FLASH_NEXT_EXPOSITION.replace('ignis_expert_cache_hits_total{class="gate_up_k2",phase="decode"} 4000', 'ignis_expert_cache_hits_total{class="gate_up_k2",phase="decode"} 4900')
      .replace('ignis_expert_cache_misses_total{class="down_k3",phase="decode"} 300', 'ignis_expert_cache_misses_total{class="down_k3",phase="decode"} 400')
      .replace("ignis_decoded_tokens_total 12400", "ignis_decoded_tokens_total 12450");
    const flashNext = applyScrape(applyScrape(initialMonitor(), { kind: "ok", text: FLASH_NEXT_EXPOSITION }, now - 10_000, 3), { kind: "ok", text: later }, now, 4);
    const html = renderToStaticMarkup(<MonitorView state={flashNext} />);
    expect(html).toContain('aria-label="Expert residency"');
    expect(html).toContain('aria-label="N-gram rows"');
    expect(html).toContain("90%");
    expect(html).toContain("gate_up_k4");
    // 2,999 of 3,000 down_k2 slots, both terms.
    expect(html).toContain("2,999");
    expect(html).not.toContain("not in ADR 0017");

    const dense = applyScrape(initialMonitor(), { kind: "ok", text: IGNIS_EXPOSITION }, now, 2);
    const html27b = renderToStaticMarkup(<MonitorView state={dense} />);
    expect(html27b).not.toContain('aria-label="Expert residency"');
    expect(html27b).not.toContain('aria-label="N-gram rows"');
  });

  it("says when the key is wanted instead of drawing charts", () => {
    const html = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "unauthorized" }, 0, 1)} />);
    expect(html).toContain("ignis wants the API key for its metrics");
    expect(html).toContain("Locked");
    expect(html).not.toContain('aria-label="Server health"');
  });
});
