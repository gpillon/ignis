import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { FLASH_NEXT_EXPOSITION, IGNIS_EXPOSITION, IMAGE_EXPOSITION, IMAGE_LOAD_EXPOSITION, KV_DISK_EXPOSITION, LOOP_GUARD_EXPOSITION, SPEECH_EXPOSITION, THINKING_FORCED_CLOSES } from "./fixture.ts";
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
    expect(html).toContain("off on this load");
    expect(html).not.toContain("not implemented");
    expect(html).toContain("Live");
    // No disk bar and no disk columns without the tier.
    expect(html).not.toContain("KV-disk");
    expect(html).not.toContain("Disk <span");
  });

  it("makes the Disk row live and draws the disk bar on a load with the KV-disk tier (spec vram-budget/03)", () => {
    const now = Date.now();
    const state = applyScrape(initialMonitor(), { kind: "ok", text: KV_DISK_EXPOSITION }, now, 3);
    const html = renderToStaticMarkup(<MonitorView state={state} />).replace(/<!-- -->/g, "");
    expect(html).not.toContain("off on this load");
    expect(html).toContain("KV-disk");
    expect(html).toContain("25% in use");
    expect(html).toContain("Disk <span");
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
    // GitHub #222: the prefix walk raises its own misses, so the row reads a
    // count in every column — device then KV-RAM, checkpoint then prefix.
    const misses = html.match(/<td[^>]*>Misses<\/td>(.*?)<\/tr>/)?.[1] ?? "";
    expect([...misses.matchAll(/<span[^>]*>([\d,]+)<\/span>/g)].map((m) => m[1])).toEqual(["11", "14", "5", "16"]);
    expect(html).not.toContain("not measured");
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
    // --reuse-prompt off hands out no slots (#215): the bound is exported, and
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

  it("hides the 27B's sibling-prefix card on Flash-Next and names the eviction row by where its retained slots live (GitHub #306)", () => {
    const now = Date.now();
    // Flash-Next: eight retained slots, all in pinned host RAM, none on the device.
    const text = FLASH_NEXT_EXPOSITION.replace('ignis_retained_slots{state="capacity"} 10', 'ignis_retained_slots{state="capacity"} 8');
    const html = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "ok", text }, now, 3)} />);
    expect(html).not.toContain("Prefix reuse");
    // The row keeps its tier name (its Live cell is sequences that left VRAM); the host-RAM note rides on the Retained cell.
    expect(html).not.toContain("Retained slots (host RAM)");
    expect(html).toContain("retained state lives in host RAM");
    // The 27B keeps both: its prefix card, and VRAM for the two device slots it has.
    const dense = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "ok", text: IGNIS_EXPOSITION }, now, 2)} />);
    expect(dense).toContain("Prefix reuse");
    expect(dense).not.toContain("retained state lives in host RAM");
    // Flash-Next with device slots keeps the VRAM wording.
    const withDevice = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "ok", text: FLASH_NEXT_EXPOSITION }, now, 3)} />);
    expect(withDevice).not.toContain("retained state lives in host RAM");
  });

  it("draws a Flash-Next load's expert residency and n-gram rows, and a 27B load's neither", () => {
    const now = Date.now();
    // Ten seconds on: 900 decode hits and 100 misses, 50 tokens decoded.
    const later = FLASH_NEXT_EXPOSITION.replace('ignis_expert_cache_hits_total{class="gate_up_k2",phase="decode"} 4000', 'ignis_expert_cache_hits_total{class="gate_up_k2",phase="decode"} 4900')
      .replace('ignis_expert_cache_misses_total{class="down_k3",phase="decode"} 300', 'ignis_expert_cache_misses_total{class="down_k3",phase="decode"} 400')
      .replace("ignis_decoded_tokens_total 12400", "ignis_decoded_tokens_total 12450")
      .replace('ignis_expert_residency_stall_seconds_total{phase="decode"} 18.6', 'ignis_expert_residency_stall_seconds_total{phase="decode"} 18.7');
    const flashNext = applyScrape(applyScrape(initialMonitor(), { kind: "ok", text: FLASH_NEXT_EXPOSITION }, now - 10_000, 3), { kind: "ok", text: later }, now, 4);
    const html = renderToStaticMarkup(<MonitorView state={flashNext} />);
    expect(html).toContain('aria-label="Expert residency"');
    expect(html).toContain('aria-label="N-gram rows"');
    expect(html).toContain("90%");
    expect(html).toContain("gate_up_k4");
    // 2,999 of 3,000 down_k2 slots, both terms.
    expect(html).toContain("2,999");
    // The capacity is the LRU's: a pinned hot set's slots are not in it (ADR 0055).
    expect(html).toContain("LRU slots in use");
    expect(html).toContain("A pinned hot set&#x27;s slots are not among them");
    // 0.1 s waited on demand copies over those 50 tokens and 10 s.
    expect(html).toContain("Stall per token");
    expect(html).toContain("2.0 ms");
    expect(html).toContain("1.0%");
    expect(html).not.toContain("not in ADR 0017");

    const dense = applyScrape(initialMonitor(), { kind: "ok", text: IGNIS_EXPOSITION }, now, 2);
    const html27b = renderToStaticMarkup(<MonitorView state={dense} />);
    expect(html27b).not.toContain('aria-label="Expert residency"');
    expect(html27b).not.toContain('aria-label="N-gram rows"');
  });

  it("draws the image jobs once the image model admitted one, and nothing of them before (GitHub #324)", () => {
    const now = Date.now();
    // Ten seconds on: ten more seconds of denoising, so the phases have a window to share.
    const later = IMAGE_EXPOSITION.replace('ignis_image_job_phase_seconds_total{phase="denoising"} 14', 'ignis_image_job_phase_seconds_total{phase="denoising"} 24');
    const state = applyScrape(applyScrape(initialMonitor(), { kind: "ok", text: IMAGE_EXPOSITION }, now - 10_000, 2), { kind: "ok", text: later }, now, 2);
    const html = renderToStaticMarkup(<MonitorView state={state} />);
    expect(html).toContain("Image jobs");
    expect(html).toContain("Median duration");
    expect(html).toContain("Time denoising");
    const text = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "ok", text: IGNIS_EXPOSITION }, now, 2)} />);
    expect(text).not.toContain("Image jobs");
  });

  // Spec speech/02: the Transcriptions card, only once the speech model was asked.
  it("draws the Transcriptions card once the speech model transcribed, and not on a server that never did", () => {
    const now = Date.now();
    // Ten seconds on: three more completed transcriptions and 60 more seconds of audio.
    const later = SPEECH_EXPOSITION.replace('executor="cpu",outcome="completed"} 4', 'executor="cpu",outcome="completed"} 7').replace("audio_seconds_total 80", "audio_seconds_total 140");
    const state = applyScrape(applyScrape(initialMonitor(), { kind: "ok", text: SPEECH_EXPOSITION }, now - 10_000, 2), { kind: "ok", text: later }, now, 2);
    const html = renderToStaticMarkup(<MonitorView state={state} />);
    expect(html).toContain("Transcriptions");
    expect(html).toContain("Median wait");
    expect(html).toContain("Sessions open");
    const text = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "ok", text: IGNIS_EXPOSITION }, now, 2)} />);
    expect(text).not.toContain("Median wait");
  });

  // GitHub #330: the Reasoning card, and the rewinds among the Request flow's cancels.
  it("draws the Reasoning card once the loop guard watched a request, with the rewinds noted on the Request flow", () => {
    const now = Date.now();
    // Ten seconds on: 12 more watched turns, 3 looped, 3 rewound and 1 closed, so the window has something to say.
    const later = LOOP_GUARD_EXPOSITION.replace("watched_requests_total 40", "watched_requests_total 52")
      .replace("looped_requests_total 6", "looped_requests_total 9")
      .replace('outcome="rewound"} 7', 'outcome="rewound"} 10')
      .replace('outcome="closed"} 1', 'outcome="closed"} 2')
      .replace('outcome="no_room"} 0', 'outcome="no_room"} 1')
      .replace('outcome="submit_refused"} 0', 'outcome="submit_refused"} 1')
      .replace("rewound_tokens_total 5120", "rewound_tokens_total 6144")
      .replace("ignis_decoded_tokens_total 12400", "ignis_decoded_tokens_total 22400");
    const state = applyScrape(applyScrape(initialMonitor(), { kind: "ok", text: LOOP_GUARD_EXPOSITION }, now - 10_000, 2), { kind: "ok", text: later }, now, 2);
    const html = renderToStaticMarkup(<MonitorView state={state} />).replace(/<!-- -->/g, "");
    expect(html).toContain('aria-label="Reasoning"');
    expect(html).toContain("3 of 12");
    expect(html).toContain("25%");
    expect(html).toContain("1,024");
    expect(html).toContain("10%");
    expect(html).toContain("Rewound");
    expect(html).toContain("Closed");
    // The loops the guard could not act on show only when there were some in the window.
    expect(html).toContain("No room");
    expect(html).not.toContain("No tokenizer");
    expect(html).toContain("Budget closes");
    expect(html).toContain("Continuation refused");
    // Three nudges, a close and a refused continuation: five cancels.
    expect(html).toContain("5 of these cancels were loop-guard rewinds");
    expect(html).toContain("Lowest loop score per turn");
  });

  it("draws no Reasoning card and no rewind note without the series, and only the budget's closes with just that counter", () => {
    const now = Date.now();
    const plain = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "ok", text: IGNIS_EXPOSITION }, now, 2)} />);
    expect(plain).not.toContain('aria-label="Reasoning"');
    expect(plain).not.toContain("loop-guard rewind");
    const budget = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "ok", text: `${IGNIS_EXPOSITION}${THINKING_FORCED_CLOSES}` }, now, 2)} />);
    expect(budget).toContain('aria-label="Reasoning"');
    expect(budget).toContain("Budget closes");
    expect(budget).not.toContain("Turns looped");
    // A guard that watched but rewound nothing in the window says nothing on the Request flow.
    const still = applyScrape(applyScrape(initialMonitor(), { kind: "ok", text: LOOP_GUARD_EXPOSITION }, now - 10_000, 2), { kind: "ok", text: LOOP_GUARD_EXPOSITION }, now, 2);
    const html = renderToStaticMarkup(<MonitorView state={still} />);
    expect(html).toContain('aria-label="Reasoning"');
    expect(html).not.toContain("loop-guard rewind");
  });

  it("lays an image load's plan out by its condition budget, not a KV pool, and reads the prefixes held against it (GitHub #324)", () => {
    const state = applyScrape(initialMonitor(), { kind: "ok", text: IMAGE_LOAD_EXPOSITION }, Date.now(), 2);
    const html = renderToStaticMarkup(<MonitorView state={state} />).replace(/<!-- -->/g, "");
    // 15.46 GB of weights, a 1.32 GB arena and a 12.3 GB condition budget in a 29.1 GB budget.
    for (const segment of ["Weights · 14.4 GiB", "Workspace · 1.23 GiB", "Condition budget · 11.46 GiB"]) expect(html).toContain(segment);
    expect(html).toContain("27.08 GiB");
    expect(html).toContain("reserved of a 27.1 GiB budget");
    expect(html).toContain("the condition budget took what the weights and the arena left");
    expect(html).not.toContain("left for the KV pool");
    expect(html).not.toContain("The pool is");
    expect(html).not.toContain("KV pool ·");
    // A running job's 1.12 GB prefix of the 12.3 GB budget.
    expect(html).toContain("Condition prefixes");
    expect(html).toContain("1.04 GiB");
    expect(html).toContain("9.1% in use");
    // A text load has neither.
    const text = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "ok", text: IGNIS_EXPOSITION }, Date.now(), 2)} />);
    expect(text).not.toContain("Condition budget");
    expect(text).not.toContain("Condition prefixes");
  });

  it("says when the key is wanted instead of drawing charts", () => {
    const html = renderToStaticMarkup(<MonitorView state={applyScrape(initialMonitor(), { kind: "unauthorized" }, 0, 1)} />);
    expect(html).toContain("ignis wants the API key for its metrics");
    expect(html).toContain("Locked");
    expect(html).not.toContain('aria-label="Server health"');
  });
});
