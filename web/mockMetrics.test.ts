import { describe, expect, it } from "vitest";
import { parseExposition } from "./src/monitor/exposition.ts";
import { readSnapshot } from "./src/monitor/snapshot.ts";
import { createMetricsSim } from "./mockMetrics.ts";

describe("createMetricsSim", () => {
  it("renders the KV-disk tier from the first scrape, inside the contract (spec vram-budget/03)", () => {
    const start = 1_000_000;
    const sim = createMetricsSim(start);
    const first = readSnapshot(parseExposition(sim.render(start)));
    expect(first.kvDisk).toMatchObject({ used: 0, spills: { device: 0, kv_ram: 0 }, failures: { write: 0, read: 0 } });
    expect(first.retained.spills.disk).toEqual({ checkpoint: 0, prefix: 0 });

    // Ten minutes of traffic: the disk fills a little and every series stays known.
    const later = readSnapshot(parseExposition(sim.render(start + 600_000)));
    expect(later.unknown).toEqual([]);
    expect(later.kvDisk?.capacity).toBeGreaterThan(0);
    expect(later.kvDisk?.used).toBeLessThanOrEqual(later.kvDisk!.capacity!);
  });
});
