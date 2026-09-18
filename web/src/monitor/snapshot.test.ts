import { describe, expect, it } from "vitest";
import { parseExposition } from "./exposition.ts";
import { IGNIS_EXPOSITION } from "./fixture.ts";
import { emptySnapshot, readSnapshot, counterValues, RETAINED_FAMILY_KEYS } from "./snapshot.ts";

describe("readSnapshot", () => {
  it("reads every ADR 0017 contract series", () => {
    const s = readSnapshot(parseExposition(IGNIS_EXPOSITION));
    expect(s.version).toBe("0.1.0");
    expect(s.waiting).toBe(2);
    expect(s.running).toBe(5);
    expect(s.accepted).toBe(42);
    expect(s.completed).toBe(30);
    expect(s.cancelled).toBe(3);
    expect(s.generatedTokens).toBe(12345);
    expect(s.decodedTokens).toBe(12400);
    expect(s.kvEvictions).toBe(4);
    expect(s.prefixReusedTokens).toBe(8192);
    expect(s.rejected).toEqual({ full: 6, unknown_model: 1, oversized: 0 });
    expect(s.unknown).toEqual([]);
  });

  it("reads a histogram as finite bounds plus +Inf, cumulative counts, sum and count", () => {
    const { ttft, duration } = readSnapshot(parseExposition(IGNIS_EXPOSITION));
    expect(ttft?.bounds).toEqual([0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, Infinity]);
    expect(ttft?.cumulative).toEqual([1, 4, 10, 20, 28, 30, 30, 30, 30, 30, 30, 30, 30]);
    expect(ttft?.sum).toBe(15.3);
    expect(ttft?.count).toBe(30);
    expect(duration?.bounds.at(-2)).toBe(600);
  });

  it("leaves absent series null", () => {
    const s = readSnapshot(parseExposition("ignis_requests_accepted_total 3\n"));
    expect(s.accepted).toBe(3);
    expect(s.version).toBeNull();
    expect(s.running).toBeNull();
    expect(s.decodedTokens).toBeNull();
    expect(s.rejected).toEqual({ full: null, unknown_model: null, oversized: null });
    expect(s.ttft).toBeNull();
  });

  it("reads what the load reserved and what is occupied of it", () => {
    const { memory } = readSnapshot(parseExposition(IGNIS_EXPOSITION));
    expect(memory.reserved.weights).toBe(17_179_869_184);
    expect(memory.reserved.media_embedding).toBe(402_653_184);
    expect(memory.reserved.residual).toBe(268_435_456);
    expect(memory.budgetBytes).toBe(31_138_512_896);
    expect(memory.kvPoolPages).toBe(4032);
    expect(memory.kvPageBytes).toBe(1_835_008);
    expect(memory.kvPoolUsedPages).toBe(1536);
    expect(memory.kvRamArena).toEqual({ capacity: 8_589_934_592, used: 2_147_483_648 });
    expect(memory.retainedSlots).toEqual({ capacity: 10, inUse: 7 });
    expect(memory.slotSkips).toEqual({ publish_skipped_no_slot: 12, capture_skipped_no_slot: 5, capture_skipped_no_page: 1 });
  });

  it("reads all six retained-state families in both tiers and both kinds", () => {
    const { retained } = readSnapshot(parseExposition(IGNIS_EXPOSITION));
    expect(RETAINED_FAMILY_KEYS).toEqual(["reusedTokens", "hits", "misses", "spills", "discards", "restores"]);
    // All twenty-four series the server writes, read against the fixture's own figures.
    expect(retained).toEqual({
      reusedTokens: { device: { checkpoint: 41_200, prefix: 9800 }, kv_ram: { checkpoint: 6400, prefix: 0 } },
      hits: { device: { checkpoint: 18, prefix: 7 }, kv_ram: { checkpoint: 3, prefix: 0 } },
      // Checkpoint-only by construction: the prefix walk raises no miss (#222).
      misses: { device: { checkpoint: 11, prefix: 0 }, kv_ram: { checkpoint: 5, prefix: 0 } },
      spills: { device: { checkpoint: 0, prefix: 0 }, kv_ram: { checkpoint: 6, prefix: 2 } },
      discards: { device: { checkpoint: 4, prefix: 9 }, kv_ram: { checkpoint: 1, prefix: 0 } },
      restores: { device: { checkpoint: 0, prefix: 0 }, kv_ram: { checkpoint: 3, prefix: 1 } },
    });
  });

  it("leaves a memory series a load does not have null, and reads a zero as a zero", () => {
    // A load without vision, without a drafter and with prompt reuse off: the
    // lines it did not reserve are exported as zeros, and a load that never
    // spilled has no kv_ram series at all.
    const text = [
      'ignis_vram_reserved_bytes{line="weights"} 17179869184',
      'ignis_vram_reserved_bytes{line="media_embedding"} 0',
      'ignis_vram_reserved_bytes{line="drafter_round"} 0',
      'ignis_retained_slots{state="capacity"} 0',
      'ignis_retained_state_spills_total{tier="device",kind="checkpoint"} 0',
      'ignis_retained_state_hits_total{tier="device",kind="prefix"} 0',
      'ignis_retained_reused_tokens_total{tier="device",kind="checkpoint"} 0',
      'ignis_retained_slot_skips_total{reason="capture_skipped_no_page"} 0',
    ].join("\n");
    const s = readSnapshot(parseExposition(`${text}\n`));
    expect(s.memory.reserved.weights).toBe(17_179_869_184);
    expect(s.memory.reserved.media_embedding).toBe(0);
    expect(s.memory.reserved.drafter_round).toBe(0);
    expect(s.memory.reserved.workspace).toBeNull();
    expect(s.memory.budgetBytes).toBeNull();
    expect(s.memory.kvPoolUsedPages).toBeNull();
    expect(s.memory.kvRamArena).toEqual({ capacity: null, used: null });
    expect(s.memory.retainedSlots).toEqual({ capacity: 0, inUse: null });
    expect(s.memory.slotSkips.publish_skipped_no_slot).toBeNull();
    expect(s.memory.slotSkips.capture_skipped_no_page).toBe(0);
    expect(s.retained.spills.device.checkpoint).toBe(0);
    expect(s.retained.spills.kv_ram.checkpoint).toBeNull();
    expect(s.retained.hits.device.prefix).toBe(0);
    expect(s.retained.hits.device.checkpoint).toBeNull();
    expect(s.retained.reusedTokens.device.checkpoint).toBe(0);
    expect(s.retained.misses.kv_ram.prefix).toBeNull();
    expect(s.unknown).toEqual([]);
  });

  it("counts the retained counters towards a restart, and the memory gauges not", () => {
    const empty = emptySnapshot();
    const moved = readSnapshot(parseExposition(IGNIS_EXPOSITION));
    // 7 plain counters, 3 reject reasons, 24 retained series, 3 skips, 2 histogram counts.
    expect(counterValues(empty)).toHaveLength(39);
    expect(counterValues(moved).filter((v) => v !== null)).toHaveLength(39);
    expect(counterValues(moved)).toContain(41_200);
    expect(counterValues(moved)).toContain(12);
    expect(counterValues(moved)).not.toContain(31_138_512_896);
  });

  it("lists series outside the contract instead of dropping them", () => {
    const text = `${IGNIS_EXPOSITION}# TYPE process_cpu_seconds_total counter\nprocess_cpu_seconds_total 3\nignis_future{a="b"} 1\n`;
    const s = readSnapshot(parseExposition(text));
    expect(s.unknown).toEqual([
      { name: "process_cpu_seconds_total", labels: {}, value: 3 },
      { name: "ignis_future", labels: { a: "b" }, value: 1 },
    ]);
  });

  it("lists a label value the contract does not name, on a metric it does", () => {
    const text = IGNIS_EXPOSITION.replace(
      'ignis_scheduler_requests{state="running"} 5',
      'ignis_scheduler_requests{state="running"} 5\nignis_scheduler_requests{state="prefilling"} 1',
    ).replace('reason="oversized"} 0', 'reason="oversized"} 0\nignis_requests_rejected_total{reason="shutdown"} 2');
    const s = readSnapshot(parseExposition(text));
    expect(s.running).toBe(5);
    expect(s.unknown).toEqual([
      { name: "ignis_scheduler_requests", labels: { state: "prefilling" }, value: 1 },
      { name: "ignis_requests_rejected_total", labels: { reason: "shutdown" }, value: 2 },
    ]);
  });
});
