import { describe, expect, it } from "vitest";
import { parseExposition } from "./exposition.ts";
import { IGNIS_EXPOSITION } from "./fixture.ts";

describe("parseExposition", () => {
  it("groups ignis's scrape into typed families with their help", () => {
    const { families, errors } = parseExposition(IGNIS_EXPOSITION);
    expect(errors).toEqual([]);
    expect(families.get("ignis_requests_accepted_total")).toMatchObject({
      type: "counter",
      help: "Accepted submissions.",
      samples: [{ name: "ignis_requests_accepted_total", labels: {}, value: 42 }],
    });
    expect(families.get("ignis_scheduler_requests")?.type).toBe("gauge");
    expect(families.get("ignis_scheduler_requests")?.samples).toEqual([
      { name: "ignis_scheduler_requests", labels: { state: "waiting" }, value: 2 },
      { name: "ignis_scheduler_requests", labels: { state: "running" }, value: 5 },
    ]);
    expect(families.get("ignis_build_info")?.samples[0].labels).toEqual({ version: "0.1.0" });
  });

  it("keeps a histogram's buckets, sum and count under its family", () => {
    const ttft = parseExposition(IGNIS_EXPOSITION).families.get("ignis_request_ttft_seconds");
    expect(ttft?.type).toBe("histogram");
    const buckets = ttft?.samples.filter((s) => s.name === "ignis_request_ttft_seconds_bucket") ?? [];
    expect(buckets).toHaveLength(13);
    expect(buckets.at(-1)).toEqual({ name: "ignis_request_ttft_seconds_bucket", labels: { le: "+Inf" }, value: 30 });
    expect(ttft?.samples.find((s) => s.name === "ignis_request_ttft_seconds_sum")?.value).toBe(15.3);
    expect(ttft?.samples.find((s) => s.name === "ignis_request_ttft_seconds_count")?.value).toBe(30);
    expect(parseExposition(IGNIS_EXPOSITION).families.has("ignis_request_ttft_seconds_bucket")).toBe(false);
  });

  it("unescapes label values and accepts a trailing comma and a timestamp", () => {
    const text = [
      String.raw`odd{path="C:\\dir",quote="say \"hi\"",nl="a\nb",} 1 1700000000000`,
    ].join("\n");
    const { families, errors } = parseExposition(text);
    expect(errors).toEqual([]);
    expect(families.get("odd")).toMatchObject({
      type: "untyped",
      samples: [{ labels: { path: "C:\\dir", quote: 'say "hi"', nl: "a\nb" }, value: 1 }],
    });
  });

  it("reads the special and scientific values", () => {
    const { families } = parseExposition("a NaN\nb +Inf\nc -Inf\nd 1.5e3\n");
    expect(families.get("a")?.samples[0].value).toBeNaN();
    expect(families.get("b")?.samples[0].value).toBe(Infinity);
    expect(families.get("c")?.samples[0].value).toBe(-Infinity);
    expect(families.get("d")?.samples[0].value).toBe(1500);
  });

  it("reports malformed lines by number and still parses the rest", () => {
    const text = [
      "# a free comment",
      "",
      'broken{state="x" 1',
      "no_value",
      "bad_value abc",
      "# TYPE lonely",
      "late 1 abc",
      "good 7",
    ].join("\n");
    const { families, errors } = parseExposition(text);
    expect(errors.map((e) => e.line)).toEqual([3, 4, 5, 6, 7]);
    expect(errors[0].text).toBe('broken{state="x" 1');
    expect(families.get("good")?.samples[0].value).toBe(7);
    expect(families.has("broken")).toBe(false);
  });
});
