import { describe, expect, it } from "vitest";
import { dateTimePrompt } from "./datetime.ts";

describe("dateTimePrompt", () => {
  it("gives the day, date, time and time zone of the moment", () => {
    const prompt = dateTimePrompt(new Date("2026-09-15T01:52:00+02:00"), "Europe/Rome");
    expect(prompt.startsWith("# Date and time\n")).toBe(true);
    for (const part of ["Tuesday", "15 September 2026", "01:52", "Europe/Rome", "GMT+02:00"]) expect(prompt).toContain(part);
  });

  it("follows the time zone it is given", () => {
    const prompt = dateTimePrompt(new Date("2026-09-15T01:52:00+02:00"), "America/New_York");
    for (const part of ["Monday", "14 September 2026", "19:52", "GMT-04:00"]) expect(prompt).toContain(part);
  });
});
