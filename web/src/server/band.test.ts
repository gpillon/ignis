import { describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import { diskLabel, diskSummary, downloadCommand, elapsed, groupBySlot, rowAction, runningFamily, slotOf, sourceLabel, speechFigures } from "./band.ts";
import { CATALOG } from "./fixture.ts";
import type { CatalogRow } from "./routes.ts";

// The model band's rows (spec playground/05 §Model band): one per catalog
// row, with where it is from, what is on disk, and the one action it offers.

const row = (id: string): CatalogRow => CATALOG.find((r) => r.id === id)!;
const ready: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 786432 };
const switching: ModelState = { state: "switching", from: "qwen3.8-27b", to: "qwen3.8-flash-next", since: 0 };

describe("rowAction", () => {
  it("is Loaded on the loaded row, Load where a switch can begin, Download where the entry is not on disk, and Resume where it is partly", () => {
    expect(rowAction(row("qwen3.8-27b"), ready)).toEqual({ kind: "loaded" });
    expect(rowAction(row("qwen3.8-flash-next"), ready)).toEqual({ kind: "load", enabled: true });
    expect(rowAction(row("qwen3.8-27b-abliterated"), ready)).toEqual({ kind: "resume", enabled: true });
    expect(rowAction(row("qwen3.8-27b-custom"), ready)).toEqual({ kind: "download", enabled: true });
  });

  // GitHub #377: a partly downloaded entry, a cancelled download's included,
  // offers Resume, out of reach exactly where Download is; a running one is
  // the row that offers Cancel.
  it("offers Resume on a partial row, out of reach while a download runs or no model serves, and the command where the page downloads nothing", () => {
    const partial = row("qwen3.8-27b-abliterated");
    expect(rowAction(partial, ready, { fromPage: true, busy: true })).toEqual({ kind: "resume", enabled: false });
    expect(rowAction(partial, switching)).toEqual({ kind: "resume", enabled: false });
    expect(rowAction(partial, { state: "error", message: "x", failed: true })).toEqual({ kind: "resume", enabled: false });
    expect(rowAction({ ...partial, download: "failed", download_reason: "connection reset" }, ready)).toEqual({ kind: "resume", enabled: true });
    expect(rowAction(partial, ready, { fromPage: false, busy: false })).toEqual({ kind: "copy" });
    expect(rowAction({ ...row("qwen3.8-27b-custom"), on_disk: "partial", bytes_on_disk: 4097 }, ready)).toEqual({ kind: "resume", enabled: true });
    expect(rowAction({ ...partial, download: "running" }, ready)).toEqual({ kind: "downloading" });
  });

  // GitHub #376: one download at a time, followed on its row; where the
  // server downloads nothing from the page, the row copies the command.
  it("shows a running download on its row and keeps every other Download out of reach meanwhile", () => {
    const busy = { fromPage: true, busy: true };
    expect(rowAction({ ...row("qwen3.8-27b-abliterated"), download: "running" }, ready, busy)).toEqual({ kind: "downloading" });
    expect(rowAction(row("qwen3.8-27b-custom"), ready, busy)).toEqual({ kind: "download", enabled: false });
    expect(rowAction(row("qwen3.8-flash-next"), ready, busy)).toEqual({ kind: "load", enabled: true });
  });

  it("offers Download again after a failure, and none while no model serves or a switch runs", () => {
    expect(rowAction({ ...row("qwen3.8-27b-custom"), download: "failed", download_reason: "401" }, ready)).toEqual({ kind: "download", enabled: true });
    expect(rowAction(row("qwen3.8-27b-custom"), switching)).toEqual({ kind: "download", enabled: false });
    expect(rowAction(row("qwen3.8-27b-custom"), { state: "error", message: "x", failed: true })).toEqual({ kind: "download", enabled: false });
    expect(rowAction(row("qwen3.8-27b-custom"), { state: "unloaded", id: "qwen3.8-27b", contextLimit: null })).toEqual({ kind: "download", enabled: true });
  });

  it("copies the download command where download.enabled or download.from_ui is off", () => {
    const off = { fromPage: false, busy: false };
    expect(rowAction(row("qwen3.8-27b-custom"), ready, off)).toEqual({ kind: "copy" });
    expect(rowAction(row("qwen3.8-27b-abliterated"), ready, off)).toEqual({ kind: "copy" });
    expect(rowAction(row("qwen3.8-flash-next"), ready, off)).toEqual({ kind: "load", enabled: true });
  });

  it("switches to an entry its download just finished, and offers no Download for it", () => {
    const landed = { ...row("qwen3.8-27b-custom"), on_disk: "complete" as const, switchable: true, download: "none" as const, bytes_on_disk: 19406949449 };
    expect(rowAction(landed, ready)).toEqual({ kind: "load", enabled: true });
  });

  // ADR 0064: an entry above this card's class offers neither Load nor
  // Download, whatever is on disk; the server's reason is its tooltip.
  it("offers nothing for an entry above the card's class, and gives the server's reason", () => {
    const reason = "qwen3.8-flash-next needs a 32 GB card or larger (its catalog entry's card_class), and this card is of the 16 GB class";
    const above = { ...row("qwen3.8-flash-next"), switchable: false, available: false, reason };
    expect(rowAction(above, ready)).toEqual({ kind: "unavailable", reason });
    expect(rowAction({ ...above, on_disk: "absent" }, ready)).toEqual({ kind: "unavailable", reason });
    expect(rowAction({ ...above, on_disk: "absent" }, ready, { fromPage: false, busy: false })).toEqual({ kind: "unavailable", reason });
    expect(rowAction({ ...above, loaded: true }, ready)).toEqual({ kind: "loaded" });
    expect(rowAction({ ...row("qwen3.8-flash-next"), available: true }, ready)).toEqual({ kind: "load", enabled: true });
  });

  // ADR 0079 (spec #474): an entry above this card's GPU generation is the
  // same as one above its class, whatever the reason says.
  it("offers nothing for an entry above the card's GPU generation, and gives the server's reason", () => {
    const reason =
      "qwen3.8-flash-next needs a card of the Blackwell generation (RTX 50, RTX PRO) or newer (its catalog entry's generation) and a 32 GB card or larger (its catalog entry's card_class), and this card, NVIDIA GeForce RTX 3090 (compute capability 8.6), is of the Ampere generation and of the 24 GB class";
    const above = { ...row("qwen3.8-flash-next"), switchable: false, available: false, reason };
    expect(rowAction(above, ready)).toEqual({ kind: "unavailable", reason });
    expect(rowAction({ ...above, on_disk: "absent" }, ready)).toEqual({ kind: "unavailable", reason });
    expect(rowAction({ ...above, on_disk: "absent" }, ready, { fromPage: false, busy: false })).toEqual({ kind: "unavailable", reason });
  });

  it("offers no download for a known model, which is no catalog entry: its artifact is just missing", () => {
    expect(rowAction({ ...row("qwen3.8-27b-fail"), on_disk: "absent", switchable: false }, ready)).toEqual({ kind: "missing" });
  });

  it("follows the server's `switchable` before what is on disk: a catalog entry the server can switch to is Load", () => {
    expect(rowAction({ ...row("qwen3.8-27b-custom"), on_disk: "absent", switchable: true }, ready)).toEqual({ kind: "load", enabled: true });
    expect(rowAction({ ...row("qwen3.8-27b-custom"), source: "built-in", on_disk: "absent", switchable: true }, ready)).toEqual({ kind: "load", enabled: true });
  });

  it("offers Load on every row, the torn-down model's included, while nothing serves", () => {
    const dead: ModelState = { state: "error", message: "Switch failed: x", failed: true };
    const torn = CATALOG.map((r) => ({ ...r, loaded: false, switchable: r.on_disk === "complete" }));
    expect(rowAction(torn[0], dead)).toEqual({ kind: "load", enabled: true });
    expect(torn.map((r) => rowAction(r, dead).kind)).not.toContain("loaded");
  });

  it("marks the model a switch is loading, and offers no Load while one runs", () => {
    expect(rowAction(row("qwen3.8-flash-next"), switching)).toEqual({ kind: "loading" });
    expect(rowAction(row("qwen3.8-27b-fail"), switching)).toEqual({ kind: "load", enabled: false });
    // A failed switch is left by switching again.
    expect(rowAction(row("qwen3.8-27b-fail"), { state: "error", message: "Switch failed: x", failed: true })).toEqual({ kind: "load", enabled: true });
  });
  it("offers Load on the unloaded model, marks it loading while it loads, and offers no Load meanwhile (spec model-switch/02)", () => {
    const unloaded: ModelState = { state: "unloaded", id: "qwen3.8-27b", contextLimit: 786432 };
    const unloadedRow = { ...row("qwen3.8-27b"), loaded: false, switchable: true };
    expect(rowAction(unloadedRow, unloaded)).toEqual({ kind: "load", enabled: true });
    const loading: ModelState = { state: "loading", to: "qwen3.8-27b", since: 0 };
    expect(rowAction(unloadedRow, loading)).toEqual({ kind: "loading" });
    expect(rowAction(row("qwen3.8-flash-next"), loading)).toEqual({ kind: "load", enabled: false });
  });
});

// GitHub #454: a speech model and a diarizer are picked by the configuration (`speech.model`, `diarizer.model`), so
// their rows offer no Load: the one in use says so, and the rest offer what their files need.
describe("rowAction on a speech or diarizer row", () => {
  const asr: CatalogRow = { id: "asr-a", source: "built-in", kind: "speech", on_disk: "complete", bytes: 10, bytes_on_disk: 10, loaded: false, switchable: false, download: "none" };
  const dia: CatalogRow = { ...asr, id: "dia-a", kind: "diarizer" };
  const hearing: ModelState = { ...ready, speech: { available: true, id: "asr-a", diarizer: { id: "dia-a", loaded: true, available: true } } };

  it("says In use on the one the configuration picks while it can serve, and offers nothing on another on disk", () => {
    expect(rowAction(asr, hearing)).toEqual({ kind: "inUse" });
    expect(rowAction(dia, hearing)).toEqual({ kind: "inUse" });
    expect(rowAction({ ...asr, id: "asr-b" }, hearing)).toEqual({ kind: "idle" });
    expect(rowAction({ ...dia, id: "dia-b" }, hearing)).toEqual({ kind: "idle" });
  });

  it("says nothing of use while the picked one cannot serve, which the status bar explains", () => {
    const deaf: ModelState = { ...ready, speech: { available: false, reason: "no files", diarizer: { id: "dia-a", loaded: false, available: false, reason: "no files" } } };
    expect(rowAction(asr, deaf)).toEqual({ kind: "idle" });
    expect(rowAction(dia, deaf)).toEqual({ kind: "idle" });
  });

  it("downloads, resumes and follows its files as any row's", () => {
    expect(rowAction({ ...asr, on_disk: "absent", bytes_on_disk: 0 }, hearing)).toEqual({ kind: "download", enabled: true });
    expect(rowAction({ ...dia, on_disk: "partial", bytes_on_disk: 4 }, hearing)).toEqual({ kind: "resume", enabled: true });
    expect(rowAction({ ...asr, on_disk: "partial", download: "running" }, hearing)).toEqual({ kind: "downloading" });
  });
});

describe("slots", () => {
  it("puts text, image and a row of no kind in the main slot, speech and diarizer rows in their own", () => {
    expect(slotOf(row("qwen3.8-27b"))).toBe("main");
    expect(slotOf({ ...row("qwen3.8-27b"), kind: "image" })).toBe("main");
    expect(slotOf(row("qwen3.8-27b-custom"))).toBe("main");
    expect(slotOf({ ...row("qwen3.8-27b"), kind: "speech" })).toBe("speech");
    expect(slotOf({ ...row("qwen3.8-27b"), kind: "diarizer" })).toBe("diarizer");
  });

  it("groups the rows main, speech, diarizer, each in catalog order, leaving out a slot no row fills", () => {
    const rows: CatalogRow[] = [
      { ...row("qwen3.8-27b"), id: "dia", kind: "diarizer" },
      row("qwen3.8-27b"),
      { ...row("qwen3.8-27b"), id: "img", kind: "image" },
      { ...row("qwen3.8-27b"), id: "asr", kind: "speech" },
    ];
    expect(groupBySlot(rows).map(({ slot, rows }) => [slot, rows.map((r) => r.id)])).toEqual([
      ["main", ["qwen3.8-27b", "img"]],
      ["speech", ["asr"]],
      ["diarizer", ["dia"]],
    ]);
    expect(groupBySlot([row("qwen3.8-27b")]).map(({ slot }) => slot)).toEqual(["main"]);
  });
});

describe("diskSummary", () => {
  it("totals what the listed models take on disk, and the room left where the server downloads them", () => {
    const rows: CatalogRow[] = [
      { ...row("qwen3.8-27b"), bytes_on_disk: 2 * 1024 ** 3 },
      { ...row("qwen3.8-27b"), id: "b", bytes_on_disk: 1024 ** 3 / 2 },
      { ...row("qwen3.8-27b"), id: "c", bytes_on_disk: undefined },
    ];
    expect(diskSummary(rows, { path: "/srv/models", free_bytes: 412 * 1024 ** 3 })).toBe("The listed models take 2.5 GiB. 412 GiB free under /srv/models.");
    // An older server says no free space.
    expect(diskSummary(rows, null)).toBe("The listed models take 2.5 GiB.");
  });
});

describe("row labels", () => {
  it("says where a model is from and what of it is on disk, in GiB", () => {
    expect(sourceLabel("built-in")).toBe("Built-in catalog");
    expect(sourceLabel("operator")).toBe("Operator catalog");
    expect(sourceLabel("known")).toBe("Known model");
    expect(diskLabel(row("qwen3.8-27b"))).toBe("On disk, 18.07 GiB");
    expect(diskLabel(row("qwen3.8-flash-next"))).toBe("On disk, 67.8 GiB");
    expect(diskLabel(row("qwen3.8-27b-abliterated"))).toBe("Partly downloaded, 6 GiB of 18.07 GiB");
    expect(diskLabel(row("qwen3.8-27b-custom"))).toBe("Not downloaded, 18.07 GiB");
    expect(diskLabel({ id: "x", source: "known", on_disk: "absent", loaded: false, switchable: false })).toBe("Artifact not found");
  });

  // GitHub #377: a partial row says how much is already there, beside its total.
  it("gives a partial row's bytes on disk beside its total, and the total alone from a server that sends none", () => {
    expect(diskLabel({ ...row("qwen3.8-27b-abliterated"), bytes_on_disk: 4097 })).toBe("Partly downloaded, 4 KiB of 18.07 GiB");
    expect(diskLabel({ ...row("qwen3.8-27b-abliterated"), bytes_on_disk: undefined })).toBe("Partly downloaded, 18.07 GiB in all");
    expect(diskLabel({ ...row("qwen3.8-27b-abliterated"), bytes: undefined })).toBe("Partly downloaded, 6 GiB");
  });

  it("copies the command that fetches a model", () => {
    expect(downloadCommand("qwen3.8-27b-abliterated")).toBe("ignis-server model download qwen3.8-27b-abliterated");
  });
});

describe("runningFamily", () => {
  it("is the loaded row's family, or unknown without a catalog", () => {
    expect(runningFamily(CATALOG)).toBe("qwen38");
    expect(runningFamily(null)).toBeNull();
  });
});

describe("elapsed", () => {
  it("counts whole seconds since the switch was first seen", () => {
    expect(elapsed(1_000, 1_400)).toBe("0 s");
    expect(elapsed(1_000, 13_200)).toBe("12 s");
    expect(elapsed(0, 72_000)).toBe("1m 12s");
  });
});

// Spec speech/08: the diarizer beside the speech model, its id, whether it is
// loaded, and its reason when unavailable.
describe("speechFigures", () => {
  it("shows the speech model and the diarizer beside it, loaded or not", () => {
    const diarizer = { id: "nemotron-3-diarization", loaded: true, available: true };
    expect(speechFigures({ available: true, id: "qwen3-asr-1.7b", diarizer })).toEqual([
      { label: "Speech", value: "qwen3-asr-1.7b", state: "Available", dot: "var(--good)" },
      { label: "Diarizer", value: "nemotron-3-diarization", state: "Loaded", dot: "var(--good)" },
    ]);
    const idle = speechFigures({ available: true, id: "qwen3-asr-1.7b", diarizer: { ...diarizer, loaded: false } });
    expect(idle[1]).toEqual({ label: "Diarizer", value: "nemotron-3-diarization", state: "Not loaded", dot: "var(--ash)" });
  });

  it("says why either cannot serve, and names no diarizer where none is named", () => {
    const missing = "nemotron-3-diarization is not on disk: `ignis-server model download nemotron-3-diarization`";
    const [speech, diarizer] = speechFigures({
      available: false,
      reason: "the speech model is turned off",
      diarizer: { id: "nemotron-3-diarization", loaded: false, available: false, reason: missing },
    });
    expect(speech).toEqual({ label: "Speech", value: "Unavailable", state: "Unavailable", dot: "var(--ash)", reason: "the speech model is turned off" });
    expect(diarizer).toEqual({ label: "Diarizer", value: "nemotron-3-diarization", state: "Unavailable", dot: "var(--ash)", reason: missing });
    const off = speechFigures({ available: true, id: "qwen3-asr-1.7b", diarizer: { loaded: false, available: false, reason: "turned off" } });
    expect(off[1]).toMatchObject({ value: "None", state: "Unavailable", reason: "turned off" });
  });

  it("shows nothing of what an older server does not report", () => {
    expect(speechFigures(undefined)).toEqual([]);
    expect(speechFigures({ available: true, id: "qwen3-asr-1.7b" })).toHaveLength(1);
  });
});
