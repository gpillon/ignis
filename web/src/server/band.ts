import { canSend, loadUnderWay, type ModelState, type SpeechStatus } from "../api/model.ts";
import { formatBytes, formatSeconds } from "../monitor/format.ts";
import { type CatalogRow, unavailableReason } from "./routes.ts";

// The model band's rows (spec playground/05 §Model band): one per
// GET /v1/models/catalog row, switched to by id alone. A catalog entry not on
// disk is downloaded onto the server from its row (GitHub #376); where the
// server downloads nothing from the page, the row copies the command that
// fetches it. A running download is cancelled from its row, and a partial
// entry resumed from it (GitHub #377).

const SOURCES: Record<CatalogRow["source"], string> = {
  "built-in": "Built-in catalog",
  operator: "Operator catalog",
  known: "Known model",
};

export function sourceLabel(source: CatalogRow["source"]): string {
  return SOURCES[source] ?? source;
}

const KINDS: Record<NonNullable<CatalogRow["kind"]>, string> = {
  text: "Text",
  image: "Image",
  speech: "Speech",
  diarizer: "Diarizer",
};

/** The model kind a row's Type column shows (GitHub #448); empty where the server tells none. */
export function kindLabel(kind: CatalogRow["kind"]): string {
  return kind === undefined ? "" : (KINDS[kind] ?? kind);
}

/** What of a model is on disk; an entry's size is its total, whatever is there. */
export function diskLabel(row: CatalogRow): string {
  const size = row.bytes === undefined ? "" : formatBytes(row.bytes);
  if (row.on_disk === "complete") return size ? `On disk, ${size}` : "On disk";
  // GitHub #377: what of a partial entry is there already, beside its total.
  if (row.on_disk === "partial" && row.bytes_on_disk !== undefined) {
    const there = formatBytes(row.bytes_on_disk);
    return size ? `Partly downloaded, ${there} of ${size}` : `Partly downloaded, ${there}`;
  }
  if (row.on_disk === "partial") return size ? `Partly downloaded, ${size} in all` : "Partly downloaded";
  // A known model is a path, not a catalog entry: there is nothing to download.
  if (row.source === "known") return "Artifact not found";
  return size ? `Not downloaded, ${size}` : "Not downloaded";
}

export function downloadCommand(id: string): string {
  return `ignis-server model download ${id}`;
}

export type RowAction =
  | { kind: "loaded" }
  /** The model the switch under way is loading. */
  | { kind: "loading" }
  | { kind: "load"; enabled: boolean }
  /** A catalog entry not on disk: download it onto the server. */
  | { kind: "download"; enabled: boolean }
  /** A catalog entry partly on disk, a cancelled download's included: download the rest, resuming what is there (GitHub #377). */
  | { kind: "resume"; enabled: boolean }
  /** The same where the server downloads nothing from the page: copy `model download`. */
  | { kind: "copy" }
  /** Its download runs now. */
  | { kind: "downloading" }
  /** A known model whose artifact is not there. */
  | { kind: "missing" }
  /** A catalog entry above this card's class (ADR 0064): neither switched to nor downloaded; `reason` is the server's. */
  | { kind: "unavailable"; reason: string };

/**
 * What a row's Download or Remove depends on beyond the row: whether the server takes it from the page, and whether
 * one is busy — for Download a download running or being sent (one at a time), for Remove a removal being sent.
 */
export type PageGate = { fromPage: boolean; busy: boolean };

/**
 * The row's one action. "Load" follows the server's `switchable`, and is out
 * of reach while a switch or a load runs whatever the last catalog read said;
 * a failed switch is left by switching again, so it stays in reach then, and
 * so does an unloaded model (spec model-switch/02), which "Load" loads.
 * "Download" is out of reach while any download runs, and wherever the
 * server would refuse it for its model's state: no model serving, or one on
 * its way. "Resume" is the same request, on a partial entry, and is out of
 * reach where "Download" is.
 */
export function rowAction(row: CatalogRow, model: ModelState, gate: PageGate = { fromPage: true, busy: false }): RowAction {
  const loading = loadUnderWay(model);
  if (loading && row.id === model.to) return { kind: "loading" };
  if (row.loaded) return { kind: "loaded" };
  // The card's class decides before anything on disk: such an entry is never
  // switchable, and `model download` refuses it.
  const unavailable = unavailableReason(row);
  if (unavailable !== null) return { kind: "unavailable", reason: unavailable };
  if (row.download === "running") return { kind: "downloading" };
  // `switchable` decides first: the server knows a switch by id alone can begin
  // (a catalog entry fetched since the last read, the model torn down by a
  // failed restore), whatever `on_disk` last said.
  if (row.switchable || row.on_disk === "complete") {
    return { kind: "load", enabled: row.switchable && !loading && model.state !== "reading" };
  }
  if (row.source === "known") return { kind: "missing" };
  if (row.on_disk === "partial" && gate.fromPage) return { kind: "resume", enabled: !gate.busy && canSend(model) };
  return gate.fromPage ? { kind: "download", enabled: !gate.busy && canSend(model) } : { kind: "copy" };
}

/** The family the loaded model runs as (its scope name), from the catalog's loaded row. */
export function runningFamily(catalog: CatalogRow[] | null): string | null {
  return catalog?.find((row) => row.loaded)?.family ?? null;
}

/** A model loaded beside the one that serves: its id, its state, and why it cannot serve while it cannot. */
export type SpeechFigure = { label: string; value: string; state: string; dot: string; reason?: string };

/**
 * The speech model and the diarizer beside it (spec speech/08), from `GET /v1/models`' `speech`: no row of the
 * catalog loads either, so the band says what they are and whether they can serve. An older server reports neither,
 * or no diarizer.
 */
export function speechFigures(speech: SpeechStatus | undefined): SpeechFigure[] {
  if (!speech) return [];
  const reason = (text: string | undefined) => (text ? { reason: text } : {});
  const figures: SpeechFigure[] = [
    speech.available
      ? { label: "Speech", value: speech.id ?? "Available", state: "Available", dot: "var(--good)" }
      : { label: "Speech", value: "Unavailable", state: "Unavailable", dot: "var(--ash)", ...reason(speech.reason) },
  ];
  const diarizer = speech.diarizer;
  if (diarizer) {
    const value = diarizer.id ?? "None";
    figures.push(
      !diarizer.available
        ? { label: "Diarizer", value, state: "Unavailable", dot: "var(--ash)", ...reason(diarizer.reason) }
        : diarizer.loaded
          ? { label: "Diarizer", value, state: "Loaded", dot: "var(--good)" }
          : { label: "Diarizer", value, state: "Not loaded", dot: "var(--ash)" },
    );
  }
  return figures;
}

/** Whole seconds since a switch was first seen: `12 s`, `1m 12s`. */
export function elapsed(since: number, now: number): string {
  const seconds = Math.max(0, Math.floor((now - since) / 1000));
  return seconds < 60 ? `${seconds} s` : formatSeconds(seconds);
}
