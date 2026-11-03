import { Fragment, type ReactNode, useState } from "react";
import type { ModelState } from "../api/model.ts";
import { diskLabel, diskSummary, downloadCommand, groupBySlot, kindLabel, type PageGate, rowAction, type Slot, sourceLabel } from "./band.ts";
import { cancelReachable, downloadFailure, progressLabel } from "./download.ts";
import type { Line } from "./outcome.ts";
import { removalPrompt, removeAction } from "./removal.ts";
import type { CatalogDisk, CatalogRow, ConfigDocument } from "./routes.ts";
import type { Loaded } from "./useServer.ts";

// The Server view's Models tab (spec playground/05 §Model band, GitHub #454):
// one row per model the server can switch to or fetch, grouped by the engine
// it is for, under a line that totals what the listed models take on disk and
// the room left where the server downloads them. The status bar above the
// tabs says what runs now; a refused Load, Download or Remove is said here,
// beside the rows that sent it.

const TONE: Record<Line["tone"], string> = { done: "text-good", warn: "text-warn", fault: "text-fault" };

export function Models({
  model,
  catalog,
  disk,
  loadingId,
  refusal,
  onLoad,
  downloads,
  removals,
  removal,
}: {
  model: ModelState;
  catalog: Loaded<CatalogRow[]>;
  /** Where the server downloads models and the room left there; unknown from an older server. */
  disk: CatalogDisk | null;
  loadingId: string | null;
  refusal: string | null;
  onLoad: (id: string) => void;
  /** What a row needs to offer Download and follow it, and to cancel it or resume it (GitHub #376, #377). */
  downloads: RowDownloads;
  /** What a row needs to offer Remove and confirm it (GitHub #380). */
  removals: RowRemovals;
  /** The last removal's line: what it freed, or why the server refused it. */
  removal: Line | null;
}) {
  return (
    <div className="flex flex-col gap-3">
      {catalog.state === "ok" && <p className="text-sm text-ash">{diskSummary(catalog.value, disk)}</p>}
      {(refusal || removal) && (
        <div className="flex flex-col gap-1 text-sm" aria-live="polite">
          {refusal && <p className="text-fault [overflow-wrap:anywhere]">{refusal}</p>}
          {removal && <p className={`${TONE[removal.tone]} [overflow-wrap:anywhere]`}>{removal.text}</p>}
        </div>
      )}
      <div className="cut bg-surface [--cut-size:14px]">
        <Catalog catalog={catalog} model={model} loadingId={loadingId} onLoad={onLoad} downloads={downloads} removals={removals} />
      </div>
    </div>
  );
}

/** Each slot's heading, and what decides the model in it. */
const SLOTS: Record<Slot, { title: string; note: ReactNode }> = {
  main: { title: "Main model", note: "Text or image, one at a time: Load switches the server to it." },
  speech: {
    title: "Speech",
    note: (
      <>
        Transcriptions use the one <code className="font-mono text-[0.95em] text-ink">speech.model</code> names, in Configuration.
      </>
    ),
  },
  diarizer: {
    title: "Diarizer",
    note: (
      <>
        Live sessions tell speakers apart with the one <code className="font-mono text-[0.95em] text-ink">diarizer.model</code> names, in Configuration.
      </>
    ),
  },
};

/** What a row needs to offer Download and follow it, and to cancel it or resume it (GitHub #377). */
type RowDownloads = {
  /** Whether the server downloads from the page, and whether a download runs. */
  gate: PageGate;
  /** The id whose Download is being sent. */
  sendingId: string | null;
  /** The running download's bytes per second, once read. */
  speed: number | null;
  onDownload: (id: string) => void;
  /** A "Cancel" of the running download is being sent. */
  cancelling: boolean;
  onCancel: () => void;
};

/** What a row needs to offer Remove and confirm it (GitHub #380). */
type RowRemovals = {
  gate: PageGate;
  /** The id whose removal is being sent. */
  removingId: string | null;
  /** The id whose Remove waits for its confirmation. */
  confirming: string | null;
  /** The running configuration, which the confirmation reads `download.enabled` from. */
  config: ConfigDocument | null;
  onAsk: (id: string) => void;
  onConfirm: (id: string) => void;
  onKeep: () => void;
};

function Catalog({
  catalog,
  model,
  loadingId,
  onLoad,
  downloads,
  removals,
}: {
  catalog: Loaded<CatalogRow[]>;
  model: ModelState;
  loadingId: string | null;
  onLoad: (id: string) => void;
  downloads: RowDownloads;
  removals: RowRemovals;
}) {
  if (catalog.state !== "ok") {
    return (
      <p className="px-5 py-3 text-sm text-ash md:px-6">
        {catalog.state === "loading" ? "Reading the models this server can load…" : `The models this server can load could not be read: ${catalog.refusal.message}`}
      </p>
    );
  }
  // GitHub #448: from the sm breakpoint the rows share the list's columns
  // (subgrid), so an action's width no longer moves the others; below it each
  // row stacks its cells beside its actions, and the heading row is hidden.
  // GitHub #454: a slot's heading spans the columns, so they line up across slots.
  return (
    <ul aria-label="Models this server can load" className="sm:grid sm:grid-cols-[minmax(0,1.5fr)_auto_minmax(0,1fr)_minmax(0,1.3fr)_auto] sm:gap-x-4">
      <li aria-hidden className="hidden px-5 pt-3 pb-1 text-[11px] text-ash sm:col-span-full sm:grid sm:grid-cols-subgrid md:px-6">
        <span>Model</span>
        <span>Type</span>
        <span>Source</span>
        <span>On disk</span>
      </li>
      {groupBySlot(catalog.value).map(({ slot, rows }) => (
        <Fragment key={slot}>
          <li className="border-t border-line px-5 pt-5 pb-2 sm:col-span-full md:px-6">
            <h3 className="font-display text-[15px] font-semibold text-ink">{SLOTS[slot].title}</h3>
            <p className="text-xs text-ash">{SLOTS[slot].note}</p>
          </li>
          {rows.map((row) => (
        <li
          key={row.id}
          className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-0.5 border-t border-line px-5 py-3 sm:col-span-full sm:grid-cols-subgrid md:px-6"
        >
          <div className="flex flex-col">
            <span className="font-display text-base font-semibold text-ink [overflow-wrap:anywhere]">{row.id}</span>
            {row.license && <LicenseLink license={row.license} />}
          </div>
          <span className="col-start-1 text-xs text-ash sm:col-start-auto sm:text-sm">{kindLabel(row.kind)}</span>
          <span className="col-start-1 text-xs text-ash sm:col-start-auto sm:text-sm">{sourceLabel(row.source)}</span>
          <span className="col-start-1 text-xs text-ash sm:col-start-auto sm:text-sm">
            {row.download === "running" ? progressLabel(row, downloads.speed) : diskLabel(row)}
          </span>
          <div className="col-start-2 row-span-4 row-start-1 flex flex-wrap items-center justify-end gap-x-1 gap-y-1 justify-self-end sm:col-start-auto sm:row-span-1 sm:row-start-auto">
            <RowActionView row={row} model={model} loadingId={loadingId} onLoad={onLoad} downloads={downloads} />
            <RemoveButton row={row} model={model} removals={removals} />
          </div>
          {row.license && (row.download === "running" || downloads.sendingId === row.id) && (
            <p className="col-span-full pt-1 text-xs text-ash">
              Downloading it, you are bound by its license, <LicenseLink license={row.license} />.
            </p>
          )}
          {downloadFailure(row) !== null && (
            <p className="col-span-full pt-1 text-xs text-fault [overflow-wrap:anywhere]">Download failed: {downloadFailure(row)}</p>
          )}
          {removals.confirming === row.id && <RemovalConfirmation row={row} model={model} removals={removals} />}
        </li>
          ))}
        </Fragment>
      ))}
    </ul>
  );
}

/** The license's identifier, linking to its text (ADR 0078): there is no acceptance step, the link is all the page adds. */
function LicenseLink({ license }: { license: NonNullable<CatalogRow["license"]> }) {
  return (
    <a href={license.url} target="_blank" rel="noreferrer" className="text-xs text-ash underline decoration-line underline-offset-2 hover:text-ember">
      {license.id}
    </a>
  );
}

function RowActionView({
  row,
  model,
  loadingId,
  onLoad,
  downloads,
}: {
  row: CatalogRow;
  model: ModelState;
  loadingId: string | null;
  onLoad: (id: string) => void;
  downloads: RowDownloads;
}) {
  const action = rowAction(row, model, downloads.gate);
  switch (action.kind) {
    case "loaded":
      return <span className="font-display text-[13px] font-semibold text-good">Loaded</span>;
    case "loading":
      return <span className="font-display text-[13px] font-semibold text-ember">Loading…</span>;
    case "load":
      return (
        <button
          type="button"
          aria-label={`Load ${row.id}`}
          disabled={!action.enabled || loadingId !== null}
          onClick={() => onLoad(row.id)}
          className="border border-line px-3 py-1.5 font-display text-[13px] font-semibold text-ink hover:border-ember hover:bg-ember/10 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-line disabled:hover:bg-transparent"
        >
          {loadingId === row.id ? "Sending…" : "Load"}
        </button>
      );
    case "download":
    case "resume": {
      // Resume is the same request on a partial entry: the server resumes what is there.
      const verb = action.kind === "resume" ? "Resume" : "Download";
      return (
        <button
          type="button"
          aria-label={`${verb} ${row.id}`}
          disabled={!action.enabled || downloads.sendingId !== null}
          onClick={() => downloads.onDownload(row.id)}
          className="border border-line px-3 py-1.5 font-display text-[13px] font-semibold text-ink hover:border-ember hover:bg-ember/10 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-line disabled:hover:bg-transparent"
        >
          {downloads.sendingId === row.id ? "Sending…" : verb}
        </button>
      );
    }
    case "copy":
      return <CopyCommand id={row.id} />;
    case "downloading":
      // Its progress is in the disk column; the action stops it, keeping what it fetched (GitHub #377), during a switch too.
      return (
        <button
          type="button"
          aria-label={`Cancel the download of ${row.id}`}
          disabled={downloads.cancelling || !cancelReachable(model)}
          onClick={downloads.onCancel}
          className="border border-line px-3 py-1.5 font-display text-[13px] font-semibold text-ember hover:border-ember hover:bg-ember/10 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-line disabled:hover:bg-transparent"
        >
          {downloads.cancelling ? "Cancelling…" : "Cancel"}
        </button>
      );
    case "inUse":
      return <span className="font-display text-[13px] font-semibold text-good">In use</span>;
    case "missing":
    case "idle":
      return null;
    case "unavailable":
      return (
        <span title={action.reason} className="font-display text-[13px] font-semibold text-ash">
          Not for this card
        </span>
      );
  }
}

/** Remove beside the row's own action (GitHub #380): it asks first, in the row. */
function RemoveButton({ row, model, removals }: { row: CatalogRow; model: ModelState; removals: RowRemovals }) {
  const action = removeAction(row, model, removals.gate);
  if (action === null) return null;
  return (
    <button
      type="button"
      aria-label={`Remove ${row.id}`}
      disabled={!action.enabled}
      onClick={() => removals.onAsk(row.id)}
      className="px-3 py-1.5 font-display text-[13px] font-semibold text-ash hover:bg-line/60 hover:text-fault disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-ash"
    >
      {removals.removingId === row.id ? "Removing…" : "Remove"}
    </button>
  );
}

/** The confirmation under the row: what goes, what the entry is to this server, and the two answers. */
function RemovalConfirmation({ row, model, removals }: { row: CatalogRow; model: ModelState; removals: RowRemovals }) {
  const action = removeAction(row, model, removals.gate);
  if (action === null) return null;
  return (
    <div role="group" aria-label={`Confirm the removal of ${row.id}`} className="col-span-full mt-2 flex flex-col gap-1.5 border-l-2 border-fault pl-3 text-xs sm:text-sm">
      {removalPrompt(row, removals.config).map((line) => (
        <p key={line} className="text-ink [overflow-wrap:anywhere]">
          {line}
        </p>
      ))}
      <div className="flex items-center gap-4 font-display text-[13px]">
        <button
          type="button"
          aria-label={`Remove ${row.id} for good`}
          disabled={!action.enabled}
          onClick={() => removals.onConfirm(row.id)}
          className="font-semibold text-fault hover:underline disabled:cursor-not-allowed disabled:opacity-40"
        >
          Remove
        </button>
        <button type="button" onClick={removals.onKeep} className="text-ash hover:text-ink">
          Keep it
        </button>
      </div>
    </div>
  );
}

/** Copies `model download <id>`; where the clipboard is out of reach (a page not on https or localhost) it shows the command to select instead. */
function CopyCommand({ id }: { id: string }) {
  const [copy, setCopy] = useState<"idle" | "copied" | "failed">("idle");
  const command = downloadCommand(id);
  if (copy === "failed") return <code className="block font-mono text-xs break-all text-ink select-all">{command}</code>;
  return (
    <button
      type="button"
      aria-label={`Copy download command for ${id}`}
      onClick={() => {
        const written = navigator.clipboard?.writeText(command);
        if (!written) return setCopy("failed");
        written.then(
          () => {
            setCopy("copied");
            setTimeout(() => setCopy("idle"), 2000);
          },
          () => setCopy("failed"),
        );
      }}
      className="px-3 py-1.5 font-display text-[13px] font-semibold text-ash hover:bg-line/60 hover:text-ink"
    >
      {copy === "copied" ? "Copied" : "Copy download command"}
    </button>
  );
}
