import { useState } from "react";
import { loadUnderWay, type ModelState } from "../api/model.ts";
import { diskLabel, downloadCommand, elapsed, kindLabel, type PageGate, rowAction, runningFamily, sourceLabel, speechFigures } from "./band.ts";
import { cancelReachable, downloadFailure, progressLabel } from "./download.ts";
import { familyName } from "./form.ts";
import { drainNote, lastSwitchLine, type Line } from "./outcome.ts";
import { removalPrompt, removeAction } from "./removal.ts";
import type { CatalogRow, ConfigDocument } from "./routes.ts";
import type { Loaded } from "./useServer.ts";

// The model band (spec playground/05 §Model band): the loaded model on the
// kiln palette the Monitor's band uses in both schemes, how the last switch
// ended, and one row per model the server can switch to or fetch. The only
// motion of a switch is the header's flame and heat bar, run by the page's
// shared model state; the band says what is happening in words.

const TONE: Record<Line["tone"], string> = { done: "text-good", warn: "text-warn", fault: "text-fault" };

export function ModelBand({
  model,
  catalog,
  loadingId,
  refusal,
  streaming,
  drain,
  now,
  onLoad,
  downloads,
  removals,
  removal,
}: {
  model: ModelState;
  catalog: Loaded<CatalogRow[]>;
  loadingId: string | null;
  refusal: string | null;
  /** A chat reply is streaming: say what a switch does to it. */
  streaming: boolean;
  /** `switch.drain_timeout` in force, when the configuration was read. */
  drain: number | null;
  now: number;
  onLoad: (id: string) => void;
  /** What a row needs to offer Download and follow it, and to cancel it or resume it (GitHub #376, #377). */
  downloads: RowDownloads;
  /** What a row needs to offer Remove and confirm it (GitHub #380). */
  removals: RowRemovals;
  /** The last removal's line: what it freed, or why the server refused it. */
  removal: Line | null;
}) {
  const rows = catalog.state === "ok" ? catalog.value : null;
  const family = runningFamily(rows);
  // Spec speech/08: the speech model and the diarizer beside the loaded model.
  const beside = speechFigures(model.speech);
  const besideReasons = beside.filter((figure) => figure.reason);
  // The last switch's line stays until the next switch begins.
  const last =
    model.state !== "reading" && !loadUnderWay(model) && model.lastSwitch ? lastSwitchLine(model.lastSwitch) : null;
  return (
    <section aria-label="Model" className="monitor-kiln cut overflow-hidden text-[#eae8e4] [--cut-size:20px]">
      <div className="flex flex-col gap-4 p-5 md:p-6">
        <Headline model={model} />
        <dl className="flex flex-wrap gap-x-8 gap-y-3">
          <Figure label="State" value={stateWord(model)} dot={stateColor(model)} />
          {(model.state === "ready" || model.state === "unloaded") && (
            <Figure label="Context" value={model.contextLimit === null ? "Not reported" : `${model.contextLimit.toLocaleString("en-US")} tokens`} />
          )}
          {model.state === "ready" && family && <Figure label="Family" value={familyName(family)} />}
          {loadUnderWay(model) && <Figure label="Elapsed" value={elapsed(model.since, now)} />}
          {beside.map((figure) => (
            <Figure key={figure.label} label={figure.label} value={figure.value} dot={figure.dot} detail={figure.state} />
          ))}
        </dl>
        {besideReasons.length > 0 && (
          <div className="flex flex-col gap-1 text-sm text-[#b9bec4]">
            {besideReasons.map((figure) => (
              <p key={figure.label} className="[overflow-wrap:anywhere]">
                {figure.label}: {figure.reason}
              </p>
            ))}
          </div>
        )}
        {(last || refusal || removal || streaming || model.state === "error" || model.state === "unloaded") && (
          <div className="flex flex-col gap-1 text-sm" aria-live="polite">
            {model.state === "error" && <p className="text-fault [overflow-wrap:anywhere]">{model.message}</p>}
            {model.state === "unloaded" && (
              <p className="text-[#b9bec4]">Unloaded after going unused (model.idle_unload). The next message loads it again; Load does it now.</p>
            )}
            {last && <p className={`${TONE[last.tone]} [overflow-wrap:anywhere]`}>{last.text}</p>}
            {refusal && <p className="text-fault [overflow-wrap:anywhere]">{refusal}</p>}
            {removal && <p className={`${TONE[removal.tone]} [overflow-wrap:anywhere]`}>{removal.text}</p>}
            {streaming && <p className="text-[#b9bec4]">{drainNote(drain, model.state === "switching" ? "The switch" : "A switch")}</p>}
          </div>
        )}
      </div>
      <Catalog
        catalog={catalog}
        model={model}
        loadingId={loadingId}
        onLoad={onLoad}
        downloads={downloads}
        removals={removals}
      />
    </section>
  );
}

function Headline({ model }: { model: ModelState }) {
  if (model.state === "switching") {
    return (
      <h2 className="font-display text-2xl leading-tight font-semibold text-white [overflow-wrap:anywhere] sm:text-[30px]">
        Switching from <span className="text-[#b9bec4]">{model.from}</span> to {model.to}
      </h2>
    );
  }
  if (model.state === "loading") {
    return (
      <h2 className="font-display text-2xl leading-tight font-semibold text-white [overflow-wrap:anywhere] sm:text-[30px]">
        Loading {model.to}
      </h2>
    );
  }
  // A reported failure lasts while the old model is loaded again, or until a
  // switch succeeds: either way nothing serves meanwhile (a refused switch
  // reads `failed` for one tick only).
  const text =
    model.state === "ready" || model.state === "unloaded"
      ? model.id
      : model.state === "reading"
        ? "Reading the model…"
        : model.failed
          ? "No model is serving"
          : "ignis is not answering";
  const muted = model.state !== "ready";
  return (
    <h2
      className={`font-display leading-none font-semibold [overflow-wrap:anywhere] ${muted ? "text-2xl text-[#b9bec4] sm:text-[30px]" : "text-[32px] text-white sm:text-[44px]"}`}
    >
      {text}
    </h2>
  );
}

function stateWord(model: ModelState): string {
  if (model.state === "ready") return "Serving";
  if (model.state === "switching") return "Switching";
  if (model.state === "loading") return "Loading";
  if (model.state === "unloaded") return "Unloaded";
  if (model.state === "reading") return "Reading";
  return model.failed ? "Not serving" : "Unreachable";
}

function stateColor(model: ModelState): string {
  if (model.state === "ready") return "var(--good)";
  if (loadUnderWay(model)) return "#ff5a1f";
  return model.state === "reading" || model.state === "unloaded" ? "var(--ash)" : "var(--fault)";
}

function Figure({ label, value, dot, detail }: { label: string; value: string; dot?: string; detail?: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] text-[#939ba4]">{label}</dt>
      <dd className="flex items-center gap-2 font-display text-lg leading-tight font-semibold text-[#eae8e4] [overflow-wrap:anywhere]">
        {dot && <span className="cut size-2.5 shrink-0 [--cut-size:4px]" style={{ background: dot }} aria-hidden />}
        {value}
      </dd>
      {detail && <dd className="text-xs text-[#939ba4]">{detail}</dd>}
    </div>
  );
}

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
      <p className="border-t border-kiln-line px-5 py-3 text-sm text-[#939ba4] md:px-6">
        {catalog.state === "loading" ? "Reading the models this server can load…" : `The models this server can load could not be read: ${catalog.refusal.message}`}
      </p>
    );
  }
  // GitHub #448: from the sm breakpoint the rows share the list's columns
  // (subgrid), so an action's width no longer moves the others; below it each
  // row stacks its cells beside its actions, and the heading row is hidden.
  return (
    <ul aria-label="Models this server can load" className="sm:grid sm:grid-cols-[minmax(0,1.5fr)_auto_minmax(0,1fr)_minmax(0,1.3fr)_auto] sm:gap-x-4">
      <li aria-hidden className="hidden border-t border-kiln-line px-5 pt-3 pb-1 text-[11px] text-[#939ba4] sm:col-span-full sm:grid sm:grid-cols-subgrid md:px-6">
        <span>Model</span>
        <span>Type</span>
        <span>Source</span>
        <span>On disk</span>
      </li>
      {catalog.value.map((row) => (
        <li
          key={row.id}
          className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-0.5 border-t border-kiln-line px-5 py-3 sm:col-span-full sm:grid-cols-subgrid md:px-6"
        >
          <span className="font-display text-base font-semibold text-white [overflow-wrap:anywhere]">{row.id}</span>
          <span className="col-start-1 text-xs text-[#939ba4] sm:col-start-auto sm:text-sm">{kindLabel(row.kind)}</span>
          <span className="col-start-1 text-xs text-[#939ba4] sm:col-start-auto sm:text-sm">{sourceLabel(row.source)}</span>
          <span className="col-start-1 text-xs text-[#939ba4] sm:col-start-auto sm:text-sm">
            {row.download === "running" ? progressLabel(row, downloads.speed) : diskLabel(row)}
          </span>
          <div className="col-start-2 row-span-4 row-start-1 flex flex-wrap items-center justify-end gap-x-1 gap-y-1 justify-self-end sm:col-start-auto sm:row-span-1 sm:row-start-auto">
            <RowActionView row={row} model={model} loadingId={loadingId} onLoad={onLoad} downloads={downloads} />
            <RemoveButton row={row} model={model} removals={removals} />
          </div>
          {downloadFailure(row) !== null && (
            <p className="col-span-full pt-1 text-xs text-fault [overflow-wrap:anywhere]">Download failed: {downloadFailure(row)}</p>
          )}
          {removals.confirming === row.id && <RemovalConfirmation row={row} model={model} removals={removals} />}
        </li>
      ))}
    </ul>
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
      return <span className="font-display text-[13px] font-semibold text-[#ff8a4c]">Loading…</span>;
    case "load":
      return (
        <button
          type="button"
          aria-label={`Load ${row.id}`}
          disabled={!action.enabled || loadingId !== null}
          onClick={() => onLoad(row.id)}
          className="border border-[#4a525c] px-3 py-1.5 font-display text-[13px] font-semibold text-white hover:border-[#ff5a1f] hover:bg-[#ff5a1f]/15 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-[#4a525c] disabled:hover:bg-transparent"
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
          className="border border-[#4a525c] px-3 py-1.5 font-display text-[13px] font-semibold text-white hover:border-[#ff5a1f] hover:bg-[#ff5a1f]/15 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-[#4a525c] disabled:hover:bg-transparent"
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
          className="border border-[#4a525c] px-3 py-1.5 font-display text-[13px] font-semibold text-[#ff8a4c] hover:border-[#ff5a1f] hover:bg-[#ff5a1f]/15 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-[#4a525c] disabled:hover:bg-transparent"
        >
          {downloads.cancelling ? "Cancelling…" : "Cancel"}
        </button>
      );
    case "missing":
      return null;
    case "unavailable":
      return (
        <span title={action.reason} className="font-display text-[13px] font-semibold text-[#939ba4]">
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
      className="px-3 py-1.5 font-display text-[13px] font-semibold text-[#b9bec4] hover:bg-kiln-line hover:text-[#ff7b6b] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-[#b9bec4]"
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
    <div role="group" aria-label={`Confirm the removal of ${row.id}`} className="col-span-full mt-2 flex flex-col gap-1.5 border-l-2 border-[#ff7b6b] pl-3 text-xs sm:text-sm">
      {removalPrompt(row, removals.config).map((line) => (
        <p key={line} className="text-[#eae8e4] [overflow-wrap:anywhere]">
          {line}
        </p>
      ))}
      <div className="flex items-center gap-4 font-display text-[13px]">
        <button
          type="button"
          aria-label={`Remove ${row.id} for good`}
          disabled={!action.enabled}
          onClick={() => removals.onConfirm(row.id)}
          className="font-semibold text-[#ff7b6b] hover:text-[#ffa196] disabled:cursor-not-allowed disabled:opacity-40"
        >
          Remove
        </button>
        <button type="button" onClick={removals.onKeep} className="text-[#939ba4] hover:text-[#eae8e4]">
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
  if (copy === "failed") return <code className="block font-mono text-xs break-all text-[#eae8e4] select-all">{command}</code>;
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
      className="px-3 py-1.5 font-display text-[13px] font-semibold text-[#b9bec4] hover:bg-kiln-line hover:text-white"
    >
      {copy === "copied" ? "Copied" : "Copy download command"}
    </button>
  );
}
