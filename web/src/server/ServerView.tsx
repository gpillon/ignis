import { type KeyboardEvent, type ReactNode, useState } from "react";
import { canSend, type ModelState } from "../api/model.ts";
import { IconClose } from "../ui/icons.tsx";
import { ConfigForm } from "./ConfigForm.tsx";
import { actionInk, actionShape } from "../ui/classes.ts";
import { changes, drainTimeout, familiesOf, needsReload, primaryLabel, waitingForOthers, waitingLine } from "./form.ts";
import { Models } from "./Models.tsx";
import { StatusBar } from "./StatusBar.tsx";
import { drainNote, type Line } from "./outcome.ts";
import { runningFamily } from "./band.ts";
import { downloadsFromPage, runningDownload } from "./download.ts";
import { removalsFromPage } from "./removal.ts";
import type { ServerActions, ServerState } from "./useServer.ts";

// The Server view (spec playground/05, GitHub #316): a status bar of what
// each engine runs, which stays put, then two tabs whose panels scroll under
// it (GitHub #454), the models the server can switch to or fetch, and the
// running configuration as a form, and a dock at the foot while the draft
// holds a change. Both panels stay mounted, so a draft, a search and a scope
// outlive a change of tab. It has no side drawers. It renders `state` and
// nothing else; useServer owns it.

export function ServerView({
  model,
  state,
  actions,
  streaming,
  now,
}: {
  model: ModelState;
  state: ServerState;
  actions: ServerActions;
  /** A chat reply is streaming somewhere on the page. */
  streaming: boolean;
  now: number;
}) {
  const config = state.config.state === "ok" ? state.config.value : null;
  const rows = state.catalog.state === "ok" ? state.catalog.value : null;
  const family = runningFamily(rows);
  const drain = drainTimeout(config);
  const count = Object.keys(changes(state.draft)).length;
  const docked = Object.keys(state.draft).length > 0 || state.result !== null || state.pending !== null;
  const [tab, setTab] = useState<ServerTab>("models");
  return (
    <main aria-label="Server" className="flex min-h-0 flex-1 flex-col">
      <div className="mx-auto w-full max-w-[1180px] shrink-0 px-4 pt-5 md:px-6">
        <StatusBar model={model} family={family} streaming={streaming} drain={drain} now={now} />
        <div role="tablist" aria-label="Server sections" onKeyDown={(e) => step(e, setTab)} className="mt-4 flex gap-6 border-b border-line">
          <Tab tab="models" selected={tab} onSelect={setTab}>
            Models
          </Tab>
          <Tab tab="config" selected={tab} onSelect={setTab}>
            Configuration
            {count > 0 && <span className="ml-1.5 text-xs text-ember tabular-nums">{count}</span>}
          </Tab>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1180px] px-4 py-5 md:px-6">
          <div role="tabpanel" id="server-panel-models" aria-labelledby="server-tab-models" hidden={tab !== "models"}>
            <Models
              model={model}
              catalog={state.catalog}
              disk={state.disk}
              loadingId={state.loadingId}
              refusal={state.loadRefusal ?? state.downloadRefusal}
              onLoad={actions.load}
              downloads={{
                gate: { fromPage: downloadsFromPage(config), busy: state.downloadingId !== null || runningDownload(rows) !== null },
                sendingId: state.downloadingId,
                speed: state.downloadSpeed,
                onDownload: actions.download,
                cancelling: state.cancelling,
                onCancel: actions.cancel,
              }}
              removals={{
                gate: { fromPage: removalsFromPage(config), busy: state.removingId !== null },
                removingId: state.removingId,
                confirming: state.confirmingRemoval,
                config,
                onAsk: actions.askRemoval,
                onConfirm: actions.confirmRemoval,
                onKeep: actions.keepModel,
              }}
              removal={state.removal}
            />
          </div>
          <div role="tabpanel" id="server-panel-config" aria-labelledby="server-tab-config" hidden={tab !== "config"}>
            <ConfigForm
              fields={state.fields}
              config={state.config}
              origins={state.origins}
              catalog={state.catalog}
              family={family}
              draft={state.draft}
              marked={state.marked}
              onEdit={actions.edit}
              onRemove={actions.remove}
              onUndo={actions.undo}
            />
          </div>
        </div>
      </div>
      {docked && <Dock model={model} state={state} actions={actions} count={count} streaming={streaming} drain={drain} family={family} />}
    </main>
  );
}

type ServerTab = "models" | "config";
const TABS: ServerTab[] = ["models", "config"];

/** A tab, in the header's look: the selected one in ink over the ember's line. */
function Tab({ tab, selected, onSelect, children }: { tab: ServerTab; selected: ServerTab; onSelect: (tab: ServerTab) => void; children: ReactNode }) {
  const on = tab === selected;
  return (
    <button
      type="button"
      role="tab"
      id={`server-tab-${tab}`}
      aria-selected={on}
      aria-controls={`server-panel-${tab}`}
      tabIndex={on ? 0 : -1}
      onClick={() => onSelect(tab)}
      className={`-mb-px flex items-baseline border-b-2 pb-2.5 font-display text-[15px] font-medium ${on ? "border-ember text-ink" : "border-transparent text-ash hover:text-ink"}`}
    >
      {children}
    </button>
  );
}

/** The arrow keys move between the tabs, as a tab list's do. */
function step(e: KeyboardEvent, onSelect: (tab: ServerTab) => void) {
  const by = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
  if (by === 0) return;
  e.preventDefault();
  const at = TABS.findIndex((tab) => document.getElementById(`server-tab-${tab}`) === document.activeElement);
  const next = TABS[(Math.max(at, 0) + by + TABS.length) % TABS.length];
  onSelect(next);
  document.getElementById(`server-tab-${next}`)?.focus();
}

const TONE: Record<Line["tone"], string> = { done: "text-good", warn: "text-warn", fault: "text-fault" };

/**
 * The dock: how many changes, "Discard", and the action whose label says what
 * it will do — "Apply", or "Apply and reload <id>" in the heat of the
 * header's bar when a changed or removed value reaches the running model. A
 * value that waits for the other model is listed, not counted as a reload.
 * No confirmation: the label is the confirmation. Then the answer, in the
 * spec's words.
 */
function Dock({
  model,
  state,
  actions,
  count,
  streaming,
  drain,
  family,
}: {
  model: ModelState;
  state: ServerState;
  actions: ServerActions;
  count: number;
  streaming: boolean;
  drain: number | null;
  family: string | null;
}) {
  const fields = state.fields.state === "ok" ? state.fields.value : [];
  const config = state.config.state === "ok" ? state.config.value : null;
  const origins = state.origins.state === "ok" ? state.origins.value : null;
  const catalog = state.catalog.state === "ok" ? state.catalog.value : null;
  const basis = { fields, doc: config, origins, family, families: familiesOf(fields, catalog, family) };
  const reload = needsReload(state.draft, basis);
  const later = waitingForOthers(state.draft, basis);
  const id =
    model.state === "ready" || model.state === "unloaded"
      ? model.id
      : model.state === "switching"
        ? model.from
        : model.state === "loading"
          ? model.to
          : (state.pending?.id ?? "the model");
  // Every route but GET answers 503 while a switch runs, so the action waits for it to land.
  const waiting =
    state.pending !== null
      ? `Reloading ${state.pending.id}…`
      : model.state === "switching"
        ? `A switch from ${model.from} to ${model.to} is under way. Apply once it lands.`
        : !canSend(model)
          ? "Apply once a model serves."
          : null;
  const blocked = count === 0 || state.applying || waiting !== null;
  return (
    <div role="region" aria-label="Changes" className="sticky bottom-0 z-10 border-t border-line bg-ground/95 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1180px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 md:px-6">
        <div className="flex min-w-0 flex-1 basis-64 flex-col gap-0.5 text-sm">
          {count > 0 && <p className="font-display font-semibold">{count === 1 ? "1 change" : `${count} changes`}</p>}
          {waiting && <p className="text-ash">{waiting}</p>}
          {later.map((entry) => (
            <p key={entry.family} className="text-ash [overflow-wrap:anywhere]">
              {waitingLine(entry)}
            </p>
          ))}
          {reload && streaming && state.pending === null && <p className="text-warn">{drainNote(drain, "The reload")}</p>}
          {state.result && (
            <ul aria-live="polite" className="flex flex-col gap-0.5">
              {state.result.map((line) => (
                <li key={line.text} className={`${TONE[line.tone]} [overflow-wrap:anywhere]`}>
                  {line.text}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
          {Object.keys(state.draft).length > 0 && (
            <button
              type="button"
              onClick={actions.discard}
              disabled={state.applying || state.pending !== null}
              className="px-3 py-2 font-display text-[13px] font-semibold text-ash hover:text-ink disabled:opacity-40"
            >
              Discard
            </button>
          )}
          {count > 0 && (
            <button
              type="button"
              onClick={actions.apply}
              disabled={blocked}
              className={`${actionShape} px-4 py-2 [--cut-size:8px] ${reload ? "heat-fill hover:brightness-110" : actionInk}`}
            >
              {state.applying ? "Applying…" : primaryLabel(reload, id)}
            </button>
          )}
          {count === 0 && state.result && state.pending === null && (
            <button type="button" aria-label="Dismiss" title="Dismiss" onClick={actions.dismiss} className="grid size-8 place-items-center text-ash hover:bg-surface hover:text-ink">
              <IconClose />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
