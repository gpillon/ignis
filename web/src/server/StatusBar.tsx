import { type ReactNode, useState } from "react";
import { loadUnderWay, type ModelState } from "../api/model.ts";
import { elapsed, speechFigures } from "./band.ts";
import { familyName } from "./form.ts";
import { drainNote, lastSwitchLine, type Line } from "./outcome.ts";

// The Server view's status bar (spec playground/05 §Model band, GitHub #454):
// what each engine runs now, a slot each (the main model, the speech model,
// the diarizer), on the kiln palette the Monitor's band uses in both schemes,
// and how the last switch ended. It stays put above the tabs while their
// panels scroll. The only motion of a switch is the header's flame and heat
// bar, run by the page's shared model state; the bar says what happens in words.

const TONE: Record<Line["tone"], string> = { done: "text-good", warn: "text-warn", fault: "text-fault" };

export function StatusBar({
  model,
  family,
  streaming,
  drain,
  now,
}: {
  model: ModelState;
  /** The running family's scope name, from the catalog's loaded row. */
  family: string | null;
  /** A chat reply is streaming: say what a switch does to it. */
  streaming: boolean;
  /** `switch.drain_timeout` in force, when the configuration was read. */
  drain: number | null;
  now: number;
}) {
  // Spec speech/08: the speech model and the diarizer beside the loaded model.
  const beside = speechFigures(model.speech);
  // The slots whose reason was asked for with Why.
  const [asked, setAsked] = useState<string[]>([]);
  const reasons = beside.filter((figure) => figure.reason && asked.includes(figure.label));
  // The last switch's line stays until the next switch begins.
  const last =
    model.state !== "reading" && !loadUnderWay(model) && model.lastSwitch ? lastSwitchLine(model.lastSwitch) : null;
  const lines = reasons.length > 0 || last !== null || streaming || model.state === "error" || model.state === "unloaded";
  return (
    <section aria-label="Model" className="monitor-kiln cut text-[#eae8e4] [--cut-size:16px]">
      <div className="flex flex-col gap-4 px-5 py-4 md:flex-row md:items-center md:px-6">
        <div className="min-w-0 flex-1">
          <Headline model={model} />
          <p className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1 text-sm">
            <span className="flex items-center gap-2 font-display font-semibold">
              <Dot color={stateColor(model)} />
              {stateWord(model)}
            </span>
            {(model.state === "ready" || model.state === "unloaded") && (
              <Fact label="Context">{model.contextLimit === null ? "Not reported" : `${model.contextLimit.toLocaleString("en-US")} tokens`}</Fact>
            )}
            {model.state === "ready" && family && <Fact label="Family">{familyName(family)}</Fact>}
            {loadUnderWay(model) && <Fact label="Elapsed">{elapsed(model.since, now)}</Fact>}
          </p>
        </div>
        {beside.length > 0 && (
          <dl className="flex flex-wrap gap-x-8 gap-y-3 md:flex-nowrap md:gap-x-0">
            {beside.map((figure) => {
              const open = asked.includes(figure.label);
              return (
                <div key={figure.label} className="min-w-0 md:border-l md:border-kiln-line md:px-6 md:last:pr-0">
                  <dt className="text-[11px] text-[#939ba4]">{figure.label}</dt>
                  <dd className="flex items-center gap-2 font-display text-base leading-tight font-semibold [overflow-wrap:anywhere]">
                    <Dot color={figure.dot} />
                    {figure.value}
                  </dd>
                  <dd className="flex items-baseline gap-2 text-xs text-[#939ba4]">
                    {figure.state}
                    {figure.reason && (
                      <button
                        type="button"
                        aria-expanded={open}
                        aria-controls={`server-why-${figure.label}`}
                        title={figure.reason}
                        onClick={() => setAsked((now) => (open ? now.filter((label) => label !== figure.label) : [...now, figure.label]))}
                        className="text-[#ff8a4c] hover:underline"
                      >
                        Why
                      </button>
                    )}
                  </dd>
                </div>
              );
            })}
          </dl>
        )}
      </div>
      {lines && (
        <div className="flex flex-col gap-1 border-t border-kiln-line px-5 py-2.5 text-sm md:px-6" aria-live="polite">
          {reasons.map((figure) => (
            <p key={figure.label} id={`server-why-${figure.label}`} className="text-[#b9bec4] [overflow-wrap:anywhere]">
              {figure.label}: {figure.reason}
            </p>
          ))}
          {model.state === "error" && <p className="text-fault [overflow-wrap:anywhere]">{model.message}</p>}
          {model.state === "unloaded" && (
            <p className="text-[#b9bec4]">Unloaded after going unused (model.idle_unload). The next message loads it again; Load does it now.</p>
          )}
          {last && <p className={`${TONE[last.tone]} [overflow-wrap:anywhere]`}>{last.text}</p>}
          {streaming && <p className="text-[#b9bec4]">{drainNote(drain, model.state === "switching" ? "The switch" : "A switch")}</p>}
        </div>
      )}
    </section>
  );
}

function Headline({ model }: { model: ModelState }) {
  if (model.state === "switching") {
    return (
      <h2 className="font-display text-xl leading-tight font-semibold text-white [overflow-wrap:anywhere] sm:text-2xl">
        Switching from <span className="text-[#b9bec4]">{model.from}</span> to {model.to}
      </h2>
    );
  }
  if (model.state === "loading") {
    return <h2 className="font-display text-xl leading-tight font-semibold text-white [overflow-wrap:anywhere] sm:text-2xl">Loading {model.to}</h2>;
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
      className={`font-display leading-none font-semibold [overflow-wrap:anywhere] ${muted ? "text-xl text-[#b9bec4] sm:text-2xl" : "text-[28px] text-white sm:text-[34px]"}`}
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

/** A figure of the main model, its label in the bar's quiet grey before it. */
function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="[overflow-wrap:anywhere]">
      <span className="text-[#939ba4]">{label}</span> <span className="font-display font-semibold">{children}</span>
    </span>
  );
}

function Dot({ color }: { color: string }) {
  return <span className="cut size-2.5 shrink-0 [--cut-size:4px]" style={{ background: color }} aria-hidden />;
}
