import { describeFigures, type Figures } from "./figures.ts";

/** One request's HTTP-observed figures, in a line under the reply (or the agent's answer). */
export function Readout({ figures }: { figures: Figures }) {
  const d = describeFigures(figures);
  const items: [string, string, boolean?][] = [
    ["TTFT", d.ttft],
    ["Decode", d.decode, true],
    ["Total", d.duration],
    ["Tokens", `${d.promptTokens} in, ${d.completionTokens} out`],
    ["Finish", d.finish],
  ];
  return (
    <dl className="flex flex-wrap items-baseline gap-x-5 gap-y-1 border-t border-line pt-2 font-display text-xs">
      {items.map(([k, v, hot]) => (
        <div key={k} className="flex items-baseline gap-1.5">
          <dt className="text-ash">{k}</dt>
          <dd className={`tabular-nums ${hot ? "font-semibold text-ember" : "text-ink"}`}>{v}</dd>
        </div>
      ))}
      <div className="ml-auto text-ash" title="Measured by the browser around the HTTP stream, not engine-internal timings">
        HTTP-observed
      </div>
    </dl>
  );
}
