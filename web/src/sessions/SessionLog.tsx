import type { ReactNode } from "react";
import { describeFigures, type Figures } from "../metrics/figures.ts";
import { EFFORT_LABELS } from "../settings/defaults.ts";
import { IconChevron } from "../ui/icons.tsx";
import type { LogRow } from "./sessions.ts";

/** The session's log at the foot of the page: one row per request, collapsible. */
export function SessionLog({ rows, open, onToggle }: { rows: LogRow[]; open: boolean; onToggle: () => void }) {
  const numeric = ["TTFT", "Decode", "Total", "Prompt", "Completion", "Finish"];
  const replies = rows.filter((row) => row.agent === undefined);
  const agentRows = rows.length - replies.length;
  const last = replies.at(-1)?.figures;
  const count = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  return (
    <section className="z-10 shrink-0 border-t border-line bg-ground" aria-label="This session">
      <div className="flex items-center gap-x-3 px-4 py-2.5 md:px-6">
        <h2 className="font-display text-sm font-semibold">This session</h2>
        <p className="min-w-0 truncate text-xs text-ash">
          {rows.length === 0
            ? "HTTP-observed figures for each reply collect here."
            : `${count(replies.length, "reply", "replies")}${agentRows ? `, ${count(agentRows, "agent request", "agent requests")}` : ""}${last ? `, last decode ${describeFigures(last).decode}` : ""}. HTTP-observed: measured by the browser, not the engine.`}
        </p>
        <button
          type="button"
          aria-expanded={open}
          aria-controls="session-log"
          aria-label={open ? "Collapse this session's log" : "Expand this session's log"}
          title={open ? "Collapse" : "Expand"}
          className="ml-auto grid size-8 shrink-0 place-items-center text-ash hover:bg-surface hover:text-ink"
          onClick={onToggle}
        >
          <IconChevron className={`transition-transform motion-reduce:transition-none ${open ? "rotate-180" : ""}`} />
        </button>
      </div>
      <div
        id="session-log"
        className={`grid transition-[grid-template-rows] duration-200 motion-reduce:transition-none ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}
        inert={!open}
      >
        <div className="min-h-0 overflow-hidden">
          {/* Open, the table scrolls within a quarter of the window, under the title row. */}
          <div className="max-h-[25dvh] overflow-auto px-4 pb-4 md:px-6">
            {rows.length === 0 ? (
              <p className="py-2 text-sm text-ash">Each reply in this session adds a row here.</p>
            ) : (
              <table className="w-full border-collapse font-display text-[13px] tabular-nums">
                <thead className="sticky top-0 bg-ground text-ash">
                  <tr className="border-b border-line">
                    {["#", "Time", "Lane tag", "Thinking", "Budget"].map((h) => (
                      <Th key={h}>{h}</Th>
                    ))}
                    {numeric.map((h) => (
                      <Th key={h} right>
                        {h}
                      </Th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.n} className="border-b border-line/60 last:border-0 hover:bg-surface">
                      <td className="px-3 py-1.5 text-ash">{row.n}</td>
                      <td className="px-3 py-1.5 whitespace-nowrap">{row.at}</td>
                      <td className="px-3 py-1.5 whitespace-nowrap">
                        {row.agent ? (
                          <>
                            agent <span className="text-ash">{row.agent}</span>
                          </>
                        ) : (
                          row.laneTag
                        )}
                      </td>
                      <td className="px-3 py-1.5">{EFFORT_LABELS[row.reasoningEffort]}</td>
                      <td className="px-3 py-1.5 whitespace-nowrap">
                        <BudgetCell row={row} />
                      </td>
                      {row.figures ? (
                        <FigureCells figures={row.figures} />
                      ) : (
                        <td colSpan={6} className="px-3 py-1.5 text-right text-fault">
                          {row.error}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

/**
 * The thinking budget a request sent, and where the budget closed its
 * reasoning when it did. Thinking off and the `max` effort send none and
 * cannot have one; any other effort that sent none ran on the server's default.
 */
function BudgetCell({ row }: { row: LogRow }) {
  const forcedAt = row.figures?.thinkingForcedAt;
  const sent =
    row.reasoningEffort === "none" || row.reasoningEffort === "max"
      ? "—"
      : row.thinkingBudget === undefined
        ? "default"
        : row.thinkingBudget === 0
          ? "off"
          : String(row.thinkingBudget);
  return (
    <>
      {sent}
      {forcedAt !== undefined && <span className="text-ember"> · reached {forcedAt}</span>}
    </>
  );
}

function Th({ children, right }: { children: ReactNode; right?: boolean }) {
  return <th className={`px-3 py-2 font-medium whitespace-nowrap ${right ? "text-right" : "text-left"}`}>{children}</th>;
}

function FigureCells({ figures }: { figures: Figures }) {
  const d = describeFigures(figures);
  return (
    <>
      {[d.ttft, d.decode, d.duration, d.promptTokens, d.completionTokens, d.finish].map((value, i) => (
        <td key={i} className={`px-3 py-1.5 text-right whitespace-nowrap ${i === 1 ? "font-semibold text-ember" : ""}`}>
          {value}
        </td>
      ))}
    </>
  );
}
