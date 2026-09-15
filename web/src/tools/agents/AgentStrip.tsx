import { describeFigures } from "../../metrics/figures.ts";
import { type AgentRun, agentSummary } from "./agents.ts";
import { StatusMark, statusLine, useNow } from "./status.tsx";

// The agents a reply started, as a strip of lanes inside the reply. A card
// opens the agent in the reader (AgentReader.tsx).

export function AgentStrip(props: { runs: AgentRun[]; openCallId: string | null; onOpen: (callId: string) => void }) {
  const now = useNow(props.runs.some((r) => r.status === "running"));
  return (
    <section aria-label="Agents" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className="font-display text-[13px] font-semibold text-ink">Agents</h3>
        <p className="font-display text-xs text-ash">{agentSummary(props.runs)}</p>
      </div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(12rem,1fr))] gap-2">
        {props.runs.map((run) => (
          <AgentCard
            key={run.callId}
            run={run}
            now={now}
            open={run.callId === props.openCallId}
            onOpen={() => props.onOpen(run.callId)}
          />
        ))}
      </div>
    </section>
  );
}

function AgentCard({ run, now, open, onOpen }: { run: AgentRun; now: number; open: boolean; onOpen: () => void }) {
  const d = run.figures ? describeFigures(run.figures) : null;
  const bar = {
    queued: "bg-line",
    running: "",
    done: "bg-ember",
    failed: "bg-fault",
    stopped: "bg-ash",
  }[run.status];
  return (
    <button
      type="button"
      aria-pressed={open}
      title={`Read ${run.name}`}
      onClick={onOpen}
      className={`cut flex min-w-0 flex-col gap-2 bg-surface px-3 pt-2.5 pb-2 text-left [--cut-size:10px] hover:bg-line/50 ${open ? "shadow-[inset_0_0_0_1px_var(--ember)]" : ""}`}
    >
      <span className="flex min-w-0 items-center gap-2">
        <StatusMark status={run.status} />
        <span className="truncate font-display text-sm font-semibold text-ink">{run.name}</span>
      </span>
      <span className="line-clamp-2 min-h-[2lh] text-xs leading-snug text-ash">{run.prompt}</span>
      {run.status === "running" ? (
        <span className="heat block" data-busy="true" aria-hidden />
      ) : (
        <span className={`block h-0.5 ${bar}`} aria-hidden />
      )}
      <span className="flex items-baseline justify-between gap-2 font-display text-xs tabular-nums">
        <span className={run.status === "failed" ? "text-fault" : "text-ash"}>
          {statusLine(run, now)}
          {run.web?.length ? `, ${run.web.length} web` : ""}
        </span>
        {d && run.status === "done" && <span className="font-semibold text-ember">{d.decode}</span>}
      </span>
    </button>
  );
}
