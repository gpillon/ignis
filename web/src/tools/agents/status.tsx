import { useEffect, useState } from "react";
import { describeFigures, formatMs } from "../../metrics/figures.ts";
import type { AgentRun, AgentStatus } from "./agents.ts";

// An agent's status, as the strip's cards and the reader both show it.

/** A clock that ticks while `active`, for the running agents' elapsed time. */
export function useNow(active: boolean) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

const STATUS_LABEL: Record<AgentStatus, string> = {
  queued: "Waiting for a lane",
  running: "Running",
  done: "Done",
  failed: "Failed",
  stopped: "Stopped",
};

export function statusLine(run: AgentRun, now: number): string {
  if (run.status === "running" && run.startedAt !== undefined) return `Running, ${formatMs(now - run.startedAt)}`;
  if (run.status === "done" && run.figures) return `Done in ${describeFigures(run.figures).duration}`;
  return STATUS_LABEL[run.status];
}

export function StatusMark({ status }: { status: AgentStatus }) {
  const color = {
    queued: "bg-line",
    running: "bg-ember animate-pulse motion-reduce:animate-none",
    done: "bg-ink",
    failed: "bg-fault",
    stopped: "bg-ash",
  }[status];
  return <span className={`cut size-2 shrink-0 [--cut-size:3px] ${color}`} aria-hidden />;
}
