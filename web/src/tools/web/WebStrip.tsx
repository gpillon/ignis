import { StatusMark } from "../agents/status.tsx";
import { FETCH_LIMIT, WEB_SEARCH_TOOL_NAME, type WebRun, webSummary } from "./web.ts";

// The web calls a reply made, as a list inside the reply. A row opens in
// place: the results of a search, the start of a fetched page, or the error.

export function WebStrip({ runs }: { runs: WebRun[] }) {
  return (
    <section aria-label="Web" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className="font-display text-[13px] font-semibold text-ink">Web</h3>
        <p className="font-display text-xs text-ash">{webSummary(runs)}</p>
      </div>
      <ul className="flex flex-col gap-1.5">
        {/* Keyed by position too: an agent's later requests may reuse call ids. */}
        {runs.map((run, i) => (
          <WebRow key={`${i}:${run.callId}`} run={run} />
        ))}
      </ul>
    </section>
  );
}

/** A short preview of a fetched page; the model gets up to FETCH_LIMIT characters. */
const PREVIEW = 1200;

const VIA_LABEL: Record<NonNullable<WebRun["via"]>, string> = { tavily: "Tavily", tiago: "tiago.zip", duckduckgo: "DuckDuckGo" };

function outcome(run: WebRun): string {
  const search = run.tool === WEB_SEARCH_TOOL_NAME;
  if (run.status === "running") return search ? "Searching…" : "Reading…";
  if (run.status === "failed") return "Failed";
  if (run.status === "stopped") return "Stopped";
  if (search) {
    const n = run.results?.length ?? 0;
    return `${n} ${n === 1 ? "result" : "results"}${run.via ? `, ${VIA_LABEL[run.via]}` : ""}`;
  }
  const chars = run.page?.trim().length ?? 0;
  return `${chars.toLocaleString()} chars${chars > FETCH_LIMIT ? ", cut" : ""}`;
}

function WebRow({ run }: { run: WebRun }) {
  const search = run.tool === WEB_SEARCH_TOOL_NAME;
  return (
    <li className="cut bg-surface [--cut-size:10px]">
      <details>
        <summary className="flex min-w-0 cursor-pointer list-none items-center gap-2 px-3 py-2 select-none hover:bg-line/50 [&::-webkit-details-marker]:hidden">
          <StatusMark status={run.status} />
          <span className="shrink-0 font-display text-xs font-semibold text-ash">{search ? "Search" : "Read"}</span>
          <span className="min-w-0 truncate text-sm text-ink">{run.target}</span>
          <span
            className={`ml-auto shrink-0 font-display text-xs tabular-nums ${run.status === "failed" ? "text-fault" : run.status === "done" ? "text-ember" : "text-ash"}`}
          >
            {outcome(run)}
          </span>
        </summary>
        <div className="border-t border-line px-3 py-2.5">
          {run.error ? (
            <p className="border-l-2 border-fault pl-3 text-sm text-fault">{run.error}</p>
          ) : run.status !== "done" ? (
            <p className="text-xs text-ash">{run.status === "running" ? "No answer yet." : "Stopped before it finished."}</p>
          ) : search ? (
            <SearchResults run={run} />
          ) : (
            <div className="flex flex-col gap-2">
              <a href={run.target} target="_blank" rel="noreferrer" className="truncate text-xs text-ember hover:underline">
                {run.target}
              </a>
              <pre className="max-h-60 overflow-y-auto border-l-2 border-line pl-3 whitespace-pre-wrap break-words font-sans text-[13px] leading-normal text-ash">
                {(run.page ?? "").trim().slice(0, PREVIEW) || "The page returned no text."}
                {(run.page ?? "").trim().length > PREVIEW && "…"}
              </pre>
            </div>
          )}
        </div>
      </details>
    </li>
  );
}

function SearchResults({ run }: { run: WebRun }) {
  const results = run.results ?? [];
  if (results.length === 0) return <p className="text-xs text-ash">No results.</p>;
  return (
    <ol className="flex flex-col gap-3">
      {results.map((r, i) => (
        <li key={`${i}:${r.url}`} className="flex min-w-0 flex-col gap-0.5">
          <a href={r.url} target="_blank" rel="noreferrer" className="truncate font-display text-sm font-semibold text-ink hover:text-ember">
            {r.title || r.url}
          </a>
          <span className="truncate text-xs text-ash">{r.url}</span>
          <span className="line-clamp-3 text-xs leading-snug text-ash">{r.content}</span>
        </li>
      ))}
    </ol>
  );
}
