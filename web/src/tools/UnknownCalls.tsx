import type { UnknownCall } from "./errors.ts";

// Calls to tools the request never declared — a model reaching for a tool it
// was not given. They do not run; the model gets the error shown here.

export function UnknownCalls({ calls }: { calls: UnknownCall[] }) {
  return (
    <section aria-label="Unknown tools" className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h3 className="font-display text-[13px] font-semibold text-ink">Unknown tools</h3>
        <p className="font-display text-xs text-ash">
          {calls.length} {calls.length === 1 ? "call" : "calls"} not run
        </p>
      </div>
      <ul className="flex flex-col gap-1.5">
        {calls.map((call, i) => (
          <li key={`${i}:${call.callId}`} className="cut flex min-w-0 flex-col gap-1 bg-surface px-3 py-2 [--cut-size:10px]">
            <span className="flex min-w-0 items-center gap-2">
              <span className="cut size-2 shrink-0 bg-fault [--cut-size:3px]" aria-hidden />
              <span className="shrink-0 font-display text-sm font-semibold text-ink">{call.name}</span>
              <span className="min-w-0 truncate font-display text-xs text-ash">{call.arguments}</span>
              <span className="ml-auto shrink-0 font-display text-xs text-fault">Not run</span>
            </span>
            <span className="text-xs leading-snug text-ash">{call.error}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
