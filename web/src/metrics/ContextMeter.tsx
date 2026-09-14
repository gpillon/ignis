import { compactTokens, type ContextUsage } from "./context.ts";

/** The context bar under the prompt, with its breakdown on hover or focus. */
export function ContextMeter({ usage: u }: { usage: ContextUsage }) {
  const n = (v: number) => v.toLocaleString("en-US");
  const label = u.limit === null ? `${compactTokens(u.used)} used` : `${compactTokens(u.used)} / ${compactTokens(u.limit)}`;
  const free = u.limit === null ? null : u.limit - u.used - (u.reserve ?? 0);
  const rows: [string, string, boolean?][] = [
    ["Context", u.limit === null ? "Not reported by ignis" : `${n(u.limit)} tokens`],
    ["Last prompt", n(u.promptTokens)],
    ["Last completion", n(u.completionTokens)],
    ["In use", u.limit === null ? n(u.used) : `${n(u.used)} (${Math.round((u.used / u.limit) * 100)}%)`],
    ["Next reply reserves", u.reserve === null ? "Engine cap" : n(u.reserve)],
    ...(free === null ? [] : [["Left after that", n(free), free < 0] as [string, string, boolean]]),
  ];
  return (
    <div className="group relative shrink-0">
      <button
        type="button"
        aria-describedby="context-details"
        className="flex items-center gap-2.5 py-1 font-display text-xs tabular-nums text-ash hover:text-ink"
      >
        <span>Context</span>
        <span className="relative flex h-1.5 w-28 overflow-hidden bg-line sm:w-40" aria-hidden>
          {u.usedShare !== null && <span className="h-full bg-[#ff5a1f]" style={{ width: `${u.usedShare * 100}%` }} />}
          {u.reserveShare !== null && (
            <span
              className={`h-full ${u.overflows ? "bg-fault" : "bg-[#ff5a1f]/35"}`}
              style={{ width: `${u.reserveShare * 100}%` }}
            />
          )}
        </span>
        <span className={u.overflows ? "text-fault" : "text-ink"}>{label}</span>
      </button>
      <div
        id="context-details"
        role="tooltip"
        className="cut invisible absolute right-0 bottom-full z-30 mb-2 w-72 bg-kiln p-4 text-[#eae8e4] opacity-0 shadow-[0_12px_32px_rgb(0_0_0/0.35)] transition-opacity [--cut-size:12px] group-focus-within:visible group-focus-within:opacity-100 group-hover:visible group-hover:opacity-100 motion-reduce:transition-none"
      >
        <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1.5 font-display text-[13px]">
          {rows.map(([k, v, bad]) => (
            <div key={k} className="contents">
              <dt className="text-[#939ba4]">{k}</dt>
              <dd className={`text-right tabular-nums ${bad ? "text-[#ff7b6b]" : ""}`}>{v}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 border-t border-kiln-line pt-3 text-xs leading-relaxed text-[#939ba4]">
          {u.overflows
            ? "The next request is over the context, so ignis will refuse it. Lower Max tokens or start a new session."
            : "From the last reply's usage in this session. Reasoning is not sent back, so the next prompt can be shorter."}
        </p>
      </div>
    </div>
  );
}
