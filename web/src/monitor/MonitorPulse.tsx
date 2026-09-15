import { headerPulse, RATE_SPAN_MS } from "./derive.ts";
import { formatCount, formatNumber, formatWindow } from "./format.ts";
import { useMonitor } from "./useMonitor.ts";

// The engine's pulse in the Playground's header while chatting (GitHub #165):
// the latest scheduler gauges and the generated-token rate, from the shared
// scraper. It opens the Monitor.

/** The live chip, once metrics answer. */
export function MonitorPulse({ onOpen }: { onOpen: () => void }) {
  const monitor = useMonitor();
  const { running, waiting, tokensPerSec } = headerPulse(monitor.points);
  if (monitor.availability !== "on" || running === null) return null;
  return <PulseChip running={running} waiting={waiting} tokensPerSec={tokensPerSec} paused={monitor.paused} onOpen={onOpen} />;
}

export function PulseChip({
  running,
  waiting,
  tokensPerSec,
  paused,
  onOpen,
}: {
  running: number;
  waiting: number | null;
  tokensPerSec: number | null;
  paused: boolean;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      title={`Open the Monitor. Tokens are counted as requests complete, over ${formatWindow(RATE_SPAN_MS)}: a long reply adds its tokens when it ends.`}
      onClick={onOpen}
      className="hidden shrink-0 items-center gap-3 border border-kiln-line px-2.5 py-1 font-display text-[13px] text-[#939ba4] hover:border-[#ff5a1f]/60 hover:text-white md:flex"
    >
      <span className={`size-2 rounded-full ${paused ? "" : "pulse-dot"} ${running > 0 ? "bg-[#ff5a1f]" : "bg-[#3ecf8e]"}`} aria-hidden />
      <span>
        <b className="font-semibold text-[#eae8e4] tabular-nums">{formatCount(running)}</b> running
      </span>
      {(waiting ?? 0) > 0 && (
        <span>
          <b className="font-semibold text-[#fab219] tabular-nums">{formatCount(waiting)}</b> waiting
        </span>
      )}
      <span className="hidden lg:inline">
        <b className="font-semibold text-[#eae8e4] tabular-nums">{formatNumber(tokensPerSec)}</b> tok/s done
      </span>
    </button>
  );
}
