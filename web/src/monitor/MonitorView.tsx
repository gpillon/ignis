import { type ReactNode, useMemo, useState } from "react";
import { forgetKey } from "../api/auth.ts";
import { IconChevron, IconHealth, IconPause, IconPlay } from "../ui/icons.tsx";
import { Distribution, Legend, type Series, Sparkline, TimeChart } from "./charts.tsx";
import { type Counter, type Dashboard, deriveDashboard, type HealthLevel, type Latency, RATE_SPAN_MS, TREND_SPAN_MS } from "./derive.ts";
import { formatAgo, formatBound, formatCount, formatNumber, formatSeconds, formatShare, formatWindow } from "./format.ts";
import { bucketCounts } from "./quantile.ts";
import type { MonitorState } from "./scrape.ts";
import { REJECT_REASONS, type RejectReason } from "./snapshot.ts";
import { setMonitorInterval, setMonitorPaused, useNow } from "./useMonitor.ts";

// The Monitor (GitHub #165): a live dashboard over ignis's Prometheus
// exposition, scraped by this browser from /ui/metrics. It leads with a
// verdict in words, then throughput and load, admission, latency, cache
// pressure and the scraper itself; every series is also in the table at the
// foot. Only ADR 0017's contract is drawn — no invented metrics.

const WINDOWS = [
  { ms: 60_000, label: "1m" },
  { ms: 300_000, label: "5m" },
  { ms: 900_000, label: "15m" },
];
const INTERVALS = [
  { ms: 2_000, label: "2s" },
  { ms: 3_000, label: "3s" },
  { ms: 5_000, label: "5s" },
];

const REASON_LABEL: Record<RejectReason, string> = { full: "Engine full", unknown_model: "Unknown model", oversized: "Oversized" };

const VERDICT: Record<HealthLevel, { label: string; color: string }> = {
  idle: { label: "Idle", color: "var(--ash)" },
  healthy: { label: "Healthy", color: "var(--good)" },
  busy: { label: "Busy", color: "var(--warn)" },
  saturated: { label: "Saturated", color: "var(--fault)" },
};

/** A series' latest known value as text. */
const latestOf = (s: Series, format: (v: number) => string) => {
  const v = s.values.at(-1);
  return v === null || v === undefined ? "—" : format(v);
};

export function MonitorView({ state }: { state: MonitorState }) {
  const [windowMs, setWindowMs] = useState(300_000);
  const now = useNow(1000);
  const dash = useMemo(() => deriveDashboard(state.points, windowMs), [state.points, windowMs]);
  const late = state.availability === "on" && !state.paused && state.last !== null && now - state.last.at > 3 * state.intervalMs + 1000;

  return (
    <main aria-label="Monitor" className="min-h-0 flex-1 overflow-y-auto">
      <ControlBar state={state} late={late} now={now} windowMs={windowMs} onWindow={setWindowMs} />
      <div className="mx-auto flex w-full max-w-[1480px] flex-col gap-4 px-4 py-5 md:px-6">
        {state.availability === "unauthorized" && (
          <Notice tone="var(--warn)" title="ignis wants the API key for its metrics">
            <p>/ui/metrics answered 401: the key is missing or was refused.</p>
            <button type="button" className="mt-2 bg-ink px-3 py-1.5 font-display text-[13px] font-semibold text-ground hover:bg-ember" onClick={forgetKey}>
              Enter the API key
            </button>
          </Notice>
        )}
        {state.availability === "unreachable" && (
          <Notice tone="var(--fault)" title="ignis is not answering">
            <p>The last scrape failed ({state.message}). The charts hold what was collected; the scraper keeps trying.</p>
          </Notice>
        )}
        {late && state.last && (
          <Notice tone="var(--warn)" title="Scrapes are late">
            <p>No answer for {formatSeconds((now - state.last.at) / 1000)}; the figures below may be out of date.</p>
          </Notice>
        )}
        {dash ? <Board dash={dash} state={state} /> : state.availability !== "unauthorized" && <p className="py-16 text-center text-sm text-ash">Waiting for the first scrape…</p>}
      </div>
    </main>
  );
}

function ControlBar({ state, late, now, windowMs, onWindow }: { state: MonitorState; late: boolean; now: number; windowMs: number; onWindow: (ms: number) => void }) {
  const last = state.last;
  const version = state.points.at(-1)?.snap.version;
  return (
    <div className="sticky top-0 z-20 border-b border-line bg-ground/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1480px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5 md:px-6">
        <LiveBadge state={state} late={late} />
        <p className="flex min-w-0 flex-wrap items-baseline gap-x-2 text-xs text-ash">
          {version && <span className="font-display text-[13px] font-semibold text-ink">ignis {version}</span>}
          {last && (
            <span key={last.at} className="scrape-flash tabular-nums">
              scraped {formatAgo(now - last.at)} · {Math.round(last.scrapeMs)} ms
            </span>
          )}
        </p>
        <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-2">
          <Pills legend="Window" options={WINDOWS} value={windowMs} onChange={onWindow} />
          <Pills legend="Every" options={INTERVALS} value={state.intervalMs} onChange={setMonitorInterval} />
          <button
            type="button"
            onClick={() => setMonitorPaused(!state.paused)}
            className={`flex items-center gap-2 px-3 py-1.5 font-display text-[13px] font-semibold ${state.paused ? "bg-ember text-white hover:brightness-110" : "bg-surface text-ink hover:bg-line"}`}
          >
            {state.paused ? <IconPlay /> : <IconPause />}
            {state.paused ? "Resume" : "Pause"}
          </button>
        </div>
      </div>
    </div>
  );
}

function LiveBadge({ state, late }: { state: MonitorState; late: boolean }) {
  const [label, color, pulse] = state.paused
    ? ["Paused", "var(--ash)", false]
    : state.availability === "unreachable"
      ? ["Offline", "var(--fault)", false]
      : state.availability === "unauthorized"
        ? ["Locked", "var(--warn)", false]
        : late
          ? ["Late", "var(--warn)", false]
          : ["Live", "var(--good)", true];
  return (
    <span className="flex items-center gap-2 font-display text-[13px] font-bold tracking-[0.18em] uppercase" role="status">
      <span className={`size-2 rounded-full ${pulse ? "pulse-dot" : ""}`} style={{ background: color }} aria-hidden />
      <span style={{ color }}>{label}</span>
    </span>
  );
}

function Pills({ legend, options, value, onChange }: { legend: string; options: { ms: number; label: string }[]; value: number; onChange: (ms: number) => void }) {
  return (
    <div role="radiogroup" aria-label={legend} className="flex items-center gap-2">
      <span className="font-display text-xs text-ash">{legend}</span>
      <div className="flex bg-surface p-0.5">
        {options.map((o) => (
          <button
            key={o.ms}
            type="button"
            role="radio"
            aria-checked={o.ms === value}
            onClick={() => onChange(o.ms)}
            className={`px-2.5 py-1 font-display text-xs font-semibold ${o.ms === value ? "bg-ink text-ground" : "text-ash hover:text-ink"}`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function Notice({ tone, title, children }: { tone: string; title: string; children: ReactNode }) {
  return (
    <section className="flex gap-3 border-l-2 bg-surface px-4 py-3 text-sm" style={{ borderColor: tone }}>
      <div>
        <h2 className="font-display font-semibold">{title}</h2>
        <div className="text-ash">{children}</div>
      </div>
    </section>
  );
}

type ChartFrame = { times: number[]; from: number; to: number; markers: number[]; live: boolean };

function Board({ dash, state }: { dash: Dashboard; state: MonitorState }) {
  const win = formatWindow(dash.windowMs);
  const chart: ChartFrame = { times: dash.times, from: dash.from, to: dash.now, markers: state.restarts, live: state.availability === "on" && !state.paused };
  const visible = <T,>(values: T[]) => values.slice(dash.visibleStart);
  const finished = (dash.completed.window ?? 0) + (dash.cancelled.window ?? 0);

  const load: Series[] = [
    { key: "running", label: "Running", color: "var(--series-1)", values: dash.runningSeries },
    { key: "waiting", label: "Waiting", color: "var(--series-2)", values: dash.waitingSeries },
  ];
  const flow: Series[] = [
    { key: "accepted", label: "Accepted", color: "var(--series-2)", values: dash.accepted.perMinSeries },
    { key: "completed", label: "Completed", color: "var(--series-3)", values: dash.completed.perMinSeries },
    { key: "cancelled", label: "Cancelled", color: "var(--series-4)", values: dash.cancelled.perMinSeries },
    { key: "rejected", label: "Rejected", color: "var(--series-1)", values: dash.rejected.perMinSeries },
  ];
  const [accepted, completed, cancelled, rejected] = flow;

  return (
    <>
      <Verdict dash={dash} />

      <Hero dash={dash} chart={chart} />

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <CounterTile series={accepted} counter={dash.accepted} win={win} spark={visible(accepted.values)}>
          admitted by the scheduler
        </CounterTile>
        <CounterTile series={completed} counter={dash.completed} win={win} spark={visible(completed.values)}>
          {finished ? `${formatShare(dash.completed.window, finished)} of finished requests` : "none finished yet"}
        </CounterTile>
        <CounterTile series={cancelled} counter={dash.cancelled} win={win} spark={visible(cancelled.values)}>
          {finished ? `${formatShare(dash.cancelled.window, finished)} of finished requests` : "client went away before the end"}
        </CounterTile>
        <CounterTile series={rejected} counter={dash.rejected} win={win}>
          <ReasonBars byReason={dash.rejected.byReason} />
        </CounterTile>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Scheduler load" subtitle="Requests by scheduler state, at each scrape">
          <Legend series={load} swatch="area" value={(s) => latestOf(s, formatCount)} />
          <TimeChart label="Scheduler load" {...chart} stacked integer height={210} format={formatCount} series={load} />
        </Card>
        <Card title="Request flow" subtitle={`Requests per minute, ${formatWindow(RATE_SPAN_MS)} rolling`}>
          <Legend series={flow} value={(s) => latestOf(s, formatNumber)} />
          <TimeChart label="Request flow per minute" {...chart} height={210} format={formatNumber} series={flow} />
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <LatencyCard title="Time to first token" subtitle="Submission to the first token" latency={dash.ttft} chart={chart} win={win} none="No first tokens" />
        <LatencyCard title="Request duration" subtitle="Submission to completion" latency={dash.duration} chart={chart} win={win} none="No requests completed" />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card title="Prefix reuse" subtitle="Prompt tokens skipped through a sibling's prefix">
          <Figures
            items={[
              ["Now", `${formatNumber(dash.prefix.perSec)} tok/s`],
              [`Last ${win}`, formatCount(dash.prefix.window)],
              ["Since start", formatCount(dash.prefix.total)],
            ]}
          />
          <TimeChart label="Prefix tokens reused per second" {...chart} area height={120} format={formatNumber} series={[{ key: "prefix", label: "Reused tok/s", color: "var(--series-3)", values: dash.prefix.perSecSeries }]} />
        </Card>
        <Card title="KV evictions" subtitle="Requests moved out to the host RAM tier">
          <Figures
            items={[
              [`Last ${win}`, formatCount(dash.evictions.window)],
              ["Per minute", formatNumber(dash.evictions.perMin)],
              ["Since start", formatCount(dash.evictions.total)],
            ]}
          />
          <TimeChart label="KV evictions per minute" {...chart} area height={120} format={formatNumber} series={[{ key: "evictions", label: "Evictions/min", color: "var(--series-4)", values: dash.evictions.perMinSeries }]} />
        </Card>
        <ScraperCard dash={dash} state={state} />
      </div>

      <SeriesTable state={state} />
    </>
  );
}

function Verdict({ dash }: { dash: Dashboard }) {
  const { level, summary, notes } = dash.health;
  const verdict = VERDICT[level];
  return (
    <section aria-label="Server health" className="cut relative flex flex-wrap items-center gap-x-6 gap-y-3 bg-surface py-4 pr-5 pl-6 [--cut-size:14px]">
      <span className="absolute inset-y-0 left-0 w-1" style={{ background: verdict.color }} aria-hidden />
      <div className="flex min-w-0 items-center gap-3.5">
        <span className="grid size-11 shrink-0 place-items-center" style={{ color: verdict.color, background: `color-mix(in oklab, ${verdict.color} 14%, transparent)` }}>
          <IconHealth level={level} />
        </span>
        <div className="min-w-0">
          <p className="font-display text-[11px] font-bold tracking-[0.18em] uppercase" style={{ color: verdict.color }}>
            {verdict.label}
          </p>
          <p className="font-display text-xl leading-tight font-semibold">{summary}</p>
        </div>
      </div>
      {notes.length > 0 && (
        <ul className="flex flex-wrap gap-2 lg:ml-auto">
          {notes.map((note) => (
            <li key={note} className="bg-ground px-2.5 py-1 text-xs text-ink">
              {note}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Hero({ dash, chart }: { dash: Dashboard; chart: ChartFrame }) {
  const win = formatWindow(dash.windowMs);
  return (
    <section aria-label="Throughput and load" className="monitor-kiln cut relative overflow-hidden text-[#eae8e4] [--cut-size:20px]">
      <div className="grid gap-6 p-5 md:p-6 lg:grid-cols-[minmax(0,1fr)_minmax(250px,320px)]">
        <div className="min-w-0">
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <h2 className="font-display text-xs font-bold tracking-[0.18em] text-[#939ba4] uppercase">Generated tokens</h2>
            <span className="text-xs text-[#939ba4]">
              {dash.tokens.live ? "counted as they are decoded" : "counted as requests complete"} · {formatWindow(dash.tokens.spanMs)} rolling
            </span>
          </div>
          <div className="mt-2 flex flex-wrap items-end gap-x-8 gap-y-3">
            <p className="hero-glow font-display text-[60px] leading-[0.9] font-semibold text-white">
              {formatNumber(dash.tokens.perSec)}
              <span className="ml-2 text-lg font-medium text-[#939ba4]">tok/s</span>
            </p>
            <dl className="flex flex-wrap gap-x-6 gap-y-1 pb-1">
              <KilnFigure label={`Last ${win}`} value={formatCount(dash.tokens.window)} />
              <KilnFigure label="Per request" value={formatCount(dash.tokens.perRequest)} />
              <KilnFigure label="Since start" value={formatCount(dash.tokens.total)} />
            </dl>
          </div>
          <div className="mt-4">
            <TimeChart
              label="Generated tokens per second"
              {...chart}
              area
              height={180}
              format={formatNumber}
              series={[{ key: "tokens", label: "Tokens/s", color: "var(--series-1)", values: dash.tokens.perSecSeries }]}
            />
          </div>
        </div>

        <div className="flex flex-col gap-5 border-t border-kiln-line pt-5 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-6">
          <h2 className="font-display text-xs font-bold tracking-[0.18em] text-[#939ba4] uppercase">Right now</h2>
          <div className="grid grid-cols-2 gap-4">
            <Gauge label="Running" hint="in a lane" value={dash.running} peak={dash.runningPeak} color="var(--series-1)" win={win} />
            <Gauge label="Waiting" hint="queued" value={dash.waiting} peak={dash.waitingPeak} color="var(--series-2)" win={win} />
          </div>
          <LoadStrip running={dash.running} waiting={dash.waiting} />
          <dl className="mt-auto grid grid-cols-2 gap-x-4 gap-y-3 border-t border-kiln-line pt-4">
            <KilnFigure label="TTFT p95" value={formatSeconds(dash.ttft.p95)} />
            <KilnFigure label="Duration p95" value={formatSeconds(dash.duration.p95)} />
            <KilnFigure label="Accepted / min" value={formatNumber(dash.accepted.perMin)} />
            <KilnFigure label="Rejected / min" value={formatNumber(dash.rejected.perMin)} />
          </dl>
        </div>
      </div>
      <div className="heat" data-busy={(dash.running ?? 0) > 0} aria-hidden />
    </section>
  );
}

function KilnFigure({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[11px] text-[#939ba4]">{label}</dt>
      <dd className="font-display text-lg leading-tight font-semibold text-[#eae8e4]">{value}</dd>
    </div>
  );
}

function Gauge({ label, hint, value, peak, color, win }: { label: string; hint: string; value: number | null; peak: number | null; color: string; win: string }) {
  return (
    <div>
      <p className="flex items-center gap-2 text-xs text-[#b9bec4]">
        <span className="size-2.5 rounded-[2px]" style={{ background: color }} aria-hidden />
        {label}
        <span className="text-[#939ba4]">{hint}</span>
      </p>
      <p className="mt-1 font-display text-[52px] leading-none font-semibold text-white">{formatCount(value)}</p>
      <p className="mt-1 text-[11px] text-[#939ba4]">
        peak {formatCount(peak)} in {win}
      </p>
    </div>
  );
}

/** The live split of requests between lanes and the queue. */
function LoadStrip({ running, waiting }: { running: number | null; waiting: number | null }) {
  const inLanes = running ?? 0;
  const queued = waiting ?? 0;
  const total = inLanes + queued;
  return (
    <div aria-hidden>
      <div className="flex h-2.5 gap-[2px] bg-[#2e343c]">
        {total > 0 && (
          <>
            <span className="h-full transition-[flex-grow] duration-500 motion-reduce:transition-none" style={{ flexGrow: inLanes, background: "var(--series-1)" }} />
            {queued > 0 && <span className="h-full rounded-r-[3px] transition-[flex-grow] duration-500 motion-reduce:transition-none" style={{ flexGrow: queued, background: "var(--series-2)" }} />}
          </>
        )}
      </div>
      <p className="mt-1.5 text-[11px] text-[#939ba4]">{total === 0 ? "No requests in the engine" : `${formatShare(inLanes, total)} of requests in a lane`}</p>
    </div>
  );
}

function Card({ title, subtitle, aside, children }: { title: string; subtitle?: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section aria-label={title} className="cut flex min-w-0 flex-col gap-3 bg-surface p-4 [--cut-size:14px] md:p-5">
      <header className="flex flex-wrap items-start gap-x-4 gap-y-1">
        <div className="min-w-0">
          <h3 className="font-display text-[15px] font-semibold">{title}</h3>
          {subtitle && <p className="text-xs text-ash">{subtitle}</p>}
        </div>
        {aside && <div className="ml-auto">{aside}</div>}
      </header>
      {children}
    </section>
  );
}

/** One admission counter: its gain over the window, its rate, and its total, keyed by the colour its series draws in. */
function CounterTile({ series, counter, win, spark, children }: { series: Series; counter: Counter; win: string; spark?: (number | null)[]; children?: ReactNode }) {
  return (
    <section aria-label={series.label} className="cut flex min-w-0 flex-col gap-2 bg-surface p-4 [--cut-size:12px]">
      <header className="flex items-center gap-2">
        <span className="size-2.5 rounded-[2px]" style={{ background: series.color }} aria-hidden />
        <h3 className="font-display text-[13px] font-semibold text-ash">{series.label}</h3>
        <span className="ml-auto font-display text-xs text-ash">
          <span className="tabular-nums text-ink">{formatCount(counter.total)}</span> since start
        </span>
      </header>
      <p className="flex flex-wrap items-baseline gap-x-2">
        <span className="font-display text-[34px] leading-none font-semibold">{formatCount(counter.window)}</span>
        <span className="text-xs text-ash">in {win}</span>
        <span className="ml-auto font-display text-sm font-semibold">
          {formatNumber(counter.perMin)}
          <span className="font-medium text-ash">/min</span>
        </span>
      </p>
      {children && <div className="text-xs text-ash">{children}</div>}
      {spark && (
        <div className="mt-auto pt-1">
          <Sparkline values={spark} color={series.color} height={34} />
        </div>
      )}
    </section>
  );
}

function ReasonBars({ byReason }: { byReason: Dashboard["rejected"]["byReason"] }) {
  const max = Math.max(1, ...REJECT_REASONS.map((r) => byReason[r].window ?? 0));
  return (
    <ul className="mt-1 flex flex-col gap-1.5">
      {REJECT_REASONS.map((r) => {
        const count = byReason[r].window ?? 0;
        return (
          <li key={r} className="grid grid-cols-[96px_1fr_auto] items-center gap-2">
            <span>{REASON_LABEL[r]}</span>
            <span className="h-1.5 bg-line/70">
              {count > 0 && <span className="block h-full rounded-r-[3px] bg-[var(--series-1)]" style={{ width: `${(100 * count) / max}%` }} />}
            </span>
            <span className="min-w-6 text-right font-display text-[13px] font-semibold tabular-nums text-ink">{formatCount(count)}</span>
          </li>
        );
      })}
    </ul>
  );
}

function Figures({ items }: { items: [string, string][] }) {
  return (
    <dl className="grid grid-cols-3 gap-3">
      {items.map(([label, value]) => (
        <div key={label}>
          <dt className="text-[11px] text-ash">{label}</dt>
          <dd className="font-display text-lg leading-tight font-semibold">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function LatencyCard({ title, subtitle, latency, chart, win, none }: { title: string; subtitle: string; latency: Latency; chart: ChartFrame; win: string; none: string }) {
  const figures: [string, number | null][] = [
    ["p50", latency.p50],
    ["p95", latency.p95],
    ["p99", latency.p99],
    ["Mean", latency.mean],
  ];
  const trend: Series[] = [
    { key: "p50", label: "p50", color: "var(--series-2)", values: latency.trendP50 },
    { key: "p95", label: "p95", color: "var(--series-1)", values: latency.trendP95 },
  ];
  return (
    <Card
      title={title}
      subtitle={`${subtitle} · last ${win}`}
      aside={
        <span className="font-display text-xs text-ash">
          <span className="font-semibold tabular-nums text-ink">{formatCount(latency.count)}</span> requests
        </span>
      }
    >
      <dl className="grid grid-cols-4 gap-3 border-b border-line pb-3">
        {figures.map(([k, v]) => (
          <div key={k}>
            <dt className="font-display text-[11px] font-bold tracking-[0.14em] text-ash uppercase">{k}</dt>
            <dd className={`font-display leading-tight font-semibold ${k === "p95" ? "text-[26px]" : "text-[22px]"}`}>{formatSeconds(v)}</dd>
          </div>
        ))}
      </dl>
      {latency.window ? (
        <Distribution
          label={`${title} distribution`}
          bounds={latency.window.bounds}
          counts={bucketCounts(latency.window)}
          color="var(--series-2)"
          marks={[
            { label: "p50", value: latency.p50 },
            { label: "p95", value: latency.p95 },
          ]}
          formatBound={formatBound}
          empty={`${none} in the last ${win}`}
        />
      ) : (
        <p className="grid h-[150px] place-items-center text-xs text-ash">Needs two scrapes in the window.</p>
      )}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
        <h4 className="font-display text-xs font-semibold text-ash">Trend · sliding {formatWindow(TREND_SPAN_MS)}</h4>
        <Legend series={trend} />
      </div>
      <TimeChart label={`${title} p50 and p95 trend`} {...chart} height={130} format={formatSeconds} series={trend} />
      <p className="text-[11px] text-ash">
        Since start: p95 {formatSeconds(latency.lifetimeP95)} over {formatCount(latency.lifetimeCount)} requests. Quantiles are interpolated inside ADR 0017's fixed buckets.
      </p>
    </Card>
  );
}

function ScraperCard({ dash, state }: { dash: Dashboard; state: MonitorState }) {
  const spanMs = state.points.length > 1 ? state.points.at(-1)!.at - state.points[0].at : 0;
  const scrapeMs = state.last?.scrapeMs ?? null;
  return (
    <Card title="Scraper" subtitle="This browser, polling /ui/metrics">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 text-sm">
        <Row label="Build" value={dash.version ?? "—"} />
        <Row label="Interval" value={state.paused ? "paused" : `${state.intervalMs / 1000} s`} />
        <Row label="Last scrape" value={scrapeMs === null ? "—" : `${Math.round(scrapeMs)} ms`} />
        <Row label="History" value={`${formatWindow(spanMs)} · ${state.points.length} scrapes`} />
        <Row label="Restarts seen" value={String(state.restarts.length)} />
        <Row label="Unreadable lines" value={String(state.last?.errors.length ?? 0)} />
      </dl>
      <div>
        <p className="mb-1 text-[11px] text-ash">Scrape round trip</p>
        <Sparkline values={dash.scrapeMsSeries.slice(dash.visibleStart)} color="var(--series-4)" height={40} />
      </div>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] text-ash">{label}</dt>
      <dd className="truncate font-display font-semibold">{value}</dd>
    </div>
  );
}

/** Every sample of the latest scrape — the table view of the whole page — plus unreadable lines and the raw text. */
function SeriesTable({ state }: { state: MonitorState }) {
  const families = state.last?.families ?? [];
  const samples = families.flatMap((f) => f.samples.map((sample) => ({ sample, type: f.type })));
  // The snapshot's unknown samples are the very objects this scrape parsed.
  const unknown = new Set(state.points.at(-1)?.snap.unknown ?? []);
  const errors = state.last?.errors ?? [];
  return (
    <details className="cut group bg-surface [--cut-size:14px]">
      <summary className="flex cursor-pointer list-none flex-wrap items-baseline gap-x-3 gap-y-1 px-5 py-3.5 [&::-webkit-details-marker]:hidden">
        <span className="font-display text-[15px] font-semibold">All series</span>
        <span className="text-xs text-ash">
          {samples.length} samples in {families.length} metrics
          {unknown.size > 0 && ` · ${unknown.size} outside the contract`}
          {errors.length > 0 && ` · ${errors.length} unreadable lines`}
        </span>
        <IconChevron className="ml-auto self-center text-ash transition-transform group-open:rotate-180 motion-reduce:transition-none" />
      </summary>
      <div className="flex flex-col gap-4 px-5 pb-5">
        <div className="max-h-[420px] overflow-auto">
          <table className="w-full border-collapse font-display text-[13px]">
            <thead className="sticky top-0 bg-surface text-left text-ash">
              <tr className="border-b border-line">
                <th className="py-1.5 pr-4 font-medium">Series</th>
                <th className="py-1.5 pr-4 font-medium">Labels</th>
                <th className="py-1.5 pr-4 font-medium">Type</th>
                <th className="py-1.5 text-right font-medium">Value</th>
              </tr>
            </thead>
            <tbody>
              {samples.map(({ sample, type }, i) => (
                <tr key={i} className="border-b border-line/50">
                  <td className="py-1 pr-4">
                    {sample.name}
                    {unknown.has(sample) && <span className="ml-2 bg-ground px-1.5 py-0.5 text-[11px] text-ash">not in ADR 0017</span>}
                  </td>
                  <td className="py-1 pr-4 text-ash">
                    {Object.entries(sample.labels)
                      .map(([k, v]) => `${k}="${v}"`)
                      .join(", ")}
                  </td>
                  <td className="py-1 pr-4 text-ash">{type}</td>
                  <td className="py-1 text-right tabular-nums">{Number.isInteger(sample.value) ? formatCount(sample.value) : String(sample.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {errors.length > 0 && (
          <ul className="flex flex-col gap-1 text-xs">
            {errors.map((e) => (
              <li key={e.line} className="text-fault">
                line {e.line}: {e.reason} — <code>{e.text}</code>
              </li>
            ))}
          </ul>
        )}
        <details>
          <summary className="cursor-pointer font-display text-xs font-semibold text-ash hover:text-ink">Raw exposition</summary>
          <pre className="md-code cut mt-2 max-h-[360px] overflow-auto p-4 font-mono text-xs leading-relaxed">{state.last?.raw}</pre>
        </details>
      </div>
    </details>
  );
}
