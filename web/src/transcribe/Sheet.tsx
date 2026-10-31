import { Fragment, memo, type ReactNode, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { AnswerAside } from "./Ask.tsx";
import { ImageAside } from "./ContextImages.tsx";
import { type ContextImage, imageAt } from "./contextImages.ts";
import { type ImageReaderState, readingStatus } from "./imageReading.ts";
import { type Answered, answerAt, type AskerState } from "./ask.ts";
import { type Hit, type HitSpan, partsOf } from "./search.ts";
import { type RewriterState, wordDiff } from "./rewrite.ts";
import { landing, onJump } from "./jump.ts";
import { ANALYSIS_FRESH } from "./suggestions.ts";
import { isAtBottom } from "../ui/scroll.ts";
import { clock, startedLine, startOf } from "./export.ts";
import type { LiveUtterance, ListeningState } from "./listening.ts";
import { titleOf } from "./sessions.ts";
import type { TranscribeActions } from "./useTranscribe.ts";
import type { Speaker, Transcript, Utterance } from "./transcript.ts";

// The Transcript as it is heard (spec speech/07): its heading (drawn by the
// view above the sheet, with the exports, so it stays in sight), then its
// utterances hung on a time spine — each one's mm:ss to the left of the
// rule, its speaker's hue as a tick along it — and the utterances under way at
// the foot, their provisional tail light (Q82), the spine burning with the
// ember there while it listens. The view follows the newest words until the
// reader scrolls up, and offers Jump to live then (Q102). Every completed
// utterance's text is editable inline, and its speaker a menu: a move, a new
// speaker, and a split at the cursor (Q92, Q96, Q98). The answers to the
// questions asked about it stand on the spine where they were asked (Q139),
// and its context images at their moments (#431).

/** A speaker's hue: Me in ink, Others in ash, each voice one of six (styles.css). */
export function speakerColor(speaker: Speaker | undefined): string {
  if (!speaker) return "var(--line)";
  if (speaker.role === "me") return "var(--ink)";
  if (speaker.role === "others") return "var(--ash)";
  return `var(--speaker-${((speaker.id - 1) % 6) + 1})`;
}

/** The label of a call's utterance under way, before the server names its speaker. */
const laneLabel = (lane: LiveUtterance["lane"], call: boolean) => (call ? (lane === "mic" ? "Me" : "Call") : null);

/** What the sheet does to a Transcript: its corrections, Undo, the rewrite's, an image's Read again. */
export type SheetActions = Pick<
  TranscribeActions,
  "moveUtterance" | "splitUtterance" | "editText" | "undoFixes" | "deleteUtterance" | "undoDelete" | "holdUndo" | "rewriteUtterance" | "acceptRewrite" | "closeRewrite" | "forgetAnswer" | "askFrom" | "readImageAgain"
>;

/** No context image being read, none failed: what the sheet says without the reader. */
const NO_READING: ImageReaderState = { reading: null, failed: [] };

/**
 * The sheet. It is handed only what it draws, not the listening's whole state, and redraws only when that changes:
 * the levels move ten times a second and the clock every second.
 */
export const Sheet = memo(function Sheet({
  transcript,
  live,
  phase,
  call,
  here,
  actions,
  deleted,
  hits,
  current,
  seek,
  autoCheck,
  rewrite,
  canRewrite,
  readTo,
  asking,
  readingFrom = null,
  canAsk = false,
  imageReading = NO_READING,
  canRead = false,
}: {
  transcript: Transcript;
  /** The utterances under way of the listening appending here; none when it appends elsewhere. */
  live: LiveUtterance[];
  phase: ListeningState["phase"];
  call: boolean;
  /** The listening appends to this Transcript. */
  here: boolean;
  actions: SheetActions;
  /** The utterance just deleted, while its Undo stands (Q137). */
  deleted: Utterance | null;
  /** The rewrite under way or proposed in this Transcript (Q138), and whether one can be asked for now. */
  rewrite: RewriterState | null;
  canRewrite: boolean;
  /** The search's hits in this Transcript, and the one it stands on (Q136). */
  hits: Hit[];
  current: number;
  /** Bumped by the view when the user searches or steps: the current hit is brought into view then. */
  seek: number;
  /** The automatic check is on (Suggest while listening): a rewrite says when it has not read the utterance yet. */
  autoCheck: boolean;
  /**
   * Where the text model's look has read up to (Q140), and when it may look next while listening (epoch ms, `null`
   * when it is not keeping up); `null` before any look.
   */
  readTo: { untilMs: number; nextAt: number | null } | null;
  /** The question about this Transcript under way or failed (Q139); `null` when there is none, or it is another's. */
  asking: AskerState | null;
  /** What did I miss? reads from this utterance on (missed.ts): the spine is tinted from it; `null`, not shown. */
  readingFrom?: number | null;
  /** A text model serves: each time picks where What did I miss? reads from. */
  canAsk?: boolean;
  /** The context image being read, and the readings that failed (#437): what each image on the spine says under it. */
  imageReading?: ImageReaderState;
  /** A text model with vision serves: an image not read yet is to be read, not waiting for one. */
  canRead?: boolean;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const [following, setFollowing] = useState(true);
  const speakers = new Map(transcript.speakers.map((s) => [s.id, s]));
  const byUtterance = useMemo(() => {
    const map = new Map<number, HitSpan[]>();
    hits.forEach((h, i) => map.set(h.utterance, [...(map.get(h.utterance) ?? []), { start: h.start, end: h.end, current: i === current }]));
    return map;
  }, [hits, current]);

  // A jump from the notes or an answer (Q141): the utterance of that moment in view, lit for a moment.
  const [lit, setLit] = useState<number | null>(null);
  useEffect(
    () =>
      onJump((ms) => {
        const id = landing(transcript.utterances, ms);
        const el = id === null ? null : scroller.current?.querySelector(`[data-utterance="${id}"]`);
        if (!el) return;
        setFollowing(false);
        el.scrollIntoView?.({ block: "center" });
        setLit(id);
      }),
    [transcript.utterances],
  );
  useEffect(() => {
    if (lit === null) return;
    const timer = setTimeout(() => setLit(null), 1_600);
    return () => clearTimeout(timer);
  }, [lit]);

  useLayoutEffect(() => setFollowing(true), [transcript.id]);
  const answering = asking?.writing ? asking : null;
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && following) el.scrollTop = el.scrollHeight;
  }, [transcript.utterances, transcript.images, live, following, answering?.text]);

  // A question just asked: its answer, at the foot above the utterances under way, in view and followed.
  const asked = answering && `${answering.question}@${answering.untilMs}`;
  useLayoutEffect(() => {
    if (asked === null) return;
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
    setFollowing(true);
  }, [asked]);

  // The answers kept, by the utterance each stands before, in the order they were asked.
  const asides = useMemo(() => {
    const at = new Map<number, Answered[]>();
    for (const a of [...(transcript.asked ?? [])].sort((x, y) => x.id - y.id)) {
      const i = answerAt(transcript.utterances, a.untilMs);
      at.set(i, [...(at.get(i) ?? []), a]);
    }
    return at;
  }, [transcript.asked, transcript.utterances]);
  const answeringAt = answering ? answerAt(transcript.utterances, answering.untilMs) : -1;
  // The context images, by the utterance each stands before, in the order they were added.
  const pictures = useMemo(() => {
    const at = new Map<number, ContextImage[]>();
    for (const image of transcript.images ?? []) {
      const i = imageAt(transcript.utterances, image.atMs);
      at.set(i, [...(at.get(i) ?? []), image]);
    }
    return at;
  }, [transcript.images, transcript.utterances]);
  const asidesAt = (i: number) => (
    <>
      {pictures.get(i)?.map((image) => (
        <ImageAside
          key={`image-${image.id}`}
          image={image}
          status={readingStatus(transcript.id, image, imageReading, canRead)}
          onRetry={() => actions.readImageAgain(image.id)}
        />
      ))}
      {asides.get(i)?.map((a) => (
        <AnswerAside key={`answer-${a.id}`} question={a.question} answer={a.answer} untilMs={a.untilMs} onForget={() => actions.forgetAnswer(a.id)} />
      ))}
      {answering && answeringAt === i && <AnswerAside question={answering.question} answer={answering.text} untilMs={answering.untilMs} writing />}
    </>
  );

  // The search's current hit in view when the user searches or steps (`seek`), not when the Transcript changes under
  // it; after the following above, so a query carried into a Transcript just opened lands on its hit.
  const sought = useRef({ hits, current });
  sought.current = { hits, current };
  useLayoutEffect(() => {
    if (seek === 0) return;
    const hit = sought.current.hits[sought.current.current];
    const el = hit && scroller.current?.querySelector(`[data-utterance="${hit.utterance}"]`);
    if (!el) return;
    setFollowing(false);
    el.scrollIntoView?.({ block: "center" });
  }, [seek]);

  // Where the look stopped reading: before the first utterance it had not read, or after the last when it read all.
  const unread = readTo === null ? -1 : transcript.utterances.findIndex((u) => u.endMs > readTo.untilMs);
  const markAt = readTo === null ? -1 : unread === -1 ? transcript.utterances.length : unread;
  const fresh = readTo === null ? 0 : transcript.utterances.filter((u) => u.endMs > readTo.untilMs).length;

  const jump = () => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
    setFollowing(true);
  };

  const empty = transcript.utterances.length === 0 && live.length === 0 && !transcript.images?.length;
  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div ref={scroller} onScroll={(e) => setFollowing(isAtBottom(e.currentTarget))} className="min-h-0 flex-1 overflow-y-auto px-4 py-6 [scrollbar-gutter:stable] md:px-10">
        <div className="mx-auto w-full max-w-3xl">
          {empty ? (
            <Empty listening={here && phase !== "idle"} />
          ) : (
            <ol aria-label="Utterances">
              {transcript.utterances.map((u, i) => (
                // Keyed by the utterance, so a delete, an Undo or a split keeps every other row as it is.
                <Fragment key={u.id}>
                {readTo && i === markAt && i > 0 && <CheckedMark nextAt={readTo.nextAt} fresh={fresh} />}
                {asidesAt(i)}
                <UtteranceRow
                  key={u.id}
                  utterance={u}
                  speaker={u.speaker === null ? undefined : speakers.get(u.speaker)}
                  speakers={transcript.speakers}
                  actions={actions}
                  hits={byUtterance.get(u.id)}
                  rewrite={rewrite?.utterance === u.id ? rewrite : undefined}
                  canRewrite={canRewrite && !rewrite?.writing}
                  unchecked={rewrite?.utterance === u.id && autoCheck && (readTo === null || u.endMs > readTo.untilMs)}
                  lit={lit === u.id}
                  missed={readingFrom === null || i < readingFrom ? null : i === readingFrom ? "from" : "in"}
                  askable={canAsk}
                />
                </Fragment>
              ))}
              {readTo && markAt === transcript.utterances.length && markAt > 0 && <CheckedMark nextAt={readTo.nextAt} fresh={0} />}
              {asidesAt(transcript.utterances.length)}
              {live.map((u) => (
                <LiveRow key={u.key} utterance={u} label={laneLabel(u.lane, call)} burning={phase === "listening" || phase === "finishing"} />
              ))}
            </ol>
          )}
        </div>
      </div>
      {/* The text goes under the foot rather than to a rule: the ask bar below has none. */}
      <span aria-hidden="true" className="pointer-events-none absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-ground" />
      {deleted && (
        <div
          role="status"
          onMouseEnter={() => actions.holdUndo(true)}
          onMouseLeave={() => actions.holdUndo(false)}
          onFocus={() => actions.holdUndo(true)}
          onBlur={() => actions.holdUndo(false)}
          className="cut absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-3 bg-kiln py-2 pr-2 pl-3.5 text-[13px] text-[#eae8e4] [--cut-size:8px]"
        >
          <span>
            Utterance at <span className="font-display tabular-nums">{startOf(deleted)}</span> deleted.
          </span>
          {/* Focus moves here from the row deleted, which is gone. */}
          <button type="button" autoFocus onClick={actions.undoDelete} className="px-2 py-0.5 font-display font-semibold text-[#ff8a4c] hover:text-[#ffa877]">
            Undo
          </button>
        </div>
      )}
      {!deleted && !following && here && phase !== "idle" && (
        <button
          type="button"
          onClick={jump}
          className="cut absolute bottom-3 left-1/2 -translate-x-1/2 bg-ember px-4 py-2 font-display text-[13px] font-semibold text-[#1c2026] [--cut-size:8px] hover:bg-[#ff7a45]"
        >
          Jump to live
        </button>
      )}
    </div>
  );
});

/** The Transcript's title, editable, its start, what was heard and by how many; `children` at the title's right. */
export function Heading({ transcript, onRename, children }: { transcript: Transcript; onRename: (title: string) => void; children?: ReactNode }) {
  const speaking = transcript.speakers.length;
  return (
    <header className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <input
          key={`${transcript.id}:${transcript.title}`}
          aria-label="Transcript title"
          defaultValue={titleOf(transcript)}
          onBlur={(e) => e.currentTarget.value.trim() !== titleOf(transcript) && onRename(e.currentTarget.value)}
          onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
          className="-ml-1 min-w-[12rem] flex-1 rounded-[2px] border border-transparent bg-transparent px-1 font-display text-2xl font-semibold text-ink hover:border-line focus:border-ember focus:outline-none"
        />
        {children}
      </div>
      <p className="flex flex-wrap gap-x-5 gap-y-1 font-display text-[13px] text-ash">
        {transcript.startedAt !== null && <span>{startedLine(transcript.startedAt)}</span>}
        {transcript.clockMs > 0 && <span className="tabular-nums">{clock(transcript.clockMs)} heard</span>}
        {speaking > 0 && <span>{speaking === 1 ? "1 speaker" : `${speaking} speakers`}</span>}
      </p>
    </header>
  );
}

function Empty({ listening }: { listening: boolean }) {
  return (
    <p className="mt-10 max-w-[60ch] text-[15px] leading-relaxed text-ash">
      {listening
        ? "Listening. Each utterance shows here as it is heard, and settles when its text is complete."
        : "Nothing heard yet. Pick a source below and start: the microphone hears the room, a call adds the audio of a tab, a window or the screen, and a file is transcribed faster than real time."}
    </p>
  );
}

/** The spine and the time beside it: one row of the sheet. */
function Row({
  time,
  tick,
  label,
  id,
  lit = false,
  missed = null,
  onTime,
  children,
}: {
  time: string;
  tick: { color?: string; burning?: boolean };
  label: ReactNode;
  id?: number;
  lit?: boolean;
  /** What did I miss? reads from this row, or it is among those it reads: the spine and the time in the ember. */
  missed?: "from" | "in" | null;
  /** Its time picks it as where What did I miss? reads from. */
  onTime?: () => void;
  children: ReactNode;
}) {
  const tone = missed === "from" ? "text-ember" : missed === "in" ? "text-ember/70" : "text-ash";
  return (
    <li
      data-utterance={id}
      className={`grid grid-cols-[3.75rem_1fr] gap-x-4 transition-colors duration-700 motion-reduce:transition-none sm:grid-cols-[4.5rem_1fr] ${lit ? "bg-ember/12" : ""}`}
    >
      {onTime ? (
        <button
          type="button"
          onClick={onTime}
          aria-label={`Ask what was said from ${time}`}
          title={missed === "from" ? "What did I miss? reads from here. Press again for the last 5 minutes." : "Ask what was said from here"}
          className={`self-start pt-[3px] text-right font-display text-[13px] tabular-nums hover:text-ember ${tone}`}
        >
          {time}
        </button>
      ) : (
        <span className="pt-[3px] text-right font-display text-[13px] text-ash tabular-nums">{time}</span>
      )}
      <div className={`relative border-l-2 pb-5 pl-4 ${missed ? "border-ember/45" : "border-line"}`}>
        <span
          aria-hidden="true"
          className={`absolute top-1 bottom-5 -left-[3px] w-1 ${tick.burning ? "spine-live" : ""}`}
          style={tick.burning ? undefined : { background: tick.color }}
        />
        {label}
        {children}
      </div>
    </li>
  );
}

/**
 * `text` with the words fixes put in it marked (Q133), each where it stands as words of its own, and the search's
 * hits (Q136), the current one in the ember.
 */
export function highlighted(text: string, fixed: string[], hits: HitSpan[] = []): ReactNode {
  if (fixed.length === 0 && hits.length === 0) return text;
  return partsOf(text, fixed, hits).map((part, i) => {
    if (!part.fixed && part.hit === null) return part.text;
    const fix = part.fixed ? "underline decoration-ember decoration-dotted decoration-2 underline-offset-4" : "";
    const hit = part.hit === "current" ? "bg-ember text-[#1c2026]" : part.hit === "other" ? "bg-ember/25 text-ink" : "bg-transparent text-ink";
    return (
      <mark key={i} data-hit={part.hit ?? undefined} className={`${hit} ${fix}`}>
        {part.text}
      </mark>
    );
  });
}

/** A rewrite of an utterance (Q138): under way, proposed with the words it changes, or failed. */
export function RewritePanel({
  utterance: u,
  rewrite,
  unchecked = false,
  actions,
}: {
  utterance: Utterance;
  rewrite: RewriterState;
  /** The automatic look has not read this utterance yet (Q140): said, since it might have fixed it by itself. */
  unchecked?: boolean;
  actions: Pick<TranscribeActions, "acceptRewrite" | "closeRewrite">;
}) {
  const unchanged = rewrite.proposal !== null && rewrite.proposal === u.text;
  return (
    <div role="status" className="relative mt-2 flex flex-col gap-2 bg-surface/70 py-2.5 pr-3 pl-3.5">
      <span aria-hidden="true" className={`absolute inset-y-0 left-0 w-[2px] ${rewrite.writing ? "spine-live" : "bg-ember"}`} />
      {rewrite.writing ? (
        <div className="flex items-center gap-3">
          <span className="text-[13px] text-ash">Rebuilding what was said from the conversation around it…</span>
          <button type="button" onClick={actions.closeRewrite} className="ml-auto font-display text-xs text-ash hover:text-ink">
            Stop
          </button>
        </div>
      ) : rewrite.failed !== null ? (
        <div className="flex items-start gap-3">
          <span className="text-[13px] text-fault">The rewrite failed: {rewrite.failed}</span>
          <button type="button" onClick={actions.closeRewrite} className="ml-auto font-display text-xs text-ash hover:text-ink">
            Close
          </button>
        </div>
      ) : unchanged ? (
        <div className="flex items-center gap-3">
          <span className="text-[13px] text-ash">The model would leave it as it is.</span>
          <button type="button" onClick={actions.closeRewrite} className="ml-auto font-display text-xs text-ash hover:text-ink">
            Close
          </button>
        </div>
      ) : (
        <>
          <p className="text-[15px] leading-relaxed text-ink">
            {wordDiff(u.text, rewrite.proposal ?? "").map((part, i) => (
              <span key={i}>
                {i > 0 && " "}
                {part.kind === "same" ? (
                  part.text
                ) : part.kind === "added" ? (
                  <ins className="text-ink underline decoration-ember decoration-2 underline-offset-4">{part.text}</ins>
                ) : (
                  <del className="text-ash decoration-ash/80">{part.text}</del>
                )}
              </span>
            ))}
          </p>
          <div className="flex items-center gap-2">
            <button type="button" onClick={actions.acceptRewrite} className="cut bg-ink px-3 py-1 font-display text-xs font-semibold text-ground [--cut-size:6px] hover:bg-ember">
              Accept
            </button>
            <button type="button" onClick={actions.closeRewrite} className="px-1.5 py-1 font-display text-xs text-ash hover:text-ink">
              Discard
            </button>
            <span className="text-xs text-ash">The text as heard is kept: Undo fix brings it back.</span>
          </div>
        </>
      )}
      {unchecked && rewrite.failed === null && <p className="text-xs leading-snug text-ash">The automatic check has not read this utterance yet: its next look might fix it too.</p>}
    </div>
  );
}

/**
 * When the next look comes, as the mark says it: in about 40 s, once more is said (the look waits for three new
 * utterances, ANALYSIS_FRESH), or any moment.
 */
export function nextCheckIn(at: number, now: number, fresh = ANALYSIS_FRESH): string {
  const seconds = Math.round((at - now) / 1_000);
  if (seconds <= 5) return fresh < ANALYSIS_FRESH ? "next check once more is said" : "next check any moment";
  return seconds >= 60 ? `next check in about ${Math.round(seconds / 60)} min` : `next check in about ${Math.ceil(seconds / 5) * 5} s`;
}

/** The line across the spine where the text model's look stopped reading (Q140), and when the next may come. */
function CheckedMark({ nextAt, fresh }: { nextAt: number | null; fresh: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (nextAt === null) return;
    const timer = setInterval(() => setNow(Date.now()), 5_000);
    return () => clearInterval(timer);
  }, [nextAt]);
  return (
    <li role="separator" aria-label="Checked by the text model up to here" className="grid grid-cols-[3.75rem_1fr] gap-x-4 sm:grid-cols-[4.5rem_1fr]">
      <span aria-hidden="true" />
      <div className="relative border-l-2 border-line pb-4 pl-4">
        <span aria-hidden="true" className="absolute top-[4px] -left-[5px] size-2 rotate-45 bg-ember" />
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="h-0 flex-1 border-t border-dashed border-ember/55" />
          <span className="shrink-0 font-display text-[11px] leading-4 text-ember">
            Checked to here{nextAt !== null && <span className="text-ash"> · {nextCheckIn(nextAt, now, fresh)}</span>}
          </span>
        </div>
      </div>
    </li>
  );
}

/** An utterance of the sheet: drawn again only when it, its speaker or the speakers change. */
const UtteranceRow = memo(function UtteranceRow({
  utterance: u,
  speaker,
  speakers,
  actions,
  hits,
  rewrite,
  canRewrite,
  unchecked,
  lit,
  missed,
  askable,
}: {
  utterance: Utterance;
  speaker: Speaker | undefined;
  speakers: Speaker[];
  actions: SheetActions;
  hits: HitSpan[] | undefined;
  rewrite: RewriterState | undefined;
  canRewrite: boolean;
  /** The text model's look has not read it yet (Q140). */
  unchecked: boolean;
  /** A jump just landed on it (Q141). */
  lit: boolean;
  /** What did I miss? reads from it, or it is among those it reads. */
  missed: "from" | "in" | null;
  /** Its time picks it as where What did I miss? reads from. */
  askable: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const cursor = useRef(0);
  const color = speakerColor(speaker);
  const label = speaker && (
    <SpeakerMenu
      value={speaker.id}
      speakers={speakers}
      color={color}
      ariaLabel={`Speaker of the utterance at ${startOf(u)}`}
      onChoose={(to) => actions.moveUtterance(u.id, to)}
    />
  );
  return (
    <Row time={startOf(u)} tick={{ color }} label={label} id={u.id} lit={lit} missed={missed} onTime={askable ? () => actions.askFrom(u.startMs) : undefined}>
      {editing ? (
        <div className="flex flex-col gap-1.5">
          <textarea
            aria-label={`Text of the utterance at ${startOf(u)}`}
            autoFocus
            value={u.text}
            onChange={(e) => actions.editText(u.id, e.currentTarget.value)}
            onSelect={(e) => (cursor.current = e.currentTarget.selectionStart)}
            onBlur={(e) => {
              // Moving to the split menu keeps the editor open: the cursor it splits at is here.
              if (!e.currentTarget.parentElement?.contains(e.relatedTarget as Node | null)) setEditing(false);
            }}
            onKeyDown={(e) => e.key === "Escape" && setEditing(false)}
            className="w-full resize-none rounded-[2px] border border-line bg-surface px-2.5 py-1.5 text-[15px] leading-relaxed text-ink [field-sizing:content] focus:border-ember focus:outline-none"
          />
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
            <SpeakerMenu
              value={null}
              speakers={speakers}
              prompt="Split at the cursor, the rest to…"
              ariaLabel="Split this utterance at the cursor"
              onChoose={(to) => {
                actions.splitUtterance(u.id, cursor.current, to);
                setEditing(false);
              }}
            />
            <button
              type="button"
              aria-disabled={!canRewrite}
              title="The text model rebuilds what was said, from the context and the conversation around it"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => canRewrite && actions.rewriteUtterance(u.id)}
              className="cut bg-ember px-3 py-1 font-display text-xs font-semibold text-[#1c2026] [--cut-size:6px] hover:bg-[#ff7a45] aria-disabled:cursor-not-allowed aria-disabled:bg-line aria-disabled:text-ash"
            >
              Rewrite
            </button>
            <button
              type="button"
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => actions.deleteUtterance(u.id)}
              className="ml-auto font-display text-xs text-ash hover:text-fault"
            >
              Delete
            </button>
          </div>
          {rewrite && <RewritePanel utterance={u} rewrite={rewrite} unchecked={unchecked} actions={actions} />}
        </div>
      ) : (
        <div>
          <p
            tabIndex={0}
            title="Edit the text"
            onClick={() => setEditing(true)}
            onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), setEditing(true))}
            className="cursor-text rounded-[2px] text-[15px] leading-relaxed whitespace-pre-wrap text-ink hover:bg-surface/70"
          >
            {u.text ? highlighted(u.text, u.fixed ?? [], hits) : <span className="text-ash">(no text)</span>}
          </p>
          {rewrite && <RewritePanel utterance={u} rewrite={rewrite} unchecked={unchecked} actions={actions} />}
          {u.heard !== undefined && (
            <button
              type="button"
              title={`Heard as: ${u.heard}`}
              onClick={() => actions.undoFixes(u.id)}
              className="font-display text-xs text-ash hover:text-ember"
            >
              Undo fix
            </button>
          )}
        </div>
      )}
    </Row>
  );
});

/**
 * A speaker as a native menu: the speakers, then a new one. Native, so it opens over a clipped `.cut` box and on a
 * phone. With `value` null it is an action menu that resets after each choice.
 */
function SpeakerMenu({
  value,
  speakers,
  color,
  prompt,
  ariaLabel,
  onChoose,
}: {
  value: number | null;
  speakers: Speaker[];
  color?: string;
  prompt?: string;
  ariaLabel: string;
  onChoose: (to: number | "new") => void;
}) {
  return (
    <select
      aria-label={ariaLabel}
      value={value === null ? "" : String(value)}
      onChange={(e) => {
        const chosen = e.currentTarget.value;
        if (chosen !== "") onChoose(chosen === "new" ? "new" : Number(chosen));
      }}
      style={color ? { color } : undefined}
      className={`max-w-full cursor-pointer appearance-none rounded-[2px] bg-transparent py-0.5 pr-1 font-display text-[13px] font-semibold hover:bg-surface focus:outline-none focus-visible:outline-2 focus-visible:outline-ember ${color ? "" : "self-start border border-line px-2 text-ash"}`}
    >
      {value === null && (
        <option value="" disabled>
          {prompt}
        </option>
      )}
      {speakers.map((s) => (
        <option key={s.id} value={s.id}>
          {s.name}
        </option>
      ))}
      <option value="new">New speaker</option>
    </select>
  );
}

function LiveRow({ utterance: u, label, burning }: { utterance: LiveUtterance; label: string | null; burning: boolean }) {
  const heard = (u.stable + u.tail).trim() !== "";
  return (
    <Row time={clock(u.startMs)} tick={{ burning }} label={label && <span className="font-display text-[13px] font-semibold text-ash">{label}</span>}>
      <p aria-live="polite" className="text-[15px] leading-relaxed">
        {heard ? (
          <>
            <span className="text-ink">{u.stable}</span>
            <span className="text-ash">{u.tail}</span>
          </>
        ) : (
          <span className="text-ash">…</span>
        )}
      </p>
    </Row>
  );
}
