import { type ReactNode, type Ref, useEffect, useMemo, useRef, useState } from "react";
import type { SpeechStatus } from "../api/model.ts";
import { caption } from "../ui/classes.ts";
import { IconBell, IconChevron, IconClose, IconDownload, IconPlus, IconSearch, IconTrash } from "../ui/icons.tsx";
import { hitsIn, matchesOf } from "./search.ts";
import { useWatched } from "./assist.ts";
import { onJump } from "./jump.ts";
import { ContextCard } from "./Context.tsx";
import { type ImageAdding, ImagesCard, useImageAdding } from "./ContextImages.tsx";
import { AskBar } from "./Ask.tsx";
import type { CallContext } from "./context.ts";
import { useSavedContext } from "./defaultContext.ts";
import { escapeClears, missedSince } from "./missed.ts";
import { Deck } from "./Deck.tsx";
import { Notes } from "./Notes.tsx";
import { Suggestions } from "./Suggestions.tsx";
import { openSuggestions } from "./suggestions.ts";
import { Switch } from "../ui/Switch.tsx";
import { HintText } from "../speech/Dictation.tsx";
import { fileStem, toMarkdown, toSrt, toText, toVtt } from "./export.ts";
import { StoreWorkspaceChooser } from "../workspace/WorkspaceChooser.tsx";
import { type Destination, sentLine } from "../workspace/send.ts";
import { pageWorkspaces, type Workspaces } from "../workspace/workspaces.ts";
import { sendTranscript, workspaceNameFor } from "./toWorkspace.ts";
import { chooseFilePace, FILE_PACE_CHOICES, useFilePace } from "./filePace.ts";
import { UTTERANCE_SILENCE_CHOICES, useUtteranceSilence } from "./silence.ts";
import { type ListeningHandle, type ListeningState, type LiveUtterance, useListening } from "./listening.ts";
import { Heading, Sheet, speakerColor } from "./Sheet.tsx";
import { titleOf, type TranscriptList } from "./sessions.ts";
import { sessionsWithSpeakers, type Transcript } from "./transcript.ts";
import type { Assistants, TranscribeActions, TranscribeState } from "./useTranscribe.ts";

// The Transcribe view (spec speech/07), in the Decide tab's three-column
// shape: the Transcripts on the left; the one open in the middle, its heading
// and exports above the sheet and the deck at its foot; on the right three
// tabs (Q129): Call, what the transcription is told (the context), its
// context images (ContextImages.tsx) and who speaks; Notes (Notes.tsx);
// Settings, the view's own.
// Below `lg` the two side columns are drawers the header opens. The state is
// the page's (useTranscribe.ts): a listening goes on while the view is hidden.

/** The utterances under way of a listening that appends elsewhere: one array, so the sheet is not drawn again. */
const NONE_LIVE: LiveUtterance[] = [];

/** Which side drawer is out, below `lg`. */
export type Drawer = "sessions" | "settings" | null;

export function TranscribeView({
  state,
  actions,
  listening: handle,
  assistants,
  speech,
  drawer,
  onDrawer,
  onOpenInChat,
  workspaces,
}: {
  state: TranscribeState;
  actions: TranscribeActions;
  listening: ListeningHandle;
  assistants: Assistants;
  speech?: SpeechStatus;
  drawer: Drawer;
  onDrawer: (drawer: Drawer) => void;
  /** A new chat, its composer seeded with the Transcript as Markdown; nothing sent (Q89). */
  onOpenInChat: (markdown: string) => void;
  /** The page's workspaces, where Send to workspace adds the Transcript; the page's own when left out. */
  workspaces?: Workspaces;
}) {
  // Outside https and localhost the browser gives no page a microphone; a test has no such flag.
  const secure = typeof globalThis.isSecureContext === "boolean" ? globalThis.isSecureContext : true;
  const listening = useListening(handle);
  const here = listening.transcript === state.active.id;
  const [tab, setTab] = useState<SideTab>("call");
  // The search in the Transcript open (Q136): its query, its hits, the one it stands on.
  const [query, setQuery] = useState("");
  const [current, setCurrent] = useState(0);
  /** Bumped when the user searches or steps, so only then the sheet brings the hit into view. */
  const [seek, setSeek] = useState(0);
  const hits = useMemo(() => hitsIn(state.active, query), [state.active, query]);
  const at = hits.length === 0 ? 0 : Math.min(current, hits.length - 1);
  const searchField = useRef<HTMLInputElement>(null);
  const saved = useSavedContext();
  // A context image (#431): a paste anywhere in the view while it shows, a file or Capture in the Call tab.
  const shown = useRef<HTMLDivElement>(null);
  const adding = useImageAdding({ transcript: state.active, listening: handle, actions, view: shown });
  // What did I miss?: what it reads, tinted on the spine while it is moved from the last five minutes or pointed at.
  const [previewing, setPreviewing] = useState(false);
  const missedFrom = state.missedFrom;
  const readingFrom = useMemo(
    () => (missedFrom !== null || previewing ? missedSince(state.active.utterances, missedFrom).index : null),
    [state.active.utterances, missedFrom, previewing],
  );
  useEffect(() => {
    if (missedFrom === null) return;
    const back = (e: globalThis.KeyboardEvent) => escapeClears(e.key, e.target as HTMLElement | null, drawer !== null) && actions.clearMissed();
    window.addEventListener("keydown", back);
    return () => window.removeEventListener("keydown", back);
  }, [missedFrom, drawer, actions]);
  // The first Start's question, for the Transcript it was raised on, until it is started (Deck.tsx).
  const [checking, setChecking] = useState<number | null>(null);
  const checkingContext = checking === state.active.id && state.active.source === null && listening.phase === "idle";
  const editContext = () => {
    setTab("call");
    onDrawer("settings");
    if (state.draft === null) actions.editContext();
  };
  const rewrite = useWatched(assistants.rewriter);
  const asking = useWatched(assistants.asker);
  const looking = useWatched(assistants.analyst);
  const keepingUp = here && listening.phase !== "idle" && state.suggestionsOn;
  // Held still between changes: the sheet is drawn again only when what it is handed changes.
  const untilMs = state.active.checkedUntilMs;
  const nextAt = keepingUp ? (looking.nextAt[state.active.id] ?? null) : null;
  const readTo = useMemo(() => (untilMs === undefined ? null : { untilMs, nextAt }), [untilMs, nextAt]);
  const search = (next: string) => {
    setQuery(next);
    setCurrent(0);
    setSeek((n) => n + 1);
  };

  // Ctrl+F (Cmd+F) searches the transcript while this view shows; the browser's own search otherwise.
  useEffect(() => {
    const find = (e: globalThis.KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "f") return;
      const field = searchField.current;
      if (!field || field.offsetParent === null) return;
      e.preventDefault();
      field.focus();
      field.select();
    };
    window.addEventListener("keydown", find);
    return () => window.removeEventListener("keydown", find);
  }, []);

  // A jump to a moment from the notes or an answer (Q141): the drawer it came from gives way to the sheet.
  useEffect(() => onJump(() => onDrawer(null)), [onDrawer]);

  useEffect(() => {
    if (!drawer) return;
    const close = (e: globalThis.KeyboardEvent) => e.key === "Escape" && onDrawer(null);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [drawer, onDrawer]);

  return (
    <div ref={shown} className="relative flex min-h-0 flex-1">
      {drawer && <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => onDrawer(null)} aria-hidden />}
      {adding.editor}

      <Transcripts
        list={state.list}
        listening={listening}
        open={drawer === "sessions"}
        notice={state.storeError}
        onNew={() => {
          actions.newTranscript();
          onDrawer(null);
        }}
        onSelect={(id, carried) => {
          actions.select(id);
          if (carried !== undefined) search(carried);
          onDrawer(null);
        }}
        onRemove={actions.remove}
      />

      <main aria-label="Transcript" className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-line px-4 pt-5 pb-3 md:px-10">
          <div className="mx-auto w-full max-w-3xl">
            <Heading transcript={state.active} onRename={(title) => actions.rename(state.active.id, title)}>
              <SearchField
                inputRef={searchField}
                query={query}
                count={hits.length}
                at={at}
                onQuery={search}
                onStep={(by) => {
                  if (hits.length === 0) return;
                  setCurrent((at + by + hits.length) % hits.length);
                  setSeek((n) => n + 1);
                }}
              />
              <Exports transcript={state.active} onOpenInChat={onOpenInChat} workspaces={workspaces} />
            </Heading>
          </div>
        </div>
        <Sheet
          transcript={state.active}
          live={here ? listening.live : NONE_LIVE}
          phase={listening.phase}
          call={listening.source === "call"}
          here={here}
          actions={actions}
          deleted={state.deleted}
          hits={hits}
          current={at}
          seek={seek}
          autoCheck={state.suggestionsOn}
          rewrite={rewrite.transcript === state.active.id ? rewrite : null}
          // One rewrite at a time for the page: none while another, in any Transcript, is under way.
          canRewrite={state.textBlocked === null && !rewrite.writing}
          readTo={readTo}
          asking={asking.transcript === state.active.id ? asking : null}
          readingFrom={readingFrom}
          canAsk={state.textBlocked === null}
        />
        <AskBar transcript={state.active} state={state} actions={actions} asking={asking} onPreview={setPreviewing} />
        <Deck
          state={state}
          listening={listening}
          actions={actions}
          speech={speech}
          secure={secure}
          here={here}
          saved={saved}
          checking={checkingContext}
          // The Call tab, where the panel stands beside the sheet; below `lg` its drawer would cover the deck.
          onCheck={() => {
            setChecking(state.active.id);
            setTab("call");
          }}
          onEditContext={editContext}
        />
      </main>

      <Side open={drawer === "settings"} tab={tab} onTab={setTab} waiting={openSuggestions(state.active).length}>
        {tab === "call" && (
          <>
            <Suggestions transcript={state.active} state={state} actions={actions} analyst={assistants.analyst} />
            <Call
              transcript={state.active}
              state={state}
              actions={actions}
              live={here && listening.phase !== "idle"}
              saved={saved}
              flagged={checkingContext}
              sharing={here && listening.sharing}
              adding={adding}
            />
          </>
        )}
        {tab === "notes" && <Notes transcript={state.active} state={state} actions={actions} notes={assistants.notes} />}
        {tab === "settings" && <Settings state={state} actions={actions} />}
      </Side>
    </div>
  );
}

/** What a Transcript's row says under its title. */
export function transcriptSummary(t: Transcript, listening: ListeningState): string {
  if (listening.transcript === t.id) return listening.phase === "paused" ? "Paused" : "Listening";
  const n = t.utterances.length;
  return n === 0 ? "Nothing heard yet" : n === 1 ? "1 utterance" : `${n} utterances`;
}

/** The heading's search (Q136): the query, "3 of 12", the previous and the next hit; Enter steps, Escape clears. */
export function SearchField({
  inputRef,
  query,
  count,
  at,
  onQuery,
  onStep,
}: {
  inputRef?: Ref<HTMLInputElement>;
  query: string;
  count: number;
  at: number;
  onQuery: (query: string) => void;
  onStep: (by: number) => void;
}) {
  const searching = query.trim() !== "";
  return (
    <div role="search" className={`flex h-[34px] items-center gap-1 border bg-ground pr-1 pl-2 ${searching ? "border-ember/70" : "border-line"} focus-within:border-ember`}>
      <span className="text-ash" aria-hidden="true">
        <IconSearch />
      </span>
      <input
        ref={inputRef}
        type="search"
        aria-label="Search the transcript"
        value={query}
        placeholder="Search"
        onChange={(e) => onQuery(e.currentTarget.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.preventDefault(), onStep(e.shiftKey ? -1 : 1));
          if (e.key === "Escape") onQuery("");
        }}
        className="w-28 min-w-0 bg-transparent font-display text-[13px] text-ink placeholder:text-ash focus:outline-none [&::-webkit-search-cancel-button]:hidden"
      />
      {searching && (
        <>
          <span aria-live="polite" className={`shrink-0 font-display text-xs tabular-nums ${count === 0 ? "text-fault" : "text-ash"}`}>
            {count === 0 ? "None" : `${at + 1} of ${count}`}
          </span>
          <button type="button" aria-label="Previous match" disabled={count === 0} onClick={() => onStep(-1)} className="grid size-6 place-items-center text-ash hover:text-ink disabled:opacity-40">
            <IconChevron />
          </button>
          <button type="button" aria-label="Next match" disabled={count === 0} onClick={() => onStep(1)} className="grid size-6 place-items-center text-ash hover:text-ink disabled:opacity-40">
            <IconChevron className="rotate-180" />
          </button>
          <button type="button" aria-label="Clear the search" onClick={() => onQuery("")} className="grid size-6 place-items-center text-ash hover:text-ink">
            <IconClose />
          </button>
        </>
      )}
    </div>
  );
}

/** What a Transcript's row says while the list is searched: its hits, or that its title holds the query. */
export function listMatch(hits: number): string {
  if (hits === 0) return "In the title";
  return hits === 1 ? "1 match" : `${hits} matches`;
}

function Transcripts({
  list,
  listening,
  open,
  notice,
  onNew,
  onSelect,
  onRemove,
}: {
  list: TranscriptList;
  listening: ListeningState;
  open: boolean;
  notice: string | null;
  onNew: () => void;
  /** `carried`, the list's query, for the Transcript's own search. */
  onSelect: (id: number, carried?: string) => void;
  onRemove: (id: number) => void;
}) {
  const [query, setQuery] = useState("");
  const searching = query.trim() !== "";
  const rows = list.transcripts.map((t) => ({ t, match: searching ? matchesOf(t, titleOf(t), query) : null })).filter(({ match }) => !match || match.hits > 0 || match.title);
  return (
    <aside
      aria-label="Transcripts"
      className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-line bg-ground transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:w-60 lg:translate-x-0 ${open ? "translate-x-0" : "max-lg:invisible max-lg:-translate-x-full"}`}
    >
      <div className="p-3">
        <button type="button" className="cut flex w-full items-center gap-2 bg-surface px-3 py-2.5 font-display text-sm font-medium text-ink hover:bg-line" onClick={onNew}>
          <IconPlus />
          New transcript
        </button>
      </div>
      <div className="px-3 pb-2">
        <label className={`flex h-8 items-center gap-1.5 border bg-ground px-2 ${searching ? "border-ember/70" : "border-line"} focus-within:border-ember`}>
          <span className="text-ash" aria-hidden="true">
            <IconSearch />
          </span>
          <input
            type="search"
            aria-label="Search the transcripts"
            value={query}
            placeholder="Search transcripts"
            onChange={(e) => setQuery(e.currentTarget.value)}
            onKeyDown={(e) => e.key === "Escape" && setQuery("")}
            className="min-w-0 flex-1 bg-transparent text-[13px] text-ink placeholder:text-ash focus:outline-none [&::-webkit-search-cancel-button]:hidden"
          />
        </label>
      </div>
      {searching && rows.length === 0 && <p className="px-6 py-2 text-xs leading-snug text-ash">No transcript holds “{query.trim()}”.</p>}
      <ul className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-3 pb-3">
        {rows.map(({ t, match }) => {
          const active = t.id === list.activeId;
          const heard = listening.transcript === t.id;
          return (
            <li key={t.id} className={`group relative flex items-stretch ${active ? "bg-surface shadow-[inset_2px_0_0_var(--ember)]" : "hover:bg-surface/60"}`}>
              <button
                type="button"
                aria-current={active ? "true" : undefined}
                className="flex min-w-0 flex-1 flex-col gap-0.5 px-3 py-2 text-left"
                onClick={() => onSelect(t.id, match && match.hits > 0 ? query.trim() : undefined)}
              >
                <span className={`truncate text-sm ${active ? "text-ink" : "text-ink/85"}`}>{titleOf(t)}</span>
                <span className={`flex items-center gap-1.5 font-display text-xs ${match ? "text-ember" : "text-ash"}`}>
                  {heard && <span className="cut size-2 bg-ember [--cut-size:3px]" aria-hidden />}
                  {match ? listMatch(match.hits) : transcriptSummary(t, listening)}
                </span>
              </button>
              <button
                type="button"
                aria-label={`Delete transcript: ${titleOf(t)}`}
                title="Delete transcript"
                disabled={heard}
                className={`grid w-8 shrink-0 place-items-center text-ash hover:text-fault focus-visible:opacity-100 disabled:hidden ${active ? "" : "opacity-0 group-hover:opacity-100 max-lg:opacity-100"}`}
                onClick={() => onRemove(t.id)}
              >
                <IconTrash />
              </button>
            </li>
          );
        })}
      </ul>
      {notice && (
        <p role="status" className="border-t border-line px-3 py-2 text-xs text-fault">
          {notice}
        </p>
      )}
    </aside>
  );
}

export { fileStem };

const EXPORTS: { label: string; extension: string; type: string; write: (t: Transcript) => string }[] = [
  { label: ".txt", extension: "txt", type: "text/plain", write: toText },
  { label: ".md", extension: "md", type: "text/markdown", write: toMarkdown },
  { label: ".srt", extension: "srt", type: "application/x-subrip", write: toSrt },
  { label: ".vtt", extension: "vtt", type: "text/vtt", write: toVtt },
];

function download(t: Transcript, { extension, type, write }: (typeof EXPORTS)[number]) {
  const url = URL.createObjectURL(new Blob([write(t)], { type: `${type};charset=utf-8` }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `${fileStem(titleOf(t))}.${extension}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

export type SideTab = "call" | "notes" | "settings";

const SIDE_TABS: [SideTab, string][] = [
  ["call", "Call"],
  ["notes", "Notes"],
  ["settings", "Settings"],
];

/**
 * The right panel, in tabs as the chat's settings are, the bell counting the open Transcript's suggestions `waiting`
 * (Q132). A drawer below `lg`, shown when `open`.
 */
function Side({ open, tab, onTab, waiting, children }: { open: boolean; tab: SideTab; onTab: (tab: SideTab) => void; waiting: number; children: ReactNode }) {
  return (
    <aside
      aria-label="Call, notes and settings"
      className={`fixed inset-y-0 right-0 z-40 flex w-80 flex-col overflow-y-auto border-l border-line bg-ground px-5 pb-5 transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:translate-x-0 xl:w-96 ${open ? "translate-x-0" : "max-lg:invisible max-lg:translate-x-full"}`}
    >
      <div role="tablist" aria-label="Panel sections" className="sticky top-0 z-10 -mx-5 mb-5 flex gap-5 border-b border-line bg-ground px-5 pt-4">
        {SIDE_TABS.map(([t, label]) => (
          <button
            key={t}
            type="button"
            role="tab"
            id={`transcribe-tab-${t}`}
            aria-selected={tab === t}
            aria-controls="transcribe-tab-panel"
            onClick={() => onTab(t)}
            className={`-mb-px border-b-2 pb-2 font-display text-[13px] font-semibold ${tab === t ? "border-ember text-ink" : "border-transparent text-ash hover:text-ink"}`}
          >
            {label}
          </button>
        ))}
        {waiting > 0 && (
          <button
            type="button"
            aria-label={waiting === 1 ? "1 suggestion" : `${waiting} suggestions`}
            title="Suggestions, in the Call tab"
            onClick={() => onTab("call")}
            className="-mb-px ml-auto flex items-center gap-1 pb-2 font-display text-xs font-semibold text-ember tabular-nums hover:text-ink"
          >
            <IconBell />
            {waiting}
          </button>
        )}
      </div>
      <div id="transcribe-tab-panel" role="tabpanel" aria-labelledby={`transcribe-tab-${tab}`} className="flex flex-1 flex-col gap-6">
        {children}
      </div>
    </aside>
  );
}

/** The heading's exports: Copy, the four downloads behind one menu, and Open in chat. */
function Exports({ transcript: t, onOpenInChat, workspaces }: { transcript: Transcript; onOpenInChat: (markdown: string) => void; workspaces?: Workspaces }) {
  const [copied, setCopied] = useState(false);
  const [choosing, setChoosing] = useState(false);
  const [sent, setSent] = useState<string | null>(null);
  const menu = useRef<HTMLDetailsElement>(null);
  const nothing = t.utterances.every((u) => u.text.trim() === "");
  return (
    <div className="flex items-center gap-1.5">
      <button
        type="button"
        disabled={nothing}
        onClick={() => {
          void navigator.clipboard?.writeText(toText(t)).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2_000);
          });
        }}
        className="cut bg-ink px-3 py-1.5 font-display text-[13px] font-semibold text-ground [--cut-size:7px] hover:bg-ember disabled:cursor-not-allowed disabled:opacity-50"
      >
        {copied ? "Copied" : "Copy"}
      </button>
      <details ref={menu} className={`group relative ${nothing ? "pointer-events-none opacity-50" : ""}`}>
        <summary
          aria-disabled={nothing}
          className="flex cursor-pointer list-none items-center gap-1 border border-line px-2.5 py-1.5 font-display text-[13px] font-medium text-ink hover:border-ember [&::-webkit-details-marker]:hidden"
        >
          <IconDownload />
          Download
          <IconChevron className="rotate-180 transition-transform group-open:rotate-0 motion-reduce:transition-none" />
        </summary>
        <ul className="absolute right-0 z-20 mt-1 flex min-w-36 flex-col border border-line bg-surface py-1 shadow-lg">
          {EXPORTS.map((format) => (
            <li key={format.extension}>
              <button
                type="button"
                onClick={() => {
                  download(t, format);
                  menu.current?.removeAttribute("open");
                }}
                className="w-full px-3 py-1.5 text-left font-display text-[13px] text-ink hover:bg-line"
              >
                {format.label}
              </button>
            </li>
          ))}
        </ul>
      </details>
      <button
        type="button"
        disabled={nothing}
        title="A new chat opens with the transcript in its prompt box. Nothing is sent until you press Send."
        onClick={() => onOpenInChat(toMarkdown(t))}
        className="px-1.5 font-display text-[13px] font-semibold text-ember hover:underline disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:no-underline"
      >
        Open in chat
      </button>
      <div className="relative">
        <button
          type="button"
          disabled={nothing}
          title="Adds the transcript, as Markdown, to a workspace the model can read. The transcript stays here."
          onClick={() => {
            setSent(null);
            setChoosing(!choosing);
          }}
          className="px-1.5 font-display text-[13px] font-semibold text-ember hover:underline disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:no-underline"
        >
          Send to workspace
        </button>
        {choosing && (
          <StoreWorkspaceChooser
            source={workspaces ?? pageWorkspaces()}
            newName={workspaceNameFor(t)}
            onCancel={() => setChoosing(false)}
            onChoose={(destination: Destination) => {
              setChoosing(false);
              sendTranscript(workspaces ?? pageWorkspaces(), t, destination).then(
                (done) => setSent(sentLine({ ok: true, file: done.written.name, workspace: done.workspace.name })),
                (err: unknown) => setSent(sentLine({ ok: false, error: err instanceof Error ? err.message : String(err) })),
              );
            }}
          />
        )}
        {sent && (
          <p role="status" className="absolute right-0 z-20 mt-1 w-64 border border-line bg-surface px-3 py-1.5 text-xs text-ink shadow-lg">
            {sent}
          </p>
        )}
      </div>
    </div>
  );
}

/** The Call tab: what the transcription is told, its context images, and who speaks. */
export function Call({
  transcript: t,
  state,
  actions,
  live,
  saved,
  flagged,
  sharing = false,
  adding,
}: {
  transcript: Transcript;
  state: TranscribeState;
  actions: TranscribeActions;
  live: boolean;
  saved: CallContext | null;
  flagged: boolean;
  /** The listening appending here is a call that shares a picture: Capture is offered. */
  sharing?: boolean;
  adding: Pick<ImageAdding, "open" | "capture" | "capturing" | "error">;
}) {
  const counts = new Map<number, number>();
  for (const u of t.utterances) if (u.speaker !== null) counts.set(u.speaker, (counts.get(u.speaker) ?? 0) + 1);

  return (
    <>
      <ContextCard transcript={t} draft={state.draft} maxTokens={state.contextTokens} live={live} saved={saved} flagged={flagged} actions={actions} />

      <ImagesCard transcript={t} sharing={sharing} adding={adding} />

      <section aria-labelledby="transcribe-speakers" className="flex flex-col gap-2.5">
        <h2 id="transcribe-speakers" className={caption}>
          Speakers
        </h2>
        {t.speakers.length === 0 ? (
          <p className="text-xs leading-snug text-ash">Each voice the diarizer tells apart gets a name here, which you can change.</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {t.speakers.map((s) => (
              <li key={s.id} className="flex items-center gap-2">
                <span aria-hidden="true" className="h-6 w-1 shrink-0" style={{ background: speakerColor(s) }} />
                <input
                  key={`${s.id}:${s.name}`}
                  aria-label={`Name of ${s.name}`}
                  defaultValue={s.name}
                  onBlur={(e) => e.currentTarget.value.trim() !== s.name && actions.renameSpeaker(s.id, e.currentTarget.value)}
                  onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
                  className="min-w-0 flex-1 rounded-[2px] border border-transparent bg-transparent px-1.5 py-1 font-display text-[13px] font-semibold text-ink hover:border-line focus:border-ember focus:outline-none"
                />
                <span className="shrink-0 font-display text-xs text-ash tabular-nums" title="Utterances">
                  {counts.get(s.id) ?? 0}
                </span>
                {t.speakers.length > 1 && (
                  <select
                    aria-label={`Merge ${s.name} into another speaker`}
                    value=""
                    onChange={(e) => e.currentTarget.value !== "" && actions.mergeSpeakers(s.id, Number(e.currentTarget.value))}
                    className="w-[5.5rem] shrink-0 cursor-pointer rounded-[2px] border border-line bg-ground px-1 py-1 font-display text-xs text-ash hover:text-ink focus:border-ember focus:outline-none"
                  >
                    <option value="" disabled>
                      Merge into
                    </option>
                    {t.speakers
                      .filter((other) => other.id !== s.id)
                      .map((other) => (
                        <option key={other.id} value={other.id}>
                          {other.name}
                        </option>
                      ))}
                  </select>
                )}
              </li>
            ))}
          </ul>
        )}
        {sessionsWithSpeakers(t) > 1 && (
          <p className="text-xs leading-snug text-ash">A Resume or a Continue numbers the voices anew: merge a speaker who came back into the one they were.</p>
        )}
      </section>
    </>
  );
}

/**
 * The Settings tab: the view's own, this browser's. Speakers and the mic held on a call stood in the deck until the
 * owner moved them here (the deck's switches came before the panel did).
 */
export function Settings({ state, actions }: { state: TranscribeState; actions: TranscribeActions }) {
  const pace = useFilePace();
  const silence = useUtteranceSilence();
  const struck = state.speakersUnavailable;
  return (
    <>
      <div className="flex items-start gap-2.5">
        {struck ? (
          <span aria-disabled="true" className="pointer-events-none flex opacity-45">
            <Switch on={false} onChange={() => {}} labelledBy="transcribe-speakers-switch" />
          </span>
        ) : (
          <Switch on={state.speakers} onChange={actions.setSpeakers} labelledBy="transcribe-speakers-switch" />
        )}
        <span className="flex flex-col gap-0.5">
          <span id="transcribe-speakers-switch" className={`font-display text-[13px] font-medium ${struck ? "text-ash line-through" : "text-ink"}`}>
            Speakers
          </span>
          <span className="text-xs leading-snug text-ash">
            {struck ? <HintText text={struck} /> : "The diarizer tells the voices apart, Speaker 1, 2 and on. From the next Start."}
          </span>
        </span>
      </div>

      <div className="flex items-start gap-2.5">
        <Switch on={state.holdMic} onChange={actions.setHoldMic} labelledBy="transcribe-hold-switch" />
        <span className="flex flex-col gap-0.5">
          <span id="transcribe-hold-switch" className="font-display text-[13px] font-medium text-ink">
            Hold my mic while they speak
          </span>
          <span className="text-xs leading-snug text-ash">
            On a call. {state.holdMic ? "The call's voices coming back into the mic are not transcribed as you." : "With headphones, nothing comes back: you can speak over them."}
          </span>
        </span>
      </div>

      <div className="flex items-start gap-2.5">
        <Switch on={state.suggestionsOn} onChange={actions.setSuggestionsOn} labelledBy="transcribe-suggestions-switch" />
        <span className="flex flex-col gap-0.5">
          <span id="transcribe-suggestions-switch" className="font-display text-[13px] font-medium text-ink">
            Suggest while listening
          </span>
          <span className="text-xs leading-snug text-ash">The languages the server heard, and about every minute and a half the text model's topic, title and terms.</span>
        </span>
      </div>


      <section aria-labelledby="transcribe-silence" className="flex flex-col gap-2">
        <h2 id="transcribe-silence" className={caption}>
          Utterance ends after
        </h2>
        <div role="radiogroup" aria-labelledby="transcribe-silence" className="flex">
          {UTTERANCE_SILENCE_CHOICES.map((ms) => (
            <button
              key={ms}
              type="button"
              role="radio"
              aria-checked={silence === ms}
              onClick={() => actions.setSilence(ms)}
              className={`border border-line px-3 py-1 font-display text-[13px] not-first:-ml-px ${silence === ms ? "relative z-10 border-ember bg-surface text-ink" : "text-ash hover:text-ink"}`}
            >
              {ms / 1_000} s
            </button>
          ))}
        </div>
        <p className="text-xs leading-snug text-ash">The silence that ends an utterance. Shorter cuts a speaker at every breath, and the model guesses the language of each piece on its own.</p>
      </section>

      <section aria-labelledby="transcribe-file-pace" className="flex flex-col gap-2">
        <h2 id="transcribe-file-pace" className={caption}>
          File pace
        </h2>
        <div role="radiogroup" aria-labelledby="transcribe-file-pace" className="flex">
          {FILE_PACE_CHOICES.map((seconds) => (
            <button
              key={seconds}
              type="button"
              role="radio"
              aria-checked={pace === seconds}
              onClick={() => chooseFilePace(seconds)}
              className={`border border-line px-3 py-1 font-display text-[13px] not-first:-ml-px ${pace === seconds ? "relative z-10 border-ember bg-surface text-ink" : "text-ash hover:text-ink"}`}
            >
              {seconds} s
            </button>
          ))}
        </div>
        <p className="text-xs leading-snug text-ash">How far ahead of its transcript a file is sent. Further ahead finishes sooner and holds more of the server's queue.</p>
      </section>
    </>
  );
}
