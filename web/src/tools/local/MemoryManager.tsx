import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";
import { IconClose } from "../../ui/icons.tsx";
import { browserMemory, MEMORY_TITLE_LIMIT, type MemoryNote, type MemoryStore, useMemoryNotes } from "./memory.ts";
import { heat, highlight, type NoteOrder, savedAgo, visibleNotes } from "./memoryView.ts";

// What ignis remembers: the saved notes in a kiln-dark sheet, each a title
// (all the model sees up front) over its body (read on demand). Each note's
// bar shows its heat — ember when just saved, cooling to ash over a week —
// and a note saved or edited here flares and cools in front of you.

const RAISED = "bg-[#232830]";
const FIELD = `${RAISED} px-3 py-2 text-[#eae8e4] placeholder:text-[#6b737d] focus:shadow-[inset_0_-2px_0_#ff5a1f] focus:outline-none`;
/** A body longer than this starts folded to a few lines. */
const FOLD_AT = 220;

/** A clock for "saved … ago", ticking every half minute. */
function useNow() {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(timer);
  }, []);
  return now;
}

function Marked({ text, query }: { text: string; query: string }) {
  return highlight(text, query).map((piece, i) =>
    piece.match ? (
      <mark key={i} className="bg-transparent text-[#ffa877] underline decoration-[#ff5a1f] decoration-2 underline-offset-4">
        {piece.text}
      </mark>
    ) : (
      <span key={i}>{piece.text}</span>
    ),
  );
}

export function MemoryManager({ onClose, store = browserMemory }: { onClose: () => void; store?: MemoryStore }) {
  const notes = useMemoryNotes(store);
  const now = useNow();
  const [query, setQuery] = useState("");
  const [order, setOrder] = useState<NoteOrder>("newest");
  const [editing, setEditing] = useState<string | null>(null);
  const [removed, setRemoved] = useState<MemoryNote | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [fresh, setFresh] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const search = useRef<HTMLInputElement>(null);
  const shown = visibleNotes(notes, query, order);

  // Focus the search on open, and hand focus back to whatever opened the sheet.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    search.current?.focus();
    return () => opener?.focus();
  }, []);

  useEffect(() => {
    const close = (e: globalThis.KeyboardEvent) => e.key === "Escape" && !e.defaultPrevented && onClose();
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);

  // An undo is offered for a few seconds.
  useEffect(() => {
    if (!removed) return;
    const timer = setTimeout(() => setRemoved(null), 6000);
    return () => clearTimeout(timer);
  }, [removed]);

  function add(e: FormEvent) {
    e.preventDefault();
    if (!body.trim()) return;
    const saved = store.save(title, body);
    setTitle("");
    setBody("");
    setQuery("");
    setOrder("newest");
    setFresh(saved.id);
  }

  const count =
    notes.length === 0
      ? "Nothing saved yet"
      : query.trim()
        ? `${shown.length} of ${notes.length} notes`
        : `${notes.length} ${notes.length === 1 ? "note" : "notes"}, kept in this browser`;

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center sm:items-center sm:p-6">
      <div className="absolute inset-0 bg-[#0e1115]/70 backdrop-blur-[2px]" onClick={onClose} aria-hidden />
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="memory-title"
        className="memory-sheet cut relative flex max-h-[90dvh] w-full max-w-2xl flex-col bg-kiln text-[#eae8e4] shadow-[0_32px_90px_rgb(0_0_0/0.55)] [--cut-size:22px]"
      >
        <header className="flex items-start gap-4 px-6 pt-7 pb-5 sm:px-9">
          <div className="min-w-0 flex-1">
            <h2 id="memory-title" className="font-display text-[26px] leading-tight font-semibold tracking-tight">
              What ignis remembers
            </h2>
            <p className="mt-1.5 text-sm text-[#939ba4]">{count}</p>
          </div>
          <button
            type="button"
            aria-label="Close memory"
            title="Close (Esc)"
            onClick={onClose}
            className="-mt-1 -mr-2 grid size-9 shrink-0 place-items-center text-[#939ba4] hover:bg-kiln-line hover:text-white"
          >
            <IconClose />
          </button>
        </header>
        <div className="memory-rule mx-6 sm:mx-9" aria-hidden />

        {notes.length > 0 && (
          <div className="flex flex-wrap items-center gap-x-5 gap-y-3 px-6 pt-5 pb-2 sm:px-9">
            <label className="min-w-0 flex-1 basis-56">
              <span className="sr-only">Search notes</span>
              <input
                ref={search}
                type="search"
                name="memory-search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search titles and notes"
                className={`w-full text-sm ${FIELD}`}
              />
            </label>
            <div role="radiogroup" aria-label="Order" className="flex gap-4 font-display text-[13px] font-medium">
              {(["newest", "oldest"] as const).map((o) => (
                <button
                  key={o}
                  type="button"
                  role="radio"
                  aria-checked={order === o}
                  onClick={() => setOrder(o)}
                  className={`border-b-2 pb-0.5 ${order === o ? "border-[#ff5a1f] text-[#eae8e4]" : "border-transparent text-[#939ba4] hover:text-[#eae8e4]"}`}
                >
                  {o === "newest" ? "Newest" : "Oldest"}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto px-6 sm:px-9">
          {notes.length === 0 ? (
            <div className="flex flex-col gap-2 py-12">
              <p className="font-display text-lg font-medium text-[#eae8e4]">Nothing remembered yet</p>
              <p className="max-w-md text-sm leading-relaxed text-[#939ba4]">
                ignis saves a note here when it learns something worth keeping about you. You can write one below too.
              </p>
            </div>
          ) : shown.length === 0 ? (
            <p className="py-10 text-sm text-[#939ba4]">No note matches “{query.trim()}”.</p>
          ) : (
            <ol className="py-2">
              {shown.map((n) => (
                <NoteRow
                  key={n.id}
                  note={n}
                  now={now}
                  query={query}
                  fresh={fresh === n.id}
                  editing={editing === n.id}
                  onEdit={() => setEditing(n.id)}
                  onCancel={() => setEditing(null)}
                  onSave={(nextTitle, nextBody) => {
                    if (store.update(n.id, nextTitle, nextBody)) setFresh(n.id);
                    setEditing(null);
                  }}
                  onDelete={() => {
                    store.remove(n.id);
                    setRemoved(n);
                  }}
                />
              ))}
            </ol>
          )}
        </div>

        {removed && (
          <div role="status" className={`mx-6 mb-3 flex items-center gap-3 ${RAISED} px-3 py-2 text-sm sm:mx-9`}>
            <span className="min-w-0 flex-1 truncate text-[#b9bec4]">Deleted “{removed.title}”</span>
            <button
              type="button"
              onClick={() => {
                store.restore(removed);
                setRemoved(null);
              }}
              className="shrink-0 font-display font-semibold text-[#ff8a4c] hover:text-[#ffa877]"
            >
              Undo
            </button>
          </div>
        )}

        <footer className="border-t border-kiln-line px-6 pt-4 pb-5 sm:px-9">
          <form onSubmit={add} className="flex flex-col gap-2">
            <input
              name="memory-new-title"
              aria-label="New note title"
              maxLength={MEMORY_TITLE_LIMIT}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Title, a few words"
              className={`font-display text-[15px] font-semibold ${FIELD}`}
            />
            <div className="flex items-end gap-2">
              <textarea
                name="memory-new-body"
                aria-label="New note"
                rows={2}
                value={body}
                onChange={(e) => setBody(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) add(e);
                }}
                placeholder="What ignis should keep. Only the title goes into every prompt."
                className={`max-h-40 min-h-16 flex-1 resize-none text-sm leading-relaxed [field-sizing:content] ${FIELD}`}
              />
              <button
                type="submit"
                disabled={!body.trim()}
                className="cut shrink-0 bg-[#ff5a1f] px-4 py-2.5 font-display text-sm font-semibold text-[#1c2026] [--cut-size:8px] hover:bg-[#ff7a45] disabled:bg-kiln-line disabled:text-[#6b737d]"
              >
                Save note
              </button>
            </div>
          </form>
          <div className="mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-2 text-xs">
            <span className="text-[#6b737d]">The model sees every title, and reads a note when it needs it.</span>
            {notes.length > 0 &&
              (confirmClear ? (
                <span className="flex items-center gap-3 font-display">
                  <span className="text-[#ff7b6b]">
                    Delete all {notes.length} {notes.length === 1 ? "note" : "notes"}?
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      store.clear();
                      setConfirmClear(false);
                      setRemoved(null);
                    }}
                    className="font-semibold text-[#ff7b6b] hover:text-[#ffa196]"
                  >
                    Delete all
                  </button>
                  <button type="button" onClick={() => setConfirmClear(false)} className="text-[#939ba4] hover:text-[#eae8e4]">
                    Keep them
                  </button>
                </span>
              ) : (
                <button type="button" onClick={() => setConfirmClear(true)} className="font-display text-[#939ba4] hover:text-[#ff7b6b]">
                  Delete all
                </button>
              ))}
          </div>
        </footer>
      </section>
    </div>
  );
}

function NoteRow(props: {
  note: MemoryNote;
  now: number;
  query: string;
  fresh: boolean;
  editing: boolean;
  onEdit: () => void;
  onCancel: () => void;
  onSave: (title: string, body: string) => void;
  onDelete: () => void;
}) {
  const { note, editing } = props;
  const [title, setTitle] = useState(note.title);
  const [body, setBody] = useState(note.body);
  const [unfolded, setUnfolded] = useState(false);
  const long = note.body.length > FOLD_AT;
  const warmth = heat(note.savedAt, props.now);

  function keys(e: KeyboardEvent<HTMLElement>) {
    if (e.key === "Escape") {
      // Esc leaves the edit, not the sheet.
      e.preventDefault();
      props.onCancel();
    } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      props.onSave(title, body);
    }
  }

  return (
    <li className={`group flex gap-4 border-b border-kiln-line/70 py-4 last:border-b-0 ${props.fresh ? "note-fresh" : ""}`}>
      <span
        className="note-heat w-[3px] shrink-0 self-stretch"
        style={{ background: `color-mix(in oklab, #ff5a1f ${warmth}%, #454c56)` }}
        title={`Saved ${savedAgo(note.savedAt, props.now)}`}
        aria-hidden
      />
      <div className="min-w-0 flex-1">
        {editing ? (
          <div className="flex flex-col gap-2" onKeyDown={keys}>
            <input
              name={`memory-edit-title-${note.id}`}
              aria-label={`Title of note ${note.id}`}
              autoFocus
              maxLength={MEMORY_TITLE_LIMIT}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className={`font-display text-base font-semibold ${FIELD}`}
            />
            <textarea
              name={`memory-edit-body-${note.id}`}
              aria-label={`Text of note ${note.id}`}
              value={body}
              onChange={(e) => setBody(e.target.value)}
              className={`min-h-20 w-full resize-none text-sm leading-relaxed [field-sizing:content] ${FIELD}`}
            />
            <div className="flex items-baseline gap-4 font-display text-[13px] font-medium">
              <button
                type="button"
                disabled={!title.trim() && !body.trim()}
                onClick={() => props.onSave(title, body)}
                className="text-[#ff8a4c] hover:text-[#ffa877] disabled:text-[#6b737d]"
              >
                Save changes
              </button>
              <button
                type="button"
                onClick={() => {
                  setTitle(note.title);
                  setBody(note.body);
                  props.onCancel();
                }}
                className="text-[#939ba4] hover:text-[#eae8e4]"
              >
                Cancel
              </button>
              <span className="ml-auto text-xs text-[#6b737d]">Ctrl+Enter saves</span>
            </div>
          </div>
        ) : (
          <>
            <h3 className="font-display text-base leading-snug font-semibold break-words text-[#eae8e4]">
              <Marked text={note.title} query={props.query} />
            </h3>
            {note.body && note.body !== note.title && (
              <p
                className={`mt-1 text-sm leading-relaxed break-words whitespace-pre-line text-[#b9bec4] ${long && !unfolded ? "line-clamp-3" : ""}`}
              >
                <Marked text={note.body} query={props.query} />
              </p>
            )}
          </>
        )}
        <p className="mt-2 flex items-baseline gap-3 font-display text-xs text-[#6b737d] tabular-nums">
          <span>{note.id}</span>
          <span>Saved {savedAgo(note.savedAt, props.now)}</span>
          {!editing && long && (
            <button type="button" onClick={() => setUnfolded((u) => !u)} className="text-[#939ba4] hover:text-[#eae8e4]">
              {unfolded ? "Show less" : "Show all"}
            </button>
          )}
        </p>
      </div>
      {!editing && (
        <div className="flex shrink-0 items-start gap-1 font-display text-[13px] font-medium sm:opacity-0 sm:group-focus-within:opacity-100 sm:group-hover:opacity-100">
          <button
            type="button"
            aria-label={`Edit note ${note.id}`}
            onClick={() => {
              setTitle(note.title);
              setBody(note.body);
              props.onEdit();
            }}
            className="px-2 py-1 text-[#939ba4] hover:bg-kiln-line hover:text-[#eae8e4]"
          >
            Edit
          </button>
          <button
            type="button"
            aria-label={`Delete note ${note.id}`}
            onClick={props.onDelete}
            className="px-2 py-1 text-[#939ba4] hover:bg-kiln-line hover:text-[#ff7b6b]"
          >
            Delete
          </button>
        </div>
      )}
    </li>
  );
}
