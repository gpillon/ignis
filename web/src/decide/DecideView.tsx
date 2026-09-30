import { type DragEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PromptImage } from "../conversation/images.ts";
import { caption } from "../ui/classes.ts";
import { IconPaperclip } from "../ui/icons.tsx";
import { Answers } from "./Answers.tsx";
import { Bench, type Mode } from "./Bench.tsx";
import { evidenceFromFile, isImageFile } from "./evidenceFile.ts";
import { EXAMPLES, type Example } from "./examples.ts";
import {
  type Draft,
  EMPTY_SPARE,
  type Evidence,
  freeId,
  keepFile,
  loadEvidence,
  newQuestion,
  type Primitive,
  readRequest,
  requestBody,
  validate,
} from "./model.ts";
import { decide } from "./request.ts";
import { activeOf, type Decision, freeDecisionId, openDecision, removeDecision, startList, updateDecision } from "./sessions.ts";
import { Sessions } from "./Sessions.tsx";

// The Decide tab (GitHub #247), in the chat's three-column shape: the decisions
// on the left, what came back in the middle, and the bench that writes the
// request on the right. Each column scrolls on its own, and below `lg` the two
// side columns are drawers.
//
// The bench is on the right and not the left for the reason the chat puts the
// transcript in the middle and its controls at the edges: the answers are what
// a reader came for. It is wider than the chat's settings panel, because a
// question is a paragraph and a list of options rather than a row of switches.
//
// A file dropped anywhere on the tab becomes the evidence (`evidenceFile.ts`):
// a reader holding a log does not want to aim it at one textarea in a drawer.

/** Which side drawer is out, below `lg`. */
export type Drawer = "sessions" | "settings" | null;

export function DecideView({ ready, drawer, onDrawer }: { ready: boolean; drawer: Drawer; onDrawer: (drawer: Drawer) => void }) {
  const [list, setList] = useState(startList);
  const [mode, setMode] = useState<Mode>("build");
  // The JSON view's text is what a reader edits while it is open; the draft
  // follows it on every parse that succeeds and stands still on one that does
  // not. It belongs to the view rather than to a decision: opening the view
  // writes it from whichever draft is active.
  const [json, setJson] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // Why the last file did not load; the next load, of whatever kind, clears it.
  const [fileError, setFileError] = useState<string | null>(null);
  // Enter and leave fire for every child the pointer crosses, so the overlay
  // is up while the count is.
  const [dropping, setDropping] = useState(false);
  const depth = useRef(0);
  // One per decision in flight, so a send in one session can be stopped while
  // another is still running.
  const controllers = useRef(new Map<number, AbortController>());

  const active = activeOf(list);
  const faults = useMemo(() => validate(active.draft), [active.draft]);

  useEffect(() => {
    if (!drawer) return;
    const close = (e: globalThis.KeyboardEvent) => e.key === "Escape" && onDrawer(null);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [drawer, onDrawer]);

  const change = useCallback(
    (id: number, apply: (decision: Decision) => Decision) => setList((l) => updateDecision(l, id, apply)),
    [],
  );

  /** Edited in place. The JSON view is rewritten only while it is the one showing. */
  function edit(draft: Draft) {
    change(active.id, (d) => ({ ...d, draft }));
    if (mode === "json") setJson(requestBody(draft));
  }

  /** A whole new draft: the evidence shapes set aside belonged to the old one, and so did its answers. */
  function replace(id: number, draft: Draft) {
    change(id, (d) => ({ ...d, draft, spare: EMPTY_SPARE, answer: null, refusal: null }));
    if (mode === "json") setJson(requestBody(draft));
    setJsonError(null);
    setFileError(null);
  }

  /**
   * Show `decision`'s request in the JSON view, whatever the view was showing
   * before. Only while the view is open: the body of a loaded log is the log,
   * and writing megabytes of it into a view nobody is looking at is a stall
   * with nothing to show for it. Opening the view writes it then.
   */
  function reread(decision: Decision, open = mode === "json") {
    if (open) setJson(requestBody(decision.draft));
    setJsonError(null);
    // A refused file was the decision's being left, not this one's.
    setFileError(null);
  }

  function editJson(text: string) {
    setJson(text);
    const read = readRequest(text);
    if (read.ok) {
      change(active.id, (d) => ({ ...d, draft: { ...read.draft, evidence: keepFile(d.draft.evidence, read.draft.evidence) } }));
      setJsonError(null);
    } else {
      setJsonError(read.message);
    }
  }

  function select(id: number) {
    const picked = list.decisions.find((d) => d.id === id);
    setList((l) => ({ ...l, activeId: id }));
    if (picked) reread(picked);
    onDrawer(null);
  }

  function newDecision() {
    const next = openDecision(list, freeDecisionId(list));
    setList(next);
    reread(activeOf(next));
    onDrawer(null);
  }

  function remove(id: number) {
    controllers.current.get(id)?.abort();
    controllers.current.delete(id);
    const next = removeDecision(list, id, freeDecisionId(list));
    setList(next);
    reread(activeOf(next));
  }

  async function pick(example: Example) {
    const id = active.id;
    setLoading(true);
    try {
      replace(id, await example.build());
    } catch (error) {
      change(id, (d) => ({ ...d, refusal: `That example did not load: ${String(error)}` }));
    } finally {
      setLoading(false);
    }
  }

  /**
   * Files dropped on the tab or chosen from it, as the active decision's
   * evidence. A text, a log or a JSON file replaces the evidence — the first
   * one, when several come at once, since a decision reads one — and pictures
   * are added to its images.
   */
  async function loadFiles(files: File[]) {
    const id = active.id;
    const text = files.find((file) => !isImageFile(file));
    const pictures = text ? [] : files.filter(isImageFile);
    const loaded = await Promise.all((text ? [text] : pictures).map(evidenceFromFile));
    const errors = loaded.flatMap((l) => (l.ok ? [] : [l.error]));
    setFileError(errors.length ? errors.join(" ") : null);
    const evidence = loaded.flatMap((l) => (l.ok && "evidence" in l ? [l.evidence] : []))[0];
    const images = loaded.flatMap((l) => (l.ok && "image" in l ? [l.image] : []));
    if (!evidence && images.length === 0) return;
    const into = (d: Decision): Evidence => evidence ?? withImages(d, images);
    const next = loadEvidence(active.draft, active.spare, into(active));
    change(id, (d) => ({ ...d, ...loadEvidence(d.draft, d.spare, into(d)) }));
    if (mode === "json") setJson(requestBody(next.draft));
  }

  function addQuestion(kind: Primitive) {
    const stem = kind === "noul" ? "answer" : kind;
    edit({ ...active.draft, questions: [...active.draft.questions, newQuestion(kind, freeId(active.draft.questions, stem))] });
  }

  async function send() {
    const id = active.id;
    const sent = active.draft;
    if (!ready || active.running || faults.length > 0 || (mode === "json" && jsonError !== null)) return;
    const controller = new AbortController();
    controllers.current.set(id, controller);
    change(id, (d) => ({ ...d, running: true, refusal: null }));
    const result = await decide(requestBody(sent), controller.signal);
    controllers.current.delete(id);
    change(id, (d) => ({
      ...d,
      running: false,
      // A refused re-send keeps the answers already paid for, and says why the
      // new ones are missing above them.
      ...(result.ok ? { answer: { run: result.run, draft: sent }, refusal: null } : { refusal: result.message }),
    }));
  }

  const carriesFiles = (e: DragEvent) => e.dataTransfer.types.includes("Files");

  return (
    <div
      className="relative flex min-h-0 flex-1"
      onDragEnter={(e) => {
        if (!carriesFiles(e)) return;
        e.preventDefault();
        depth.current += 1;
        setDropping(true);
      }}
      onDragOver={(e) => {
        if (!carriesFiles(e)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
      }}
      onDragLeave={(e) => {
        if (!carriesFiles(e)) return;
        depth.current = Math.max(0, depth.current - 1);
        if (depth.current === 0) setDropping(false);
      }}
      onDrop={(e) => {
        if (!carriesFiles(e)) return;
        e.preventDefault();
        depth.current = 0;
        setDropping(false);
        void loadFiles([...e.dataTransfer.files]);
      }}
    >
      {drawer && <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => onDrawer(null)} aria-hidden />}
      {dropping && <DropOverlay />}

      <Sessions list={list} open={drawer === "sessions"} onNew={newDecision} onSelect={select} onRemove={remove} />

      <main aria-label="Answers" className="flex min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 md:px-6">
          <div className="mx-auto flex w-full max-w-[64rem] flex-col gap-5">
            {active.refusal && (
              <Refused message={active.refusal} onDismiss={() => change(active.id, (d) => ({ ...d, refusal: null }))} />
            )}
            {active.running && <Deciding count={active.draft.questions.length} again={active.answer !== null} />}
            {active.answer ? (
              <Answers key={active.answer.run.raw} draft={active.answer.draft} run={active.answer.run} />
            ) : (
              !active.running &&
              !active.refusal && (
                <Opening
                  onPick={(example) => void pick(example)}
                  onFiles={(files) => void loadFiles(files)}
                  fileError={fileError}
                  loading={loading}
                />
              )
            )}
          </div>
        </div>
      </main>

      <Bench
        draft={active.draft}
        spare={active.spare}
        mode={mode}
        json={json}
        jsonError={jsonError}
        faults={faults}
        ready={ready}
        busy={active.running}
        loading={loading}
        open={drawer === "settings"}
        onEdit={edit}
        onSpare={(spare) => change(active.id, (d) => ({ ...d, spare }))}
        onFiles={(files) => void loadFiles(files)}
        fileError={fileError}
        onMode={(next) => {
          if (next === "json") reread(active, true);
          setMode(next);
        }}
        onJson={editJson}
        onAdd={addQuestion}
        onSend={() => void send()}
        onStop={() => controllers.current.get(active.id)?.abort()}
      />
    </div>
  );
}

/** The wait, which is one prefill. `again` says the answers below it are the last run's. */
function Deciding({ count, again }: { count: number; again: boolean }) {
  return (
    <div className="flex flex-col gap-2">
      <p className="font-display text-[14px] text-ash">
        {again
          ? "Deciding again; the answers below are the last run's."
          : `Prefilling the evidence once, for ${count} question${count === 1 ? "" : "s"}…`}
      </p>
      <span className="h-[2px] w-full overflow-hidden bg-line">
        <span className="heat block h-full w-full" data-busy="true" />
      </span>
    </div>
  );
}

/** `decision`'s evidence with `images` added: to the ones it has, or in place of a text it sets aside. */
function withImages(decision: Decision, images: PromptImage[]): Evidence {
  const { evidence } = decision.draft;
  return evidence.mode === "image"
    ? { ...evidence, images: [...evidence.images, ...images] }
    : { mode: "image", images, text: decision.spare.words };
}

/**
 * What the tab shows while a file is held over it: where it will go, and what
 * it will be read as. Over everything, drawers included, so the drop lands
 * here and nowhere in particular — and outside every `.cut`, whose clip-path
 * would swallow it.
 */
function DropOverlay() {
  return (
    <div className="absolute inset-0 z-50 grid place-items-center bg-[#1c2026]/80 p-6">
      <div className="cut w-full max-w-[30rem] bg-kiln [--cut-size:14px]">
        <span className="heat block" data-busy="true" />
        <div className="px-6 pb-7 pt-6">
          <p className="font-display text-[24px] font-semibold leading-tight text-[#eae8e4]">Drop it to make it the evidence</p>
          <p className="mt-2 max-w-[48ch] text-[13px] leading-relaxed text-[#939ba4]">
            A log or a document is read as text, a JSON file as a record, a picture as an image. Nothing is sent until you press Decide.
          </p>
        </div>
      </div>
    </div>
  );
}

/** The opening screen: what the endpoint does, the reader's own file, and the requests that show it. */
function Opening({
  onPick,
  onFiles,
  fileError,
  loading,
}: {
  onPick: (example: Example) => void;
  onFiles: (files: File[]) => void;
  fileError: string | null;
  loading: boolean;
}) {
  const picker = useRef<HTMLInputElement>(null);
  return (
    <div className="py-4">
      <h2 className="max-w-[26ch] font-display text-[30px] font-semibold leading-[1.15] tracking-tight text-ink sm:text-[38px]">
        Nothing is generated. The answer was already in the prompt.
      </h2>
      <p className="mt-3 max-w-[64ch] text-[15px] leading-relaxed text-ash">
        A decision asks for the probability of each answer you name, read straight off one position of a single prefill. Twenty
        questions over one piece of evidence cost about what one question costs, and the reply carries a distribution rather than a
        sentence you have to parse.
      </p>
      <div className="mt-7">
        <p className={caption}>Start from your own file, or from one of these</p>
        <ul className="mt-2 flex flex-col gap-px">
          <li>
            <button
              type="button"
              onClick={() => picker.current?.click()}
              className="group flex w-full flex-col items-baseline gap-1 border-b border-line py-3 text-left hover:bg-surface sm:flex-row sm:gap-4"
            >
              <span className="flex min-w-[11rem] items-center gap-1.5 self-start font-display text-[15px] font-semibold text-ember [&_svg]:size-4">
                <IconPaperclip />
                Your own file
              </span>
              <span className="min-w-0 flex-1 text-[13px] leading-snug text-ash">
                A log, a document or a JSON file — choose one, or drop it anywhere on this tab. It becomes the evidence, and a locate
                finds the line in it that answers your question.
              </span>
            </button>
            <input
              ref={picker}
              type="file"
              name="opening-file"
              aria-label="Open a file as the evidence"
              className="hidden"
              onChange={(e) => {
                onFiles([...(e.target.files ?? [])]);
                e.target.value = "";
              }}
            />
            {fileError && <p className="py-2 text-[13px] leading-snug text-warn">{fileError}</p>}
          </li>
          {EXAMPLES.map((example) => (
            <li key={example.id}>
              <button
                type="button"
                disabled={loading}
                onClick={() => onPick(example)}
                className="group flex w-full flex-col items-baseline gap-1 border-b border-line py-3 text-left hover:bg-surface disabled:opacity-50 sm:flex-row sm:gap-4"
              >
                <span className="min-w-[11rem] font-display text-[15px] font-semibold text-ink group-hover:text-ember">{example.name}</span>
                <span className="min-w-0 flex-1 text-[13px] leading-snug text-ash">{example.shows}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function Refused({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div className="flex items-start justify-between gap-4 border-l-2 border-fault bg-surface py-2 pl-4 pr-2">
      <div>
        <h2 className="font-display text-[14px] font-semibold text-ink">ignis did not answer this one</h2>
        <p className="mt-1 text-[13px] leading-snug text-ash">{message}</p>
      </div>
      <button type="button" onClick={onDismiss} className="shrink-0 px-2 py-1 font-display text-[12px] text-ash hover:text-ink">
        Dismiss
      </button>
    </div>
  );
}
