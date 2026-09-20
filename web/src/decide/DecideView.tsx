import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { caption } from "../ui/classes.ts";
import { Answers } from "./Answers.tsx";
import { Bench, type Mode } from "./Bench.tsx";
import { EXAMPLES, type Example } from "./examples.ts";
import { type Draft, EMPTY_SPARE, freeId, newQuestion, type Primitive, readRequest, requestBody, validate } from "./model.ts";
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
    setJson(requestBody(draft));
    setJsonError(null);
  }

  /** Show `decision`'s request in the JSON view, whatever the view was showing before. */
  function reread(decision: Decision) {
    setJson(requestBody(decision.draft));
    setJsonError(null);
  }

  function editJson(text: string) {
    setJson(text);
    const read = readRequest(text);
    if (read.ok) {
      change(active.id, (d) => ({ ...d, draft: read.draft }));
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

  return (
    <div className="relative flex min-h-0 flex-1">
      {drawer && <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => onDrawer(null)} aria-hidden />}

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
              !active.running && !active.refusal && <Opening onPick={(example) => void pick(example)} loading={loading} />
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
        onMode={(next) => {
          if (next === "json") reread(active);
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

/** The opening screen: what the endpoint does, and the requests that show it. */
function Opening({ onPick, loading }: { onPick: (example: Example) => void; loading: boolean }) {
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
        <p className={caption}>Start from one of these</p>
        <ul className="mt-2 flex flex-col gap-px">
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
