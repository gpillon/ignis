import { useCallback, useMemo, useRef, useState } from "react";
import { caption } from "../ui/classes.ts";
import { IconPlus } from "../ui/icons.tsx";
import { Answers } from "./Answers.tsx";
import { EvidenceEditor } from "./EvidenceEditor.tsx";
import { EXAMPLES, type Example } from "./examples.ts";
import {
  type Draft,
  EMPTY_DRAFT,
  type Fault,
  freeId,
  newQuestion,
  type Primitive,
  PRIMITIVES,
  type Question,
  readRequest,
  requestBody,
  validate,
} from "./model.ts";
import { QuestionCard } from "./QuestionCard.tsx";
import { decide, type Run } from "./request.ts";

// The Decide tab (GitHub #247): build a decision, send it, read what came
// back.
//
// The bench is on the left and the answers on the right, because that is the
// order the work happens in. The whole request goes in one call — the endpoint
// prefills the evidence once and the questions share it, and a button per
// question would hide the only property worth showing.

type State =
  | { at: "idle" }
  | { at: "running"; draft: Draft }
  | { at: "answered"; draft: Draft; run: Run }
  | { at: "refused"; message: string };

type Mode = "build" | "json";

export function DecideView({ ready }: { ready: boolean }) {
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT);
  const [mode, setMode] = useState<Mode>("build");
  // While the JSON view is open its text is what the reader is editing; the
  // draft follows it on every parse that succeeds, and stands still on one
  // that does not.
  const [json, setJson] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [state, setState] = useState<State>({ at: "idle" });
  const [loading, setLoading] = useState(false);
  const running = useRef<AbortController | null>(null);

  const faults = useMemo(() => validate(draft), [draft]);
  const blocked = faults.length > 0 || (mode === "json" && jsonError !== null);
  const jsonFault = faults.find((f) => f.code === "invalid_state_json");
  const loose = faults.filter((f) => f.uid === undefined && f.code !== "invalid_state_json");

  const replace = useCallback((next: Draft) => {
    setDraft(next);
    setJson(requestBody(next));
    setJsonError(null);
  }, []);

  function edit(next: Draft) {
    setDraft(next);
    // The JSON view is regenerated when it opens, so it does not need to keep
    // up with every keystroke in the builder.
    if (mode === "json") setJson(requestBody(next));
  }

  function openJson() {
    setJson(requestBody(draft));
    setJsonError(null);
    setMode("json");
  }

  function editJson(text: string) {
    setJson(text);
    const read = readRequest(text);
    if (read.ok) {
      setDraft(read.draft);
      setJsonError(null);
    } else {
      setJsonError(read.message);
    }
  }

  async function pick(example: Example) {
    setLoading(true);
    try {
      replace(await example.build());
      setState({ at: "idle" });
    } catch (error) {
      setState({ at: "refused", message: `That example did not load: ${String(error)}` });
    } finally {
      setLoading(false);
    }
  }

  function addQuestion(kind: Primitive) {
    edit({ ...draft, questions: [...draft.questions, newQuestion(kind, freeId(draft.questions, kind === "noul" ? "answer" : kind))] });
  }

  async function send() {
    if (blocked || !ready) return;
    const sent = draft;
    const controller = new AbortController();
    running.current = controller;
    setState({ at: "running", draft: sent });
    const result = await decide(requestBody(sent), controller.signal);
    running.current = null;
    setState(result.ok ? { at: "answered", draft: sent, run: result.run } : { at: "refused", message: result.message });
  }

  return (
    <main aria-label="Decide" className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto grid w-full max-w-[1480px] grid-cols-1 gap-x-8 gap-y-6 px-4 py-5 md:px-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)] xl:grid-cols-[minmax(0,30rem)_minmax(0,1fr)]">
        {/* The answers come first on a narrow screen once there are any. */}
        <div className={state.at === "idle" ? "lg:order-1" : "order-2 lg:order-1"}>
          <Bench
            draft={draft}
            mode={mode}
            json={json}
            jsonError={jsonError}
            jsonFault={jsonFault?.message}
            faults={faults}
            loose={loose}
            blocked={blocked}
            ready={ready}
            busy={state.at === "running"}
            loading={loading}
            onEdit={edit}
            onMode={(next) => (next === "json" ? openJson() : setMode("build"))}
            onJson={editJson}
            onAdd={addQuestion}
            onSend={() => void send()}
            onStop={() => running.current?.abort()}
            onClear={() => {
              replace(EMPTY_DRAFT);
              setState({ at: "idle" });
            }}
          />
        </div>

        <div className={state.at === "idle" ? "lg:order-2" : "order-1 lg:order-2"}>
          {state.at === "answered" ? (
            <Answers key={state.run.raw} draft={state.draft} run={state.run} />
          ) : state.at === "running" ? (
            <Waiting draft={state.draft} />
          ) : state.at === "refused" ? (
            <Refused message={state.message} />
          ) : (
            <Opening onPick={(example) => void pick(example)} loading={loading} />
          )}
        </div>
      </div>
    </main>
  );
}

function Bench({
  draft,
  mode,
  json,
  jsonError,
  jsonFault,
  faults,
  loose,
  blocked,
  ready,
  busy,
  loading,
  onEdit,
  onMode,
  onJson,
  onAdd,
  onSend,
  onStop,
  onClear,
}: {
  draft: Draft;
  mode: Mode;
  json: string;
  jsonError: string | null;
  jsonFault?: string;
  faults: Fault[];
  loose: Fault[];
  blocked: boolean;
  ready: boolean;
  busy: boolean;
  loading: boolean;
  onEdit: (draft: Draft) => void;
  onMode: (mode: Mode) => void;
  onJson: (text: string) => void;
  onAdd: (kind: Primitive) => void;
  onSend: () => void;
  onStop: () => void;
  onClear: () => void;
}) {
  return (
    <div className="flex flex-col gap-5">
      <EvidenceEditor evidence={draft.evidence} onChange={(evidence) => onEdit({ ...draft, evidence })} invalid={jsonFault} />

      <section>
        <div className="flex items-center justify-between gap-3">
          <h2 className="font-display text-[15px] font-semibold text-ink">
            Questions
            {draft.questions.length > 0 && <span className="ml-2 font-normal tabular-nums text-ash">{draft.questions.length}</span>}
          </h2>
          <div role="group" aria-label="How to edit the request" className="flex">
            {(["build", "json"] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                aria-pressed={mode === m}
                onClick={() => onMode(m)}
                className={`px-2.5 py-1 font-display text-[12px] font-medium ${
                  mode === m ? "bg-ink text-ground" : "text-ash hover:bg-line hover:text-ink"
                }`}
              >
                {m === "build" ? "Build" : "JSON"}
              </button>
            ))}
          </div>
        </div>

        {mode === "build" ? (
          <Builder draft={draft} faults={faults} onEdit={onEdit} onAdd={onAdd} />
        ) : (
          <JsonEditor text={json} error={jsonError} onChange={onJson} />
        )}
      </section>

      {loose.length > 0 && (
        <ul className="flex flex-col gap-1">
          {loose.map((fault) => (
            <li key={fault.code} className="text-[13px] leading-snug text-warn">
              {fault.message}
            </li>
          ))}
        </ul>
      )}

      <div className="flex items-center gap-3 border-t border-line pt-4">
        {busy ? (
          <button type="button" onClick={onStop} className="cut bg-surface px-4 py-2 font-display text-[14px] font-semibold text-ink hover:bg-line">
            Stop
          </button>
        ) : (
          <button
            type="button"
            disabled={blocked || !ready || loading}
            onClick={onSend}
            className="cut bg-ink px-5 py-2 font-display text-[14px] font-semibold text-ground hover:bg-ember disabled:cursor-not-allowed disabled:bg-line disabled:text-ash"
          >
            Decide
          </button>
        )}
        {draft.questions.length > 0 && (
          <button type="button" onClick={onClear} className="font-display text-[13px] text-ash hover:text-ink">
            Clear
          </button>
        )}
        {!ready && <span className="text-[13px] text-ash">Waiting for the model.</span>}
      </div>
    </div>
  );
}

function Builder({
  draft,
  faults,
  onEdit,
  onAdd,
}: {
  draft: Draft;
  faults: Fault[];
  onEdit: (draft: Draft) => void;
  onAdd: (kind: Primitive) => void;
}) {
  const replace = (index: number, question: Question) => onEdit({ ...draft, questions: draft.questions.map((q, i) => (i === index ? question : q)) });
  const move = (index: number, by: -1 | 1) => {
    const questions = [...draft.questions];
    const [moved] = questions.splice(index, 1);
    questions.splice(index + by, 0, moved);
    onEdit({ ...draft, questions });
  };
  return (
    <>
      <ol className="mt-2 flex flex-col gap-2">
        {draft.questions.map((question, index) => (
          <QuestionCard
            key={question.uid}
            question={question}
            faults={faults.filter((f) => f.uid === question.uid)}
            first={index === 0}
            last={index === draft.questions.length - 1}
            onChange={(next) => replace(index, next)}
            onRemove={() => onEdit({ ...draft, questions: draft.questions.filter((_, i) => i !== index) })}
            onMove={(by) => move(index, by)}
          />
        ))}
      </ol>
      <div className="mt-3">
        <p className={caption}>Add a question</p>
        <div className="mt-1 flex flex-wrap gap-1.5">
          {PRIMITIVES.map((kind) => (
            <button
              key={kind}
              type="button"
              onClick={() => onAdd(kind)}
              className="cut flex items-center gap-1 bg-surface px-2.5 py-1.5 font-display text-[13px] font-medium text-ink hover:bg-line [--cut-size:6px]"
            >
              <IconPlus />
              {kind}
            </button>
          ))}
        </div>
      </div>
    </>
  );
}

function JsonEditor({ text, error, onChange }: { text: string; error: string | null; onChange: (text: string) => void }) {
  return (
    <div className="mt-2">
      <textarea
        value={text}
        onChange={(e) => onChange(e.target.value)}
        name="decide-request"
        aria-label="The request, as JSON"
        spellCheck={false}
        rows={24}
        className={`w-full resize-y rounded-[2px] border bg-surface px-2.5 py-2 font-mono text-[12.5px] leading-relaxed text-ink focus:outline-none ${
          error ? "border-warn" : "border-line focus:border-ember"
        }`}
      />
      {error ? (
        <p className="mt-1 text-[12px] leading-snug text-warn">
          {error}. The builder still holds the last version that parsed.
        </p>
      ) : (
        <p className="mt-1 max-w-[68ch] text-[12px] leading-snug text-ash">
          This is the body that will be sent. Option order is part of the prompt, so it is kept exactly as written here — even for keys
          that look like numbers.
        </p>
      )}
    </div>
  );
}

/** The opening screen: what the endpoint does, and five requests that show it. */
function Opening({ onPick, loading }: { onPick: (example: Example) => void; loading: boolean }) {
  return (
    <div>
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
                className="group flex w-full items-baseline gap-4 border-b border-line py-3 text-left hover:bg-surface disabled:opacity-50"
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

function Waiting({ draft }: { draft: Draft }) {
  return (
    <div className="flex flex-col gap-3 py-10">
      <p className="font-display text-[15px] text-ash">
        Prefilling the evidence once, for {draft.questions.length} question{draft.questions.length === 1 ? "" : "s"}…
      </p>
      <span className="h-[2px] w-full overflow-hidden bg-line">
        <span className="heat block h-full w-full" data-busy="true" />
      </span>
    </div>
  );
}

function Refused({ message }: { message: string }) {
  return (
    <div className="border-l-2 border-fault pl-4 py-2">
      <h2 className="font-display text-[15px] font-semibold text-ink">ignis did not answer this one</h2>
      <p className="mt-1 max-w-[68ch] text-[13px] leading-snug text-ash">{message}</p>
    </div>
  );
}
