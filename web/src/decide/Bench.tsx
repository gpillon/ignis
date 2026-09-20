import { caption } from "../ui/classes.ts";
import { IconPlus } from "../ui/icons.tsx";
import { EvidenceEditor } from "./EvidenceEditor.tsx";
import { type Draft, type Fault, type Primitive, PRIMITIVES, type Question, type Spare } from "./model.ts";
import { QuestionCard } from "./QuestionCard.tsx";

// The bench (GitHub #247): the right-hand column where a decision is written,
// in the shape the chat's settings panel has — a drawer below `lg`, a static
// column above it, its own scroll, and the action pinned at the foot the way
// the composer is pinned under the transcript.
//
// Wider than the chat's panel, because a question is a paragraph and a list of
// options rather than a row of switches.

export type Mode = "build" | "json";

export function Bench({
  draft,
  spare,
  mode,
  json,
  jsonError,
  faults,
  ready,
  busy,
  loading,
  open,
  onEdit,
  onSpare,
  onMode,
  onJson,
  onAdd,
  onSend,
  onStop,
}: {
  draft: Draft;
  spare: Spare;
  mode: Mode;
  json: string;
  jsonError: string | null;
  faults: Fault[];
  ready: boolean;
  busy: boolean;
  loading: boolean;
  open: boolean;
  onEdit: (draft: Draft) => void;
  onSpare: (spare: Spare) => void;
  onMode: (mode: Mode) => void;
  onJson: (text: string) => void;
  onAdd: (kind: Primitive) => void;
  onSend: () => void;
  onStop: () => void;
}) {
  const jsonFault = faults.find((f) => f.code === "invalid_state_json");
  // `no_questions` is what the footer already says, and saying it to somebody
  // who has not written anything yet is scolding, not helping.
  const loose = faults.filter((f) => f.uid === undefined && f.code !== "invalid_state_json" && f.code !== "no_questions");
  const blocked = faults.length > 0 || jsonError !== null;

  return (
    <aside
      aria-label="Questions"
      className={`fixed inset-y-0 right-0 z-40 flex w-[22rem] flex-col border-l border-line bg-ground transition-transform motion-reduce:transition-none sm:w-[28rem] lg:static lg:z-auto lg:translate-x-0 xl:w-[33rem] ${open ? "translate-x-0" : "max-lg:invisible max-lg:translate-x-full"}`}
    >
      <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto overflow-x-hidden px-4 py-4">
        <EvidenceEditor
          evidence={draft.evidence}
          spare={spare}
          onChange={(evidence) => onEdit({ ...draft, evidence })}
          onSpare={onSpare}
          invalid={jsonFault?.message}
        />

        <section>
          <div className="flex items-center justify-between gap-3">
            <h2 className="font-display text-[15px] font-semibold text-ink">
              Questions
              {draft.questions.length > 0 && <span className="ml-2 font-normal tabular-nums text-ash">{draft.questions.length}</span>}
            </h2>
            <div role="group" aria-label="How to edit the request" className="flex border border-line bg-surface p-0.5">
              {(["build", "json"] as Mode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  aria-pressed={mode === m}
                  onClick={() => onMode(m)}
                  className={`px-2.5 py-1 font-display text-[12px] font-medium ${
                    mode === m ? "bg-ink text-ground" : "text-ash hover:text-ink"
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
            <JsonEditor text={json} error={jsonError} faults={faults} onChange={onJson} />
          )}
        </section>

        {mode === "build" && loose.length > 0 && (
          <ul className="flex flex-col gap-1">
            {loose.map((fault) => (
              <li key={fault.code} className="text-[13px] leading-snug text-warn">
                {fault.message}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Pinned, like the composer: the action does not scroll away from the
          questions it is about. */}
      <div className="shrink-0 border-t border-line px-4 py-3">
        {busy ? (
          <button type="button" onClick={onStop} className="cut w-full bg-surface px-4 py-2.5 font-display text-[14px] font-semibold text-ink hover:bg-line">
            Stop
          </button>
        ) : (
          <button
            type="button"
            disabled={blocked || !ready || loading}
            onClick={onSend}
            className="cut w-full bg-ink px-4 py-2.5 font-display text-[14px] font-semibold text-ground hover:bg-ember disabled:cursor-not-allowed disabled:bg-line disabled:text-ash"
          >
            Decide
          </button>
        )}
        <p className="mt-1.5 text-center font-display text-[11px] text-ash">
          {!ready
            ? "Waiting for the model."
            : busy
              ? "One prefill of the evidence, shared by every question."
              : draft.questions.length === 0
                ? "Add a question, or pick an example."
                : blocked
                  ? "Something above needs finishing."
                  : `${draft.questions.length} question${draft.questions.length === 1 ? "" : "s"}, one request.`}
        </p>
      </div>
    </aside>
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
  const replace = (index: number, question: Question) =>
    onEdit({ ...draft, questions: draft.questions.map((q, i) => (i === index ? question : q)) });
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

function JsonEditor({
  text,
  error,
  faults,
  onChange,
}: {
  text: string;
  error: string | null;
  /** Every fault, not only the loose ones: the builder that shows the rest is not on screen. */
  faults: Fault[];
  onChange: (text: string) => void;
}) {
  return (
    <div className="mt-2">
      <textarea
        value={text}
        onChange={(e) => onChange(e.target.value)}
        name="decide-request"
        aria-label="The request, as JSON"
        spellCheck={false}
        rows={22}
        className={`w-full resize-y rounded-[2px] border bg-surface px-2.5 py-2 font-mono text-[12.5px] leading-relaxed text-ink focus:outline-none ${
          error ? "border-warn" : "border-line focus:border-ember"
        }`}
      />
      {error ? (
        <p className="mt-1 text-[12px] leading-snug text-warn">{error}. The builder still holds the last version that parsed.</p>
      ) : (
        <p className="mt-1 text-[12px] leading-snug text-ash">
          This is the body that will be sent. Option order is part of the prompt, so it is kept exactly as written here — even for keys
          that look like numbers.
        </p>
      )}
      {faults.length > 0 && (
        <ul className="mt-2 flex flex-col gap-1">
          {faults.map((fault) => (
            <li key={fault.code + fault.message} className="text-[12px] leading-snug text-warn">
              {fault.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
