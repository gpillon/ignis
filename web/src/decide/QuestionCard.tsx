import { caption, field } from "../ui/classes.ts";
import { IconClose, IconPlus, IconTrash } from "../ui/icons.tsx";
import { asText, jsonString, writeOrdered } from "./json.ts";
import {
  emptyOption,
  type Fault,
  isConstrained,
  MAX_DIGITS,
  MIN_DIGITS,
  type Option,
  PRIMITIVE_BLURB,
  PRIMITIVES,
  type Primitive,
  type Question,
} from "./model.ts";

// One question in the builder (GitHub #247).
//
// Every primitive's fields live on the question at once, so switching the type
// keeps what was already written: the selector is not a destructive control.
// The order of the options is editable because it is part of the prompt — each
// one is bound to the answer token at its position — and not a presentation.

export function QuestionCard({
  question,
  faults,
  first,
  last,
  onChange,
  onRemove,
  onMove,
}: {
  question: Question;
  faults: Fault[];
  first: boolean;
  last: boolean;
  onChange: (next: Question) => void;
  onRemove: () => void;
  onMove: (by: -1 | 1) => void;
}) {
  const set = <K extends keyof Question>(key: K, value: Question[K]) => onChange({ ...question, [key]: value });
  const bad = faults.length > 0;

  return (
    <li className={`cut border-l-2 bg-surface p-3 ${bad ? "border-l-warn" : "border-l-line"}`}>
      {/* The type and the controls; what the question *asks* is the field
          below, because that is what a reader is here to write. */}
      <div className="flex items-center gap-2">
        <select
          value={question.kind}
          onChange={(e) => set("kind", e.target.value as Primitive)}
          name={`${question.uid}-type`}
          aria-label="Question type"
          className={`${field} w-[7.5rem] shrink-0 font-display text-[13px]`}
        >
          {PRIMITIVES.map((kind) => (
            <option key={kind} value={kind}>
              {kind}
            </option>
          ))}
        </select>
        <span className="min-w-0 flex-1" />
        <div className="flex shrink-0 items-center">
          <MoveButton label="Move up" disabled={first} onClick={() => onMove(-1)}>
            ↑
          </MoveButton>
          <MoveButton label="Move down" disabled={last} onClick={() => onMove(1)}>
            ↓
          </MoveButton>
          <button
            type="button"
            aria-label="Delete this question"
            title="Delete this question"
            onClick={onRemove}
            className="grid size-8 place-items-center text-ash hover:bg-fault hover:text-ground"
          >
            <IconTrash />
          </button>
        </div>
      </div>

      <p className="mt-1.5 text-[12px] leading-snug text-ash">{PRIMITIVE_BLURB[question.kind]}</p>

      {question.instructions.kind === "string" ? (
        <textarea
          value={question.instructions.value}
          onChange={(e) => set("instructions", jsonString(e.target.value))}
          name={`${question.uid}-instructions`}
          aria-label="What the model should decide"
          placeholder="What should the model decide?"
          rows={2}
          className={`${field} mt-2 resize-y text-[15px]`}
        />
      ) : (
        <div className="mt-2">
          <pre className="max-h-24 overflow-auto border border-line bg-ground p-2 font-mono text-[12px] text-ink">
            {writeOrdered(question.instructions, 2)}
          </pre>
          <button
            type="button"
            onClick={() => set("instructions", jsonString(asText(question.instructions)))}
            className="mt-1 font-display text-[12px] text-ember hover:underline"
          >
            Edit as text
          </button>
        </div>
      )}

      {question.kind === "noul" && (
        <div className="mt-2 grid grid-cols-2 gap-2">
          <Labelled label="A yes means">
            <input value={question.yes} onChange={(e) => set("yes", e.target.value)} name={`${question.uid}-yes`} placeholder="Yes" className={field} />
          </Labelled>
          <Labelled label="A no means">
            <input value={question.no} onChange={(e) => set("no", e.target.value)} name={`${question.uid}-no`} placeholder="No" className={field} />
          </Labelled>
        </div>
      )}

      {question.kind === "choice" && (
        <OptionList options={question.options} stem={question.uid} onChange={(options) => set("options", options)} />
      )}

      {question.kind === "score" && <LevelList levels={question.levels} stem={question.uid} onChange={(levels) => set("levels", levels)} />}

      {isConstrained(question.kind) && (
        <div className="mt-2 flex items-end gap-3">
          <Labelled label="Digits per number">
            <input
              type="number"
              min={MIN_DIGITS}
              max={MAX_DIGITS}
              value={question.digits}
              onChange={(e) => set("digits", Number(e.target.value))}
              name={`${question.uid}-digits`}
              aria-label="Digits per number"
              className={`${field} w-20 tabular-nums`}
            />
          </Labelled>
          <p className="pb-2 text-[12px] leading-snug text-ash">
            {question.kind === "number"
              ? `Up to ${"9".repeat(Math.max(1, Math.min(MAX_DIGITS, question.digits)))}.`
              : `The prompt declares a 0–${"9".repeat(Math.max(1, Math.min(MAX_DIGITS, question.digits)))} scale on each axis.`}
          </p>
        </div>
      )}

      <label className="mt-3 flex items-baseline gap-2">
        <span className={`${caption} shrink-0`}>Answer name</span>
        <input
          value={question.id}
          onChange={(e) => set("id", e.target.value)}
          name={`${question.uid}-id`}
          title="The key this question's answer comes back under"
          placeholder="answer_name"
          className={`${field} min-w-0 flex-1 py-1 font-mono text-[12px]`}
        />
      </label>

      {bad && (
        <ul className="mt-2 flex flex-col gap-1">
          {faults.map((fault) => (
            <li key={fault.code + fault.message} className="text-[12px] leading-snug text-warn">
              {fault.message}
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

/** The options of a `choice`, in the order they will be sent. */
function OptionList({ options, stem, onChange }: { options: Option[]; stem: string; onChange: (options: Option[]) => void }) {
  const replace = (index: number, option: Option) => onChange(options.map((o, i) => (i === index ? option : o)));
  const move = (index: number, by: -1 | 1) => {
    const next = [...options];
    const [moved] = next.splice(index, 1);
    next.splice(index + by, 0, moved);
    onChange(next);
  };
  return (
    <div className="mt-2">
      <p className={caption}>Options, in the order the prompt lists them</p>
      <ol className="mt-1 flex flex-col gap-1">
        {options.map((option, index) => (
          <li key={index} className="flex items-center gap-1.5">
            <input
              value={option.key}
              onChange={(e) => replace(index, { ...option, key: e.target.value })}
              name={`${stem}-option-${index}-key`}
              aria-label={`Option ${index + 1} name`}
              placeholder="name"
              className={`${field} w-[7.5rem] shrink-0 font-mono text-[13px]`}
            />
            <input
              value={option.description}
              onChange={(e) => replace(index, { ...option, description: e.target.value })}
              name={`${stem}-option-${index}-description`}
              aria-label={`Option ${index + 1} description`}
              placeholder="what it means — blank lets the name speak for itself"
              className={`${field} min-w-0 flex-1`}
            />
            <MoveButton label="Move up" disabled={index === 0} onClick={() => move(index, -1)}>
              ↑
            </MoveButton>
            <MoveButton label="Move down" disabled={index === options.length - 1} onClick={() => move(index, 1)}>
              ↓
            </MoveButton>
            <RemoveButton label={`Remove option ${index + 1}`} onClick={() => onChange(options.filter((_, i) => i !== index))} />
          </li>
        ))}
      </ol>
      <AddButton onClick={() => onChange([...options, emptyOption()])}>Add an option</AddButton>
    </div>
  );
}

/** The levels of a `score`, low to high; the answer is a weighted average of their positions. */
function LevelList({ levels, stem, onChange }: { levels: string[]; stem: string; onChange: (levels: string[]) => void }) {
  return (
    <div className="mt-2">
      <p className={caption}>Levels, lowest first</p>
      <ol className="mt-1 flex flex-col gap-1">
        {levels.map((level, index) => (
          <li key={index} className="flex items-center gap-1.5">
            <span className="w-5 shrink-0 text-right font-display text-[12px] tabular-nums text-ash">{index}</span>
            <input
              value={level}
              onChange={(e) => onChange(levels.map((l, i) => (i === index ? e.target.value : l)))}
              name={`${stem}-level-${index}`}
              aria-label={`Level ${index}`}
              placeholder="what this level means"
              className={`${field} min-w-0 flex-1`}
            />
            <RemoveButton label={`Remove level ${index}`} onClick={() => onChange(levels.filter((_, i) => i !== index))} />
          </li>
        ))}
      </ol>
      <AddButton onClick={() => onChange([...levels, ""])}>Add a level</AddButton>
    </div>
  );
}

function Labelled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className={`${caption} block`}>{label}</span>
      <span className="mt-1 block">{children}</span>
    </label>
  );
}

function MoveButton({ label, disabled, onClick, children }: { label: string; disabled: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="grid size-7 shrink-0 place-items-center font-display text-[13px] text-ash hover:bg-line hover:text-ink disabled:opacity-30 disabled:hover:bg-transparent"
    >
      {children}
    </button>
  );
}

function RemoveButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="grid size-7 shrink-0 place-items-center text-ash hover:bg-line hover:text-fault"
    >
      <IconClose />
    </button>
  );
}

function AddButton({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="mt-1.5 flex items-center gap-1.5 font-display text-[13px] font-medium text-ember hover:underline"
    >
      <IconPlus />
      {children}
    </button>
  );
}
