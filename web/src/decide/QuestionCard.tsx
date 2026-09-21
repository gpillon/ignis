import { useState } from "react";
import { caption, field, fieldLook } from "../ui/classes.ts";
import { IconClose, IconPlus, IconTrash } from "../ui/icons.tsx";
import { asText, jsonString, writeOrdered } from "./json.ts";
import {
  emptyOption,
  type Fault,
  hasFixedWidth,
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
          className={`${fieldLook} w-[7.5rem] shrink-0 font-display text-[13px]`}
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

      {hasFixedWidth(question.kind) && (
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
              className={`${fieldLook} w-20 tabular-nums`}
            />
          </Labelled>
          <p className="pb-2 text-[12px] leading-snug text-ash">
            {question.kind === "number"
              ? `Up to ${"9".repeat(Math.max(1, Math.min(MAX_DIGITS, question.digits)))}.`
              : `The prompt declares a 0–${"9".repeat(Math.max(1, Math.min(MAX_DIGITS, question.digits)))} scale on each axis.`}
          </p>
        </div>
      )}

      {/* A scalar's field is a **ceiling**, so blank is its default and blank
          is a legal request: the run closes itself when the number is
          complete, and the caller who does not know the magnitude is the one
          this primitive exists for. */}
      {question.kind === "scalar" && (
        <div className="mt-2 flex items-end gap-3">
          <Labelled label="Digits at most">
            <input
              type="number"
              min={MIN_DIGITS}
              max={MAX_DIGITS}
              value={question.ceiling ?? ""}
              onChange={(e) => set("ceiling", e.target.value.trim() === "" ? null : Number(e.target.value))}
              name={`${question.uid}-ceiling`}
              aria-label="Digits at most"
              placeholder="any"
              className={`${fieldLook} w-20 tabular-nums`}
            />
          </Labelled>
          <p className="pb-2 text-[12px] leading-snug text-ash">
            A ceiling, not a width — the answer ends as soon as the number is complete, and may have a decimal part or a sign.
            Leave it empty when you do not know the magnitude; {MAX_DIGITS} digits is the widest served either way.
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
          className={`${fieldLook} min-w-0 flex-1 py-1 font-mono text-[12px]`}
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

/**
 * The options of a `choice`, in the order they will be sent.
 *
 * A description is a sentence in a column with room for about six words, so
 * when one is too long to read its neighbours fold away and it takes the whole
 * row. They fold back the moment focus leaves.
 *
 * Only when it is **actually clipped**: a description that fits is left where
 * it is, because moving a row that was already readable is motion for its own
 * sake. And once folded it stays folded until focus leaves — re-checking would
 * find the text fits *because* the row widened, fold it back, find it clipped
 * again, and oscillate.
 *
 * The width rides on a wrapper rather than on the controls themselves: a
 * second width utility on an element that already carries one does not win —
 * Tailwind emits its own in its own order — so the collapsing box is a box of
 * its own, and the control inside it keeps its size and is clipped.
 *
 * Everything stays reachable by keyboard: the name and the delete button are
 * still in the tab order while folded, and focusing either unfolds the row
 * again.
 */
function OptionList({ options, stem, onChange }: { options: Option[]; stem: string; onChange: (options: Option[]) => void }) {
  const [writing, setWriting] = useState<number | null>(null);
  const replace = (index: number, option: Option) => onChange(options.map((o, i) => (i === index ? option : o)));
  /** Widen this row if what it holds does not fit; never narrow it. */
  const widenIfClipped = (index: number, field: HTMLInputElement) => {
    if (isClipped(field)) setWriting(index);
  };
  const fold = "shrink-0 overflow-hidden transition-[width,margin] duration-200 motion-reduce:transition-none";
  return (
    <div className="mt-2">
      <p className={caption}>Options, in the order the prompt lists them</p>
      <ol className="mt-1 flex flex-col gap-1">
        {options.map((option, index) => {
          const wide = writing === index;
          return (
            <li key={index} className="flex items-center">
              <span className={`${fold} ${wide ? "mr-0 w-0" : "mr-1.5 w-[7.5rem]"}`}>
                <input
                  value={option.key}
                  onChange={(e) => replace(index, { ...option, key: e.target.value })}
                  onFocus={() => setWriting(null)}
                  name={`${stem}-option-${index}-key`}
                  aria-label={`Option ${index + 1} name`}
                  placeholder="name"
                  className={`${fieldLook} w-[7.5rem] font-mono text-[13px]`}
                />
              </span>
              <input
                value={option.description}
                onChange={(e) => {
                  replace(index, { ...option, description: e.target.value });
                  // The element already holds the new text, so this measures
                  // the line as it now is: a row widens the moment its
                  // description outgrows it.
                  widenIfClipped(index, e.currentTarget);
                }}
                onFocus={(e) => widenIfClipped(index, e.currentTarget)}
                onBlur={() => setWriting((current) => (current === index ? null : current))}
                name={`${stem}-option-${index}-description`}
                aria-label={`Option ${index + 1} description`}
                placeholder="what it means — blank lets the name speak for itself"
                className={`${fieldLook} min-w-0 flex-1`}
              />
              <span className={`${fold} ${wide ? "ml-0 w-0" : "ml-1.5 w-7"}`} onFocus={() => setWriting(null)}>
                <RemoveButton label={`Remove option ${index + 1}`} onClick={() => onChange(options.filter((_, i) => i !== index))} />
              </span>
            </li>
          );
        })}
      </ol>
      <AddButton onClick={() => onChange([...options, emptyOption()])}>Add an option</AddButton>
    </div>
  );
}

/**
 * Whether a field is showing less than it holds.
 *
 * A pixel of rounding is not an overflow: a box whose content is one pixel
 * over reads as full, and treating that as clipped would widen rows nobody
 * would call unreadable.
 */
export const isClipped = (field: { scrollWidth: number; clientWidth: number }): boolean =>
  field.scrollWidth > field.clientWidth + 1;

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
              className={`${fieldLook} min-w-0 flex-1`}
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
