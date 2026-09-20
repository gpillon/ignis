import { IconPlus, IconTrash } from "../ui/icons.tsx";
import { counts, type Decision, type DecisionList, titleOf } from "./sessions.ts";

// The left-hand list of decisions (GitHub #247), in the chat's own shape: a
// drawer below `lg`, a static column above it, with its own scroll.

export function Sessions({
  list,
  onNew,
  onSelect,
  onRemove,
  open,
}: {
  list: DecisionList;
  onNew: () => void;
  onSelect: (id: number) => void;
  onRemove: (id: number) => void;
  /** Below `lg` the column is a drawer; this is whether it is out. */
  open: boolean;
}) {
  return (
    <aside
      aria-label="Decisions"
      className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-line bg-ground transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:w-60 lg:translate-x-0 ${open ? "translate-x-0" : "max-lg:invisible max-lg:-translate-x-full"}`}
    >
      <div className="p-3">
        <button
          type="button"
          className="cut flex w-full items-center gap-2 bg-surface px-3 py-2.5 font-display text-sm font-medium text-ink hover:bg-line"
          onClick={onNew}
        >
          <IconPlus />
          New decision
        </button>
      </div>
      <ul className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-3 pb-3">
        {list.decisions.map((decision) => (
          <Item
            key={decision.id}
            decision={decision}
            active={decision.id === list.activeId}
            onSelect={() => onSelect(decision.id)}
            onRemove={() => onRemove(decision.id)}
          />
        ))}
      </ul>
    </aside>
  );
}

function Item({
  decision,
  active,
  onSelect,
  onRemove,
}: {
  decision: Decision;
  active: boolean;
  onSelect: () => void;
  onRemove: () => void;
}) {
  const { asked, answered } = counts(decision);
  return (
    <li className={`group relative flex items-stretch ${active ? "bg-surface shadow-[inset_2px_0_0_var(--ember)]" : "hover:bg-surface/60"}`}>
      <button
        type="button"
        aria-current={active ? "true" : undefined}
        className="flex min-w-0 flex-1 flex-col gap-0.5 px-3 py-2 text-left"
        onClick={onSelect}
      >
        <span className={`truncate text-sm ${active ? "text-ink" : "text-ink/85"}`}>{titleOf(decision)}</span>
        <span className="flex items-center gap-1.5 font-display text-xs text-ash">
          {decision.running && <span className="cut size-2 bg-ember [--cut-size:3px]" aria-hidden />}
          {decision.running ? "Deciding" : summary(asked, answered, decision.refusal !== null)}
        </span>
      </button>
      <button
        type="button"
        aria-label={`Delete decision: ${titleOf(decision)}`}
        title="Delete decision"
        disabled={decision.running}
        className={`grid w-8 shrink-0 place-items-center text-ash hover:text-fault focus-visible:opacity-100 disabled:hidden ${active ? "" : "opacity-0 group-hover:opacity-100 max-lg:opacity-100"}`}
        onClick={onRemove}
      >
        <IconTrash />
      </button>
    </li>
  );
}

function summary(asked: number, answered: number, refused: boolean): string {
  if (refused) return "Refused";
  if (answered > 0) return answered === 1 ? "1 answer" : `${answered} answers`;
  if (asked === 0) return "Nothing asked yet";
  return asked === 1 ? "1 question" : `${asked} questions`;
}
