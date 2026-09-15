import { IconClose, IconPlus } from "../ui/icons.tsx";
import type { Session, SessionList } from "./sessions.ts";

/** The left-hand session list. A drawer below `lg`, shown when `open`. */
export function SessionsPanel(props: {
  open: boolean;
  list: SessionList;
  activeId: number;
  streamingId: number | null;
  onNew: () => void;
  onSelect: (id: number) => void;
  onRemove: (id: number) => void;
}) {
  return (
    <aside
      aria-label="Sessions"
      className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-line bg-ground transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:w-60 lg:translate-x-0 ${props.open ? "translate-x-0" : "max-lg:invisible max-lg:-translate-x-full"}`}
    >
      <div className="p-3">
        <button
          type="button"
          className="cut flex w-full items-center gap-2 bg-surface px-3 py-2.5 font-display text-sm font-medium text-ink hover:bg-line"
          onClick={props.onNew}
        >
          <IconPlus />
          New session
        </button>
      </div>
      <ul className="flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-3 pb-3">
        {props.list.sessions.map((s) => (
          <SessionItem
            key={s.id}
            session={s}
            active={s.id === props.activeId}
            streaming={s.id === props.streamingId}
            onSelect={() => props.onSelect(s.id)}
            onRemove={() => props.onRemove(s.id)}
          />
        ))}
      </ul>
    </aside>
  );
}

function SessionItem(props: {
  session: Session;
  active: boolean;
  streaming: boolean;
  onSelect: () => void;
  onRemove: () => void;
}) {
  const { session: s, active, streaming } = props;
  const replies = s.log.filter((row) => row.agent === undefined).length;
  return (
    <li
      className={`group relative flex items-stretch ${active ? "bg-surface shadow-[inset_2px_0_0_var(--ember)]" : "hover:bg-surface/60"}`}
    >
      <button
        type="button"
        aria-current={active ? "true" : undefined}
        className="flex min-w-0 flex-1 flex-col gap-0.5 px-3 py-2 text-left"
        onClick={props.onSelect}
      >
        <span className={`truncate text-sm ${active ? "text-ink" : "text-ink/85"}`}>{s.title}</span>
        <span className="flex items-center gap-1.5 font-display text-xs text-ash">
          {streaming && <span className="cut size-2 bg-ember [--cut-size:3px]" aria-hidden />}
          {streaming ? "Streaming" : replies === 0 ? "No replies yet" : replies === 1 ? "1 reply" : `${replies} replies`}
        </span>
      </button>
      <button
        type="button"
        aria-label={`Delete session: ${s.title}`}
        title="Delete session"
        disabled={streaming}
        className={`grid w-8 shrink-0 place-items-center text-ash hover:text-fault focus-visible:opacity-100 disabled:hidden ${active ? "" : "opacity-0 group-hover:opacity-100 max-lg:opacity-100"}`}
        onClick={props.onRemove}
      >
        <IconClose />
      </button>
    </li>
  );
}
