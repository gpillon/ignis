import type { ReactNode } from "react";

/** What every turn can do; `canRerun` is false while any reply streams or without a model. */
export type TurnActions = {
  canRerun: boolean;
  onSave: (text: string) => void;
  onFork: () => void;
};

export const RERUN_BLOCKED = "Wait for the current reply to end";

/**
 * Whether a turn would render as it did: every prop the same, the handlers
 * aside — the transcript's always reach the page's current ones — and of the
 * actions only whether a rerun is allowed. While a reply streams, this is
 * what keeps every finished turn out of each frame's render (GitHub #283).
 */
export function sameTurnProps<P extends { actions: TurnActions }>(prev: P, next: P): boolean {
  const keys = Object.keys(next) as (keyof P)[];
  if (keys.length !== Object.keys(prev).length) return false;
  return keys.every((key) =>
    key === "actions" ? prev.actions.canRerun === next.actions.canRerun : typeof next[key] === "function" || prev[key] === next[key],
  );
}

/** A turn's actions: shown on hover or focus on wide screens, always on the last reply and on touch widths. */
export function ActionRow({ children, className = "", pinned = false }: { children: ReactNode; className?: string; pinned?: boolean }) {
  return (
    <div
      className={`flex flex-wrap items-center gap-0.5 transition-opacity motion-reduce:transition-none ${pinned ? "" : "lg:opacity-0 lg:group-focus-within/turn:opacity-100 lg:group-hover/turn:opacity-100"} ${className}`}
    >
      {children}
    </div>
  );
}

export function ActionButton(props: { icon: ReactNode; label: string; onClick: () => void; disabled?: boolean; title?: string }) {
  return (
    <button
      type="button"
      title={props.title ?? props.label}
      disabled={props.disabled}
      onClick={props.onClick}
      className="flex items-center gap-1.5 px-2 py-1 font-display text-xs font-medium text-ash hover:bg-surface hover:text-ink disabled:cursor-default disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-ash"
    >
      {props.icon}
      {props.label}
    </button>
  );
}

export function EditedMark() {
  return (
    <span className="px-2 font-display text-xs text-ash" title="Changed by hand; any figures measure the original">
      Edited
    </span>
  );
}
