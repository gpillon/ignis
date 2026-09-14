import { type KeyboardEvent, useState } from "react";

export type EditorButton = { label: string; primary?: boolean; disabled?: boolean; title?: string; onClick: (text: string) => void };

/** In-place message editor. Esc cancels; Ctrl+Enter runs the primary button. */
export function MessageEditor({ initial, buttons, onCancel }: { initial: string; buttons: EditorButton[]; onCancel: () => void }) {
  const [draft, setDraft] = useState(initial);
  const empty = draft.trim() === "";
  const primary = buttons.find((b) => b.primary);

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && primary && !primary.disabled && !empty) {
      e.preventDefault();
      primary.onClick(draft);
    }
  }

  return (
    <div className="cut flex w-full flex-col gap-2 bg-surface p-2 shadow-[inset_0_-2px_0_var(--ember)] [--cut-size:12px]">
      <textarea
        autoFocus
        name="message"
        aria-label="Edit message"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKeyDown}
        onFocus={(e) => e.currentTarget.setSelectionRange(e.currentTarget.value.length, e.currentTarget.value.length)}
        className="max-h-[60vh] min-h-24 w-full resize-none bg-transparent px-2 py-1.5 text-[15px] leading-normal text-ink [field-sizing:content] focus:outline-none"
      />
      <div className="flex flex-wrap items-center justify-end gap-2">
        {primary && (
          <span className="mr-auto px-2 text-xs text-ash">
            Esc to cancel, Ctrl+Enter to {primary.label.toLowerCase()}.
          </span>
        )}
        <button type="button" className="px-3 py-1.5 font-display text-sm font-medium text-ash hover:text-ink" onClick={onCancel}>
          Cancel
        </button>
        {buttons.map((b) => (
          <button
            key={b.label}
            type="button"
            title={b.title}
            disabled={b.disabled || empty}
            onClick={() => b.onClick(draft)}
            className={`cut px-4 py-1.5 font-display text-sm disabled:cursor-default disabled:bg-line disabled:text-ash ${
              b.primary
                ? "bg-[#ff5a1f] font-semibold text-[#1c2026] hover:bg-[#ff7a45]"
                : "bg-ground font-medium text-ink hover:bg-line"
            }`}
          >
            {b.label}
          </button>
        ))}
      </div>
    </div>
  );
}
