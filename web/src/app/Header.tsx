import type { ReactNode } from "react";
import type { ModelState } from "../api/model.ts";
import flame from "../brand/flame.webp";
import wordmark from "../brand/wordmark-light.webp";
import { useMemoryNotes } from "../tools/local/memory.ts";
import { IconSessions, IconSliders } from "../ui/icons.tsx";

/** The kiln bar: brand, model status, and the drawer buttons below `lg`. */
export function Header({
  model,
  busy,
  onOpen,
  onForgetKey,
  onOpenMemory,
}: {
  model: ModelState;
  busy: boolean;
  onOpen: (drawer: "sessions" | "settings") => void;
  /** Present when an API key is in use: drops it and returns to the key prompt. */
  onForgetKey?: () => void;
  onOpenMemory: () => void;
}) {
  return (
    <header className="z-20 shrink-0 bg-kiln text-[#eae8e4]">
      <div className="flex items-center gap-3 px-4 py-3 sm:gap-4 md:px-6">
        <HeaderButton label="Sessions" onClick={() => onOpen("sessions")}>
          <IconSessions />
        </HeaderButton>
        <img src={flame} alt="" className="flame h-9 w-auto" data-busy={busy} />
        <img src={wordmark} alt="ignis" className="h-[18px] w-auto" />
        <span className="hidden h-6 w-px bg-kiln-line sm:block" aria-hidden />
        <span className="hidden font-display text-[15px] font-medium tracking-wide text-[#b9bec4] sm:block">Playground</span>
        <ModelStatus model={model} busy={busy} />
        <MemoryButton onOpen={onOpenMemory} />
        {onForgetKey && (
          <button
            type="button"
            title="Forget the API key"
            className="shrink-0 px-2 py-1 font-display text-[13px] font-medium text-[#939ba4] hover:bg-kiln-line hover:text-white"
            onClick={onForgetKey}
          >
            Lock
          </button>
        )}
        <HeaderButton label="Settings" onClick={() => onOpen("settings")}>
          <IconSliders />
        </HeaderButton>
      </div>
      <div className="heat" data-busy={busy} aria-hidden />
    </header>
  );
}

/** Opens what ignis remembers; the count is how many notes it keeps. */
function MemoryButton({ onOpen }: { onOpen: () => void }) {
  const count = useMemoryNotes().length;
  return (
    <button
      type="button"
      title="What ignis remembers"
      className="flex shrink-0 items-baseline gap-1.5 px-2 py-1 font-display text-[13px] font-medium text-[#939ba4] hover:bg-kiln-line hover:text-white"
      onClick={onOpen}
    >
      Memory
      {count > 0 && <span className="text-[#ff8a4c] tabular-nums">{count}</span>}
    </button>
  );
}

function HeaderButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className="grid size-9 shrink-0 place-items-center text-[#b9bec4] hover:bg-kiln-line hover:text-white lg:hidden"
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function ModelStatus({ model, busy }: { model: ModelState; busy: boolean }) {
  return (
    <span className="ml-auto flex min-w-0 items-center gap-2 font-display text-sm">
      {model.state === "ready" && (
        <>
          <span className={`cut size-2.5 shrink-0 [--cut-size:4px] ${busy ? "bg-[#ff5a1f]" : "bg-[#3ecf8e]"}`} aria-hidden />
          <span className="hidden truncate text-[#eae8e4] sm:inline">{model.id}</span>
          <span className="sr-only">{busy ? "generating" : "ready"}</span>
        </>
      )}
      {model.state === "loading" && <span className="text-[#939ba4]">Loading model…</span>}
      {model.state === "error" && <span className="truncate text-[#ff7b6b]">{model.message}</span>}
    </span>
  );
}
