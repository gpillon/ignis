import { useState } from "react";
import { caption, field } from "../ui/classes.ts";

/** The system prompt in two tabs: the owner's, editable, and the ignis one the enabled tools write, read-only. */
export function SystemPromptField({ value, onChange, ignis }: { value: string; onChange: (value: string) => void; ignis: string }) {
  const [tab, setTab] = useState<"user" | "ignis">("user");
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-end justify-between gap-2">
        <span id="system-prompt-label" className={caption}>
          System prompt
        </span>
        <div role="tablist" aria-labelledby="system-prompt-label" className="flex gap-3">
          {(["user", "ignis"] as const).map((t) => (
            <button
              key={t}
              type="button"
              role="tab"
              id={`system-tab-${t}`}
              aria-selected={tab === t}
              aria-controls="system-prompt-panel"
              onClick={() => setTab(t)}
              className={`flex items-center gap-1.5 border-b-2 pb-0.5 font-display text-[13px] font-medium ${tab === t ? "border-ember text-ink" : "border-transparent text-ash hover:text-ink"}`}
            >
              {t === "user" ? "User" : "ignis"}
              {t === "ignis" && ignis && (
                <>
                  <span className="cut size-1.5 bg-ember [--cut-size:2px]" aria-hidden />
                  <span className="sr-only">(tools add text here)</span>
                </>
              )}
            </button>
          ))}
        </div>
      </div>
      <div id="system-prompt-panel" role="tabpanel" aria-labelledby={`system-tab-${tab}`}>
        {tab === "user" ? (
          <textarea
            className={`${field} resize-y leading-relaxed`}
            rows={5}
            placeholder="None"
            name="system-prompt"
            aria-label="Your system prompt"
            value={value}
            onChange={(e) => onChange(e.target.value)}
          />
        ) : ignis ? (
          <textarea
            readOnly
            className={`${field} resize-y border-dashed bg-ground text-xs leading-relaxed text-ash focus:border-line`}
            rows={10}
            name="ignis-system-prompt"
            aria-label="The ignis system prompt, read-only"
            value={ignis}
          />
        ) : (
          <p className="border border-dashed border-line px-2.5 py-2 text-xs leading-relaxed text-ash">
            Empty. Turn on a tool and ignis writes its instructions here, sent ahead of yours.
          </p>
        )}
      </div>
    </div>
  );
}
