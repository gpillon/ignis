import { useState } from "react";
import type { ReasoningEffort } from "../api/request.ts";
import { field } from "../ui/classes.ts";
import { Segmented } from "../ui/Segmented.tsx";

// The thinking budget (spec playground/04): how many tokens a reply may
// reason before ignis closes its thinking (spec server/08). The server's
// default sends nothing, Off sends 0, a count sends that count. The presets
// cover the common counts; the box takes any other. An effort that cannot use
// a budget — thinking off, or `max`, which ignores one — disables the control
// and shows no choice, while the setting itself is kept for the next effort.

/** The preset counts, in tokens: 2K, 4K, 8K, 16K. */
export const BUDGET_PRESETS = [2048, 4096, 8192, 16384] as const;

/** The largest budget ignis accepts: its field is a u32, and past it the request is a 400. */
export const MAX_THINKING_BUDGET = 4_294_967_295;

const WHOLE = "Type a whole number of tokens, 1 or more.";

/**
 * What the box holds, read as a budget: a whole count of 1 or more, or the
 * reason it is not one. An empty box is no count and no fault either — the
 * reason is null. 0 is refused here because Off is the way to say it.
 */
export function readBudgetInput(text: string): { ok: true; tokens: number } | { ok: false; error: string | null } {
  const trimmed = text.trim();
  if (trimmed === "") return { ok: false, error: null };
  if (!/^\d+$/.test(trimmed)) return { ok: false, error: WHOLE };
  const tokens = Number(trimmed);
  if (tokens === 0) return { ok: false, error: `${WHOLE} For no budget, pick Off.` };
  if (tokens > MAX_THINKING_BUDGET) return { ok: false, error: `ignis takes at most ${MAX_THINKING_BUDGET} tokens.` };
  return { ok: true, tokens };
}

type Choice = "default" | "off" | `${(typeof BUDGET_PRESETS)[number]}`;

const OPTIONS: { value: Choice; label: string }[] = [
  { value: "default", label: "Default" },
  { value: "off", label: "Off" },
  ...BUDGET_PRESETS.map((tokens) => ({ value: `${tokens}` as Choice, label: `${tokens / 1024}K` })),
];

/** The option a budget checks, or null for a typed count that is no preset. */
function choiceOf(budget: number | null): Choice | null {
  if (budget === null) return "default";
  if (budget === 0) return "off";
  return OPTIONS.find((o) => o.value === `${budget}`)?.value ?? null;
}

export function ThinkingBudgetField(props: {
  budget: number | null;
  effort: ReasoningEffort;
  maxTokens: number | null;
  onChange: (budget: number | null) => void;
}) {
  const { budget, maxTokens } = props;
  // What is being typed, half-typed and wrong included; null once the box is left, which shows the setting again.
  const [draft, setDraft] = useState<string | null>(null);
  const inert = props.effort === "none" ? "thinking off" : props.effort === "max" ? "no budget (max)" : null;
  const typed = draft === null ? null : readBudgetInput(draft);
  const fault = typed && !typed.ok ? typed.error : null;
  const tokens = inert === null && budget !== null && budget > 0 ? budget : null;
  const over = tokens !== null && maxTokens !== null && tokens > maxTokens;

  const pick = (choice: Choice) => {
    setDraft(null);
    props.onChange(choice === "default" ? null : choice === "off" ? 0 : Number(choice));
  };
  const type = (text: string) => {
    setDraft(text);
    const read = readBudgetInput(text);
    if (read.ok) props.onChange(read.tokens);
  };

  return (
    <div className="flex flex-col gap-2">
      <Segmented
        legend="Thinking budget"
        name="thinking-budget-choice"
        value={inert === null ? choiceOf(budget) : null}
        options={OPTIONS}
        onChange={pick}
        disabled={inert !== null}
      />
      <input
        className={`${field} font-display tabular-nums disabled:cursor-not-allowed disabled:opacity-50 ${fault ? "border-fault" : ""}`}
        type="text"
        inputMode="numeric"
        name="thinking-budget"
        aria-label="Thinking budget in tokens"
        aria-invalid={fault ? true : undefined}
        placeholder={inert ?? "Tokens"}
        disabled={inert !== null}
        value={inert !== null ? "" : (draft ?? (tokens === null ? "" : String(tokens)))}
        onChange={(e) => type(e.target.value)}
        onBlur={() => setDraft(null)}
      />
      {fault && <span className="border-l-2 border-fault pl-2 text-xs leading-snug text-fault">{fault}</span>}
      {over && (
        <span className="border-l-2 border-ember pl-2 text-xs leading-snug text-ash">
          More than max tokens ({maxTokens}): ignis shortens the budget to leave the answer room.
        </span>
      )}
      <span className="text-xs leading-snug text-ash">
        How long a reply may reason before ignis closes its thinking. Default is the server's own; Off is no budget.
      </span>
    </div>
  );
}
