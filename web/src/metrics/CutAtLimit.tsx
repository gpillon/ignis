/** Whether a reply ended at the token limit (finish reason `length`), the cap or a passed deadline. */
export const endedAtTokenLimit = (figures?: { finishReason: string | null } | null): boolean => figures?.finishReason === "length";

/** The tool result of a call the turn did not run because its reply was cut at the token limit. */
export const CUT_NOT_RUN = "Not run: the reply was cut by the token limit.";

/** The mark at the end of a reply that stopped at its token limit; it looks like the budget's. */
export function CutAtLimit() {
  return (
    <span
      className="inline-flex items-center gap-1.5 font-display text-xs font-medium text-ember"
      title="The reply reached its token limit and stopped here."
    >
      <span className="cut size-1.5 shrink-0 bg-ember [--cut-size:2px]" aria-hidden />
      Cut at the token limit
    </span>
  );
}
