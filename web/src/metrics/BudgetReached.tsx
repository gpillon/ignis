/**
 * The mark on a reasoning block that ignis closed because the thinking
 * budget was spent (spec server/08's `thinking_budget_forced_at`), rather
 * than the model closing it itself. `at` is the reasoning tokens the reply
 * had emitted when the forced close began.
 */
export function BudgetReached({ at }: { at: number }) {
  return (
    <span
      className="ml-2 inline-flex items-center gap-1.5 font-display text-xs font-medium text-ember"
      title={`The thinking budget closed the reasoning after ${at} reasoning tokens`}
    >
      <span className="cut size-1.5 shrink-0 bg-ember [--cut-size:2px]" aria-hidden />
      Budget reached
    </span>
  );
}
