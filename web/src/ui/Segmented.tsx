import { caption } from "./classes.ts";

/**
 * A row of mutually exclusive options. `value` null checks none of them;
 * `disabled` greys the row and takes every option out of reach;
 * `legendHidden` keeps the legend for screen readers only, for a row that
 * names the field beside it. An option's `accent` fills it with the heat,
 * its words said to a screen reader after the label (the running family's
 * "running").
 */
export function Segmented<T extends string>(props: {
  legend: string;
  name: string;
  value: T | null;
  options: { value: T; label: string; accent?: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
  legendHidden?: boolean;
}) {
  const disabled = props.disabled ?? false;
  return (
    // Relative: the hidden legend is `position: absolute`, and with no box of its own to sit in it is the page's, which
    // a scrolling container does not clip, so a row below the fold scrolled the whole page.
    <fieldset className="relative flex flex-col gap-2" disabled={disabled}>
      <legend className={props.legendHidden ? "sr-only" : `${caption} mb-2`}>{props.legend}</legend>
      <div className={`flex border border-line bg-surface p-0.5 ${disabled ? "opacity-50" : ""}`}>
        {props.options.map((o) => (
          <label key={o.value} className="relative flex-1">
            <input
              type="radio"
              name={props.name}
              className="peer sr-only"
              checked={props.value === o.value}
              onChange={() => props.onChange(o.value)}
            />
            <span
              // The heat's own text colour: a second colour utility beside it would win, Tailwind's order over the class string's.
              className={`block px-1 py-1.5 text-center font-display text-[13px] font-medium ${o.accent ? "heat-fill peer-checked:ring-2 peer-checked:ring-ink peer-checked:ring-inset" : "text-ash peer-checked:bg-ink peer-checked:text-ground"} peer-focus-visible:outline-2 peer-focus-visible:outline-ember ${disabled ? "cursor-not-allowed" : o.accent ? "cursor-pointer hover:brightness-110" : "cursor-pointer hover:text-ink peer-checked:hover:text-ground"}`}
            >
              {o.label}
              {o.accent && <span className="sr-only"> ({o.accent})</span>}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
