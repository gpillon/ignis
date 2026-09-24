import { caption } from "./classes.ts";

/**
 * A row of mutually exclusive options. `value` null checks none of them;
 * `disabled` greys the row and takes every option out of reach.
 */
export function Segmented<T extends string>(props: {
  legend: string;
  name: string;
  value: T | null;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  const disabled = props.disabled ?? false;
  return (
    <fieldset className="flex flex-col gap-2" disabled={disabled}>
      <legend className={`${caption} mb-2`}>{props.legend}</legend>
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
              className={`block px-1 py-1.5 text-center font-display text-[13px] font-medium text-ash peer-checked:bg-ink peer-checked:text-ground peer-focus-visible:outline-2 peer-focus-visible:outline-ember ${disabled ? "cursor-not-allowed" : "cursor-pointer hover:text-ink peer-checked:hover:text-ground"}`}
            >
              {o.label}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
