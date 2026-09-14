import { caption } from "./classes.ts";

export function Segmented<T extends string>(props: {
  legend: string;
  name: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className={`${caption} mb-2`}>{props.legend}</legend>
      <div className="flex border border-line bg-surface p-0.5">
        {props.options.map((o) => (
          <label key={o.value} className="relative flex-1">
            <input
              type="radio"
              name={props.name}
              className="peer sr-only"
              checked={props.value === o.value}
              onChange={() => props.onChange(o.value)}
            />
            <span className="block cursor-pointer px-1 py-1.5 text-center font-display text-[13px] font-medium text-ash peer-checked:bg-ink peer-checked:text-ground peer-focus-visible:outline-2 peer-focus-visible:outline-ember hover:text-ink peer-checked:hover:text-ground">
              {o.label}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
