import type { CSSProperties } from "react";
import { caption } from "./classes.ts";
import { trackFill } from "./slider.ts";

export function Slider(props: { label: string; name: string; min: number; max: number; step: number; value: number; onChange: (value: number) => void }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="flex items-baseline justify-between">
        <span className={caption}>{props.label}</span>
        <output className="font-display text-sm font-medium tabular-nums text-ink">{props.value.toFixed(2)}</output>
      </span>
      <input
        type="range"
        name={props.name}
        className="range w-full"
        style={{ "--fill": trackFill(props.value, props.min, props.max) } as CSSProperties}
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}
