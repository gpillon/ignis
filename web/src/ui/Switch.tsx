export function Switch({ on, onChange, labelledBy }: { on: boolean; onChange: (on: boolean) => void; labelledBy: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-labelledby={labelledBy}
      className="mt-0.5 shrink-0 cursor-pointer"
      onClick={() => onChange(!on)}
    >
      <span
        className={`cut relative block h-6 w-11 transition-colors [--cut-size:7px] motion-reduce:transition-none ${on ? "bg-[#ff5a1f]" : "bg-line"}`}
      >
        <span
          className={`cut absolute top-1 left-1 size-4 transition-transform duration-200 [--cut-size:5px] motion-reduce:transition-none ${on ? "translate-x-5 bg-[#1c2026]" : "bg-ink"}`}
        />
      </span>
    </button>
  );
}
