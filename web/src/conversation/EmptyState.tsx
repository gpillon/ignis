import type { ModelState } from "../api/model.ts";
import flame from "../brand/flame.webp";

export function EmptyState({ model }: { model: ModelState }) {
  return (
    <div className="m-auto flex max-w-md flex-col items-center gap-7 py-10 text-center">
      <img src={flame} alt="" className="h-44 w-auto drop-shadow-[0_18px_40px_rgb(200_22_29/0.35)]" />
      <div className="flex flex-col gap-2">
        <p className="font-display text-4xl font-semibold tracking-tight text-ink">
          {model.state === "ready" ? "Light it up" : model.state === "loading" ? "Warming up" : "No model to talk to"}
        </p>
        <p className="text-sm text-ash">
          {model.state === "ready" && (
            <>
              Send a prompt to <span className="font-display text-ink">{model.id}</span>. Timings for each reply show up
              under it and in this session's log below.
            </>
          )}
          {model.state === "loading" && "Asking ignis which model it serves."}
          {model.state === "error" && "Start ignis-server, then reload this page."}
        </p>
      </div>
    </div>
  );
}
