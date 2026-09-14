import { useEffect, useState } from "react";

// The model ignis serves, asked once from GET /v1/models when the page opens.

export type ModelState =
  | { state: "loading" }
  | { state: "ready"; id: string; contextLimit: number | null }
  | { state: "error"; message: string };

export function useModel(): ModelState {
  const [model, setModel] = useState<ModelState>({ state: "loading" });

  useEffect(() => {
    fetch("/v1/models")
      .then(async (res) => {
        if (!res.ok) throw new Error(`GET /v1/models: ${res.status}`);
        const body = (await res.json()) as { data?: { id: string; max_model_len?: number }[] };
        const id = body.data?.[0]?.id;
        if (!id) throw new Error("GET /v1/models: no model listed");
        setModel({ state: "ready", id, contextLimit: body.data?.[0]?.max_model_len ?? null });
      })
      .catch((err: unknown) => setModel({ state: "error", message: String(err) }));
  }, []);

  return model;
}
