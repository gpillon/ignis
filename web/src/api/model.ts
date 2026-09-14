import { useEffect, useState } from "react";
import { authHeaders, keyRequired, useAuth } from "./auth.ts";

// The model ignis serves, asked from GET /v1/models when the page opens, and
// again whenever the API key changes.

export type ModelState =
  | { state: "loading" }
  | { state: "ready"; id: string; contextLimit: number | null }
  | { state: "error"; message: string };

export function useModel(): ModelState {
  const [model, setModel] = useState<ModelState>({ state: "loading" });
  const { key, needsKey } = useAuth();

  useEffect(() => {
    if (needsKey) return;
    setModel({ state: "loading" });
    fetch("/v1/models", { headers: authHeaders() })
      .then(async (res) => {
        if (res.status === 401) {
          keyRequired();
          throw new Error("GET /v1/models: ignis wants an API key");
        }
        if (!res.ok) throw new Error(`GET /v1/models: ${res.status}`);
        const body = (await res.json()) as { data?: { id: string; max_model_len?: number }[] };
        const id = body.data?.[0]?.id;
        if (!id) throw new Error("GET /v1/models: no model listed");
        setModel({ state: "ready", id, contextLimit: body.data?.[0]?.max_model_len ?? null });
      })
      .catch((err: unknown) => setModel({ state: "error", message: String(err) }));
  }, [key, needsKey]);

  return model;
}
