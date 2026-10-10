import { useCallback, useEffect, useState } from "react";
import { authHeaders, keyRequired, useAuth } from "./auth.ts";

// The model ignis serves, asked from GET /v1/models when the page opens,
// whenever the API key changes, and when a turn starts: another client may
// have switched the server since. The Playground only follows it — its
// requests name no model, so none of them can switch it.

export type ModelState =
  | { state: "loading" }
  | { state: "ready"; id: string; contextLimit: number | null }
  | { state: "error"; message: string };

/** The model loaded now, or why it could not be read. */
export async function readModel(fetcher: typeof fetch = fetch): Promise<ModelState> {
  try {
    const res = await fetcher("/v1/models", { headers: authHeaders() });
    if (res.status === 401) {
      keyRequired();
      throw new Error("GET /v1/models: ignis wants an API key");
    }
    if (!res.ok) throw new Error(`GET /v1/models: ${res.status}`);
    const body = (await res.json()) as { data?: { id: string; max_model_len?: number }[] };
    const id = body.data?.[0]?.id;
    if (!id) throw new Error("GET /v1/models: no model listed");
    return { state: "ready", id, contextLimit: body.data?.[0]?.max_model_len ?? null };
  } catch (err) {
    return { state: "error", message: String(err) };
  }
}

/** What a re-read leaves on screen: a model already shown is not lost to a read that failed. */
export function afterRefresh(shown: ModelState, read: ModelState): ModelState {
  return read.state === "error" && shown.state === "ready" ? shown : read;
}

export function useModel(): [ModelState, () => void] {
  const [model, setModel] = useState<ModelState>({ state: "loading" });
  const { key, needsKey } = useAuth();

  useEffect(() => {
    if (needsKey) return;
    setModel({ state: "loading" });
    void readModel().then(setModel);
  }, [key, needsKey]);

  const refresh = useCallback(() => {
    if (needsKey) return;
    void readModel().then((read) => setModel((shown) => afterRefresh(shown, read)));
  }, [needsKey]);

  return [model, refresh];
}
