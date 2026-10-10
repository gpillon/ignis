import { describe, expect, it } from "vitest";
import { afterRefresh, readModel, type ModelState } from "./model.ts";
import { keyRequired, saveKey } from "./auth.ts";

const listing = (id: string, max?: number) =>
  (async () => new Response(JSON.stringify({ data: [{ id, max_model_len: max }] }))) as typeof fetch;

describe("readModel", () => {
  it("reads the loaded model and its context limit from GET /v1/models", async () => {
    saveKey("sk");
    expect(await readModel(listing("qwen3.8-27b", 786432))).toEqual({ state: "ready", id: "qwen3.8-27b", contextLimit: 786432 });
  });

  it("reads again what is loaded now, so the page follows a switch another client made", async () => {
    saveKey("sk");
    const first = await readModel(listing("qwen3.8-27b", 786432));
    const second = await readModel(listing("flash-next", 589824));
    expect(first).toMatchObject({ id: "qwen3.8-27b" });
    expect(second).toEqual({ state: "ready", id: "flash-next", contextLimit: 589824 });
  });

  it("is an error when nothing is listed or ignis refuses", async () => {
    saveKey("sk");
    const empty = (async () => new Response(JSON.stringify({ data: [] }))) as typeof fetch;
    expect(await readModel(empty)).toEqual({ state: "error", message: "Error: GET /v1/models: no model listed" });
    const down = (async () => new Response("", { status: 503 })) as typeof fetch;
    expect(await readModel(down)).toEqual({ state: "error", message: "Error: GET /v1/models: 503" });
    const unauthorized = (async () => new Response("", { status: 401 })) as typeof fetch;
    expect((await readModel(unauthorized)).state).toBe("error");
    keyRequired();
  });
});

describe("afterRefresh", () => {
  const ready: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: null };
  it("takes the model read", () => {
    const next: ModelState = { state: "ready", id: "flash-next", contextLimit: 1 };
    expect(afterRefresh(ready, next)).toBe(next);
  });
  it("keeps a model already shown when the re-read fails, as a turn must not lose it to a blip", () => {
    expect(afterRefresh(ready, { state: "error", message: "x" })).toBe(ready);
    const error: ModelState = { state: "error", message: "x" };
    expect(afterRefresh({ state: "loading" }, error)).toBe(error);
  });
});
