import { describe, expect, it } from "vitest";
import type { CatalogRow } from "../server/routes.ts";
import { createVisionWatch, readVision, visionModel, visionOf } from "./vision.ts";

// Whether the loaded model has vision, as the page learns it from what the
// server reports (#394): the running family's `vision.enabled` in
// `GET /v1/config`, read again after every switch. Unknown is no vision.

const row = (id: string, family: string, loaded: boolean): CatalogRow => ({ id, family, loaded, source: "built-in", on_disk: "complete", switchable: true });

const catalog = (loaded: string) => [
  row("qwen3.8-27b", "qwen38", loaded === "qwen3.8-27b"),
  row("qwen3.8-flash-next", "qwen38flashnext", loaded === "qwen3.8-flash-next"),
  row("ignis-image-2.1-turbo", "qwenimage21", loaded === "ignis-image-2.1-turbo"),
];

describe("the vision of the loaded model", () => {
  it("is its family's vision.enabled, from the family's own section, else the group's", () => {
    expect(visionOf({ vision: { qwen38: { enabled: true, max_tokens: 32768 } } }, catalog("qwen3.8-27b"))).toBe(true);
    expect(visionOf({ vision: { qwen38: { enabled: false } } }, catalog("qwen3.8-27b"))).toBe(false);
    expect(visionOf({ vision: { enabled: true } }, catalog("qwen3.8-27b"))).toBe(true);
    expect(visionOf({ vision: { enabled: true, qwen38: { enabled: false } } }, catalog("qwen3.8-27b"))).toBe(false);
  });

  it("is none for a family the server gives no vision, and whenever it is not known", () => {
    expect(visionOf({ vision: { qwen38: { enabled: true } } }, catalog("qwen3.8-flash-next"))).toBe(false);
    expect(visionOf({ vision: { qwen38: { enabled: true } } }, catalog("ignis-image-2.1-turbo"))).toBe(false);
    expect(visionOf({ vision: { qwen38: { enabled: true } } }, catalog("none"))).toBe(false);
    expect(visionOf(null, catalog("qwen3.8-27b"))).toBe(false);
    expect(visionOf({ vision: { qwen38: { enabled: true } } }, null)).toBe(false);
    expect(visionOf({ vision: { qwen38: { enabled: "yes" } } }, catalog("qwen3.8-27b"))).toBe(false);
  });

  it("is read from the server's running configuration and its catalog, and is none when either cannot be read", async () => {
    const json = (body: object, status = 200) => new Response(JSON.stringify(body), { status });
    const server = (config: Response) =>
      (async (path: string) => {
        if (path === "/v1/config") return config;
        if (path === "/v1/models/catalog") return json({ data: catalog("qwen3.8-27b") });
        throw new Error(`unexpected ${path}`);
      }) as unknown as typeof fetch;
    expect(await readVision(server(json({ vision: { qwen38: { enabled: true } } })))).toBe(true);
    expect(await readVision(server(json({ error: { message: "no config" } }, 501)))).toBe(false);
  });
});

describe("the page's watch on vision", () => {
  /** A server whose answers the test hands out one at a time. */
  function answers() {
    const pending: ((vision: boolean) => void)[] = [];
    const read = () => new Promise<boolean>((resolve) => pending.push(resolve));
    return { read, pending };
  }

  it("knows none until the server answers for the model loaded", async () => {
    const server = answers();
    const watch = createVisionWatch(server.read);
    const read = watch.seen("qwen3.8-27b");
    expect(watch.has()).toBe(false);
    server.pending[0](true);
    await read;
    expect(watch.has()).toBe(true);
  });

  it("reads again after a switch, the same model's return included, and knows none meanwhile", async () => {
    const server = answers();
    const watch = createVisionWatch(server.read);
    const first = watch.seen("qwen3.8-27b");
    server.pending[0](true);
    await first;
    // make_image's visit to the image model: nothing ready to chat with, then the same text model again.
    await watch.seen(null);
    expect(watch.has()).toBe(false);
    const back = watch.seen("qwen3.8-27b");
    expect(server.pending).toHaveLength(2);
    expect(watch.has()).toBe(false);
    server.pending[1](false);
    await back;
    expect(watch.has()).toBe(false);
    // The same model seen again without a switch between is not read again.
    void watch.seen("qwen3.8-27b");
    expect(server.pending).toHaveLength(2);
  });

  it("drops an answer for a model no longer loaded, and takes a failed read as no vision", async () => {
    const server = answers();
    const watch = createVisionWatch(server.read);
    const old = watch.seen("qwen3.8-27b");
    const now = watch.seen("qwen3.8-flash-next");
    server.pending[0](true);
    await old;
    expect(watch.has()).toBe(false);
    server.pending[1](false);
    await now;
    expect(watch.has()).toBe(false);
    const failing = createVisionWatch(() => Promise.reject(new Error("offline")));
    await failing.seen("qwen3.8-27b");
    expect(failing.has()).toBe(false);
  });
});

describe("the model whose vision the Transcribe view follows (#437)", () => {
  it("is the text model ready, read then; kept while it is unloaded idle; none for the image model, a switch or nothing", () => {
    expect(visionModel({ state: "ready", id: "qwen3.8-27b", contextLimit: 262_144 })).toEqual({ id: "qwen3.8-27b", read: true });
    // Unloaded, the catalog loads no row: what was read for it holds, and its next request loads it again.
    expect(visionModel({ state: "unloaded", id: "qwen3.8-27b", contextLimit: 262_144 })).toEqual({ id: "qwen3.8-27b", read: false });
    expect(visionModel({ state: "ready", id: "ignis-image-2.1-turbo", contextLimit: null, kind: "image" })).toEqual({ id: null, read: false });
    expect(visionModel({ state: "switching", from: "qwen3.8-27b", to: "qwen3.8-flash-next", since: 0 })).toEqual({ id: null, read: false });
    expect(visionModel({ state: "error", message: "down" })).toEqual({ id: null, read: false });
  });
});
