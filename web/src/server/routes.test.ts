import { describe, expect, it } from "vitest";
import { getAuth, saveKey } from "../api/auth.ts";
import { cancelDownload, patchConfig, readCatalog, readConfigFile, readFields, readOrigins, refusalOf, removeModel, startDownload, switchTo } from "./routes.ts";

// The Server view's requests (spec playground/05): what each sends, and how
// each answer is read.

type Sent = { url: string; init: RequestInit };

/** A fetch that records what was sent and answers `status` with `body`. */
function answering(status: number, body: unknown, sent: Sent[] = []): typeof fetch {
  return (async (url: string, init: RequestInit) => {
    sent.push({ url, init });
    return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
  }) as typeof fetch;
}

describe("readConfigFile", () => {
  it("asks for the YAML and keeps it as text, not parsed", async () => {
    saveKey("sk-test");
    const sent: Sent[] = [];
    const yaml = "server:\n  bind: 127.0.0.1:8000\n";
    const answer = await readConfigFile(answering(200, yaml, sent));
    expect(answer).toEqual({ ok: true, status: 200, value: yaml });
    expect(sent[0].url).toBe("/v1/config?format=yaml");
    expect(sent[0].init.headers).toMatchObject({ Authorization: "Bearer sk-test" });
  });

  it("reads a refusal as the server's message", async () => {
    const answer = await readConfigFile(answering(400, { error: { message: "unknown format", code: "config_invalid" } }));
    expect(answer).toEqual({ ok: false, status: 400, message: "unknown format", code: "config_invalid" });
  });
});

describe("readOrigins", () => {
  it("asks GET /v1/config/origins with the key and reads the rows of the list envelope", async () => {
    saveKey("sk-test");
    const sent: Sent[] = [];
    const rows = [{ key: "vram.headroom_bytes", scope: "qwen38", source: "profile", scoped: true, spelling: "rtx5090" }];
    const answer = await readOrigins(answering(200, { object: "list", data: rows }, sent));
    expect(answer).toEqual({ ok: true, status: 200, value: rows });
    expect(sent[0].url).toBe("/v1/config/origins");
    expect(sent[0].init.headers).toMatchObject({ Authorization: "Bearer sk-test" });
  });

  it("is a refusal where the server has no running configuration or no such route", async () => {
    expect(await readOrigins(answering(501, { error: { message: "no configuration", code: "config_unavailable" } }))).toEqual({ ok: false, status: 501, message: "no configuration", code: "config_unavailable" });
    expect(await readOrigins(answering(404, ""))).toMatchObject({ ok: false, status: 404 });
  });
});

describe("switchTo", () => {
  it("sends POST /v1/models/switch with the id alone, never a path, and the key", async () => {
    saveKey("sk-test");
    const sent: Sent[] = [];
    const answer = await switchTo("qwen3.8-flash-next", answering(202, { status: "switching", from: "a", to: "qwen3.8-flash-next" }, sent));
    expect(answer).toEqual({ ok: true, status: 202, value: { status: "switching", from: "a", to: "qwen3.8-flash-next" } });
    expect(sent[0].url).toBe("/v1/models/switch");
    expect(sent[0].init.method).toBe("POST");
    expect(JSON.parse(sent[0].init.body as string)).toEqual({ model: "qwen3.8-flash-next" });
    expect(sent[0].init.headers).toMatchObject({ Authorization: "Bearer sk-test", "Content-Type": "application/json" });
  });

  it("reads a refusal as the server's message and code", async () => {
    saveKey("sk-test");
    const body = { error: { message: "model x is in the catalog but not on disk", type: "model_not_found", code: "model_not_found" } };
    expect(await switchTo("x", answering(404, body))).toEqual({ ok: false, status: 404, message: "model x is in the catalog but not on disk", code: "model_not_found" });
  });
});

describe("startDownload", () => {
  it("sends POST /v1/models/download with the catalog id alone, never a token, and the key", async () => {
    saveKey("sk-test");
    const sent: Sent[] = [];
    const accepted = { status: "downloading", model: "qwen3.8-27b-abliterated", bytes: 19406984175, bytes_on_disk: 0 };
    const answer = await startDownload("qwen3.8-27b-abliterated", answering(202, accepted, sent));
    expect(answer).toEqual({ ok: true, status: 202, value: accepted });
    expect(sent[0].url).toBe("/v1/models/download");
    expect(sent[0].init.method).toBe("POST");
    expect(JSON.parse(sent[0].init.body as string)).toEqual({ model: "qwen3.8-27b-abliterated" });
    expect(sent[0].init.headers).toMatchObject({ Authorization: "Bearer sk-test", "Content-Type": "application/json" });
  });

  it("reads a refusal as the server's message and code", async () => {
    saveKey("sk-test");
    const body = { error: { message: "a download of qwen3.8-flash-next is already running; downloads are one at a time, never queued", code: "download_in_progress" } };
    expect(await startDownload("x", answering(409, body))).toEqual({ ok: false, status: 409, message: body.error.message, code: "download_in_progress" });
  });
});

// GitHub #377: the running download stopped, its partial files kept.
describe("cancelDownload", () => {
  it("sends POST /v1/models/download/cancel with no body, and the key", async () => {
    saveKey("sk-test");
    const sent: Sent[] = [];
    const cancelled = { status: "cancelled", model: "qwen3.8-27b-abliterated", bytes: 19406984175, bytes_on_disk: 6442450944 };
    const answer = await cancelDownload(answering(200, cancelled, sent));
    expect(answer).toEqual({ ok: true, status: 200, value: cancelled });
    expect(sent[0].url).toBe("/v1/models/download/cancel");
    expect(sent[0].init.method).toBe("POST");
    expect(sent[0].init.body).toBeUndefined();
    expect(sent[0].init.headers).toMatchObject({ Authorization: "Bearer sk-test" });
  });

  it("reads a refusal as the server's message and code", async () => {
    saveKey("sk-test");
    const body = { error: { message: "no download is running; GET /v1/models/catalog shows each entry's state", code: "download_not_running" } };
    expect(await cancelDownload(answering(409, body))).toEqual({ ok: false, status: 409, message: body.error.message, code: "download_not_running" });
  });
});

describe("patchConfig", () => {
  it("sends PATCH /v1/config with the body as given", async () => {
    saveKey("sk-test");
    const sent: Sent[] = [];
    const body = { model: { qwen38: { max_context: "524288" } } };
    const answer = await patchConfig(body, answering(202, { applied: "reload", persisted: false }, sent));
    expect(answer).toMatchObject({ ok: true, status: 202, value: { applied: "reload" } });
    expect(sent[0]).toMatchObject({ url: "/v1/config", init: { method: "PATCH" } });
    expect(JSON.parse(sent[0].init.body as string)).toEqual(body);
  });
});

describe("the list routes", () => {
  it("read the `data` of the list envelope", async () => {
    saveKey("sk-test");
    const fields = await readFields(answering(200, { object: "list", data: [{ file_key: "server.bind" }] }));
    expect(fields).toMatchObject({ ok: true, value: [{ file_key: "server.bind" }] });
    const catalog = await readCatalog(answering(200, { object: "list", data: [{ id: "a", loaded: true }] }));
    expect(catalog).toMatchObject({ ok: true, value: [{ id: "a", loaded: true }] });
  });

  it("keep the catalog's disk beside its rows, and none from a server that sends none (GitHub #453)", async () => {
    saveKey("sk-test");
    const disk = { path: "/srv/models", free_bytes: 1024 };
    const read = await readCatalog(answering(200, { object: "list", data: [{ id: "a" }], disk }));
    expect(read.ok && read.value.disk).toEqual(disk);
    const older = await readCatalog(answering(200, { object: "list", data: [{ id: "a" }] }));
    expect(older.ok && older.value.disk).toBeUndefined();
  });

  it("show the key prompt on a 401, and say when ignis cannot be reached", async () => {
    saveKey("sk-test");
    const refused = await readFields(answering(401, { error: { message: "invalid api key", code: "invalid_api_key" } }));
    expect(refused).toMatchObject({ ok: false, status: 401 });
    expect(getAuth().needsKey).toBe(true);
    saveKey("sk-test");
    const down = await readCatalog((async () => {
      throw new TypeError("fetch failed");
    }) as typeof fetch);
    expect(down).toEqual({ ok: false, status: 0, message: "Could not reach ignis: TypeError: fetch failed" });
  });
});

describe("refusalOf", () => {
  it("is the error's message and code, else the raw body, else the status", () => {
    expect(refusalOf(400, JSON.stringify({ error: { message: "m", code: "c" } }))).toEqual({ status: 400, message: "m", code: "c" });
    expect(refusalOf(502, "Bad gateway")).toEqual({ status: 502, message: "Bad gateway" });
    expect(refusalOf(503, "")).toEqual({ status: 503, message: "503" });
  });
});

// GitHub #380: a catalog entry's files removed from the server's models folder.
describe("removeModel", () => {
  it("sends POST /v1/models/remove with the catalog id alone, and the key", async () => {
    saveKey("sk-test");
    const sent: Sent[] = [];
    const removed = { status: "removed", model: "qwen3.8-flash-next", bytes_freed: 72799980571, files_removed: ["qwen3_8_flash_next.ninfer"], files_kept: [] };
    const answer = await removeModel("qwen3.8-flash-next", answering(200, removed, sent));
    expect(answer).toEqual({ ok: true, status: 200, value: removed });
    expect(sent[0].url).toBe("/v1/models/remove");
    expect(sent[0].init.method).toBe("POST");
    expect(JSON.parse(sent[0].init.body as string)).toEqual({ model: "qwen3.8-flash-next" });
    expect(sent[0].init.headers).toMatchObject({ Authorization: "Bearer sk-test", "Content-Type": "application/json" });
  });

  it("reads a refusal as the server's message and code", async () => {
    saveKey("sk-test");
    const body = { error: { message: "qwen3.8-27b is the model loaded (qwen3.8-27b), which serves from its files; load another model before removing it", code: "model_in_use" } };
    expect(await removeModel("qwen3.8-27b", answering(409, body))).toEqual({ ok: false, status: 409, message: body.error.message, code: "model_in_use" });
  });
});
