import { describe, expect, it } from "vitest";
import { authHeaders, checkKey, forgetKey, getAuth, keyRequired, saveKey } from "./auth.ts";

const answering = (status: number, body = "") => (async () => new Response(body, { status })) as typeof fetch;

describe("the API key", () => {
  it("is sent as a bearer header once saved", () => {
    saveKey("sk-one");
    expect(authHeaders()).toEqual({ Authorization: "Bearer sk-one" });
    expect(getAuth()).toEqual({ key: "sk-one", needsKey: false, rejected: false });
  });

  it("is dropped as refused when ignis answers 401 to it", () => {
    saveKey("sk-wrong");
    keyRequired();
    expect(getAuth()).toEqual({ key: null, needsKey: true, rejected: true });
    expect(authHeaders()).toEqual({});
  });

  it("asks without calling it refused when no key was sent", () => {
    forgetKey();
    expect(getAuth()).toEqual({ key: null, needsKey: true, rejected: false });
    saveKey("sk-two");
    forgetKey();
    keyRequired();
    expect(getAuth().rejected).toBe(false);
  });

  it("is checked against ignis before it is kept", async () => {
    expect(await checkKey("sk", answering(200))).toEqual({ ok: true });
    expect(await checkKey("sk", answering(401))).toEqual({ ok: false, message: "ignis refused this key." });
    const down = await checkKey("sk", answering(503, '{"error":{"message":"engine full"}}'));
    expect(down).toEqual({ ok: false, message: "503: engine full" });
  });
});
