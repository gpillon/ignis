import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { STORED_FLAGS } from "../app/storedFlag.ts";

// The transport choice is this browser's and survives a reload; the mock,
// which serves no socket, talks HTTP whatever was chosen (GitHub #283).
// Each case loads the module afresh, as a reload does.

/** A browser's storage, as much of it as the choice touches. */
function fakeStorage() {
  const items = new Map<string, string>();
  return { items, getItem: (k: string) => items.get(k) ?? null, setItem: (k: string, v: string) => void items.set(k, v) };
}

function useStorage(storage: unknown) {
  Object.defineProperty(globalThis, "localStorage", { value: storage, configurable: true, writable: true });
}

const load = async () => {
  vi.resetModules();
  return import("./transport.ts");
};

describe("the transport choice", () => {
  let storage: ReturnType<typeof fakeStorage>;
  beforeEach(() => useStorage((storage = fakeStorage())));
  afterEach(() => {
    useStorage(undefined);
    vi.unstubAllEnvs();
  });

  it("starts on the WebSocket, and keeps HTTP across a reload once it is picked", async () => {
    const first = await load();
    expect(first.getTransport().choice).toBe("websocket");
    expect(first.activeTransport()).toBe("websocket");
    first.chooseTransport("http");
    expect(storage.items.get(STORED_FLAGS.httpTransport)).toBe("1");

    const reloaded = await load();
    expect(reloaded.getTransport().choice).toBe("http");
    expect(reloaded.activeTransport()).toBe("http");
    reloaded.chooseTransport("websocket");
    expect((await load()).activeTransport()).toBe("websocket");
  });

  it("does not keep a fallback across a reload: the next page tries the socket again", async () => {
    const page = await load();
    page.fallBackToHttp();
    expect(page.activeTransport()).toBe("http");
    expect((await load()).activeTransport()).toBe("websocket");
  });

  it("talks HTTP under the mock whatever was chosen, and says nothing of it", async () => {
    vi.stubEnv("MODE", "mock");
    const mock = await load();
    expect(mock.getTransport()).toMatchObject({ choice: "websocket", fellBack: false, notice: false });
    expect(mock.activeTransport()).toBe("http");
  });
});
