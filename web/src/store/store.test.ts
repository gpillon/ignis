import { IDBFactory } from "fake-indexeddb";
import { describe, expect, it } from "vitest";
import { type ImageStore, namedBy, sha256, unreferenced } from "./imageStore.ts";
import { openIndexedDbStore } from "./indexeddb.ts";
import { createMemoryStore } from "./memory.ts";

// The contract every ImageStore keeps (store/imageStore.ts), run on each
// adapter: the memory one, and IndexedDB on fake-indexeddb with a fresh
// factory per case. A new adapter — a local folder, the server — joins the
// list below and passes the same cases.

const ADAPTERS: [string, () => ImageStore][] = [
  ["memory", createMemoryStore],
  ["indexeddb", () => openIndexedDbStore(new IDBFactory(), "test")],
];

const png = (...bytes: number[]) => new Blob([new Uint8Array(bytes)], { type: "image/png" });
const bytesOf = async (blob: Blob | null) => (blob ? [...new Uint8Array(await blob.arrayBuffer())] : null);

for (const [name, make] of ADAPTERS) {
  describe(`the ${name} store`, () => {
    it("keeps an image by the SHA-256 of its bytes, and gives the bytes and their type back", async () => {
      const store = make();
      const stored = await store.putImage(png(1, 2, 3), { width: 4, height: 3 });
      expect(stored.id).toBe(await sha256(png(1, 2, 3)));
      expect(stored).toMatchObject({ type: "image/png", bytes: 3, width: 4, height: 3 });
      const back = await store.getImage(stored.id);
      expect(back?.type).toBe("image/png");
      expect(await bytesOf(back)).toEqual([1, 2, 3]);
      expect(await store.imageInfo(stored.id)).toEqual(stored);
    });

    it("keeps the same bytes once, and the first one's time", async () => {
      const store = make();
      const first = await store.putImage(png(9), { width: 1, height: 1 });
      const again = await store.putImage(png(9), { width: 1, height: 1 });
      expect(again).toEqual(first);
      expect(await store.listImages()).toHaveLength(1);
      expect(await store.usage()).toMatchObject({ bytes: 1, images: 1 });
    });

    it("keeps a thumbnail beside the image, given once", async () => {
      const store = make();
      const { id } = await store.putImage(png(1), { width: 1, height: 1 });
      expect(await store.getThumb(id)).toBeNull();
      await store.putImage(png(1), { width: 1, height: 1, thumb: new Blob([new Uint8Array([7])], { type: "image/webp" }) });
      const thumb = await store.getThumb(id);
      expect(thumb?.type).toBe("image/webp");
      expect(await bytesOf(thumb)).toEqual([7]);
    });

    it("forgets an image, its bytes and its thumbnail", async () => {
      const store = make();
      const { id } = await store.putImage(png(5), { width: 1, height: 1, thumb: png(6) });
      await store.deleteImage(id);
      expect(await store.getImage(id)).toBeNull();
      expect(await store.getThumb(id)).toBeNull();
      expect(await store.imageInfo(id)).toBeNull();
      expect(await store.usage()).toMatchObject({ bytes: 0, images: 0 });
    });

    it("keeps documents by collection and id, as copies", async () => {
      const store = make();
      const doc = { title: "Fox", pool: [{ n: 1, id: "abc" }] };
      await store.putDoc("image-sessions", "s1", doc);
      await store.putDoc("image-sessions", "s2", { title: "Logo", pool: [] });
      await store.putDoc("other", "s1", { title: "elsewhere" });
      doc.title = "changed after";
      expect(await store.getDoc("image-sessions", "s1")).toEqual({ title: "Fox", pool: [{ n: 1, id: "abc" }] });
      expect((await store.listDocs<{ title: string }>("image-sessions")).map((d) => d.title).sort()).toEqual(["Fox", "Logo"]);
      await store.putDoc("image-sessions", "s1", { title: "Fox 2", pool: [] });
      expect(await store.getDoc<{ title: string }>("image-sessions", "s1")).toMatchObject({ title: "Fox 2" });
      await store.deleteDoc("image-sessions", "s1");
      expect(await store.getDoc("image-sessions", "s1")).toBeNull();
      expect(await store.listDocs("image-sessions")).toHaveLength(1);
      expect(await store.getDoc("other", "s1")).toEqual({ title: "elsewhere" });
    });

    it("keeps bytes that are not a picture by their SHA-256, once, apart from the images", async () => {
      const store = make();
      const pdf = () => new Blob([new Uint8Array([37, 80, 68, 70])], { type: "application/pdf" });
      const stored = await store.putBlob(pdf());
      expect(stored).toMatchObject({ id: await sha256(pdf()), type: "application/pdf", bytes: 4 });
      expect(await store.putBlob(pdf())).toEqual(stored);
      const back = await store.getBlob(stored.id);
      expect(back?.type).toBe("application/pdf");
      expect(await bytesOf(back)).toEqual([37, 80, 68, 70]);
      expect(await store.getImage(stored.id)).toBeNull();
      expect(await store.usage()).toMatchObject({ bytes: 0, images: 0 });
      await store.deleteBlob(stored.id);
      expect(await store.getBlob(stored.id)).toBeNull();
    });
  });
}

describe("the IndexedDB store's upgrade", () => {
  it("opens a database of the first version, keeping what it holds, and gains the byte store", async () => {
    const factory = new IDBFactory();
    await new Promise<void>((resolve, reject) => {
      const request = factory.open("old", 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        for (const name of ["images", "bytes", "thumbs"]) db.createObjectStore(name, { keyPath: "id" });
        db.createObjectStore("docs", { keyPath: ["collection", "id"] }).createIndex("collection", "collection");
      };
      request.onsuccess = () => {
        const db = request.result;
        const tx = db.transaction("docs", "readwrite");
        tx.objectStore("docs").put({ collection: "chat-sessions", id: "k", value: { title: "Kept" } });
        tx.oncomplete = () => {
          db.close();
          resolve();
        };
        tx.onerror = () => reject(tx.error);
      };
      request.onerror = () => reject(request.error);
    });
    const store = openIndexedDbStore(factory, "old");
    expect(await store.getDoc("chat-sessions", "k")).toEqual({ title: "Kept" });
    const { id } = await store.putBlob(new Blob([new Uint8Array([1])], { type: "audio/wav" }));
    expect(await bytesOf(await store.getBlob(id))).toEqual([1]);
  });

  it("says so when a page of the older version holds the database open, rather than waiting for good", async () => {
    const factory = new IDBFactory();
    // An older page's connection: it never closes for an upgrade.
    await new Promise<void>((resolve, reject) => {
      const request = factory.open("held", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("docs", { keyPath: ["collection", "id"] }).createIndex("collection", "collection");
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
    await expect(openIndexedDbStore(factory, "held").listDocs("chat-sessions")).rejects.toThrow(
      "another tab of the Playground holds this browser's store at an older version: close it, then reload this page",
    );
  });

  it("lets a newer page upgrade the database it holds open", async () => {
    const factory = new IDBFactory();
    const store = openIndexedDbStore(factory, "mine");
    await store.putDoc("chat-sessions", "k", { title: "Kept" });
    const newer = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = factory.open("mine", 99);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error("blocked"));
    });
    expect(newer.version).toBe(99);
    newer.close();
  });
});

describe("what can go", () => {
  it("is every image asked about that nothing keeps", () => {
    expect(unreferenced(["a", "b", "a", "c"], new Set(["b"]))).toEqual(["a", "c"]);
  });

  it("is never what another view names, and nothing while one of them cannot say yet", () => {
    expect(namedBy(["a"], new Set(["b"]), [])).toEqual(["a", "b"]);
    expect(namedBy(["a"], null)).toBeNull();
  });
});
