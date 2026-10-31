// Where the Playground keeps pictures, and the documents that name them (the
// sessions of the Image view, the chat and the Decide tab). Three kinds of
// thing, kept apart:
//   - images: bytes named by their SHA-256, so the same picture is stored
//     once however many sessions use it, with a small thumbnail beside;
//   - blobs: any other bytes (a workspace's PDF or audio file), named the
//     same way, with no width, height or thumbnail;
//   - documents: JSON values by collection and id, naming images and blobs by id.
// The page talks to an `ImageStore` and never to what is under it. Today that
// is IndexedDB in the browser (indexeddb.ts) and memory in tests (memory.ts);
// a local folder (the File System Access API) or the server are adapters to
// come, held to the same contract tests (store.test.ts).
//
// IndexedDB belongs to an origin: the Vite dev server (localhost:5173) and the
// Playground the server embeds (127.0.0.1:8000/ui) keep two separate stores.

export type ImageId = string;

/** What the store knows of an image without reading its bytes. */
export type StoredImage = { id: ImageId; type: string; bytes: number; width: number; height: number; added: number };

/** Bytes that are not a picture, by the SHA-256 of their bytes like an image. */
export type BlobId = string;

/** What the store knows of a blob without reading its bytes. */
export type StoredBlob = { id: BlobId; type: string; bytes: number; added: number };

export type StoreUsage = {
  /** The images' bytes, thumbnails left out. */
  bytes: number;
  images: number;
  /** What the browser grants the origin, when it says. */
  quota?: number;
};

export interface ImageStore {
  readonly kind: "memory" | "indexeddb";
  /** Keeps a picture; the same bytes again are the same image, and the first keeps its `added`. */
  putImage(blob: Blob, info: { width: number; height: number; thumb?: Blob }): Promise<StoredImage>;
  getImage(id: ImageId): Promise<Blob | null>;
  /** The thumbnail kept with the image, if one was. */
  getThumb(id: ImageId): Promise<Blob | null>;
  imageInfo(id: ImageId): Promise<StoredImage | null>;
  listImages(): Promise<StoredImage[]>;
  deleteImage(id: ImageId): Promise<void>;
  /** Keeps bytes that are not a picture; the same bytes again are the same blob, and the first keeps its `added`. */
  putBlob(blob: Blob): Promise<StoredBlob>;
  getBlob(id: BlobId): Promise<Blob | null>;
  deleteBlob(id: BlobId): Promise<void>;
  putDoc<T>(collection: string, id: string, value: T): Promise<void>;
  getDoc<T>(collection: string, id: string): Promise<T | null>;
  listDocs<T>(collection: string): Promise<T[]>;
  deleteDoc(collection: string, id: string): Promise<void>;
  usage(): Promise<StoreUsage>;
}

/** An image's or a blob's id: the SHA-256 of its bytes, in hex. */
export async function sha256(blob: Blob): Promise<ImageId> {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** The images no document names any more, of those the caller asks about. */
export function unreferenced(candidates: Iterable<ImageId>, keep: Set<ImageId>): ImageId[] {
  return [...new Set(candidates)].filter((id) => !keep.has(id));
}

/**
 * The images several views name together, for one view's deletes to keep (the store holds every view's pictures
 * once); null while any of them cannot say yet, which deletes nothing.
 */
export function namedBy(...views: (Iterable<ImageId> | null)[]): ImageId[] | null {
  if (views.some((v) => v === null)) return null;
  return views.flatMap((v) => [...(v as Iterable<ImageId>)]);
}
