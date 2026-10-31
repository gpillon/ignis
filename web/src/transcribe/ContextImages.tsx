import { type ReactNode, type RefObject, useEffect, useId, useRef, useState } from "react";
import { pictureRefusal } from "../conversation/images.ts";
import { Picture } from "../image/pictures.tsx";
import { RedactEditor } from "../markup/MarkupEditor.tsx";
import { caption } from "../ui/classes.ts";
import { type ContextImage, imageMoment, pastedPicture } from "./contextImages.ts";
import { clock } from "./export.ts";
import type { ListeningHandle } from "./listening.ts";
import type { Transcript } from "./transcript.ts";
import type { TranscribeActions } from "./useTranscribe.ts";

// A transcript's context images in the view (spec #428): added in the Call
// tab by a file, by a paste anywhere in the view, or by Capture, a frame of
// what a call shares; each opens in the marking editor's crop and black-out
// first, and only the picture saved there is kept, at the moment the picture
// came in. The spine shows each at its moment.

/** A picture on its way to the editor: the Transcript it was added to, and its moment there, taken when it came in. */
type Editing = { url: string; transcript: number; atMs: number };

export type ImageAdding = {
  /** A picked, pasted or captured picture, to the editor; `named` says it in a refusal. */
  open: (picture: Blob, named: string, atMs?: number) => void;
  /** Takes a frame of what the call shares, to the editor. */
  capture: () => void;
  capturing: boolean;
  error: string | null;
  /** The editor, while a picture is in it. */
  editor: ReactNode;
};

/**
 * Adding a context image to the open Transcript. A paste anywhere in the view while it shows (`view` is rendered)
 * adds the picture it carries, unless a field takes the paste's text; the original stays in this page's memory
 * until the editor closes, and only what the editor saves reaches the store.
 */
export function useImageAdding({
  transcript,
  listening,
  actions,
  view,
}: {
  transcript: Transcript;
  listening: ListeningHandle;
  actions: Pick<TranscribeActions, "addImage" | "captureFrame">;
  view: RefObject<HTMLElement | null>;
}): ImageAdding {
  const [editing, setEditing] = useState<Editing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);
  const now = useRef({ transcript, editing });
  now.current = { transcript, editing };

  function open(picture: Blob, named: string, atMs?: number) {
    if (now.current.editing) return;
    const refused = pictureRefusal(picture, named);
    setError(refused);
    if (refused !== null) return;
    const t = now.current.transcript;
    setEditing({ url: URL.createObjectURL(picture), transcript: t.id, atMs: atMs ?? imageMoment(t, listening.getState()) });
  }
  const openNow = useRef(open);
  openNow.current = open;

  function close() {
    const was = now.current.editing;
    if (was) URL.revokeObjectURL(was.url);
    setEditing(null);
  }

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      if (!view.current || view.current.offsetParent === null || !e.clipboardData) return;
      const target = e.target as HTMLElement | null;
      const intoField = !!target && (target.isContentEditable || target.tagName === "INPUT" || target.tagName === "TEXTAREA");
      const picture = pastedPicture([...e.clipboardData.files], { intoField, hasText: e.clipboardData.types.includes("text/plain") });
      if (!picture) return;
      e.preventDefault();
      openNow.current(picture, "The pasted picture");
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [view]);

  // A picture still in the editor when the view goes is let go with it.
  useEffect(() => () => void (now.current.editing && URL.revokeObjectURL(now.current.editing.url)), []);

  function capture() {
    if (capturing) return;
    // The moment of the click: the frame is what the call showed then.
    const atMs = imageMoment(now.current.transcript, listening.getState());
    setCapturing(true);
    setError(null);
    actions.captureFrame().then(
      (frame) => {
        setCapturing(false);
        if (frame) open(frame, "The captured frame", atMs);
        else setError("Nothing to capture: the call shares no tab, window or screen now.");
      },
      (err: unknown) => {
        setCapturing(false);
        setError(`The capture failed: ${err instanceof Error ? err.message : String(err)}.`);
      },
    );
  }

  const editor = editing && (
    <RedactEditor
      url={editing.url}
      title="Add a context image"
      saveLabel="Add image"
      onSave={async (edited) => {
        await actions.addImage(editing.transcript, edited, editing.atMs);
        close();
      }}
      onClose={close}
    />
  );
  return { open, capture, capturing, error, editor };
}

/** The Call tab's images: a file, a paste or Capture adds one, and how many the spine holds. */
export function ImagesCard({ transcript: t, sharing, adding }: { transcript: Transcript; sharing: boolean; adding: Pick<ImageAdding, "open" | "capture" | "capturing" | "error"> }) {
  const headingId = useId();
  const picker = useRef<HTMLInputElement>(null);
  const count = t.images?.length ?? 0;
  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-2.5">
      <h2 id={headingId} className={caption}>
        Images
      </h2>
      <p className="text-xs leading-snug text-ash">
        What the call is looking at, a slide, a shared screen, a whiteboard: a picture of it stands on the transcript from this moment. You crop it and black
        out what must not be kept before it is added.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {sharing && (
          <button
            type="button"
            disabled={adding.capturing}
            onClick={adding.capture}
            title="A frame of the tab, window or screen the call shares, taken now"
            className="cut bg-ink px-3 py-1.5 font-display text-[13px] font-semibold text-ground [--cut-size:7px] hover:bg-ember disabled:cursor-wait disabled:opacity-60"
          >
            {adding.capturing ? "Capturing…" : "Capture"}
          </button>
        )}
        <button
          type="button"
          onClick={() => picker.current?.click()}
          className="border border-line px-2.5 py-1.5 font-display text-[13px] font-medium text-ink hover:border-ember"
        >
          From a file
        </button>
        <input
          ref={picker}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            const file = e.currentTarget.files?.[0];
            e.currentTarget.value = "";
            if (file) adding.open(file, file.name || "The picked file");
          }}
        />
        <span className="text-xs text-ash">or paste one</span>
      </div>
      {adding.error && <p className="text-xs leading-snug text-fault">{adding.error}</p>}
      <p className="text-xs leading-snug text-ash">
        {count > 0 ? `${count === 1 ? "1 image" : `${count} images`} on the spine. ` : ""}
        {!sharing && "On a call, Capture takes a frame of what it shares."}
      </p>
    </section>
  );
}

/** A context image on the time spine: its moment in the time column, a frame on the rule, the picture beside them. */
export function ImageAside({ image }: { image: ContextImage }) {
  const at = clock(image.atMs);
  return (
    <li aria-label={`Context image at ${at}`} className="grid grid-cols-[3.75rem_1fr] gap-x-4 sm:grid-cols-[4.5rem_1fr]">
      <span className="pt-[3px] text-right font-display text-[13px] text-ash tabular-nums">{at}</span>
      <div className="relative border-l-2 border-line pb-5 pl-4">
        <span aria-hidden="true" className="absolute top-[6px] -left-[6px] size-2.5 border-2 border-ink bg-ground" />
        <figure className="flex flex-col items-start gap-1">
          <Picture id={image.image} alt={`Context image at ${at}`} className="block h-36 w-auto max-w-full border border-line object-contain" />
          <figcaption className="font-display text-xs text-ash tabular-nums">
            Context image, {image.width}×{image.height}
          </figcaption>
        </figure>
      </div>
    </li>
  );
}
