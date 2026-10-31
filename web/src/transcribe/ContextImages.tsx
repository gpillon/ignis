import { Fragment, type ReactNode, type RefObject, useEffect, useId, useRef, useState } from "react";
import { pictureRefusal } from "../conversation/images.ts";
import { Picture } from "../image/pictures.tsx";
import { RedactEditor } from "../markup/MarkupEditor.tsx";
import { caption } from "../ui/classes.ts";
import { type ContextImage, imageMoment, pastedPicture } from "./contextImages.ts";
import { clock } from "./export.ts";
import type { ReadingStatus } from "./imageReading.ts";
import type { ListeningHandle } from "./listening.ts";
import type { Transcript } from "./transcript.ts";
import type { TranscribeActions } from "./useTranscribe.ts";

// A transcript's context images in the view (spec #428): added in the Call
// tab by a file, by a paste anywhere in the view, or by Capture, a frame of
// what a call shares; each opens in the marking editor's crop and black-out
// first, and only the picture saved there is kept, at the moment the picture
// came in. The spine shows each at its moment, with what the text model with
// vision read in it, or why it is not read yet (#437).

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

/**
 * The Call tab's images: a file, a paste or Capture adds one, how many the spine holds, and how many wait to be read
 * while no text model with vision serves (`canRead` false).
 */
export function ImagesCard({
  transcript: t,
  sharing,
  canRead,
  adding,
}: {
  transcript: Transcript;
  sharing: boolean;
  canRead: boolean;
  adding: Pick<ImageAdding, "open" | "capture" | "capturing" | "error">;
}) {
  const headingId = useId();
  const picker = useRef<HTMLInputElement>(null);
  const count = t.images?.length ?? 0;
  const unread = (t.images ?? []).filter((i) => !i.reading).length;
  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-2.5">
      <h2 id={headingId} className={caption}>
        Images
      </h2>
      <p className="text-xs leading-snug text-ash">
        What the call is looking at, a slide, a shared screen, a whiteboard: a picture of it stands on the transcript from this moment. You crop it and black
        out what must not be kept before it is added. The text model with vision reads each picture once, and what it read reaches the notes, Ask and the
        suggestions.
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
      {!canRead && unread > 0 && (
        <p className="text-xs leading-snug text-ink">
          {unread === 1 ? "1 image waits" : `${unread} images wait`} to be read: no text model with vision serves. They are read once one does.
        </p>
      )}
    </section>
  );
}

/**
 * A context image on the time spine: its moment in the time column, a frame on the rule, the picture beside them, and
 * under it what was read in it, or where its reading stands; the rule burns while it is read, as under an answer.
 */
export function ImageAside({ image, status, onRetry }: { image: ContextImage; status: ReadingStatus; onRetry?: () => void }) {
  const at = clock(image.atMs);
  const reading = status.kind === "reading";
  return (
    <li aria-label={`Context image at ${at}`} aria-busy={reading} className="grid grid-cols-[3.75rem_1fr] gap-x-4 sm:grid-cols-[4.5rem_1fr]">
      <span className="pt-[3px] text-right font-display text-[13px] text-ash tabular-nums">{at}</span>
      <div className="relative border-l-2 border-line pb-5 pl-4">
        <span aria-hidden="true" className="absolute top-[6px] -left-[6px] size-2.5 border-2 border-ink bg-ground" />
        {reading && <span aria-hidden="true" className="spine-live absolute top-5 bottom-5 -left-[3px] w-1" />}
        <figure className="flex flex-col items-start gap-1">
          <Picture id={image.image} alt={`Context image at ${at}`} className="block h-36 w-auto max-w-full border border-line object-contain" />
          <figcaption className="font-display text-xs text-ash tabular-nums">
            Context image, {image.width}×{image.height}
          </figcaption>
        </figure>
        <ImageRead status={status} onRetry={onRetry} />
      </div>
    </li>
  );
}

/** A sentence's end on a message: a full stop when it has none. */
const ended = (text: string) => (/[.!?…]$/.test(text) ? text : `${text}.`);

/** Under a context image: what was read in it, what it shows first; else where its reading stands. */
function ImageRead({ status, onRetry }: { status: ReadingStatus; onRetry?: () => void }) {
  switch (status.kind) {
    case "read": {
      const { shows, topic, names, terms } = status.reading;
      const rows = [
        ["Topic", topic ?? ""],
        ["Names", names.join(", ")],
        ["Terms", terms.join(", ")],
      ].filter(([, value]) => value !== "");
      return (
        <div className="mt-2 flex max-w-[60ch] flex-col gap-1.5">
          {shows !== "" && <p className="text-[13px] leading-snug text-ink">{shows}</p>}
          {rows.length > 0 && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-2.5 gap-y-0.5 text-xs leading-snug">
              {rows.map(([label, value]) => (
                <Fragment key={label}>
                  <dt className="font-display text-ash">{label}</dt>
                  <dd className="text-ink">{value}</dd>
                </Fragment>
              ))}
            </dl>
          )}
        </div>
      );
    }
    case "reading":
      return <p className="mt-2 text-xs text-ash">Reading the picture.</p>;
    case "queued":
      return <p className="mt-2 text-xs text-ash">To be read.</p>;
    case "waiting":
      return <p className="mt-2 max-w-[60ch] text-xs leading-snug text-ash">Not read yet: no text model with vision serves. It is read once one does.</p>;
    case "failed":
      return (
        <p className="mt-2 flex max-w-[60ch] flex-wrap items-baseline gap-x-3 text-xs leading-snug">
          <span className="text-fault">{`The reading failed: ${ended(status.message)}`}</span>
          {onRetry && (
            <button type="button" onClick={onRetry} className="font-display text-xs text-ash hover:text-ember">
              Read again
            </button>
          )}
        </p>
      );
  }
}
