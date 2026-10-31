import { useEffect, useMemo, useRef, useState } from "react";
import { keyRequired } from "../api/auth.ts";
import { STORED_FLAGS, useStoredFlag } from "../app/storedFlag.ts";
import type { SpeechStatus } from "../api/model.ts";
import { speakersUnavailable } from "../speech/availability.ts";
import { captureStream, grabFrame, pickSharedAudio, startCapture } from "../speech/capture.ts";
import { openListening } from "../speech/realtime.ts";
import { mergeLoaded } from "../store/docs.ts";
import { pageStore } from "../store/index.ts";
import type { ImageId } from "../store/imageStore.ts";
import { keepPicture } from "../store/pictures.ts";
import type { RedactedPicture } from "../markup/markup.ts";
import { useStoredSessions } from "../store/useStoredSessions.ts";
import { decodeFile } from "./file.ts";
import { filePace } from "./filePace.ts";
import { chooseUtteranceSilence, utteranceSilence } from "./silence.ts";
import { chooseNoteKind, kindOf, type NoteKind, type NoteLine, noNotes, type Notes } from "./notes.ts";
import { addLine, editLine, removeLine } from "./noteEdits.ts";
import { withRecap } from "./recap.ts";
import { type CallContext, contextAtStart, contextOf, contextTokensOf, copiedDraft, DEFAULT_CONTEXT_TOKENS, mergeDraft, told } from "./context.ts";
import { savedContext, saveContext } from "./defaultContext.ts";
import { readConfig } from "../server/routes.ts";
import { type NotesForm, type NotesHandle, NotesWriter } from "./notesWriter.ts";
import { Analyst, type AnalystHandle } from "./analyst.ts";
import { acceptRewrite, Rewriter, type RewriterHandle } from "./rewrite.ts";
import { Asker, type AskerHandle, withAnswer } from "./ask.ts";
import { AwayWatch, type MissedFrom, pickFrom } from "./missed.ts";
import { accept, analysisOffers, dismiss, dropTerm, followFixes, languageOffer, suggest } from "./suggestions.ts";
import { ListeningController, type ListeningHandle, type ListeningSource } from "./listening.ts";
import { activeOf, freeTranscriptId, openTranscript, removeTranscript, startList, type TranscriptList, updateTranscript } from "./sessions.ts";
import { restoreTranscripts, TRANSCRIBE_SESSIONS, toStored, worthKeeping } from "./stored.ts";
import { addContextImage, freedBy, picturesOf } from "./contextImages.ts";
import { ImageReader, type ImageReaderHandle, landed } from "./imageReading.ts";
import { redrawn } from "../conversation/images.ts";
import {
  addUtterance,
  beginListening,
  clockOf,
  editText,
  heardUntil,
  mergeSpeakers,
  moveUtterance,
  removeUtterance,
  renameSpeaker,
  restoreUtterance,
  splitUtterance,
  type Transcript,
  undoFixes,
  type Utterance,
} from "./transcript.ts";

// The Transcribe view's state (spec speech/07), held by the page and not by
// the view, as the Image view's is: the header shows a listening from every
// view, and a listening goes on while the user looks at another one. The
// Transcripts are kept in the page's store (stored.ts); the listening is one
// ListeningController, whose completed utterances land in the Transcript it
// started on, whichever one is open by then.

export type TranscribeState = {
  list: TranscriptList;
  active: Transcript;
  /**
   * The open Transcript's context as it is being edited, `null` when it is not (Q128): its own, applied, is
   * `active.context`. A Start applies it first.
   */
  draft: CallContext | null;
  /** The Speakers toggle, read at a Start (Q112): this browser's, on at first. */
  speakers: boolean;
  /** The utterance just deleted from the open Transcript, while its Undo stands (Q137). */
  deleted: Utterance | null;
  /** Why the Speakers toggle is struck, when it is: the diarizer the server reports. */
  speakersUnavailable: string | null;
  /** On a call, hold the mic while the far side speaks (Q121): this browser's, on at first. */
  holdMic: boolean;
  /** Keep a listening's notes up as it goes (Q126): this browser's, off at first. */
  notesLive: boolean;
  /** Why the notes and the suggestions cannot ask the text model now (notes.ts), `null` while one serves. */
  textBlocked: string | null;
  /** The text model has vision, as the server reports it: the context images are read (#437). */
  vision: boolean;
  /** What the server reads of a context, `speech.context_max_tokens` (512 until the configuration is read). */
  contextTokens: number;
  /** Suggest while listening (Q132): this browser's, on at first. */
  suggestionsOn: boolean;
  /** Where the open Transcript's What did I miss? reads from (missed.ts), `null` for the last five minutes. */
  missedFrom: MissedFrom | null;
  /** What keeping the Transcripts in this browser ran into. */
  storeError: string | null;
};

export type TranscribeActions = {
  select: (id: number) => void;
  newTranscript: () => void;
  remove: (id: number) => void;
  rename: (id: number, title: string) => void;
  /** Start, or Continue on a Transcript that holds utterances (Q97); inside the press. */
  start: (source: ListeningSource) => void;
  pause: () => void;
  resume: () => void;
  stop: () => void;
  /** Edits the open Transcript's context: a draft, from what it was told. */
  editContext: () => void;
  changeContext: (draft: CallContext) => void;
  /** Applies the draft: the Transcript keeps it, and a listening into it is told from the next utterance on. */
  applyContext: () => void;
  cancelContext: () => void;
  /** Opens a draft of the open Transcript's context from Transcript `from`'s, to check before a Start applies it. */
  copyContext: (from: number) => void;
  /** Keeps `context` as this browser's default, what a Transcript never started tells; a blank one clears it. */
  saveDefaultContext: (context: CallContext) => void;
  setSpeakers: (on: boolean) => void;
  setHoldMic: (on: boolean) => void;
  /** The silence that ends an utterance (Q122): the viewer's, and every session's at once. */
  setSilence: (silenceMs: number) => void;
  renameSpeaker: (speaker: number, name: string) => void;
  moveUtterance: (utterance: number, to: number | "new") => void;
  mergeSpeakers: (from: number, into: number) => void;
  splitUtterance: (utterance: number, at: number, to: number | "new") => void;
  editText: (utterance: number, text: string) => void;
  /** Puts an utterance's text back as it was heard, before its fixes (Q133). */
  undoFixes: (utterance: number) => void;
  /** Deletes an utterance of the open Transcript; Undo puts it back for a few seconds (Q137). */
  deleteUtterance: (utterance: number) => void;
  undoDelete: () => void;
  /** Holds the Undo while it is pointed at or focused, and gives it its time again after. */
  holdUndo: (hold: boolean) => void;
  /** Asks the text model to rebuild a misheard utterance of the open Transcript (Q138); its proposal waits. */
  rewriteUtterance: (utterance: number) => void;
  /** Takes the proposal: the utterance's text, its text as heard kept. */
  acceptRewrite: () => void;
  /** Stops the rewrite, or drops its proposal. */
  closeRewrite: () => void;
  /** Asks the text model about the open Transcript (Q139), sending `sent` in the question's place when given; the answer is kept with it. */
  askAbout: (question: string, sent?: string) => void;
  stopAsking: () => void;
  /** Drops a question and its answer from the open Transcript. */
  forgetAnswer: (id: number) => void;
  /** What did I miss? reads the open Transcript from the utterance at `startMs`, or the last five minutes again when it did. */
  askFrom: (startMs: number) => void;
  /** What did I miss? reads the open Transcript's last five minutes again. */
  clearMissed: () => void;
  /** The open Transcript's kind of notes, and this browser's next default. */
  setNoteKind: (kind: NoteKind) => void;
  /** Updates the open Transcript's running notes now. */
  writeNotes: () => void;
  /** Writes the open Transcript's running notes again from scratch: the model's lines dropped, the user's kept. */
  rewriteNotes: () => void;
  /** Writes the open Transcript's recap now, from the whole of it. */
  writeRecap: () => void;
  /** Stops the writing under way, the notes' or the recap's. */
  stopNotes: () => void;
  setNotesLive: (on: boolean) => void;
  /** A line of the open Transcript's running notes or recap, as the user saw it, edited by hand and now theirs; emptied, removed. */
  editNote: (form: NotesForm, line: NoteLine, text: string) => void;
  /** The user's line added at the end of a section of the open Transcript's running notes or recap. */
  addNote: (form: NotesForm, section: string, text: string) => void;
  /** A line of the open Transcript's running notes or recap removed by hand: it never comes back. */
  removeNote: (form: NotesForm, line: NoteLine) => void;
  /** The open Transcript's suggestion accepted; `always`, and every later one of its kind (Q132). */
  acceptSuggestion: (id: number, always?: boolean) => void;
  dismissSuggestion: (id: number) => void;
  /** Takes a term out of a terms suggestion, never to be offered again. */
  dropSuggestedTerm: (id: number, term: string) => void;
  /** Reads the open Transcript for suggestions now. */
  suggestNow: () => void;
  stopSuggesting: () => void;
  setSuggestionsOn: (on: boolean) => void;
  /**
   * Keeps an edited picture in the page's store and adds it to Transcript `id` at `atMs` as a context image (#431);
   * rejects with why the store refused it.
   */
  addImage: (id: number, picture: RedactedPicture, atMs: number) => Promise<void>;
  /** A frame of what the call shares, for a context image; null while no call shares one (#431). */
  captureFrame: () => Promise<Blob | null>;
  /** Reads again a context image of the open Transcript whose reading failed (#437). */
  readImageAgain: (image: number) => void;
};

/** The pictures the Transcripts name in the page's store, or null until they are read: what the other views' deletes keep. */
export type TranscribePictures = () => Set<ImageId> | null;

/** The page's helpers on the text model: the notes' writer, the look for suggestions, the context images' reader. */
export type Assistants = { notes: NotesHandle; analyst: AnalystHandle; rewriter: RewriterHandle; asker: AskerHandle; reader: ImageReaderHandle };

/** The text model, as the notes need it: why none serves, its context, and whether it has vision (#437). */
export type TextModel = { blocked: string | null; contextLimit: number | null; vision: boolean };

/** How long a deleted utterance's Undo stands. */
export const UNDO_DELETE_MS = 8_000;

/** How often keeping the notes up looks whether it is time. */
const NOTES_LOOK_MS = 5_000;

export function useTranscribe({
  speech,
  text,
  onUnavailable,
  elsewhere,
}: {
  speech?: SpeechStatus;
  text: TextModel;
  onUnavailable: () => void;
  /**
   * The pictures the other views name now (the chat's sessions, the Image view's, the workspaces'), or null while one
   * cannot say yet: deleting a Transcript keeps them, and while null frees no picture.
   */
  elsewhere?: () => Promise<Iterable<ImageId> | null>;
}): [TranscribeState, TranscribeActions, ListeningHandle, Assistants, TranscribePictures] {
  const [list, setList] = useState<TranscriptList>(() => startList());
  /** The draft, the Transcript it is of, and its context when Edit opened it: an Apply merges onto what it holds then. */
  const [drafting, setDrafting] = useState<{ id: number; base: CallContext; context: CallContext } | null>(null);
  const [contextTokens, setContextTokens] = useState(DEFAULT_CONTEXT_TOKENS);
  /** The utterance deleted last, the Transcript it was in, and its Undo's timer. */
  const [deleted, setDeleted] = useState<{ id: number; utterance: Utterance } | null>(null);
  const deletedNow = useRef(deleted);
  deletedNow.current = deleted;
  const undoTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(undoTimer.current), []);
  useEffect(() => {
    void readConfig().then((answer) => answer.ok && setContextTokens(contextTokensOf(answer.value)));
  }, []);
  const draftNow = useRef(drafting);
  draftNow.current = drafting;
  const [speakers, storeSpeakers] = useStoredFlag(STORED_FLAGS.transcribeSpeakers, true);
  const [holdMic, storeHoldMic] = useStoredFlag(STORED_FLAGS.transcribeHoldMic, true);
  const holdMicRef = useRef(holdMic);
  holdMicRef.current = holdMic;
  const listNow = useRef(list);
  listNow.current = list;
  const latest = useRef({ onUnavailable, text, elsewhere });
  latest.current = { onUnavailable, text, elsewhere };
  const [notesLive, storeNotesLive] = useStoredFlag(STORED_FLAGS.transcribeNotesLive, false);
  const [suggestionsOn, storeSuggestionsOn] = useStoredFlag(STORED_FLAGS.transcribeSuggestions, true);

  const [controller] = useState(
    () =>
      new ListeningController({
        open: (options, onEvent) => openListening(options, onEvent),
        captureMic: (onChunk) => startCapture(onChunk),
        pickShared: (onEnded) => pickSharedAudio(onEnded),
        captureShared: (stream, onChunk) => captureStream(stream, onChunk),
        grabFrame: (track) => grabFrame(track),
        decode: (file) => decodeFile(file),
        filePaceSeconds: filePace,
        onUtterance: (id, utterance) => setList((l) => updateTranscript(l, id, (t) => addUtterance(t, utterance))),
        onClock: (id, clockMs) => setList((l) => updateTranscript(l, id, (t) => ({ ...t, clockMs: Math.max(t.clockMs, clockMs) }))),
        onKeyRequired: keyRequired,
        onUnavailable: () => latest.current.onUnavailable(),
      }),
  );
  useEffect(() => () => controller.dispose(), [controller]);

  const [writer] = useState(
    () =>
      new NotesWriter({
        // Each call the notes thread applies lands as it comes, on the notes the Transcript holds then.
        onNotes: (id, change) => setList((l) => updateTranscript(l, id, (t) => ({ ...t, notes: change(t.notes) }))),
        onRecap: (id, written) => setList((l) => updateTranscript(l, id, (t) => ({ ...t, recap: withRecap(t.recap, written) }))),
      }),
  );
  useEffect(() => () => writer.stop(), [writer]);

  const [analyst] = useState(
    () =>
      new Analyst({
        onFound: (id, found, readUntilMs) =>
          setList((l) =>
            updateTranscript(l, id, (t) => ({
              ...analysisOffers(t, found).reduce((x, offer) => suggest(x, offer, Date.now()), t),
              checkedUntilMs: Math.max(t.checkedUntilMs ?? 0, readUntilMs),
            })),
          ),
      }),
  );
  useEffect(() => () => analyst.stop(), [analyst]);
  const [rewriter] = useState(() => new Rewriter());
  useEffect(() => () => rewriter.close(), [rewriter]);
  const [asker] = useState(
    () => new Asker({ onAnswered: (id, question, answer, untilMs) => setList((l) => updateTranscript(l, id, (t) => ({ ...t, asked: withAnswer(t.asked, question, answer, Date.now(), untilMs) }))) }),
  );
  useEffect(() => () => asker.stop(), [asker]);
  const [reader] = useState(
    () =>
      new ImageReader({
        // The picture as the chat's go to the model: redrawn, its long edge bounded, as a JPEG.
        picture: async (id) => {
          const blob = await pageStore().getImage(id);
          if (!blob) throw new Error("it is no longer in this browser");
          return redrawn(blob, "context-image.png");
        },
        // Its reading on the image, and its terms and topic offered, or applied when accepted always.
        onRead: (id, image, reading) => setList((l) => updateTranscript(l, id, (t) => landed(t, image, reading, Date.now()))),
      }),
  );
  useEffect(() => () => reader.stop(), [reader]);
  // The context images, read one at a time as they come while a text model with vision serves; until one does, they
  // wait (#437). Each is read once, whatever Transcript it is in.
  const canRead = text.blocked === null && text.vision;
  const readAll = useMemo(
    () => () => void reader.keepUp(() => listNow.current.transcripts, () => latest.current.text.blocked === null && latest.current.text.vision),
    [reader],
  );
  useEffect(() => {
    if (canRead) readAll();
  }, [list, canRead, readAll]);

  // Where What did I miss? reads from (missed.ts): from what the user had seen when they left the page, hidden or its
  // window left, for a minute or more while listening; or from a moment they picked.
  const [missed, setMissed] = useState<{ transcript: number; from: MissedFrom } | null>(null);
  useEffect(() => {
    const watch = new AwayWatch();
    const leave = () => {
      const listening = controller.getState();
      const t = listening.phase === "listening" ? listNow.current.transcripts.find((x) => x.id === listening.transcript) : undefined;
      if (t) watch.leave(Date.now(), t.id, heardUntil(t.utterances));
    };
    const back = () => {
      if (document.visibilityState === "hidden") return;
      const away = watch.back(Date.now());
      if (away) setMissed(away);
    };
    const onVisibility = () => (document.visibilityState === "hidden" ? leave() : back());
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("blur", leave);
    window.addEventListener("focus", back);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("blur", leave);
      window.removeEventListener("focus", back);
    };
  }, [controller]);

  // Keeping up while listening, a look every few seconds: the languages the server heard, then, while a text model
  // serves, the notes and the look for suggestions, each on its own cadence.
  useEffect(() => {
    if (!notesLive && !suggestionsOn) return;
    const look = () => {
      const { text: model } = latest.current;
      const listening = controller.getState();
      const on = listening.phase === "idle" ? null : listening.transcript;
      const find = (id: number) => listNow.current.transcripts.find((t) => t.id === id);
      const t = on === null ? undefined : find(on);
      const offer = suggestionsOn && t ? languageOffer(t) : null;
      if (t && offer && suggest(t, offer, Date.now()) !== t) setList((l) => updateTranscript(l, t.id, (x) => suggest(x, offer, Date.now())));
      // A fix accepted reaches the utterances heard since that hold the same misheard words.
      if (suggestionsOn && t && followFixes(t, Date.now()) !== t) setList((l) => updateTranscript(l, t.id, (x) => followFixes(x, Date.now())));
      if (model.blocked !== null) return;
      if (notesLive) void writer.look(on, find, kindOf, model.contextLimit);
      if (suggestionsOn) void analyst.look(on, find, model.contextLimit);
    };
    const timer = setInterval(look, NOTES_LOOK_MS);
    return () => clearInterval(timer);
  }, [notesLive, suggestionsOn, controller, writer, analyst]);

  const kept = useStoredSessions({
    store: pageStore(),
    collection: TRANSCRIBE_SESSIONS,
    view: "transcribe",
    items: list.transcripts,
    activeId: list.activeId,
    toStored,
    worthKeeping,
    adopt: async (values, activeKey) => {
      const restored = restoreTranscripts(values, freeTranscriptId(listNow.current));
      const reopened = [...restored.stamps].find(([, stamp]) => stamp.key === activeKey)?.[0];
      setList((l) => {
        const merged = mergeLoaded(l.transcripts, l.activeId, restored.transcripts, reopened, worthKeeping);
        return { transcripts: merged.items, activeId: merged.activeId };
      });
      return { loaded: restored.transcripts, stamps: restored.stamps };
    },
    // A Transcript's context images go with it, unless another Transcript or another view names the same picture.
    onRemoved: (gone, left) => {
      const others = latest.current.elsewhere?.() ?? Promise.resolve([]);
      void others.then(
        (named) => {
          for (const id of freedBy(gone, left, named)) void pageStore().deleteImage(id).catch(() => {});
        },
        () => {},
      );
    },
  });

  const struck = speech ? speakersUnavailable(speech) : null;
  // A listening is told what its Transcript holds, applied or renamed since: each session sends a change alone.
  useEffect(() => {
    const listening = controller.getState();
    if (listening.phase === "idle") return;
    const t = list.transcripts.find((x) => x.id === listening.transcript);
    if (t) controller.setContext(told(t.context, t.speakers));
  }, [list, controller]);

  // Read at the press: the toggle as struck by the server's word.
  const speakersRef = useRef(speakers && struck === null);
  speakersRef.current = speakers && struck === null;
  const active = activeOf(list);
  const change = (id: number, apply: (t: Transcript) => Transcript) => setList((l) => updateTranscript(l, id, apply));
  /** A correction is the open Transcript's. */
  const correct = (apply: (t: Transcript) => Transcript) => setList((l) => updateTranscript(l, l.activeId, apply));
  /**
   * The open Transcript's notes edited by hand (noteEdits.ts): the running notes through the writer, so an update under
   * way holds the edit before the model's next call; the recap, which no thread holds, in place.
   */
  const byHand = (form: NotesForm, edit: (notes: Notes) => Notes) => {
    if (form === "running") writer.editNotes(listNow.current.activeId, edit);
    else correct((t) => (t.recap ? { ...t, recap: edit(t.recap) } : t));
  };

  const actions = useMemo<TranscribeActions>(
    () => ({
      select: (id) => setList((l) => ({ ...l, activeId: id })),
      newTranscript: () => setList((l) => openTranscript(l, freeTranscriptId(l))),
      // The Transcript a listening appends to stays until the listening ends.
      remove: (id) => controller.getState().transcript !== id && setList((l) => removeTranscript(l, id, freeTranscriptId(l))),
      rename: (id, title) => {
        const trimmed = title.trim();
        if (trimmed !== "") change(id, (t) => ({ ...t, title: trimmed }));
      },
      start: (source) => {
        if (controller.getState().phase !== "idle") return;
        const pending = draftNow.current;
        const opened = activeOf(listNow.current);
        // What the Call tab shows is what the listening is told: a context being edited, else the default a
        // Transcript never started shows, applied by the Start.
        const mine = pending?.id === opened.id ? pending : null;
        const t = { ...opened, context: contextAtStart(opened, mine, savedContext(), Date.now()) };
        if (mine) setDrafting(null);
        const start = { source: source.kind, at: new Date(), ...(source.kind === "file" ? { fileName: source.file.name } : {}) };
        setList((l) => updateTranscript(l, t.id, (current) => beginListening({ ...current, context: t.context }, start)));
        controller.start({
          transcript: t.id,
          clockMs: clockOf(t),
          source,
          speakers: speakersRef.current,
          context: told(t.context, t.speakers),
          silenceMs: utteranceSilence(),
          holdMic: holdMicRef.current,
        });
      },
      pause: () => controller.pause(),
      resume: () => controller.resume(),
      stop: () => void controller.stop(),
      editContext: () => {
        const t = activeOf(listNow.current);
        const base = contextOf(t, savedContext());
        setDrafting({ id: t.id, base, context: base });
      },
      copyContext: (from) => {
        const l = listNow.current;
        const t = activeOf(l);
        const theirs = l.transcripts.find((x) => x.id === from)?.context;
        if (theirs && from !== t.id) setDrafting({ id: t.id, ...copiedDraft(t, theirs, savedContext()) });
      },
      saveDefaultContext: (context) => saveContext(context),
      changeContext: (draft) => setDrafting((d) => (d && d.id === listNow.current.activeId ? { ...d, context: draft } : d)),
      applyContext: () => {
        const pending = draftNow.current;
        if (!pending) return;
        setDrafting(null);
        change(pending.id, (t) => ({ ...t, context: mergeDraft(pending.base, pending.context, contextOf(t, savedContext()), Date.now()) }));
      },
      cancelContext: () => setDrafting(null),
      setSpeakers: (on) => storeSpeakers(on),
      setHoldMic: (on) => {
        holdMicRef.current = on;
        storeHoldMic(on);
        controller.setHoldMic(on);
      },
      setSilence: (silenceMs) => {
        chooseUtteranceSilence(silenceMs);
        controller.setSilence(silenceMs);
      },
      renameSpeaker: (speaker, name) => correct((t) => renameSpeaker(t, speaker, name)),
      moveUtterance: (utterance, to) => correct((t) => moveUtterance(t, utterance, to)),
      mergeSpeakers: (from, into) => correct((t) => mergeSpeakers(t, from, into)),
      splitUtterance: (utterance, at, to) => correct((t) => splitUtterance(t, utterance, at, to)),
      editText: (utterance, text) => correct((t) => editText(t, utterance, text)),
      undoFixes: (utterance) => correct((t) => undoFixes(t, utterance)),
      deleteUtterance: (utterance) => {
        const t = activeOf(listNow.current);
        const gone = t.utterances.find((u) => u.id === utterance);
        if (!gone) return;
        change(t.id, (x) => removeUtterance(x, utterance));
        setDeleted({ id: t.id, utterance: gone });
        clearTimeout(undoTimer.current);
        undoTimer.current = setTimeout(() => setDeleted(null), UNDO_DELETE_MS);
      },
      rewriteUtterance: (utterance) => {
        if (latest.current.text.blocked !== null) return;
        const t = activeOf(listNow.current);
        const u = t.utterances.find((x) => x.id === utterance);
        if (u) void rewriter.rewrite(t, u);
      },
      acceptRewrite: () => {
        const { transcript, utterance, proposal } = rewriter.getState();
        if (transcript === null || utterance === null || proposal === null) return;
        change(transcript, (t) => acceptRewrite(t, utterance, proposal));
        rewriter.close();
      },
      closeRewrite: () => rewriter.close(),
      askAbout: (question, sent) => {
        const { text: model } = latest.current;
        if (model.blocked === null) void asker.ask(activeOf(listNow.current), question, model.contextLimit, sent);
      },
      stopAsking: () => asker.stop(),
      forgetAnswer: (id) => correct((t) => ({ ...t, asked: (t.asked ?? []).filter((a) => a.id !== id) })),
      askFrom: (startMs) =>
        setMissed((m) => {
          const id = listNow.current.activeId;
          const from = pickFrom(m?.transcript === id ? m.from : null, startMs);
          return from ? { transcript: id, from } : null;
        }),
      clearMissed: () => setMissed((m) => (m?.transcript === listNow.current.activeId ? null : m)),
      holdUndo: (hold) => {
        clearTimeout(undoTimer.current);
        if (!hold && deletedNow.current) undoTimer.current = setTimeout(() => setDeleted(null), UNDO_DELETE_MS);
      },
      undoDelete: () => {
        const last = deletedNow.current;
        if (!last) return;
        clearTimeout(undoTimer.current);
        setDeleted(null);
        change(last.id, (t) => restoreUtterance(t, last.utterance));
      },
      setNoteKind: (kind) => {
        chooseNoteKind(kind);
        correct((t) => ({ ...t, notes: { ...(t.notes ?? noNotes(kind)), kind } }));
      },
      writeNotes: () => {
        const { text: model } = latest.current;
        if (model.blocked !== null) return;
        const t = activeOf(listNow.current);
        void writer.write(t, kindOf(t), model.contextLimit);
      },
      rewriteNotes: () => {
        const { text: model } = latest.current;
        if (model.blocked !== null) return;
        const t = activeOf(listNow.current);
        void writer.rewrite(t, kindOf(t), model.contextLimit);
      },
      writeRecap: () => {
        const { text: model } = latest.current;
        if (model.blocked !== null) return;
        const t = activeOf(listNow.current);
        void writer.writeRecap(t, kindOf(t), model.contextLimit);
      },
      stopNotes: () => writer.stop(),
      setNotesLive: (on) => storeNotesLive(on),
      editNote: (form, line, text) => byHand(form, (n) => editLine(n, line, text)),
      addNote: (form, section, text) => byHand(form, (n) => addLine(n, section, text)),
      removeNote: (form, line) => byHand(form, (n) => removeLine(n, line)),
      acceptSuggestion: (id, always = false) => correct((t) => accept(t, id, Date.now(), { always })),
      dismissSuggestion: (id) => correct((t) => dismiss(t, id)),
      dropSuggestedTerm: (id, term) => correct((t) => dropTerm(t, id, term)),
      suggestNow: () => {
        const { text: model } = latest.current;
        if (model.blocked === null) void analyst.analyze(activeOf(listNow.current), model.contextLimit);
      },
      stopSuggesting: () => analyst.stop(),
      setSuggestionsOn: (on) => storeSuggestionsOn(on),
      addImage: async (id, picture, atMs) => {
        const stored = await keepPicture(pageStore(), picture.blob);
        change(id, (t) => addContextImage(t, { image: stored.id, atMs, width: picture.width, height: picture.height }, Date.now()));
      },
      captureFrame: () => controller.captureFrame(),
      readImageAgain: (image) => {
        reader.retry(listNow.current.activeId, image);
        readAll();
      },
    }),
    // The stored flags' setters are stable for their keys.
    [controller, writer, analyst, rewriter, asker, reader, readAll, storeSpeakers, storeHoldMic, storeNotesLive, storeSuggestionsOn],
  );

  return [
    {
      list,
      active,
      draft: drafting?.id === active.id ? drafting.context : null,
      deleted: deleted?.id === active.id ? deleted.utterance : null,
      speakers: speakers && struck === null,
      speakersUnavailable: struck,
      holdMic,
      notesLive,
      textBlocked: text.blocked,
      vision: text.vision,
      contextTokens,
      suggestionsOn,
      missedFrom: missed?.transcript === active.id ? missed.from : null,
      storeError: kept.error,
    },
    actions,
    controller,
    { notes: writer, analyst, rewriter, asker, reader },
    () => (kept.loaded ? new Set(listNow.current.transcripts.flatMap((t) => [...picturesOf(t)])) : null),
  ];
}
