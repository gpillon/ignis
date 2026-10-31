import { useSyncExternalStore } from "react";
import { MIC_BLOCKED, unavailable } from "../speech/availability.ts";
import { CallNotShared, type Capture, MicBlocked, NoSharedAudio } from "../speech/capture.ts";
import { setListening } from "../speech/listening.ts";
import { CHUNK_SAMPLES, levelOf, TARGET_RATE } from "../speech/pcm.ts";
import { BUSY_RETRY_MAX_SECONDS, BUSY_RETRY_SECONDS, type Tone } from "../speech/dictation.ts";
import {
  CONFIGURATION_EVENT,
  DIARIZER_FAILED,
  DIARIZER_UNAVAILABLE,
  type Diarization,
  type Listening,
  type ListeningEvent,
  type ListeningOptions,
  NOTHING_TOLD,
  QUEUE_FULL,
  type Refusal,
  type Told,
} from "../speech/realtime.ts";
import { allowance, backlogMs, estimate } from "./pacing.ts";
import type { Heard, SourceKind, Voice } from "./transcript.ts";

// One listening in the Transcribe view (spec speech/07): the state the view
// reads, and what drives it. It knows the server only through the Realtime
// sessions it opens (`openListening`), the audio only through the captures and
// the decoder it is handed, so it runs in a test with fakes for all of them
// (listening.test.ts). What it hears goes to a Transcript by its id: the
// view owns the Transcripts, their corrections and their saving.
//
// - **Lanes.** A listening has one lane per audio source, each with its own
//   Realtime session: the microphone, a call's shared audio beside it (two
//   sessions, never mixed: Me and the others, Q93), or a file. A lane's
//   session numbers its speakers on its own, so each session is a stream of
//   its own for the Transcript's speakers. A call's pick shares a picture
//   beside its audio, which the listening never reads: Capture takes a frame
//   of it, on a click (#431).
// - **Times** are the Transcript's clock: a lane's session counts its audio
//   from its first sample, and that sample's place on the clock is the lane's
//   origin. The clock starts where the Transcript's stopped (a Continue) and
//   stands still while paused.
// - **An end** (Stop, a file's end) flushes each session first: the VAD's
//   events for the last audio sent may still be on their way, so the
//   utterance under way is known, and committed, only once the server has read
//   all of it. The session closes once every utterance's text has come.

// What the hint line says (spec speech/07; spec speech/04's states).
export const CONNECTION_LOST = "Listening stopped: the connection to the server closed. Every completed utterance is kept.";
export const SHARING_ENDED = "The call's sharing ended, and the listening with it.";
export const CALL_NOT_SHARED = "Nothing was shared: start again and pick the call's tab, its window or the screen.";
export const NO_SHARED_AUDIO = "The pick shares no audio: start again and tick the browser's Share audio (for an app on Windows, share the whole screen with its system audio).";
export const cannotStart = (why: string) => `Listening could not start: ${why.replace(/\.$/, "")}.`;
export const NO_WEBSOCKET = "Listening could not start: the server is up, but this address does not carry WebSockets.";
export const busyRetry = (seconds: number) => `Speech is busy: every session is taken. Trying again in ${seconds} s.`;
export const STILL_BUSY = "The listening still could not open: speech is busy. Press Start to try again.";
export const speakersOff = (why: string) => `Speakers are off for this listening: ${why.replace(/\.$/, "")}.`;
export const itemFailed = (why: string) => `An utterance could not be transcribed: ${why.replace(/\.$/, "")}.`;
export const fileDone = (name: string) => `Transcribed ${name}.`;

/** A file's appends are named by their first sample, so a refused one says where to send again from. */
const APPEND_PREFIX = "audio_";

/** How often a file's sending looks, and the most audio one look sends. */
const FEED_EVERY_MS = 100;
const BURST_MS = 5_000;

/** How long a refused append waits before the file is sent again from it. */
const QUEUE_FULL_BACKOFF_MS = 1_000;

/** Bars in a lane's level meter. */
export const METER_BARS = 20;

/**
 * A call holds the mic while the far side speaks (Q121): a shared chunk at this level or over (`levelOf`'s scale,
 * about -34 dBFS) holds it, and it stays held this long after the last one, since the far voice comes back into the
 * mic late, through the speakers and the room.
 */
export const HOLD_LEVEL = 0.1;
export const HOLD_AFTER_MS = 500;

const msOf = (samples: number) => (samples / TARGET_RATE) * 1_000;

export type ListeningSource = { kind: "microphone" } | { kind: "call" } | { kind: "file"; file: File };

/** An utterance under way: its text so far, the settled part and the provisional tail (Q82). */
export type LiveUtterance = { key: string; startMs: number; stable: string; tail: string; lane: LaneName };

export type LaneName = "mic" | "shared" | "file";

export type ListeningState = {
  phase: "idle" | "listening" | "paused" | "finishing" | "waiting";
  /** The Transcript it appends to. */
  transcript: number | null;
  source: SourceKind | null;
  /** The speakers asked for: off from the start, or turned off by the server's refusal. */
  speakers: boolean;
  /** The clock: where the next utterance would start. */
  clockMs: number;
  live: LiveUtterance[];
  /** Each live lane's level of its last METER_BARS chunks, 0 to 1, the newest last. */
  levels: Partial<Record<LaneName, number[]>>;
  /** The hint line's sentence: why it stopped, what the server refused. */
  notice: { text: string; tone: Tone } | null;
  /** Seconds left before the retry of a busy refusal. */
  retryIn: number;
  /** On a call, the mic is held: its session is sent silence while the far side speaks (Q121). */
  held: boolean;
  /** A file's progress: the share transcribed and the time left, none before there is a pace to read. */
  file: { name: string; fraction: number; leftMs: number | null } | null;
  /** On a call, the pick shares a picture, a tab, a window or the screen, that Capture can take a frame of (#431). */
  sharing: boolean;
};

export const IDLE: ListeningState = {
  phase: "idle",
  transcript: null,
  source: null,
  speakers: true,
  clockMs: 0,
  live: [],
  levels: {},
  notice: null,
  retryIn: 0,
  held: false,
  file: null,
  sharing: false,
};

export type ListeningDeps = {
  open: (options: ListeningOptions, onEvent: (event: ListeningEvent) => void) => Listening;
  captureMic: (onChunk: (pcm: Int16Array) => void) => Promise<Capture>;
  /** A call's one pick; `onEnded` is the browser's own Stop sharing. */
  pickShared: (onEnded: () => void) => Promise<MediaStream>;
  captureShared: (stream: MediaStream, onChunk: (pcm: Int16Array) => void) => Promise<Capture>;
  /** A frame of a call's shared picture, as a PNG: what Capture adds as a context image (#431). */
  grabFrame?: (track: MediaStreamTrack) => Promise<Blob>;
  decode: (file: File) => Promise<Int16Array>;
  /** N: the most audio not yet transcribed a file is sent ahead (Q99), read at the start. */
  filePaceSeconds: () => number;
  now?: () => number;
  /** A completed utterance, for the Transcript it belongs to. */
  onUtterance: (transcript: number, utterance: Heard) => void;
  /** The Transcript's clock as the listening leaves it: where a Continue starts. */
  onClock: (transcript: number, clockMs: number) => void;
  onKeyRequired?: () => void;
  /** The server said the speech model cannot serve: the page reads GET /v1/models again. */
  onUnavailable?: () => void;
};

export type StartOptions = {
  transcript: number;
  /** The Transcript's clock so far. */
  clockMs: number;
  source: ListeningSource;
  speakers: boolean;
  /** What the transcription is told (context.ts). */
  context: Told;
  /** The silence that ends an utterance, in ms (Q122); absent or `null`, the server's. */
  silenceMs?: number | null;
  /** On a call, hold the mic while the far side speaks (Q121). */
  holdMic?: boolean;
};

/** How a lane's utterances are labelled: a room's by its speakers, a call's Me, a call's far side by its speakers or as Others. */
type Labels = "room" | "me" | "far";

type Item = { startMs: number; endMs: number | null; open: boolean; stable: string; tail: string; speaker: number | null; language: string | null };

type Lane = {
  name: LaneName;
  labels: Labels;
  diarization: Diarization | null;
  session: Listening | null;
  /** The session's stream, for the Transcript's speakers: a new session numbers its speakers afresh. */
  stream: string;
  /** The session's first sample on the clock; `null` before any audio. */
  origin: number | null;
  /** The session's audio sent, in ms. */
  sentMs: number;
  items: Map<string, Item>;
  capture: Capture | null;
  /** Flushes asked for and not answered yet (`openListening`'s `flush`). */
  flushes: number;
  /** Ending: once its last flush is answered, its utterance under way is committed. */
  ending: boolean;
};

const quiet = (): number[] => Array(METER_BARS).fill(0);

/** The picture a call's pick shares beside its audio, while it lasts. */
const sharedPicture = (stream: MediaStream): MediaStreamTrack | null => stream.getVideoTracks?.().find((track) => track.readyState === "live") ?? null;

/** What the view and the header read of a listening: its state, and word of each change. */
export type ListeningHandle = Pick<ListeningController, "getState" | "subscribe">;

/**
 * A listening's state, for the component that draws it. The page does not hold it: the levels move ten times a
 * second, and only what shows them redraws.
 */
export function useListening(handle: ListeningHandle): ListeningState {
  return useSyncExternalStore(handle.subscribe, handle.getState, handle.getState);
}

export class ListeningController {
  private state: ListeningState = IDLE;
  private readonly listeners = new Set<() => void>();
  private readonly now: () => number;
  private lanes: Lane[] = [];
  private sessions = 0;
  /** Bumped by every capture's start and stop, so a capture a Pause overtook is let go. */
  private runs = 0;
  /** A call's pick, asked once at the press and kept until the listening ends: a Resume needs no new one. */
  private picked: Promise<MediaStream> | null = null;
  /** What the hint line says once a Stop the page made itself is over. */
  private stopNotice: ListeningState["notice"] = null;
  /** This start was refused busy once already. */
  private retried = false;
  /** Bumped by every start and every end, not by a busy retry: a file decoded across a retry is still this listening's. */
  private listenings = 0;
  /** A file's audio once decoded, its name, and how far into it the page has sent. */
  private pcm: Int16Array | null = null;
  /** The sending reached the file's end: a refused append after it sends again. */
  private fileEnded = false;
  private fileName: string | null = null;
  private cursor = 0;
  /** Where in the file the file lane's session starts, and the Transcript's clock at the file's start. */
  private sessionStart = 0;
  private fileBase = 0;
  private paceMs = 0;
  /** The sending looked last, the time spent sending so far (for the time left), and a refused append's wait. */
  private lastFeed = 0;
  private feedMs = 0;
  private backoffUntil = 0;
  /** Bumped by every start and every end, so a late answer from an earlier one is dropped. */
  private attempt = 0;
  private context: Told = NOTHING_TOLD;
  private silenceMs: number | null = null;
  private holdMic = false;
  /** Until when the far side's last loud chunk holds the mic. */
  private farUntil = 0;
  /** The clock when it last started running, and what it held then. */
  private runningSince = 0;
  private clockAtRun = 0;
  private ticker: ReturnType<typeof setInterval> | undefined;

  constructor(private readonly deps: ListeningDeps) {
    this.now = deps.now ?? (() => Date.now());
  }

  getState = (): ListeningState => this.state;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  /** Start: inside the press's user activation, which the call's pick and the mic need. */
  start(options: StartOptions) {
    if (this.state.phase !== "idle") return;
    this.attempt++;
    this.context = options.context;
    this.silenceMs = options.silenceMs ?? null;
    this.holdMic = options.holdMic ?? false;
    this.farUntil = 0;
    this.clockAtRun = options.clockMs;
    this.runningSince = this.now();
    const diarization: Diarization | null = options.speakers ? "live" : null;
    this.stopNotice = null;
    this.retried = false;
    this.pcm = null;
    this.fileName = null;
    this.fileEnded = false;
    this.listenings++;
    if (options.source.kind === "file") {
      this.picked = null;
      this.fileName = options.source.file.name;
      this.cursor = 0;
      this.sessionStart = 0;
      this.fileBase = options.clockMs;
      this.feedMs = 0;
      this.backoffUntil = 0;
      this.paceMs = this.deps.filePaceSeconds() * 1_000;
      this.lanes = [this.lane("file", "room", options.speakers ? "file" : null)];
    } else if (options.source.kind === "call") {
      // Asked first, inside the press's activation, which a pick needs; the mic's own ask follows at once.
      this.picked = this.deps.pickShared(() => this.sharingEnded());
      // Kept caught: a refused pick is said when its lane's capture reads it.
      this.picked.catch(() => {});
      const listening = this.listenings;
      void this.picked.then((stream) => listening === this.listenings && this.update({ sharing: sharedPicture(stream) !== null }), () => {});
      this.lanes = [this.lane("mic", "me", null), this.lane("shared", "far", diarization)];
    } else {
      this.picked = null;
      this.lanes = [this.lane("mic", "room", diarization)];
    }
    setListening(true);
    this.update({
      ...IDLE,
      phase: "listening",
      transcript: options.transcript,
      source: options.source.kind,
      speakers: options.speakers,
      clockMs: options.clockMs,
      levels: Object.fromEntries(this.lanes.filter((lane) => lane.name !== "file").map((lane) => [lane.name, quiet()])),
      file: this.fileName === null ? null : { name: this.fileName, fraction: 0, leftMs: null },
    });
    for (const lane of this.lanes) this.open(lane);
    if (options.source.kind === "file") void this.decode(options.source.file);
    else this.run();
  }

  /**
   * A frame of the picture the call shares, the tab, the window or the screen its pick took (#431): taken only when
   * asked, and none while no call shares one.
   */
  async captureFrame(): Promise<Blob | null> {
    if (!this.state.sharing || !this.picked || !this.deps.grabFrame) return null;
    const track = sharedPicture(await this.picked);
    return track ? this.deps.grabFrame(track) : null;
  }

  /** What the transcription is told, from the next utterance on, in every session (Q87, Q128). */
  setContext(context: Told) {
    this.context = context;
    for (const lane of this.lanes) lane.session?.setContext(context);
  }

  /** The silence that ends an utterance, in every session, at once (Q122). */
  setSilence(silenceMs: number) {
    this.silenceMs = silenceMs;
    for (const lane of this.lanes) lane.session?.setSilence(silenceMs);
  }

  /** Whether a call holds the mic while the far side speaks (Q121); turned off, the mic goes at once. */
  setHoldMic(on: boolean) {
    this.holdMic = on;
    if (!on && this.state.held) this.update({ held: false });
  }

  /** Pause: the capture stops and the utterance under way is committed; the sessions stay open, the clock still. */
  pause() {
    if (this.state.phase !== "listening") return;
    this.clockAtRun = this.clock();
    this.halt();
    // A file's utterance goes on when the sending does: committed now, it would be cut in two.
    for (const lane of this.lanes) if (lane.name !== "file") this.commitOpen(lane);
    setListening(false);
    this.update({ phase: "paused", clockMs: this.clockAtRun });
  }

  /** Resume: the capture again, on a new session where the server closed the old one (`server.socket_idle_close`). */
  resume() {
    if (this.state.phase !== "paused") return;
    setListening(true);
    this.runningSince = this.now();
    this.update({ phase: "listening", notice: null });
    for (const lane of this.lanes) {
      if (lane.session) continue;
      // A file's new session starts where the utterances the old one lost did.
      if (lane.name === "file") this.sessionStart = this.cursor;
      this.open(lane);
    }
    this.run();
  }

  /**
   * Stop: the rest of the audio goes, the utterance under way is committed, and the listening finishes while
   * the text of the ones committed comes (with speakers, `completed` waits for the diarizer: spec speech/08).
   * Pressed again while finishing, it closes at once.
   */
  async stop() {
    const { phase } = this.state;
    // Pressed again while finishing, or while a busy server is waited on: over at once.
    if (phase === "finishing" || phase === "waiting") return this.finish();
    if (phase !== "listening" && phase !== "paused") return;
    const attempt = this.attempt;
    this.clockAtRun = this.clock();
    this.update({ phase: "finishing", clockMs: this.clockAtRun });
    setListening(false);
    this.stopTicker();
    await Promise.all(this.lanes.map((lane) => lane.capture?.flush()));
    if (attempt !== this.attempt) return;
    this.halt();
    for (const lane of this.lanes) this.endLane(lane);
    this.finishIfDone();
  }

  /** A lane's end: flushed, then its utterance under way committed (the class comment's "An end"). */
  private endLane(lane: Lane) {
    if (!lane.session) return;
    lane.ending = true;
    lane.flushes++;
    lane.session.flush();
  }

  /** The page is going away: whatever is under way stops, and the mic is let go. */
  dispose() {
    if (this.state.phase === "idle") return;
    this.end();
    this.update(IDLE);
  }

  /** The clock now: where the next audio falls on the Transcript; a file's is how far into it the page has sent. */
  private clock(): number {
    if (this.fileName !== null) return this.fileBase + msOf(this.cursor);
    return this.state.phase === "listening" ? this.clockAtRun + (this.now() - this.runningSince) : this.clockAtRun;
  }

  private lane(name: LaneName, labels: Labels, diarization: Diarization | null): Lane {
    return { name, labels, diarization, session: null, stream: "", origin: null, sentMs: 0, items: new Map(), capture: null, flushes: 0, ending: false };
  }

  /** A new session for the lane: a stream of its own, its origin at its first audio. */
  private open(lane: Lane) {
    const attempt = this.attempt;
    lane.stream = `${lane.name}-${++this.sessions}`;
    lane.origin = null;
    lane.sentMs = 0;
    lane.items = new Map();
    lane.flushes = 0;
    lane.ending = false;
    lane.session = this.deps.open({ context: this.context, diarization: lane.diarization, silenceMs: this.silenceMs }, (event) => attempt === this.attempt && this.heard(lane, event));
  }

  /** The clock runs, and every lane's capture with it; a file's sending instead. */
  private run() {
    this.stopTicker();
    if (this.fileName !== null) {
      this.lastFeed = this.now();
      this.ticker = setInterval(() => this.feed(), FEED_EVERY_MS);
      return;
    }
    this.ticker = setInterval(() => this.update({ clockMs: this.clock() }), 1_000);
    const runs = ++this.runs;
    for (const lane of this.lanes) void this.capture(lane, runs);
  }

  private async decode(file: File) {
    const listening = this.listenings;
    try {
      const pcm = await this.deps.decode(file);
      if (listening !== this.listenings) return;
      this.pcm = pcm;
    } catch (err) {
      if (listening === this.listenings) this.finish({ text: cannotStart(err instanceof Error ? err.message : String(err)), tone: "fault" });
      return;
    }
    // A busy server waited on: the retry runs the sending.
    if (this.state.phase === "listening") this.run();
  }

  /**
   * A look at a file's sending (Q99, pacing.ts): a burst while the audio not yet transcribed is under N, real
   * time past it while an utterance is under way, nothing while utterances wait for their text. Each append is
   * named by its first sample. At the file's end the utterance under way is committed and the listening
   * finishes.
   */
  private feed() {
    const lane = this.lanes[0];
    const pcm = this.pcm;
    if (!lane?.session || !pcm || this.state.phase !== "listening") return;
    const now = this.now();
    const since = now - this.lastFeed;
    this.lastFeed = now;
    this.feedMs += since;
    if (now >= this.backoffUntil) {
      const items = [...lane.items.values()];
      let budget = allowance({ backlogMs: backlogMs(lane.sentMs, items), limitMs: this.paceMs, open: items.some((item) => item.open), sinceLastMs: since, burstMs: BURST_MS });
      lane.origin ??= this.fileBase + msOf(this.sessionStart);
      while (budget > 0 && this.cursor < pcm.length) {
        const chunk = pcm.subarray(this.cursor, Math.min(this.cursor + CHUNK_SAMPLES, pcm.length));
        lane.session.append(chunk, `${APPEND_PREFIX}${this.cursor}`);
        this.cursor += chunk.length;
        lane.sentMs = msOf(this.cursor - this.sessionStart);
        budget -= msOf(chunk.length);
      }
    }
    this.showProgress(lane, pcm);
    if (this.cursor < pcm.length) return;
    this.stopTicker();
    this.fileEnded = true;
    this.stopNotice = { text: fileDone(this.fileName ?? "the file"), tone: "ash" };
    this.update({ phase: "finishing" });
    setListening(false);
    this.endLane(lane);
  }

  private showProgress(lane: Lane, pcm: Int16Array) {
    const doneMs = msOf(this.cursor) - backlogMs(lane.sentMs, [...lane.items.values()]);
    const { fraction, leftMs } = estimate({ doneMs, totalMs: msOf(pcm.length), elapsedMs: this.feedMs });
    this.update({ file: { name: this.fileName ?? "", fraction, leftMs }, clockMs: this.clock() });
  }

  private async capture(lane: Lane, runs: number) {
    const attempt = this.attempt;
    const onChunk = (pcm: Int16Array) => attempt === this.attempt && runs === this.runs && this.append(lane, pcm);
    let capture: Capture;
    try {
      if (lane.name === "shared") {
        if (!this.picked) return;
        capture = await this.deps.captureShared(await this.picked, onChunk);
      } else {
        capture = await this.deps.captureMic(onChunk);
      }
    } catch (err) {
      if (attempt === this.attempt) this.refusedCapture(err);
      return;
    }
    if (attempt !== this.attempt || runs !== this.runs || this.state.phase !== "listening") return capture.stop();
    lane.capture = capture;
  }

  /** A capture that could not start: the listening ends before it began, saying why. */
  private refusedCapture(err: unknown) {
    if (err instanceof MicBlocked) return this.finish({ text: MIC_BLOCKED, tone: "ash" });
    if (err instanceof CallNotShared) return this.finish({ text: CALL_NOT_SHARED, tone: "ash" });
    if (err instanceof NoSharedAudio) return this.finish({ text: NO_SHARED_AUDIO, tone: "ash" });
    this.finish({ text: cannotStart(err instanceof Error ? err.message : String(err)), tone: "fault" });
  }

  /** The browser's own Stop sharing: the call is over, and the listening stops as Stop does, saying why. */
  private sharingEnded() {
    this.stopNotice = { text: SHARING_ENDED, tone: "ash" };
    if (this.state.phase === "waiting") return this.finish(this.stopNotice);
    void this.stop();
  }

  /** The captures stop: a Pause, a Stop, the end. */
  private halt() {
    this.runs++;
    this.stopTicker();
    for (const lane of this.lanes) {
      lane.capture?.stop();
      lane.capture = null;
    }
  }

  private stopTicker() {
    clearInterval(this.ticker);
    this.ticker = undefined;
  }

  /** Commits the lane's utterance under way, if one is: a commit with none would make an item of the silence. */
  private commitOpen(lane: Lane) {
    if (lane.session && [...lane.items.values()].some((item) => item.open)) lane.session.commit();
  }

  /** Finishing, every session flushed and nothing waiting for its text: done. */
  private finishIfDone() {
    if (this.state.phase !== "finishing") return;
    if (this.lanes.every((lane) => !lane.session || (!lane.ending && lane.items.size === 0))) this.finish(this.stopNotice);
  }

  /** The listening is over: the sessions close, and the Transcript keeps the clock. */
  private finish(notice: ListeningState["notice"] = null) {
    const transcript = this.state.transcript;
    const clockMs = this.clock();
    this.end();
    if (transcript !== null) this.deps.onClock(transcript, clockMs);
    this.update({ ...IDLE, notice });
  }

  /**
   * A chunk of the lane's audio: to its session, on the clock. On a call that holds the mic, a loud far side holds
   * it, and the mic's session gets silence of the same length in its place: its times stay the clock's, and the far
   * voice that comes back through the speakers is never transcribed as Me (Q121).
   */
  private append(lane: Lane, pcm: Int16Array) {
    if (!lane.session) return;
    const ms = (pcm.length / TARGET_RATE) * 1_000;
    // The chunk's first sample, but never before the clock last started running.
    lane.origin ??= Math.max(this.clockAtRun, this.clock() - ms);
    const call = this.state.source === "call";
    if (call && lane.name === "shared" && levelOf(pcm) >= HOLD_LEVEL) this.farUntil = this.now() + HOLD_AFTER_MS;
    const held = call && lane.name === "mic" && this.holdMic && this.now() < this.farUntil;
    const sent = held ? new Int16Array(pcm.length) : pcm;
    lane.session.append(sent);
    lane.sentMs += ms;
    const levels = this.state.levels[lane.name] ?? quiet();
    this.update({
      levels: { ...this.state.levels, [lane.name]: [...levels.slice(1), levelOf(sent)] },
      clockMs: this.clock(),
      ...(lane.name === "mic" ? { held } : {}),
    });
  }

  /** The lane's item by its id, made at its first event. */
  private item(lane: Lane, id: string): Item {
    let item = lane.items.get(id);
    if (!item) lane.items.set(id, (item = { startMs: lane.sentMs, endMs: null, open: true, stable: "", tail: "", speaker: null, language: null }));
    return item;
  }

  private heard(lane: Lane, event: ListeningEvent) {
    switch (event.type) {
      case "started":
        this.item(lane, event.item).startMs = event.atMs;
        return this.showLive();
      case "stopped": {
        const item = this.item(lane, event.item);
        item.endMs = event.atMs;
        item.open = false;
        return;
      }
      case "committed": {
        const item = this.item(lane, event.item);
        // A commit of the page's, during speech: the item ends where the audio sent did.
        item.endMs ??= lane.sentMs;
        item.open = false;
        return;
      }
      case "delta":
        this.item(lane, event.item).stable += event.text;
        return this.showLive();
      case "tail":
        this.item(lane, event.item).tail = event.text;
        return this.showLive();
      case "language":
        this.item(lane, event.item).language = event.language;
        break;
      case "speaker":
        this.item(lane, event.item).speaker = event.speaker;
        return;
      case "completed":
        this.completed(lane, event.item, event.transcript);
        return this.finishIfDone();
      case "item_failed":
        lane.items.delete(event.item);
        this.showLive();
        this.update({ notice: { text: itemFailed(event.message), tone: "warn" } });
        return this.finishIfDone();
      case "flushed":
        lane.flushes = Math.max(0, lane.flushes - 1);
        if (lane.flushes > 0 || !lane.ending) return;
        lane.ending = false;
        this.commitOpen(lane);
        return this.finishIfDone();
      case "error":
        return this.refusedEvent(lane, event);
      case "refused":
        return this.refused(event.refusal);
      case "closed":
        return this.closed(lane);
    }
  }

  /** An event of the page's the server refused. The session goes on, unless its configuration was refused. */
  private refusedEvent(lane: Lane, { code, message, eventId }: { code: string | null; message: string; eventId: string | null }) {
    if (code === QUEUE_FULL && lane.name === "file" && eventId?.startsWith(APPEND_PREFIX)) {
      // Sent again from the first refused append, once the session has caught up a little.
      const from = Number(eventId.slice(APPEND_PREFIX.length));
      if (from < this.cursor) {
        this.cursor = from;
        lane.sentMs = msOf(from - this.sessionStart);
      }
      this.backoffUntil = this.now() + QUEUE_FULL_BACKOFF_MS;
      // Refused after the file's last append went: the sending goes on, and the end comes again.
      if (this.fileEnded && this.state.phase === "finishing" && this.cursor < (this.pcm?.length ?? 0)) {
        this.fileEnded = false;
        this.stopNotice = null;
        lane.ending = false;
        setListening(true);
        this.update({ phase: "listening" });
        this.run();
      }
      return;
    }
    if (code === DIARIZER_UNAVAILABLE || code === DIARIZER_FAILED) {
      // Without speakers from here on, and on a session a Resume opens.
      for (const lane of this.lanes) lane.diarization = null;
      return this.update({ speakers: false, notice: { text: speakersOff(message), tone: "warn" } });
    }
    // The audio waits for a configuration the server takes: one it refused leaves nothing to listen with.
    if (eventId === CONFIGURATION_EVENT) return this.finish({ text: cannotStart(message), tone: "fault" });
    this.update({ notice: { text: message, tone: "warn" } });
  }

  /** Why a session did not open: spec speech/04's states, and one retry of a busy server. */
  private refused(refusal: Refusal) {
    switch (refusal.kind) {
      case "busy":
        // Stopped before the refusal came: nothing to wait for.
        if (this.state.phase === "finishing") return this.finish(this.stopNotice);
        if (this.retried) return this.finish({ text: STILL_BUSY, tone: "warn" });
        this.retried = true;
        return this.countDown(Math.min(BUSY_RETRY_MAX_SECONDS, Math.ceil(refusal.retryAfter ?? BUSY_RETRY_SECONDS)));
      case "unavailable": {
        this.deps.onUnavailable?.();
        const cause = unavailable({ secure: true, speech: { available: false, reason: refusal.reason }, blocked: false });
        return this.finish(cause && { text: cause.sentence, tone: "ash" });
      }
      case "key":
        this.deps.onKeyRequired?.();
        return this.finish();
      case "unreachable":
        return this.finish({ text: cannotStart(refusal.message), tone: "fault" });
      case "no_websocket":
        return this.finish({ text: NO_WEBSOCKET, tone: "fault" });
    }
  }

  /** The busy wait: the sessions and the captures go (a call's pick stays), a second at a time, then one retry. */
  private countDown(seconds: number) {
    this.clockAtRun = this.clock();
    this.attempt++;
    this.halt();
    for (const lane of this.lanes) {
      lane.session?.close();
      lane.session = null;
    }
    this.update({ phase: "waiting", retryIn: seconds, live: [], notice: { text: busyRetry(seconds), tone: "warn" }, clockMs: this.clockAtRun });
    this.ticker = setInterval(() => {
      const left = this.state.retryIn - 1;
      if (left > 0) return this.update({ retryIn: left, notice: { text: busyRetry(left), tone: "warn" } });
      this.stopTicker();
      this.runningSince = this.now();
      this.update({ phase: "listening", retryIn: 0, notice: null });
      // What a file sent into the refused session waited there for a configuration that never came.
      this.cursor = this.sessionStart;
      for (const lane of this.lanes) this.open(lane);
      this.run();
    }, 1_000);
  }

  /** The server closed a lane's session: while paused, Resume opens another; while listening, the listening ends. */
  private closed(lane: Lane) {
    // A file is sent again from the oldest utterance the session took with it.
    const lost = [...lane.items.values()].map((item) => item.startMs);
    if (lane.name === "file" && lost.length > 0) this.cursor = Math.min(this.cursor, this.sessionStart + Math.round((Math.min(...lost) * TARGET_RATE) / 1_000));
    const unsettled = lost.length > 0 || lane.ending;
    lane.session = null;
    lane.items = new Map();
    lane.ending = false;
    this.showLive();
    if (this.state.phase === "paused") return;
    // Finishing with nothing lost is an end like any other; with utterances lost, the connection is.
    if (this.state.phase === "finishing" && !unsettled) return this.finishIfDone();
    this.finish({ text: CONNECTION_LOST, tone: "fault" });
  }

  private completed(lane: Lane, id: string, transcript: string) {
    const item = this.item(lane, id);
    lane.items.delete(id);
    const text = transcript.trim();
    const transcriptId = this.state.transcript;
    if (text !== "" && transcriptId !== null) {
      const origin = lane.origin ?? 0;
      this.deps.onUtterance(transcriptId, {
        stream: lane.stream,
        startMs: origin + item.startMs,
        endMs: origin + (item.endMs ?? lane.sentMs),
        text,
        voice: this.voice(lane, item),
        ...(item.language === null ? {} : { language: item.language }),
      });
    }
    this.showLive();
  }

  private voice(lane: Lane, item: Item): Voice | null {
    if (lane.labels === "me") return { kind: "me" };
    if (item.speaker !== null) return { kind: "speaker", index: item.speaker };
    return lane.labels === "far" ? { kind: "others" } : null;
  }

  /** The utterances under way, every lane's, in the order they started. */
  private showLive() {
    const live: LiveUtterance[] = [];
    for (const lane of this.lanes) {
      for (const [id, item] of lane.items) {
        live.push({ key: `${lane.stream}:${id}`, startMs: (lane.origin ?? 0) + item.startMs, stable: item.stable, tail: item.tail, lane: lane.name });
      }
    }
    this.update({ live: live.sort((a, b) => a.startMs - b.startMs) });
  }

  /** Lets every session and capture go, and a call's pick; later answers from them go unheard. */
  private end() {
    this.attempt++;
    this.listenings++;
    this.halt();
    for (const lane of this.lanes) {
      lane.session?.close();
      lane.session = null;
    }
    void this.picked?.then((stream) => stream.getTracks?.().forEach((track) => track.stop()), () => {});
    this.picked = null;
    setListening(false);
  }

  private update(patch: Partial<ListeningState>) {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener();
  }
}

