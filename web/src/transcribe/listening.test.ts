import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CallNotShared, type Capture, MicBlocked, NoSharedAudio } from "../speech/capture.ts";
import { MIC_BLOCKED, TURNED_OFF } from "../speech/availability.ts";
import { BUSY_RETRY_SECONDS } from "../speech/dictation.ts";
import { listening } from "../speech/listening.ts";
import { type Listening, type ListeningEvent, type ListeningOptions, NOTHING_TOLD, type Told } from "../speech/realtime.ts";
import { busyRetry, CALL_NOT_SHARED, HOLD_AFTER_MS, fileDone, CONNECTION_LOST, itemFailed, NO_WEBSOCKET, speakersOff, STILL_BUSY, ListeningController, type ListeningDeps, type ListeningState, cannotStart, NO_SHARED_AUDIO, SHARING_ENDED } from "./listening.ts";
import type { Heard } from "./transcript.ts";

/** A context of a prompt alone. */
const said = (prompt: string): Told => ({ prompt, keywords: [], language: null });

// A listening in the Transcribe view (spec speech/07) through the controller
// the view reads: the Realtime sessions, the captures, the file's decoder and
// the clock are fakes the test drives.

type FakeSession = {
  options: ListeningOptions;
  emit: (event: ListeningEvent) => void;
  calls: string[];
  appended: { samples: number; eventId?: string }[];
  /** Every append's audio, to tell sound from silence. */
  pcm: Int16Array[];
  contexts: Told[];
  silences: number[];
};
type FakeCapture = { chunk: (samples: number, value?: number) => void; stopped: boolean };

const made: ListeningController[] = [];

function harness(options: { pace?: number; decoded?: number; refuseMic?: Error; refusePick?: Error; decodeGate?: Promise<void>; video?: MediaStreamTrack } = {}) {
  const sessions: FakeSession[] = [];
  const mics: FakeCapture[] = [];
  const shared: FakeCapture[] = [];
  const heard: [number, Heard][] = [];
  const clocks: [number, number][] = [];
  const events: string[] = [];
  const frames: MediaStreamTrack[] = [];
  let sharingEnded: (() => void) | null = null;
  const capture = (list: FakeCapture[]) => (onChunk: (pcm: Int16Array) => void): Capture => {
    const fake: FakeCapture = { chunk: (samples, value = 0) => onChunk(new Int16Array(samples).fill(value)), stopped: false };
    list.push(fake);
    return { flush: async () => {}, stop: () => (fake.stopped = true) };
  };
  const deps: ListeningDeps = {
    open: (opts, onEvent) => {
      const session: FakeSession = { options: opts, emit: onEvent, calls: [], appended: [], pcm: [], contexts: [], silences: [] };
      sessions.push(session);
      const handle: Listening = {
        append: (pcm, eventId) => {
          session.pcm.push(pcm);
          session.appended.push({ samples: pcm.length, ...(eventId === undefined ? {} : { eventId }) });
        },
        setContext: (context) => session.contexts.push(context),
        setSilence: (ms) => session.silences.push(ms),
        commit: () => session.calls.push("commit"),
        flush: () => session.calls.push("flush"),
        close: () => session.calls.push("close"),
      };
      return handle;
    },
    captureMic: async (onChunk) => {
      events.push("mic");
      if (options.refuseMic) throw options.refuseMic;
      return capture(mics)(onChunk);
    },
    pickShared: async (onEnded) => {
      events.push("pick");
      if (options.refusePick) throw options.refusePick;
      sharingEnded = onEnded;
      const video = options.video;
      return (video ? { getVideoTracks: () => [video], getTracks: () => [video] } : {}) as MediaStream;
    },
    grabFrame: async (track) => {
      frames.push(track);
      return new Blob(["a frame"], { type: "image/png" });
    },
    captureShared: async (_stream, onChunk) => capture(shared)(onChunk),
    decode: async (file) => {
      events.push(`decode ${file.name}`);
      await options.decodeGate;
      if (options.decoded === undefined) throw new Error("The browser could not decode this file as audio.");
      return new Int16Array(options.decoded);
    },
    filePaceSeconds: () => options.pace ?? 60,
    now: () => Date.now(),
    onUtterance: (transcript, utterance) => heard.push([transcript, utterance]),
    onClock: (transcript, clockMs) => clocks.push([transcript, clockMs]),
    onKeyRequired: () => events.push("key"),
    onUnavailable: () => events.push("unavailable"),
  };
  const controller = new ListeningController(deps);
  made.push(controller);
  const state = (): ListeningState => controller.getState();
  return { controller, sessions, mics, shared, heard, clocks, events, frames, state, session: (i = -1) => sessions.at(i)!, endSharing: () => sharingEnded?.() };
}

/** Lets the captures' promises settle. */
const settle = () => vi.advanceTimersByTimeAsync(0);

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(2026, 9, 25, 15, 42));
});
afterEach(() => {
  for (const controller of made.splice(0)) controller.dispose();
  vi.useRealTimers();
});

describe("a listening on the microphone", () => {
  it("opens one session with the context and the live speakers, takes the mic, and holds the page's speech", async () => {
    const h = harness();
    h.controller.start({ transcript: 7, clockMs: 0, source: { kind: "microphone" }, speakers: true, context: said("Standup.") });
    expect(h.state().phase).toBe("listening");
    expect(listening()).toBe(true);
    expect(h.sessions).toHaveLength(1);
    expect(h.session().options).toEqual({ context: said("Standup."), diarization: "live", silenceMs: null });
    await settle();
    h.mics[0].chunk(1_600);
    expect(h.session().appended).toEqual([{ samples: 1_600 }]);
  });

  it("shows each utterance under way, then hands it to its Transcript on the clock from the listening's start", async () => {
    const h = harness();
    h.controller.start({ transcript: 7, clockMs: 90_000, source: { kind: "microphone" }, speakers: true, context: NOTHING_TOLD });
    await settle();
    // The mic's first audio comes 2 s after the press: the session's 0 is there.
    vi.advanceTimersByTime(2_100);
    h.mics[0].chunk(1_600);
    const s = h.session();
    s.emit({ type: "started", item: "item_1", atMs: 480 });
    s.emit({ type: "tail", item: "item_1", text: "Buongiorno a" });
    expect(h.state().live).toEqual([{ key: "mic-1:item_1", startMs: 92_480, stable: "", tail: "Buongiorno a", lane: "mic" }]);
    s.emit({ type: "stopped", item: "item_1", atMs: 3_040 });
    s.emit({ type: "committed", item: "item_1" });
    s.emit({ type: "started", item: "item_2", atMs: 3_040 });
    s.emit({ type: "delta", item: "item_1", text: "Buongiorno a tutti." });
    s.emit({ type: "tail", item: "item_1", text: "" });
    s.emit({ type: "speaker", item: "item_1", speaker: 1 });
    s.emit({ type: "language", item: "item_1", language: "it" });
    s.emit({ type: "completed", item: "item_1", transcript: " Buongiorno a tutti. " });
    expect(h.heard).toEqual([[7, { stream: "mic-1", startMs: 92_480, endMs: 95_040, text: "Buongiorno a tutti.", voice: { kind: "speaker", index: 1 }, language: "it" }]]);
    expect(h.state().live.map((u) => u.key)).toEqual(["mic-1:item_2"]);
  });

  it("drops an utterance with no words, and one with no speaker has none", async () => {
    const h = harness();
    h.controller.start({ transcript: 1, clockMs: 0, source: { kind: "microphone" }, speakers: true, context: NOTHING_TOLD });
    await settle();
    h.mics[0].chunk(1_600);
    const s = h.session();
    s.emit({ type: "started", item: "a", atMs: 0 });
    s.emit({ type: "stopped", item: "a", atMs: 900 });
    s.emit({ type: "completed", item: "a", transcript: "  " });
    s.emit({ type: "started", item: "b", atMs: 1_000 });
    s.emit({ type: "stopped", item: "b", atMs: 2_000 });
    s.emit({ type: "completed", item: "b", transcript: "Sì." });
    expect(h.heard.map(([, u]) => [u.text, u.voice])).toEqual([["Sì.", null]]);
  });

  it("asks for no speakers with Speakers off", async () => {
    const h = harness();
    h.controller.start({ transcript: 1, clockMs: 0, source: { kind: "microphone" }, speakers: false, context: NOTHING_TOLD });
    expect(h.session().options.diarization).toBeNull();
  });
});

describe("Pause, Resume and Stop (Q97)", () => {
  async function talking() {
    const h = harness();
    h.controller.start({ transcript: 3, clockMs: 0, source: { kind: "microphone" }, speakers: true, context: NOTHING_TOLD });
    await settle();
    h.mics[0].chunk(1_600);
    return h;
  }

  it("Pause lets the mic go and commits the utterance under way; the clock stands still, and the other views' mics are free", async () => {
    const h = await talking();
    vi.advanceTimersByTime(4_900);
    h.mics[0].chunk(78_400);
    h.session().emit({ type: "started", item: "a", atMs: 1_000 });
    h.controller.pause();
    expect(h.state().phase).toBe("paused");
    expect(h.mics[0].stopped).toBe(true);
    expect(h.session().calls).toEqual(["commit"]);
    expect(listening()).toBe(false);
    const paused = h.state().clockMs;
    vi.advanceTimersByTime(60_000);
    expect(h.state().clockMs).toBe(paused);
    // Its text still comes while paused.
    h.session().emit({ type: "committed", item: "a" });
    h.session().emit({ type: "completed", item: "a", transcript: "Fermo qui." });
    // Committed by the page during speech: it ends where the audio sent did.
    expect(h.heard.map(([, u]) => [u.text, u.startMs, u.endMs])).toEqual([["Fermo qui.", 1_000, 5_000]]);
  });

  it("Pause commits nothing while no one speaks", async () => {
    const h = await talking();
    h.controller.pause();
    expect(h.session().calls).toEqual([]);
  });

  it("Resume takes the mic again on the same session, the clock going on from where it stood", async () => {
    const h = await talking();
    vi.advanceTimersByTime(900);
    h.controller.pause();
    vi.advanceTimersByTime(60_000);
    h.controller.resume();
    expect(h.state().phase).toBe("listening");
    expect(listening()).toBe(true);
    await settle();
    expect(h.sessions).toHaveLength(1);
    h.mics[1].chunk(1_600);
    expect(h.session().appended).toHaveLength(2);
    vi.advanceTimersByTime(1_000);
    expect(h.state().clockMs).toBe(1_900);
  });

  it("Resume opens a new session when the server closed the old one while paused, its speakers a new stream, its times continuing", async () => {
    const h = await talking();
    vi.advanceTimersByTime(9_900);
    h.controller.pause();
    h.session().emit({ type: "closed" });
    expect(h.state().phase).toBe("paused");
    vi.advanceTimersByTime(300_000);
    h.controller.resume();
    expect(h.sessions).toHaveLength(2);
    await settle();
    vi.advanceTimersByTime(200);
    h.mics[1].chunk(1_600);
    const s = h.session();
    s.emit({ type: "started", item: "b", atMs: 0 });
    s.emit({ type: "stopped", item: "b", atMs: 1_000 });
    s.emit({ type: "speaker", item: "b", speaker: 0 });
    s.emit({ type: "completed", item: "b", transcript: "Rieccomi." });
    expect(h.heard).toEqual([[3, { stream: "mic-2", startMs: 10_000, endMs: 11_000, text: "Rieccomi.", voice: { kind: "speaker", index: 0 } }]]);
  });

  it("Stop lets the mic go, commits the utterance under way, finishes while its text comes, then closes and leaves the clock to the Transcript", async () => {
    const h = await talking();
    vi.advanceTimersByTime(5_900);
    h.session().emit({ type: "started", item: "a", atMs: 2_000 });
    await h.controller.stop();
    expect(h.mics[0].stopped).toBe(true);
    // Flushed first: the VAD's events for the last audio sent may still be on their way.
    expect(h.session().calls).toEqual(["flush"]);
    expect(h.state().phase).toBe("finishing");
    expect(listening()).toBe(false);
    h.session().emit({ type: "flushed" });
    expect(h.session().calls).toEqual(["flush", "commit"]);
    h.session().emit({ type: "committed", item: "a" });
    h.session().emit({ type: "completed", item: "a", transcript: "Chiudo." });
    expect(h.session().calls).toEqual(["flush", "commit", "close"]);
    expect(h.state().phase).toBe("idle");
    expect(h.clocks).toEqual([[3, 5_900]]);
  });

  it("Stop with nothing under way closes once the session is flushed", async () => {
    const h = await talking();
    await h.controller.stop();
    expect(h.state().phase).toBe("finishing");
    h.session().emit({ type: "flushed" });
    expect(h.session().calls).toEqual(["flush", "close"]);
    expect(h.state().phase).toBe("idle");
  });

  it("Stop commits an utterance whose start was still on its way when it was pressed", async () => {
    const h = await talking();
    await h.controller.stop();
    h.session().emit({ type: "started", item: "late", atMs: 50 });
    h.session().emit({ type: "flushed" });
    expect(h.session().calls).toEqual(["flush", "commit"]);
    expect(h.state().phase).toBe("finishing");
  });

  it("a lost connection ends the listening, every completed utterance kept", async () => {
    const h = await talking();
    const s = h.session();
    s.emit({ type: "started", item: "a", atMs: 0 });
    s.emit({ type: "stopped", item: "a", atMs: 500 });
    s.emit({ type: "completed", item: "a", transcript: "Tenuto." });
    s.emit({ type: "started", item: "b", atMs: 600 });
    s.emit({ type: "closed" });
    expect(h.state().phase).toBe("idle");
    expect(h.state().notice).toEqual({ text: CONNECTION_LOST, tone: "fault" });
    expect(h.mics[0].stopped).toBe(true);
    expect(listening()).toBe(false);
    expect(h.heard.map(([, u]) => u.text)).toEqual(["Tenuto."]);
    expect(h.state().live).toEqual([]);
  });
});

describe("a call (Q93, Q112)", () => {
  it("asks for the pick and the mic at the press, and opens two sessions, never mixed: the mic's with no speakers, the shared audio's with them", async () => {
    const h = harness();
    h.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers: true, context: said("Weekly.") });
    expect(h.events).toEqual(["pick", "mic"]);
    expect(h.sessions.map((s) => s.options)).toEqual([
      { context: said("Weekly."), diarization: null, silenceMs: null },
      { context: said("Weekly."), diarization: "live", silenceMs: null },
    ]);
    await settle();
    h.mics[0].chunk(1_600);
    h.shared[0].chunk(3_200);
    expect(h.sessions.map((s) => s.appended)).toEqual([[{ samples: 1_600 }], [{ samples: 3_200 }]]);
  });

  it("labels the mic Me, and the shared audio by its speakers or, with Speakers off, Others; both on one clock", async () => {
    for (const speakers of [true, false]) {
      const h = harness();
      h.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers, context: NOTHING_TOLD });
      await settle();
      vi.advanceTimersByTime(1_100);
      h.mics[0].chunk(1_600);
      vi.advanceTimersByTime(500);
      h.shared[0].chunk(1_600);
      const [mic, far] = h.sessions;
      far.emit({ type: "started", item: "x", atMs: 0 });
      far.emit({ type: "stopped", item: "x", atMs: 800 });
      if (speakers) far.emit({ type: "speaker", item: "x", speaker: 2 });
      far.emit({ type: "completed", item: "x", transcript: "Mi senti?" });
      mic.emit({ type: "started", item: "y", atMs: 640 });
      mic.emit({ type: "stopped", item: "y", atMs: 1_200 });
      mic.emit({ type: "completed", item: "y", transcript: "Sì." });
      expect(h.heard.map(([, u]) => [u.text, u.stream, u.startMs, u.voice])).toEqual([
        ["Mi senti?", "shared-2", 1_500, speakers ? { kind: "speaker", index: 2 } : { kind: "others" }],
        ["Sì.", "mic-1", 1_640, { kind: "me" }],
      ]);
      h.controller.dispose();
    }
  });

  it("keeps the pick across a Pause: Resume needs no new one", async () => {
    const h = harness();
    h.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers: true, context: NOTHING_TOLD });
    await settle();
    h.controller.pause();
    expect(h.shared[0].stopped).toBe(true);
    h.controller.resume();
    await settle();
    expect(h.events).toEqual(["pick", "mic", "mic"]);
    expect(h.shared).toHaveLength(2);
  });

  it("stops when the browser's own Stop sharing ends the shared audio", async () => {
    const h = harness();
    h.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers: true, context: NOTHING_TOLD });
    await settle();
    h.endSharing();
    await settle();
    for (const s of h.sessions) s.emit({ type: "flushed" });
    expect(h.state().phase).toBe("idle");
    expect(h.state().notice).toEqual({ text: SHARING_ENDED, tone: "ash" });
    expect(h.sessions.map((s) => s.calls)).toEqual([["flush", "close"], ["flush", "close"]]);
  });

  it("takes a frame of what the call shares only when asked, and only while it shares a picture (#431)", async () => {
    const screen = { kind: "video", readyState: "live", stop: () => {} } as unknown as MediaStreamTrack;
    const h = harness({ video: screen });
    h.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers: true, context: NOTHING_TOLD });
    expect(h.state().sharing).toBe(false);
    await settle();
    expect(h.state().sharing).toBe(true);
    // A capture runs only on a click: nothing was taken by the pick itself.
    expect(h.frames).toEqual([]);
    expect(await h.controller.captureFrame()).toBeInstanceOf(Blob);
    expect(h.frames).toEqual([screen]);
    void h.controller.stop();
    await settle();
    for (const s of h.sessions) s.emit({ type: "flushed" });
    expect(h.state()).toMatchObject({ phase: "idle", sharing: false });
    expect(await h.controller.captureFrame()).toBeNull();
    expect(h.frames).toHaveLength(1);
  });

  it("has no frame to take from the microphone, nor from a pick whose picture has ended (#431)", async () => {
    const mic = harness();
    mic.controller.start({ transcript: 4, clockMs: 0, source: { kind: "microphone" }, speakers: true, context: NOTHING_TOLD });
    await settle();
    expect(mic.state().sharing).toBe(false);
    expect(await mic.controller.captureFrame()).toBeNull();
    const ended = harness({ video: { kind: "video", readyState: "ended", stop: () => {} } as unknown as MediaStreamTrack });
    ended.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers: true, context: NOTHING_TOLD });
    await settle();
    expect(ended.state().sharing).toBe(false);
    expect(await ended.controller.captureFrame()).toBeNull();
    expect(mic.frames.length + ended.frames.length).toBe(0);
  });

  it("says the sharing ended when it ends while a busy server is waited on", async () => {
    const h = harness();
    h.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers: true, context: NOTHING_TOLD });
    await settle();
    h.session().emit({ type: "refused", refusal: { kind: "busy" } });
    h.endSharing();
    expect(h.state()).toMatchObject({ phase: "idle", notice: { text: SHARING_ENDED, tone: "ash" } });
  });

  it("does not start when the pick is cancelled or shares no audio, and lets the mic go", async () => {
    for (const [refusal, sentence] of [
      [new CallNotShared("cancelled"), CALL_NOT_SHARED],
      [new NoSharedAudio("no audio"), NO_SHARED_AUDIO],
    ] as const) {
      const h = harness({ refusePick: refusal });
      h.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers: true, context: NOTHING_TOLD });
      await settle();
      expect(h.state().phase).toBe("idle");
      expect(h.state().notice).toEqual({ text: sentence, tone: "ash" });
      expect(h.mics.every((m) => m.stopped)).toBe(true);
      expect(h.sessions.map((s) => s.calls)).toEqual([["close"], ["close"]]);
      expect(listening()).toBe(false);
    }
  });
});

describe("a microphone the browser refuses", () => {
  it("ends the listening before it starts, saying why", async () => {
    const blocked = harness({ refuseMic: new MicBlocked("denied") });
    blocked.controller.start({ transcript: 1, clockMs: 0, source: { kind: "microphone" }, speakers: true, context: NOTHING_TOLD });
    await settle();
    expect(blocked.state().phase).toBe("idle");
    expect(blocked.state().notice).toEqual({ text: MIC_BLOCKED, tone: "ash" });
    const busy = harness({ refuseMic: Object.assign(new Error("Could not start audio source"), { name: "NotReadableError" }) });
    busy.controller.start({ transcript: 1, clockMs: 0, source: { kind: "microphone" }, speakers: true, context: NOTHING_TOLD });
    await settle();
    expect(busy.state().notice).toEqual({ text: cannotStart("Could not start audio source"), tone: "fault" });
  });
});

describe("what the server refuses (AC 15: spec speech/04's states)", () => {
  const start = (h: ReturnType<typeof harness>, kind: "microphone" | "call" = "microphone") =>
    h.controller.start({ transcript: 2, clockMs: 0, source: kind === "call" ? { kind: "call" } : { kind: "microphone" }, speakers: true, context: NOTHING_TOLD });

  it("a busy server: waits, counting down, then tries once more; still busy, says so", async () => {
    const h = harness();
    start(h);
    await settle();
    h.session().emit({ type: "refused", refusal: { kind: "busy" } });
    expect(h.state().phase).toBe("waiting");
    expect(h.state().retryIn).toBe(BUSY_RETRY_SECONDS);
    expect(h.state().notice).toEqual({ text: busyRetry(BUSY_RETRY_SECONDS), tone: "warn" });
    expect(h.mics[0].stopped).toBe(true);
    await vi.advanceTimersByTimeAsync(BUSY_RETRY_SECONDS * 1_000);
    expect(h.state().phase).toBe("listening");
    expect(h.sessions).toHaveLength(2);
    h.session().emit({ type: "refused", refusal: { kind: "busy", retryAfter: 9 } });
    expect(h.state().phase).toBe("idle");
    expect(h.state().notice).toEqual({ text: STILL_BUSY, tone: "warn" });
  });

  it("a busy refusal that comes after Stop does not start it again", async () => {
    const h = harness();
    start(h);
    await h.controller.stop();
    h.session().emit({ type: "refused", refusal: { kind: "busy" } });
    expect(h.state().phase).toBe("idle");
    await vi.advanceTimersByTimeAsync(BUSY_RETRY_SECONDS * 1_000);
    expect(h.sessions).toHaveLength(1);
    expect(h.mics.every((m) => m.stopped)).toBe(true);
  });

  it("Stop while it waits gives the retry up", async () => {
    const h = harness();
    start(h);
    h.session().emit({ type: "refused", refusal: { kind: "busy" } });
    await h.controller.stop();
    expect(h.state().phase).toBe("idle");
    await vi.advanceTimersByTimeAsync(BUSY_RETRY_SECONDS * 1_000);
    expect(h.sessions).toHaveLength(1);
  });

  it("a busy call keeps its pick for the retry", async () => {
    const h = harness();
    start(h, "call");
    await settle();
    h.session(1).emit({ type: "refused", refusal: { kind: "busy", retryAfter: 2 } });
    h.session(0).emit({ type: "refused", refusal: { kind: "busy" } });
    await vi.advanceTimersByTimeAsync(2_000);
    expect(h.events.filter((e) => e === "pick")).toHaveLength(1);
    expect(h.sessions).toHaveLength(4);
    expect(h.state().phase).toBe("listening");
  });

  it("an unavailable speech model: the page reads GET /v1/models again, and the hint says why", async () => {
    const h = harness();
    start(h);
    h.session().emit({ type: "refused", refusal: { kind: "unavailable", reason: "the speech model is turned off (`speech.model` is `off`)" } });
    expect(h.events).toContain("unavailable");
    expect(h.state()).toMatchObject({ phase: "idle", notice: { text: TURNED_OFF, tone: "ash" } });
  });

  it("a missing key asks for it; a lost upgrade says the address carries no WebSockets", async () => {
    const key = harness();
    start(key);
    key.session().emit({ type: "refused", refusal: { kind: "key", message: "missing API key" } });
    expect(key.events).toContain("key");
    expect(key.state().phase).toBe("idle");
    const proxy = harness();
    start(proxy);
    proxy.session().emit({ type: "refused", refusal: { kind: "no_websocket" } });
    expect(proxy.state().notice).toEqual({ text: NO_WEBSOCKET, tone: "fault" });
  });

  it("a diarizer the server refuses or loses: Speakers off with its reason, the listening going on", async () => {
    const h = harness();
    start(h);
    await settle();
    h.mics[0].chunk(1_600);
    h.session().emit({ type: "error", code: "ignis_diarizer_unavailable", message: "the diarizer is turned off (`diarizer.model` is `off`)", eventId: "listening_configuration" });
    expect(h.state()).toMatchObject({ phase: "listening", speakers: false, notice: { text: speakersOff("the diarizer is turned off (`diarizer.model` is `off`)"), tone: "warn" } });
    const failed = harness();
    start(failed);
    failed.session().emit({ type: "error", code: "ignis_diarizer_failed", message: "the diarizer failed: out of memory", eventId: null });
    expect(failed.state()).toMatchObject({ phase: "listening", speakers: false });
  });

  it("an utterance the server could not transcribe is said and dropped; the listening goes on", async () => {
    const h = harness();
    start(h);
    h.session().emit({ type: "started", item: "a", atMs: 0 });
    h.session().emit({ type: "item_failed", item: "a", message: "the speech model failed" });
    expect(h.state()).toMatchObject({ phase: "listening", live: [], notice: { text: itemFailed("the speech model failed"), tone: "warn" } });
  });

  it("a configuration the server refuses ends the listening with the server's words", async () => {
    const h = harness();
    start(h);
    h.session().emit({ type: "error", code: "unsupported_value", message: "server_vad is not served yet", eventId: "listening_configuration" });
    expect(h.state()).toMatchObject({ phase: "idle", notice: { text: cannotStart("server_vad is not served yet"), tone: "fault" } });
  });
});

describe("the context (Q87)", () => {
  it("changed mid-listening goes to every session, from the next utterance", async () => {
    const h = harness();
    h.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers: true, context: NOTHING_TOLD });
    h.controller.setContext(said("Names: Ada."));
    expect(h.sessions.map((s) => s.contexts)).toEqual([[said("Names: Ada.")], [said("Names: Ada.")]]);
  });
});

describe("a file (Q94, Q99)", () => {
  const SECONDS = 16_000;
  const file = new File([new Uint8Array([1])], "standup.m4a");
  /** The ms of audio a session was sent. */
  const sentMs = (session: FakeSession) => session.appended.reduce((n, a) => n + a.samples, 0) / 16;

  async function feeding(options: { speakers?: boolean; clockMs?: number } = {}) {
    const h = harness({ decoded: 120 * SECONDS, pace: 60 });
    h.controller.start({ transcript: 5, clockMs: options.clockMs ?? 0, source: { kind: "file", file }, speakers: options.speakers ?? true, context: NOTHING_TOLD });
    await settle();
    return h;
  }

  it("decodes the file, opens one session with the file speakers and no mic, and sends a burst at a time", async () => {
    const h = await feeding();
    expect(h.events).toEqual(["decode standup.m4a"]);
    expect(h.sessions.map((s) => s.options.diarization)).toEqual(["file"]);
    await vi.advanceTimersByTimeAsync(100);
    expect(sentMs(h.session())).toBe(5_000);
    expect(h.session().appended.slice(0, 2)).toEqual([
      { samples: 1_600, eventId: "audio_0" },
      { samples: 1_600, eventId: "audio_1600" },
    ]);
    expect((await feeding({ speakers: false })).session().options.diarization).toBeNull();
  });

  it("sends while the audio not yet transcribed stays under N, at real time past it while an utterance is under way, and stops while one waits for its text", async () => {
    const h = await feeding();
    await vi.advanceTimersByTimeAsync(100);
    const s = h.session();
    s.emit({ type: "started", item: "a", atMs: 1_000 });
    await vi.advanceTimersByTimeAsync(2_000);
    // Up to 60 s past the utterance's start, then 100 ms a tick for the 8 ticks left.
    expect(sentMs(s)).toBe(61_800);
    s.emit({ type: "stopped", item: "a", atMs: 40_000 });
    s.emit({ type: "committed", item: "a" });
    await vi.advanceTimersByTimeAsync(1_000);
    expect(sentMs(s)).toBe(61_800);
    s.emit({ type: "completed", item: "a", transcript: "Iniziamo." });
    await vi.advanceTimersByTimeAsync(100);
    expect(sentMs(s)).toBe(66_800);
    expect(h.state().file).toMatchObject({ name: "standup.m4a", fraction: 66_800 / 120_000 });
    expect(h.state().file?.leftMs).toBeGreaterThan(0);
  });

  it("at the file's end commits the utterance under way, finishes, and leaves the clock at the file's length", async () => {
    const h = await feeding({ clockMs: 30_000 });
    await vi.advanceTimersByTimeAsync(2_300);
    const s = h.session();
    s.emit({ type: "started", item: "z", atMs: 110_000 });
    await vi.advanceTimersByTimeAsync(200);
    expect(sentMs(s)).toBe(120_000);
    expect(s.calls).toEqual(["flush"]);
    expect(h.state().phase).toBe("finishing");
    s.emit({ type: "flushed" });
    expect(s.calls).toEqual(["flush", "commit"]);
    s.emit({ type: "committed", item: "z" });
    s.emit({ type: "completed", item: "z", transcript: "Fine." });
    expect(h.heard.map(([, u]) => [u.text, u.startMs, u.endMs])).toEqual([["Fine.", 140_000, 150_000]]);
    expect(h.state()).toMatchObject({ phase: "idle", notice: { text: fileDone("standup.m4a"), tone: "ash" } });
    expect(h.clocks).toEqual([[5, 150_000]]);
  });

  it("a refused append (the server's queue limit) sends again from it, after a moment", async () => {
    const h = await feeding();
    await vi.advanceTimersByTimeAsync(100);
    const s = h.session();
    s.emit({ type: "error", code: "ignis_audio_queue_full", message: "the session holds 120.0 s", eventId: "audio_48000" });
    s.emit({ type: "error", code: "ignis_audio_queue_full", message: "the session holds 120.0 s", eventId: "audio_49600" });
    s.appended.length = 0;
    await vi.advanceTimersByTimeAsync(500);
    expect(s.appended).toEqual([]);
    await vi.advanceTimersByTimeAsync(600);
    expect(s.appended[0]).toEqual({ samples: 1_600, eventId: "audio_48000" });
    expect(h.state().notice).toBeNull();
  });

  it("Pause stops the sending and commits nothing; Resume goes on where it was", async () => {
    const h = await feeding();
    await vi.advanceTimersByTimeAsync(100);
    const s = h.session();
    s.emit({ type: "started", item: "a", atMs: 1_000 });
    h.controller.pause();
    await vi.advanceTimersByTimeAsync(5_000);
    expect(sentMs(s)).toBe(5_000);
    expect(s.calls).toEqual([]);
    h.controller.resume();
    await vi.advanceTimersByTimeAsync(100);
    expect(sentMs(s)).toBe(10_000);
  });

  it("keeps a file decoded while a busy server is waited on, and sends it once the retry opens", async () => {
    let decoded!: () => void;
    const h = harness({ decoded: 120 * SECONDS, pace: 60, decodeGate: new Promise<void>((resolve) => (decoded = resolve)) });
    h.controller.start({ transcript: 5, clockMs: 0, source: { kind: "file", file }, speakers: true, context: NOTHING_TOLD });
    h.session().emit({ type: "refused", refusal: { kind: "busy" } });
    decoded();
    await settle();
    await vi.advanceTimersByTimeAsync(3_100);
    expect(h.sessions).toHaveLength(2);
    expect(sentMs(h.session())).toBe(5_000);
  });

  it("sends again an append the queue limit refused after the file's last one went", async () => {
    const h = await feeding();
    await vi.advanceTimersByTimeAsync(2_400);
    const s = h.session();
    expect(sentMs(s)).toBe(120_000);
    expect(h.state().phase).toBe("finishing");
    s.emit({ type: "error", code: "ignis_audio_queue_full", message: "full", eventId: `audio_${118 * SECONDS}` });
    expect(h.state().phase).toBe("listening");
    s.appended.length = 0;
    await vi.advanceTimersByTimeAsync(1_100);
    expect(s.appended[0]).toEqual({ samples: 1_600, eventId: `audio_${118 * SECONDS}` });
    // The first end's flush, then the second's.
    s.emit({ type: "flushed" });
    expect(h.state().phase).toBe("finishing");
    s.emit({ type: "flushed" });
    expect(h.state()).toMatchObject({ phase: "idle", notice: { text: fileDone("standup.m4a") } });
  });

  it("says the connection was lost when it closes while utterances still wait for their text", async () => {
    const h = await feeding();
    await vi.advanceTimersByTimeAsync(100);
    const s = h.session();
    s.emit({ type: "started", item: "a", atMs: 1_000 });
    await h.controller.stop();
    s.emit({ type: "flushed" });
    expect(h.state().phase).toBe("finishing");
    s.emit({ type: "closed" });
    expect(h.state()).toMatchObject({ phase: "idle", notice: { text: CONNECTION_LOST, tone: "fault" } });
  });

  it("says when the browser cannot decode the file", async () => {
    const h = harness();
    h.controller.start({ transcript: 5, clockMs: 0, source: { kind: "file", file }, speakers: true, context: NOTHING_TOLD });
    await settle();
    expect(h.state()).toMatchObject({ phase: "idle", notice: { text: cannotStart("The browser could not decode this file as audio."), tone: "fault" } });
    expect(h.session().calls).toEqual(["close"]);
  });
});

describe("the silence that ends an utterance (Q122)", () => {
  it("goes to every session at the start, and to every session when it changes", () => {
    const h = harness();
    h.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers: true, context: NOTHING_TOLD, silenceMs: 1_500 });
    expect(h.sessions.map((s) => s.options.silenceMs)).toEqual([1_500, 1_500]);
    h.controller.setSilence(2_000);
    expect(h.sessions.map((s) => s.silences)).toEqual([[2_000], [2_000]]);
  });
});

describe("a call holds the mic while the far side speaks (Q121)", () => {
  const silent = (pcm: Int16Array) => pcm.every((sample) => sample === 0);

  async function call(holdMic: boolean) {
    const h = harness();
    h.controller.start({ transcript: 4, clockMs: 0, source: { kind: "call" }, speakers: true, context: NOTHING_TOLD, holdMic });
    await settle();
    return h;
  }

  it("sends the mic's session silence while the shared audio is loud, and the mic again once it has been quiet a moment", async () => {
    const h = await call(true);
    const [mic] = h.sessions;
    h.shared[0].chunk(1_600, 8_000);
    h.mics[0].chunk(1_600, 3_000);
    expect(silent(mic.pcm[0])).toBe(true);
    expect(h.state().held).toBe(true);
    // The echo comes back late: still held just after the far side falls quiet.
    await vi.advanceTimersByTimeAsync(HOLD_AFTER_MS - 100);
    h.shared[0].chunk(1_600, 0);
    h.mics[0].chunk(1_600, 3_000);
    expect(silent(mic.pcm[1])).toBe(true);
    await vi.advanceTimersByTimeAsync(200);
    h.mics[0].chunk(1_600, 3_000);
    expect(silent(mic.pcm[2])).toBe(false);
    expect(h.state().held).toBe(false);
    // The far side is sent as it is, always.
    expect(silent(h.sessions[1].pcm[0])).toBe(false);
  });

  it("sends the mic as it is with the switch off, and lets it go when the switch is turned off mid-call", async () => {
    const off = await call(false);
    off.shared[0].chunk(1_600, 8_000);
    off.mics[0].chunk(1_600, 3_000);
    expect(silent(off.sessions[0].pcm[0])).toBe(false);

    const on = await call(true);
    on.shared[0].chunk(1_600, 8_000);
    on.controller.setHoldMic(false);
    expect(on.state().held).toBe(false);
    on.mics[0].chunk(1_600, 3_000);
    expect(silent(on.sessions[0].pcm[0])).toBe(false);
  });

  it("never holds a room's mic", async () => {
    const h = harness();
    h.controller.start({ transcript: 1, clockMs: 0, source: { kind: "microphone" }, speakers: true, context: NOTHING_TOLD, holdMic: true });
    await settle();
    h.mics[0].chunk(1_600, 3_000);
    expect(silent(h.session().pcm[0])).toBe(false);
  });
});
