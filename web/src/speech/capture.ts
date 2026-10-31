import { CHUNK_SAMPLES, PROCESSOR, TARGET_RATE, WORKLET_SOURCE } from "./pcm.ts";

// The mic, for one dictation (spec speech/04, "Capture"): asked for at the
// press, run through the worklet in pcm.ts at the context's own rate, and let
// go on every way out — Done, Esc, a closed connection, the page leaving — so
// the browser's recording indicator never outlives a dictation (Q13). A
// listening in the Transcribe view takes the mic the same way, and on a call
// the shared audio of one `getDisplayMedia` pick too (spec speech/07), whose
// picture Capture takes a frame of (#431). The browser's objects are
// injectable, so the cleanup is testable without one.

/** The browser, or the person, refused the mic: the cause pressing it again can clear. */
export class MicBlocked extends Error {}

export type Capture = {
  /** Hands over the part chunk still in the worklet, then resolves. */
  flush: () => Promise<void>;
  /** Lets the mic go. */
  stop: () => void;
};

export type CaptureDeps = {
  getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
  createContext?: () => AudioContext;
  createNode?: (context: AudioContext, name: string, options: AudioWorkletNodeOptions) => AudioWorkletNode;
  /** A URL the worklet's source loads from; revoked once loaded. */
  moduleUrl?: () => string;
};

/** The person cancelled the call's pick, or the browser refused it. */
export class CallNotShared extends Error {}

/** The picked tab, window or screen shares no audio. */
export class NoSharedAudio extends Error {}

const defaults = (deps: CaptureDeps) => ({
  createContext: deps.createContext ?? (() => new AudioContext()),
  createNode: deps.createNode ?? ((context, name, options) => new AudioWorkletNode(context, name, options)),
  moduleUrl: deps.moduleUrl ?? (() => URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "text/javascript" }))),
});

export async function startCapture(onChunk: (pcm: Int16Array) => void, deps: CaptureDeps = {}): Promise<Capture> {
  const getUserMedia = deps.getUserMedia ?? ((constraints) => navigator.mediaDevices.getUserMedia(constraints));
  // Made before the first await, inside the press's user activation: a context made later can stay suspended.
  const context = defaults(deps).createContext();
  let stream: MediaStream;
  try {
    stream = await getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  } catch (err) {
    void context.close();
    if (err instanceof Error && (err.name === "NotAllowedError" || err.name === "SecurityError")) throw new MicBlocked(err.message);
    throw err;
  }
  return run(context, stream, onChunk, deps, true);
}

/**
 * A stream someone else picked, through the same worklet: a call's shared audio (spec speech/07). Its stop
 * lets the worklet go and leaves the stream's tracks to whoever picked them, so a Pause needs no new pick.
 */
export async function captureStream(stream: MediaStream, onChunk: (pcm: Int16Array) => void, deps: CaptureDeps = {}): Promise<Capture> {
  return run(defaults(deps).createContext(), stream, onChunk, deps, false);
}

/**
 * A call's other side (spec speech/07, Q93): one `getDisplayMedia` pick with its audio — a tab's, the whole
 * screen's system audio (on Windows every application's sound, mixed), or a window's where the browser offers
 * it. The browser wants video with it; the page reads only the audio. `onEnded` is the browser's own "Stop
 * sharing". Asked inside the press's user activation.
 */
export async function pickSharedAudio(
  onEnded: () => void,
  deps: { getDisplayMedia?: (options: DisplayMediaStreamOptions) => Promise<MediaStream> } = {},
): Promise<MediaStream> {
  const getDisplayMedia = deps.getDisplayMedia ?? ((options) => navigator.mediaDevices.getDisplayMedia(options));
  let stream: MediaStream;
  try {
    stream = await getDisplayMedia({
      video: true,
      // The far side as it is: the page's own processing is for a voice in the room.
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      systemAudio: "include",
      // Chrome 141+: a window's own audio, where the browser offers it.
      windowAudio: "window",
    } as DisplayMediaStreamOptions);
  } catch (err) {
    if (err instanceof Error && (err.name === "NotAllowedError" || err.name === "AbortError")) throw new CallNotShared(err.message);
    throw err;
  }
  const audio = stream.getAudioTracks();
  if (audio.length === 0) {
    for (const track of stream.getTracks()) track.stop();
    throw new NoSharedAudio("the pick shares no audio");
  }
  for (const track of audio) track.addEventListener("ended", onEnded);
  return stream;
}

/** How long a frame of the shared picture is waited for before Capture says it got none. */
const FRAME_WAIT_MS = 5_000;

/**
 * A frame of a call's shared picture (#431): its video track, which the pick asked for and the listening never reads,
 * played into a video element the page never shows and drawn once on a canvas, as a PNG. The track stays the pick's,
 * live: only the element is let go. Browser only.
 */
export async function grabFrame(track: MediaStreamTrack): Promise<Blob> {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.srcObject = new MediaStream([track]);
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const shown = new Promise<void>((resolve) => {
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) resolve();
      else video.addEventListener("loadeddata", () => resolve(), { once: true });
    });
    const late = new Promise<never>((_, reject) => (timer = setTimeout(() => reject(new Error("the shared picture gave no frame")), FRAME_WAIT_MS)));
    await Promise.race([video.play().then(() => shown), late]);
    const { videoWidth: width, videoHeight: height } = video;
    if (width === 0 || height === 0) throw new Error("the shared picture gave no frame");
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("this browser gives no 2D canvas");
    ctx.drawImage(video, 0, 0, width, height);
    return await new Promise<Blob>((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("the canvas gave no PNG"))), "image/png"));
  } finally {
    clearTimeout(timer);
    video.pause();
    video.srcObject = null;
  }
}

/** The worklet on `stream`, in `context`; `owned` streams are let go with it. */
async function run(context: AudioContext, stream: MediaStream, onChunk: (pcm: Int16Array) => void, deps: CaptureDeps, owned: boolean): Promise<Capture> {
  const { createNode, moduleUrl } = defaults(deps);
  let stopped = false;
  let flushed: (() => void) | null = null;
  const stop = () => {
    if (stopped) return;
    stopped = true;
    if (owned) for (const track of stream.getTracks()) track.stop();
    void context.close();
    flushed?.();
  };
  try {
    const url = moduleUrl();
    try {
      await context.audioWorklet.addModule(url);
    } finally {
      if (!deps.moduleUrl) URL.revokeObjectURL(url);
    }
    const source = context.createMediaStreamSource(stream);
    // One output, to the destination: a node nothing pulls is not processed. It writes silence.
    const node = createNode(context, PROCESSOR, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      processorOptions: { targetRate: TARGET_RATE, chunkSamples: CHUNK_SAMPLES },
    });
    node.port.onmessage = (event: MessageEvent<{ pcm?: Int16Array; flushed?: true }>) => {
      if (event.data.pcm && !stopped) onChunk(event.data.pcm);
      if (event.data.flushed) flushed?.();
    };
    source.connect(node);
    node.connect(context.destination);
    void context.resume();
    return {
      flush: () =>
        stopped
          ? Promise.resolve()
          : new Promise<void>((resolve) => {
              flushed = () => {
                flushed = null;
                resolve();
              };
              node.port.postMessage("flush");
            }),
      stop,
    };
  } catch (err) {
    stop();
    throw err;
  }
}
