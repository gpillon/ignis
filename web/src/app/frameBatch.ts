// Streaming changes, one state update per frame (GitHub #283). A reply
// streams a delta at a time, in every session at once, and a state update
// per delta re-rendered the whole page for each — hundreds of times a second
// with many sessions, for a screen that repaints sixty. So a change that
// only shows progress waits here, under a key (a reply, a reply's tools),
// where a later change replaces an earlier one, and every waiting change is
// applied in one update when the next frame comes. A change that must land
// now (a turn's end, anything that is not a delta) flushes what waits first,
// so changes land in the order they were made.
//
// A background tab gets no frames, so a timeout stands in for one; its text
// still arrives, only less often. Nothing here times a request: the figures
// are read in the transport, when the events arrive.

/** Calls `run` later; the function returned cancels it. */
export type Schedule = (run: () => void) => () => void;

export type FrameBatchDeps = {
  /** The next frame. */
  frame?: Schedule;
  /** What lands the changes when the frame does not come: a hidden tab. */
  fallback?: Schedule;
};

/** How long a change waits for a frame before the fallback lands it. */
export const FRAME_FALLBACK_MS = 100;

const nextFrame: Schedule = (run) => {
  if (typeof requestAnimationFrame !== "function") return () => {};
  const handle = requestAnimationFrame(run);
  return () => cancelAnimationFrame(handle);
};

const afterFallback: Schedule = (run) => {
  const handle = setTimeout(run, FRAME_FALLBACK_MS);
  return () => clearTimeout(handle);
};

export type FrameBatch<S> = {
  /** A change that can wait for the next frame; a later one under the same key replaces it. */
  set: (key: string, update: (state: S) => S) => void;
  /** Applies every waiting change now, in one update. */
  flush: () => void;
};

/** Changes to one piece of state, applied through `commit` once per frame. */
export function createFrameBatch<S>(commit: (update: (state: S) => S) => void, deps: FrameBatchDeps = {}): FrameBatch<S> {
  const frame = deps.frame ?? nextFrame;
  const fallback = deps.fallback ?? afterFallback;
  const waiting = new Map<string, (state: S) => S>();
  let cancels: (() => void)[] = [];

  function flush() {
    for (const cancel of cancels) cancel();
    cancels = [];
    if (waiting.size === 0) return;
    const updates = [...waiting.values()];
    waiting.clear();
    commit((state) => updates.reduce((next, update) => update(next), state));
  }

  return {
    set(key, update) {
      // Deleted first, so the key moves to the end: a change lands after those made before it.
      waiting.delete(key);
      waiting.set(key, update);
      if (cancels.length === 0) cancels = [frame(flush), fallback(flush)];
    },
    flush,
  };
}
