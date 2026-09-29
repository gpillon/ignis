import { describe, expect, it } from "vitest";
import { createFrameBatch, type FrameBatchDeps } from "./frameBatch.ts";

// Streaming changes wait for the next frame and land as one state update
// for every session; a change that must land now takes the waiting ones with
// it, in order. A background tab, whose frames stop, still gets its text.

/** A fake clock of frames and timeouts the test fires by hand. */
function fakeScheduler() {
  const frames = new Set<() => void>();
  const timeouts = new Set<() => void>();
  const deps: FrameBatchDeps = {
    frame: (run) => (frames.add(run), () => frames.delete(run)),
    fallback: (run) => (timeouts.add(run), () => timeouts.delete(run)),
  };
  const fire = (set: Set<() => void>) => {
    const due = [...set];
    set.clear();
    for (const run of due) run();
  };
  return { deps, frames, timeouts, frame: () => fire(frames), timeout: () => fire(timeouts) };
}

type State = Record<string, string>;

/** A batch over a state of text per session, and every update it committed. */
function harness() {
  const clock = fakeScheduler();
  let state: State = {};
  const commits: State[] = [];
  const batch = createFrameBatch<State>((update) => {
    state = update(state);
    commits.push(state);
  }, clock.deps);
  /** What the page does per delta: the reply's text so far, set under the reply's key. */
  const texts: State = {};
  const delta = (session: string, text: string) => {
    texts[session] = (texts[session] ?? "") + text;
    const next = texts[session];
    batch.set(session, (s) => ({ ...s, [session]: next }));
  };
  return { clock, batch, commits, delta, state: () => state };
}

describe("createFrameBatch", () => {
  it("lands many deltas of one frame as one update, with the text in order", () => {
    const h = harness();
    for (const piece of ["He", "llo", ", ", "world"]) h.delta("s1", piece);
    expect(h.commits).toHaveLength(0);
    h.clock.frame();
    expect(h.commits).toEqual([{ s1: "Hello, world" }]);
  });

  it("lands the deltas of every session in one update per frame", () => {
    const h = harness();
    h.delta("s1", "a");
    h.delta("s2", "x");
    h.delta("s1", "b");
    h.clock.frame();
    expect(h.commits).toEqual([{ s1: "ab", s2: "x" }]);
    h.delta("s2", "y");
    h.clock.frame();
    expect(h.commits).toHaveLength(2);
    expect(h.state()).toEqual({ s1: "ab", s2: "xy" });
  });

  it("flushes at once for a turn's end, taking what waits with it, and leaves the frame nothing to do", () => {
    const h = harness();
    h.delta("s1", "partial ");
    h.delta("s1", "answer");
    h.batch.flush();
    expect(h.commits).toEqual([{ s1: "partial answer" }]);
    expect(h.clock.frames.size).toBe(0);
    h.clock.frame();
    h.clock.timeout();
    expect(h.commits).toHaveLength(1);
  });

  it("keeps everything received before a stop: the stop's flush lands it", () => {
    const h = harness();
    h.delta("s1", "Hel");
    h.delta("s2", "other");
    h.delta("s1", "lo");
    // Stopped here: the turn's last change is applied at once, after what waited.
    h.batch.flush();
    expect(h.state()).toEqual({ s1: "Hello", s2: "other" });
  });

  it("does nothing on a flush with nothing waiting", () => {
    const h = harness();
    h.batch.flush();
    expect(h.commits).toHaveLength(0);
  });

  it("still delivers in a background tab, whose frames do not come: the fallback timeout lands it", () => {
    const h = harness();
    h.delta("s1", "hidden ");
    h.delta("s1", "tab");
    h.clock.timeout();
    expect(h.commits).toEqual([{ s1: "hidden tab" }]);
    // The frame that finally comes finds nothing waiting.
    h.clock.frame();
    expect(h.commits).toHaveLength(1);
    // And a turn's end lands without either.
    h.delta("s1", "!");
    h.batch.flush();
    expect(h.state()).toEqual({ s1: "hidden tab!" });
  });

  it("asks for one frame however many deltas arrive before it", () => {
    const h = harness();
    for (let i = 0; i < 50; i++) h.delta(`s${i % 5}`, "t");
    expect(h.clock.frames.size).toBe(1);
    expect(h.clock.timeouts.size).toBe(1);
    h.clock.frame();
    expect(h.clock.timeouts.size).toBe(0);
    expect(h.commits).toHaveLength(1);
  });
});
