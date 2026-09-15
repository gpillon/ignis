import { describe, expect, it } from "vitest";
import type { ChatRequest, Settings } from "../../api/request.ts";
import type { StreamOptions, StreamResult } from "../../api/stream.ts";
import type { Timeline } from "../../metrics/figures.ts";
import { type Attachment, readPiece, uniqueName } from "./attachments.ts";
import { type LocalContext, type LocalRun, memoryPrompt, runLocalCalls } from "./local.ts";
import { createMemoryStore } from "./memory.ts";
import { JS_CHECK_SYSTEM_PROMPT, parseVerdict, runJs, type WorkerFactory, workerMain, type WorkerScope } from "./runjs.ts";

const settings: Settings = {
  model: "m",
  systemPrompt: "",
  temperature: 1,
  topP: 0.95,
  maxTokens: 100,
  reasoningEffort: "xhigh",
  laneTag: "interactive",
};

const timeline: Timeline = { sentAt: 0, firstTokenAt: 1, lastTokenAt: 2, endedAt: 3, finishReason: "stop", stopped: false };

/** The real worker program, run in this thread. */
const inlineWorker: WorkerFactory = (onMessage) => {
  const scope: WorkerScope = { postMessage: (message) => onMessage(message), onmessage: null };
  workerMain(scope);
  return { post: (code) => scope.onmessage?.({ data: { code } }), terminate: () => {} };
};

const reviewer = (reply: string, bodies: ChatRequest[] = []) =>
  async (o: StreamOptions): Promise<StreamResult> => {
    bodies.push(o.body as ChatRequest);
    o.onEvent({ kind: "content", text: reply });
    return { ok: true, timeline };
  };

const call = (name: string, args: object, id = "c1") => ({ id, name, arguments: JSON.stringify(args) });

function context(change: Partial<LocalContext> = {}): LocalContext {
  return {
    settings,
    signal: new AbortController().signal,
    jsSafetyCheck: false,
    attachments: [],
    memory: createMemoryStore(null),
    createWorker: inlineWorker,
    ...change,
  };
}

const runOne = async (name: string, args: object, change: Partial<LocalContext> = {}) => {
  const updates: LocalRun[] = [];
  const [run] = await runLocalCalls([call(name, args)], context(change), (r) => updates.push(r));
  return { run, updates };
};

describe("run_js", () => {
  it("returns an expression's value or a body's return, with the console", async () => {
    expect((await runOne("run_js", { code: "2 ** 10" })).run).toMatchObject({ status: "done", output: "1024", result: "Result: 1024" });
    const { run } = await runOne("run_js", { code: "const xs = [3, 1, 2];\nconsole.log('sorting', xs.length);\nreturn xs.sort();" });
    expect(run).toMatchObject({ status: "done", output: "[1,2,3]", logs: ["sorting 3"] });
    expect(run.result).toBe("Result: [1,2,3]\n\nConsole:\nsorting 3");
  });

  it("reports a thrown error and a syntax error", async () => {
    expect((await runOne("run_js", { code: "throw new RangeError('nope')" })).run).toMatchObject({ status: "failed", error: "RangeError: nope" });
    expect((await runOne("run_js", { code: "return (" })).run.status).toBe("failed");
  });

  it("stops code that runs out of time", async () => {
    const silent: WorkerFactory = () => ({ post() {}, terminate() {} });
    expect(await runJs("while (true) {}", { signal: new AbortController().signal, createWorker: silent, timeoutMs: 10 })).toEqual({ status: "timeout" });
  });

  it("asks the reviewer on an agent lane first, and runs only safe code", async () => {
    const bodies: ChatRequest[] = [];
    const checks: unknown[] = [];
    const { run, updates } = await runOne(
      "run_js",
      { code: "1 + 1" },
      { jsSafetyCheck: true, stream: reviewer('{"verdict":"safe","reason":"Arithmetic."}', bodies), onCheck: (f) => checks.push(f) },
    );
    expect(updates[0].status).toBe("checking");
    expect(run).toMatchObject({ status: "done", output: "2", check: { verdict: "safe", reason: "Arithmetic." } });
    expect(bodies[0]).toMatchObject({ class: "agent", reasoning_effort: "none", temperature: 0, top_p: 1 });
    expect(bodies[0].messages[0]).toEqual({ role: "system", content: JS_CHECK_SYSTEM_PROMPT });
    expect(checks).toHaveLength(1);

    const blocked = await runOne(
      "run_js",
      { code: "fetch('https://x')" },
      { jsSafetyCheck: true, stream: reviewer('{"verdict":"unsafe","reason":"It reaches the network."}') },
    );
    expect(blocked.run).toMatchObject({ status: "blocked", result: "Not run: the safety check blocked the code. It reaches the network." });
    expect(blocked.run.output).toBeUndefined();
  });

  it("reads anything but a clear safe verdict as unsafe", () => {
    expect(parseVerdict('Sure: {"verdict": "SAFE", "reason": "ok"}')).toEqual({ verdict: "safe", reason: "ok" });
    expect(parseVerdict("looks safe to me").verdict).toBe("unsafe");
    expect(parseVerdict('{"verdict":"maybe"}').verdict).toBe("unsafe");
  });
});

describe("update_plan, memory, files", () => {
  it("keeps valid steps and counts the done ones", async () => {
    const { run } = await runOne("update_plan", {
      steps: [{ step: "Search", status: "done" }, { step: " Read ", status: "in_progress" }, { step: "", status: "done" }, { step: "Write" }],
    });
    expect(run.steps).toEqual([
      { step: "Search", status: "done" },
      { step: "Read", status: "in_progress" },
      { step: "Write", status: "pending" },
    ]);
    expect(run.result).toBe("Plan updated: 1 of 3 steps done.");
    expect((await runOne("update_plan", { steps: [] })).run.status).toBe("failed");
  });

  it("saves notes with a title, lists only the titles in the prompt, and reads bodies on demand", async () => {
    const memory = createMemoryStore(null);
    const saved = await runOne("memory_save", { title: "Language", body: "Prefers answers in Italian, technical terms in English." }, { memory });
    expect(saved.run.result).toBe('Saved as m1: "Language".');
    await runOne("memory_save", { title: "Team", body: "Supports AS Roma." }, { memory });
    const prompt = memoryPrompt(memory.list());
    expect(prompt).toContain("- [m1] Language\n- [m2] Team");
    expect(prompt).not.toContain("Supports AS Roma");

    const read = await runOne("memory_read", { ids: ["m2", "m7"] }, { memory });
    expect(read.run.result).toBe("[m2] Team\nSupports AS Roma.\n\nNo saved note has the id m7.");
    expect(read.run.recalled?.map((n) => n.id)).toEqual(["m2"]);
    expect((await runOne("memory_read", { ids: ["m7"] }, { memory })).run.status).toBe("failed");
    expect((await runOne("memory_save", { title: "Empty", body: " " }, { memory })).run.status).toBe("failed");

    expect((await runOne("memory_delete", { id: "m1" }, { memory })).run.result).toBe("Deleted m1.");
    expect((await runOne("memory_delete", { id: "m9" }, { memory })).run.error).toBe('No saved note has the id "m9".');
    expect(memory.list().map((n) => n.id)).toEqual(["m2"]);
    expect(memoryPrompt([])).toContain("No notes saved yet.");
  });

  it("creates a file with a safe name", async () => {
    const { run } = await runOne("create_file", { name: "out/report.md", content: "# Hi" });
    expect(run).toMatchObject({ status: "done", file: { name: "out_report.md", content: "# Hi" } });
  });
});

describe("read_file and attachments", () => {
  const doc: Attachment = { name: "notes.txt", size: 30, kind: "text", text: "a".repeat(25_000) };

  it("reads a piece and says where the rest starts", async () => {
    const { run } = await runOne("read_file", { name: "notes.txt" }, { attachments: [doc] });
    expect(run.read).toEqual({ name: "notes.txt", from: 0, to: 20_000, total: 25_000 });
    expect(run.result?.endsWith("[notes.txt: characters 0–20000 of 25000. Call read_file with offset 20000 for the rest.]")).toBe(true);
    expect(readPiece(doc, 24_990, 50)).toMatchObject({ from: 24_990, to: 25_000 });
  });

  it("names the attached files when the name is wrong", async () => {
    expect((await runOne("read_file", { name: "x.pdf" }, { attachments: [doc] })).run.error).toBe(
      'No attached file is named "x.pdf". The files are: "notes.txt".',
    );
    expect((await runOne("read_file", { name: "x" })).run.error).toBe("The user has not attached any file.");
  });

  it("gives a second file of the same name a number", () => {
    expect(uniqueName("a.txt", ["a.txt"])).toBe("a (2).txt");
    expect(uniqueName("a.txt", ["a.txt", "a (2).txt"])).toBe("a (3).txt");
    expect(uniqueName("README", ["README"])).toBe("README (2)");
  });
});
