import { describe, expect, it } from "vitest";
import { buildChatRequest, type ChatRequest, type ConversationRequest, type Settings } from "../../api/request.ts";
import type { StreamOptions, StreamResult } from "../../api/stream.ts";
import type { Timeline } from "../../metrics/figures.ts";
import { type LocalContext, type LocalRun, runLocalCalls } from "./local.ts";
import { createMemoryStore, memoryNotice } from "./memory.ts";
import { checkJs, JS_CHECK_FORMAT, JS_CHECK_SYSTEM_PROMPT, parseVerdict, readVerdict, runJs, type WorkerFactory, workerMain, type WorkerScope } from "./runjs.ts";

/** The chat body a request goes out as over HTTP. */
const chatBody = ({ settings, turns, extras, responseFormat }: ConversationRequest) => buildChatRequest(settings, turns, extras, responseFormat);

const settings: Settings = {
  systemPrompt: "",
  temperature: 1,
  topP: 0.95,
  maxTokens: 100,
  reasoningEffort: "xhigh",
  thinkingBudget: null,
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
    bodies.push(chatBody(o.request));
    o.onEvent({ kind: "content", text: reply });
    return { ok: true, timeline };
  };

const call = (name: string, args: object, id = "c1") => ({ id, name, arguments: JSON.stringify(args) });

function context(change: Partial<LocalContext> = {}): LocalContext {
  return {
    settings,
    signal: new AbortController().signal,
    jsSafetyCheck: false,
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

  it("reviews without thinking, so the session's thinking budget never rides the check", async () => {
    const bodies: ChatRequest[] = [];
    await runOne(
      "run_js",
      { code: "1 + 1" },
      { settings: { ...settings, thinkingBudget: 4096 }, jsSafetyCheck: true, stream: reviewer('{"verdict":"safe","reason":"ok"}', bodies) },
    );
    expect(bodies[0].reasoning_effort).toBe("none");
    expect("thinking_budget" in bodies[0]).toBe(false);
  });

  it("holds the verdict to a strict schema and reads it as written", async () => {
    const bodies: ChatRequest[] = [];
    const check = await checkJs("1 + 1", { settings, signal: new AbortController().signal, stream: reviewer('{"verdict":"safe","reason":" Arithmetic. "}', bodies) });
    expect(bodies[0].response_format).toEqual({ type: "json_schema", json_schema: { ...JS_CHECK_FORMAT, strict: true } });
    expect(JS_CHECK_FORMAT.schema).toMatchObject({ required: ["verdict", "reason"], additionalProperties: false });
    expect(check).toMatchObject({ verdict: "safe", reason: "Arithmetic." });
    // No fence is stripped from a held reply: a fenced one is no verdict.
    expect(readVerdict('```json\n{"verdict":"safe","reason":"ok"}\n```').verdict).toBe("unsafe");
  });

  it("calls a cut reply unsafe, even when its prefix reads safe", async () => {
    const cut = async (o: StreamOptions): Promise<StreamResult> => {
      o.onEvent({ kind: "content", text: '{"verdict":"safe","reason":"Arith' });
      return { ok: true, timeline: { ...timeline, finishReason: "length" } };
    };
    expect((await checkJs("1 + 1", { settings, signal: new AbortController().signal, stream: cut })).verdict).toBe("unsafe");
  });

  // Flash-Next's refusal, as each wire words it: chat completions names
  // `response_format`, the Responses socket (the default transport) `text.format`.
  for (const [wire, message] of [
    ["HTTP", "400: response_format: structured output is not served by Qwen3.8-Flash-Next yet (GitHub #581); send {\"type\": \"text\"} or omit it"],
    ["the socket", "400: text.format: structured output is not served by Qwen3.8-Flash-Next yet (GitHub #581); send {\"type\": \"text\"} or omit it"],
  ]) {
    it(`asks again without the schema where the model does not serve one, over ${wire}, so Flash-Next sees no new failure`, async () => {
      const bodies: ChatRequest[] = [];
      const unsupported = async (o: StreamOptions): Promise<StreamResult> => {
        bodies.push(chatBody(o.request));
        if (o.request.responseFormat) return { ok: false, message, code: "model_unsupported", timeline };
        o.onEvent({ kind: "content", text: 'Sure: {"verdict":"safe","reason":"ok"}' });
        return { ok: true, timeline };
      };
      const check = await checkJs("1 + 1", { settings, signal: new AbortController().signal, stream: unsupported });
      expect(check).toMatchObject({ verdict: "safe", reason: "ok" });
      expect(bodies.map((body) => "response_format" in body)).toEqual([true, false]);
    });
  }

  it("asks nothing again for any other refusal of the held check", async () => {
    let asked = 0;
    const refused = async (): Promise<StreamResult> => {
      asked += 1;
      return { ok: false, message: "400: response_format: json_schema refused, `allOf` at #: ...", code: "unsupported_keyword", timeline };
    };
    const check = await checkJs("1 + 1", { settings, signal: new AbortController().signal, stream: refused });
    expect(check.verdict).toBe("unsafe");
    expect(asked).toBe(1);
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

  it("saves notes with a title, tells a chat only their titles, and reads bodies on demand", async () => {
    const memory = createMemoryStore(null);
    const saved = await runOne("memory_save", { title: "Language", body: "Prefers answers in Italian, technical terms in English." }, { memory });
    expect(saved.run.result).toBe('Saved as m1: "Language".');
    await runOne("memory_save", { title: "Team", body: "Supports AS Roma." }, { memory });
    const notice = memoryNotice(undefined, memory.list()).text;
    expect(notice).toContain("- [m1] Language\n- [m2] Team");
    expect(notice).not.toContain("Supports AS Roma");

    const read = await runOne("memory_read", { ids: ["m2", "m7"] }, { memory });
    expect(read.run.result).toBe("[m2] Team\nSupports AS Roma.\n\nNo saved note has the id m7.");
    expect(read.run.recalled?.map((n) => n.id)).toEqual(["m2"]);
    expect((await runOne("memory_read", { ids: ["m7"] }, { memory })).run.status).toBe("failed");
    expect((await runOne("memory_save", { title: "Empty", body: " " }, { memory })).run.status).toBe("failed");

    expect((await runOne("memory_delete", { id: "m1" }, { memory })).run.result).toBe("Deleted m1.");
    expect((await runOne("memory_delete", { id: "m9" }, { memory })).run.error).toBe('No saved note has the id "m9".');
    expect(memory.list().map((n) => n.id)).toEqual(["m2"]);
    expect(memoryNotice(undefined, []).text).toBeUndefined();
  });

});

describe("the workspace's tools", () => {
  // Their own tests drive them over a workspace (workspace/workspace.test.ts); here, the router's side.
  it("fail where no workspace is given, and create_file is no local tool any more", async () => {
    expect((await runOne("write_file", { name: "a.md", content: "x" })).run.error).toBe("The workspace is not available here.");
    expect((await runOne("create_file", { name: "a.md", content: "x" })).run.error).toBe('"create_file" is not a local tool.');
  });
});
