import { buildChatRequest, type Settings } from "../../api/request.ts";
import { streamChat } from "../../api/stream.ts";
import { computeFigures, type Figures } from "../../metrics/figures.ts";

// run_js: JavaScript the model writes runs in a Web Worker made for the one
// call — no DOM, no page state, network and storage switched off, and a time
// limit, after which the worker is terminated. With the safety check on, the
// code first goes to the model on an agent lane, with a reviewer's prompt;
// code it does not call safe never runs.

export const JS_TIMEOUT_MS = 10_000;
/** The most of a result or console output the model receives. */
export const JS_OUTPUT_LIMIT = 10_000;

export type JsOutcome =
  | { status: "done"; result?: string; logs: string[] }
  | { status: "failed"; error: string; logs: string[] }
  | { status: "timeout" }
  | { status: "stopped" };

export type WorkerScope = {
  postMessage: (message: unknown) => void;
  onmessage: ((event: { data: { code: string } }) => void) | null;
};

/**
 * The worker's whole program. It is turned into source with toString, so it
 * must not use anything from outside itself.
 */
export function workerMain(scope: WorkerScope) {
  for (const name of ["fetch", "XMLHttpRequest", "WebSocket", "EventSource", "importScripts", "indexedDB", "caches"]) {
    try {
      Object.defineProperty(scope, name, { value: undefined, writable: false, configurable: false });
    } catch {
      // Not every global can be redefined; the safety check covers the rest.
    }
  }
  const show = (value: unknown): string => {
    if (typeof value === "string") return value;
    try {
      return JSON.stringify(value, (_key, v) => (typeof v === "bigint" ? `${v}n` : v)) ?? String(value);
    } catch {
      return String(value);
    }
  };
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor as new (...args: string[]) => (
    console: object,
  ) => Promise<unknown>;
  scope.onmessage = async (event) => {
    const logs: string[] = [];
    const log = (...args: unknown[]) => {
      logs.push(args.map(show).join(" "));
    };
    const console = { log, info: log, warn: log, error: log, debug: log };
    let run: (console: object) => Promise<unknown>;
    try {
      // A single expression gives its value; anything else runs as a body that may `return`.
      run = new AsyncFunction("console", `return (${event.data.code}\n);`);
    } catch {
      try {
        run = new AsyncFunction("console", event.data.code);
      } catch (err) {
        scope.postMessage({ ok: false, error: err instanceof Error ? `${err.name}: ${err.message}` : show(err), logs });
        return;
      }
    }
    try {
      const value = await run(console);
      scope.postMessage({ ok: true, result: value === undefined ? undefined : show(value), logs });
    } catch (err) {
      scope.postMessage({ ok: false, error: err instanceof Error ? `${err.name}: ${err.message}` : show(err), logs });
    }
  };
}

export type WorkerFactory = (
  onMessage: (data: unknown) => void,
  onError: (message: string) => void,
) => { post(code: string): void; terminate(): void };

const browserWorker: WorkerFactory = (onMessage, onError) => {
  const url = URL.createObjectURL(new Blob([`(${workerMain.toString()})(self);`], { type: "text/javascript" }));
  const worker = new Worker(url);
  worker.onmessage = (event) => onMessage(event.data);
  worker.onerror = (event) => {
    event.preventDefault();
    onError(event.message || "The worker failed.");
  };
  return {
    post: (code) => worker.postMessage({ code }),
    terminate: () => {
      worker.terminate();
      URL.revokeObjectURL(url);
    },
  };
};

/** Runs `code` in a fresh worker; resolves once it answers, fails, runs out of time or the signal stops it. */
export function runJs(code: string, options: { signal: AbortSignal; timeoutMs?: number; createWorker?: WorkerFactory }): Promise<JsOutcome> {
  return new Promise((resolve) => {
    if (options.signal.aborted) return resolve({ status: "stopped" });
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let worker: { post(code: string): void; terminate(): void } | undefined;
    const finish = (outcome: JsOutcome) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      options.signal.removeEventListener("abort", onAbort);
      worker?.terminate();
      resolve(outcome);
    };
    const onAbort = () => finish({ status: "stopped" });
    worker = (options.createWorker ?? browserWorker)(
      (data) => {
        const message = data as { ok?: boolean; result?: string; error?: string; logs?: string[] };
        const logs = message.logs ?? [];
        finish(message.ok ? { status: "done", result: message.result, logs } : { status: "failed", error: message.error ?? "Unknown error.", logs });
      },
      (error) => finish({ status: "failed", error, logs: [] }),
    );
    timer = setTimeout(() => finish({ status: "timeout" }), options.timeoutMs ?? JS_TIMEOUT_MS);
    options.signal.addEventListener("abort", onAbort, { once: true });
    worker.post(code);
  });
}

export const JS_CHECK_SYSTEM_PROMPT = `You review JavaScript that an AI assistant wants to run in a sandboxed Web Worker. The code should only compute: arithmetic, data and text processing, dates, algorithms, and logging the results.
Call it unsafe if it tries to reach the network or load code (fetch, XMLHttpRequest, WebSocket, EventSource, import(), importScripts), use browser storage (indexedDB, caches, localStorage), probe or escape the sandbox, hide what it does (obfuscation, eval or Function on built strings), or waste resources on purpose (endless loops, huge allocations, mining).
Reply with only a JSON object: {"verdict": "safe" or "unsafe", "reason": "one short sentence"}.`;

export type JsCheck = { verdict: "safe" | "unsafe"; reason: string };

/** The reviewer's reply as a verdict. Anything but a clear "safe" is unsafe. */
export function parseVerdict(text: string): JsCheck {
  const json = /\{[\s\S]*\}/.exec(text)?.[0];
  if (json) {
    try {
      const parsed = JSON.parse(json) as { verdict?: unknown; reason?: unknown };
      const verdict = String(parsed.verdict ?? "").trim().toLowerCase();
      if (verdict === "safe" || verdict === "unsafe") {
        const reason = typeof parsed.reason === "string" && parsed.reason.trim() ? parsed.reason.trim() : "No reason given.";
        return { verdict, reason };
      }
    } catch {
      // Not JSON after all.
    }
  }
  return { verdict: "unsafe", reason: "The safety check gave no clear verdict." };
}

function abortableSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(timer), resolve()), { once: true });
  });
}

/** Asks the model whether `code` is safe to run: greedy, thinking off, on an agent lane; a full engine is waited out. */
export async function checkJs(
  code: string,
  options: {
    settings: Settings;
    signal: AbortSignal;
    stream?: typeof streamChat;
    sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  },
): Promise<JsCheck & { figures: Figures | null; error?: string }> {
  const stream = options.stream ?? streamChat;
  const sleep = options.sleep ?? abortableSleep;
  const body = buildChatRequest(
    {
      ...options.settings,
      systemPrompt: JS_CHECK_SYSTEM_PROMPT,
      laneTag: "agent",
      reasoningEffort: "none",
      temperature: 0,
      topP: 1,
      maxTokens: 300,
    },
    [{ role: "user", content: `Review this code:\n\n\`\`\`js\n${code}\n\`\`\`` }],
  );
  for (let attempt = 0; ; attempt++) {
    let content = "";
    const result = await stream({
      body,
      signal: options.signal,
      onEvent: (event) => {
        if (event.kind === "content") content += event.text;
      },
    });
    if (!result.ok && /engine_full|all lanes in use/i.test(result.message) && attempt < 60 && !options.signal.aborted) {
      await sleep(1000, options.signal);
      continue;
    }
    if (!result.ok) {
      return { verdict: "unsafe", reason: `The safety check could not run: ${result.message}`, figures: null, error: result.message };
    }
    return { ...parseVerdict(content), figures: computeFigures(result.timeline) };
  }
}
