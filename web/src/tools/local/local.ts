import type { Settings, ToolDefinition } from "../../api/request.ts";
import type { ToolCall } from "../../api/sse.ts";
import type { streamChat } from "../../api/stream.ts";
import type { Figures } from "../../metrics/figures.ts";
import { type Attachment, READ_LIMIT, readPiece } from "./attachments.ts";
import { browserMemory, MEMORY_PROMPT_LIMIT, type MemoryNote, type MemoryStore } from "./memory.ts";
import { checkJs, type JsCheck, JS_OUTPUT_LIMIT, JS_TIMEOUT_MS, runJs, type WorkerFactory } from "./runjs.ts";

// The local tools: everything that runs in the browser with no network.
// run_js computes in a sandboxed worker, update_plan shows the model's plan,
// memory_save and memory_delete keep notes across sessions, create_file hands
// the user a file, read_file reads the files the user attached.

export const RUN_JS = "run_js";
export const UPDATE_PLAN = "update_plan";
export const MEMORY_SAVE = "memory_save";
export const MEMORY_READ = "memory_read";
export const MEMORY_DELETE = "memory_delete";
export const CREATE_FILE = "create_file";
export const READ_FILE = "read_file";

const LOCAL_TOOL_NAMES: readonly string[] = [RUN_JS, UPDATE_PLAN, MEMORY_SAVE, MEMORY_READ, MEMORY_DELETE, CREATE_FILE, READ_FILE];

export const isLocalTool = (name: string) => LOCAL_TOOL_NAMES.includes(name);

const tool = (name: string, description: string, properties: object, required: string[]): ToolDefinition => ({
  type: "function",
  function: { name, description, parameters: { type: "object", properties, required } },
});

export const RUN_JS_TOOL = tool(
  RUN_JS,
  `Run JavaScript in a sandboxed worker with no network, DOM or storage, for up to ${JS_TIMEOUT_MS / 1000} s. Returns the value you return, or the value of a single expression, and everything logged with console.log.`,
  { code: { type: "string", description: "The code. Use `return` for the result, or pass one expression. Top-level await works." } },
  ["code"],
);

export const UPDATE_PLAN_TOOL = tool(
  UPDATE_PLAN,
  "Replace the plan the user sees with this list of steps and their status.",
  {
    steps: {
      type: "array",
      description: "Every step of the plan, in order, each time.",
      items: {
        type: "object",
        properties: { step: { type: "string" }, status: { type: "string", enum: ["pending", "in_progress", "done"] } },
        required: ["step", "status"],
      },
    },
  },
  ["steps"],
);

export const MEMORY_SAVE_TOOL = tool(
  MEMORY_SAVE,
  "Save a note that stays across sessions: a lasting fact about the user, a preference, or project context. Only its title is listed in every prompt; the body is read on demand.",
  {
    title: { type: "string", description: "A few words that say what the note is about." },
    body: { type: "string", description: "The full note, self-contained." },
  },
  ["title", "body"],
);

export const MEMORY_READ_TOOL = tool(
  MEMORY_READ,
  "Read the full text of saved notes.",
  { ids: { type: "array", items: { type: "string" }, description: "The notes' ids, like m3." } },
  ["ids"],
);

export const MEMORY_DELETE_TOOL = tool(
  MEMORY_DELETE,
  "Delete a saved note that is wrong or outdated.",
  { id: { type: "string", description: "The note's id, like m3." } },
  ["id"],
);

export const CREATE_FILE_TOOL = tool(
  CREATE_FILE,
  "Give the user a text file to download, shown in your reply.",
  {
    name: { type: "string", description: "The file name with its extension, like report.md or data.csv." },
    content: { type: "string", description: "The whole content of the file." },
  },
  ["name", "content"],
);

export const READ_FILE_TOOL = tool(
  READ_FILE,
  `Read part of a file the user attached, up to ${READ_LIMIT} characters at a time.`,
  {
    name: { type: "string", description: "The attached file's name." },
    offset: { type: "integer", description: "The character to start from; 0 by default." },
    length: { type: "integer", description: `How many characters to read, at most ${READ_LIMIT}.` },
  },
  ["name"],
);

export function runJsPrompt(safetyCheck: boolean): string {
  return `# Running JavaScript
You can run JavaScript with \`run_js\`, in a sandboxed worker: no network, no DOM, no storage, at most ${JS_TIMEOUT_MS / 1000} s per call.
- Use it instead of working things out in your head: arithmetic, counting, dates, unit conversions, parsing and transforming data, checking what a snippet prints.
- \`return\` the result or pass a single expression; \`console.log\` output comes back too.${safetyCheck ? "\n- Every call is reviewed before it runs; code that does more than compute is blocked." : ""}`;
}

export const PLAN_PROMPT = `# Plan
For a task with several steps, keep a plan the user can follow with \`update_plan\`.
- Write the plan before you start, then update it as you go: a step is \`in_progress\` while you work on it and \`done\` once finished, one step in progress at a time.
- Send every step each time; the list replaces the previous one.
- Skip it for simple requests.`;

export function memoryPrompt(notes: MemoryNote[]): string {
  const lines: string[] = [];
  let used = 0;
  for (const n of [...notes].reverse()) {
    const line = `- [${n.id}] ${n.title}`;
    if (used + line.length > MEMORY_PROMPT_LIMIT) break;
    lines.unshift(line);
    used += line.length + 1;
  }
  const hidden = notes.length - lines.length;
  const saved =
    notes.length === 0
      ? "No notes saved yet."
      : `Saved notes, by title:\n${lines.join("\n")}${hidden > 0 ? `\n(${hidden} older notes not listed.)` : ""}`;
  return `# Memory
You keep notes that last across sessions. Only their titles are listed here: read the full text with \`memory_read\` when a title looks relevant, before you rely on it.
- Save a note with \`memory_save\` when you learn something that will still matter later — who the user is, their preferences, their projects — not the details of this conversation. Give it a title that says what it is about.
- Delete a note with \`memory_delete\` when it turns out wrong or outdated; to change one, delete it and save the new version.
${saved}`;
}

export const FILES_PROMPT = `# Files
You can give the user a file with \`create_file\`: code, CSV, a long document — anything they will want to keep. Mention the file in your reply instead of repeating its content.`;

export function attachmentsPrompt(attachments: Attachment[]): string {
  const list = attachments.map((a) => `- ${a.name} (${a.kind === "pdf" ? "PDF, as text" : "text"}, ${a.text.length} characters)`);
  return `# Attached files
The user attached these files. Read them with \`read_file\`, up to ${READ_LIMIT} characters at a time, before you answer about them:
${list.join("\n")}`;
}

export type PlanStep = { step: string; status: "pending" | "in_progress" | "done" };

export type CreatedFile = { name: string; content: string };

export type LocalStatus = "checking" | "running" | "done" | "failed" | "blocked" | "stopped";

/** One local tool call as the Playground shows it. */
export type LocalRun = {
  callId: string;
  /** The tool's name, as the model called it. */
  tool: string;
  status: LocalStatus;
  code?: string;
  check?: JsCheck;
  output?: string;
  logs?: string[];
  steps?: PlanStep[];
  note?: MemoryNote;
  recalled?: MemoryNote[];
  deletedId?: string;
  file?: CreatedFile;
  read?: { name: string; from: number; to: number; total: number };
  error?: string;
  /** What the model receives, once the call ends. */
  result?: string;
};

export type LocalContext = {
  settings: Settings;
  signal: AbortSignal;
  /** Review run_js code with the model before it runs. */
  jsSafetyCheck: boolean;
  attachments: Attachment[];
  memory?: MemoryStore;
  stream?: typeof streamChat;
  createWorker?: WorkerFactory;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  /** Each safety check's request, for the session log. */
  onCheck?: (figures: Figures | null, error?: string) => void;
};

/** A local call just made, before it runs. */
export const startedRun = (call: ToolCall): LocalRun => ({ callId: call.id, tool: call.name, status: "running" });

/** Runs every call at once; resolves with each run ended, its `result` set. */
export function runLocalCalls(calls: ToolCall[], context: LocalContext, onUpdate: (run: LocalRun) => void): Promise<LocalRun[]> {
  return Promise.all(calls.map((call) => runLocal(call, context, onUpdate)));
}

/** The tool result the model receives for an ended run. */
export const localToolResult = (run: LocalRun) => run.result ?? `The ${run.tool} call did not finish.`;

const cut = (text: string) =>
  text.length > JS_OUTPUT_LIMIT ? `${text.slice(0, JS_OUTPUT_LIMIT)}\n[Cut: ${text.length} characters in all.]` : text;

function parseArgs(call: ToolCall): Record<string, unknown> | null {
  try {
    const args = JSON.parse(call.arguments) as unknown;
    return typeof args === "object" && args !== null && !Array.isArray(args) ? (args as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

async function runLocal(call: ToolCall, context: LocalContext, onUpdate: (run: LocalRun) => void): Promise<LocalRun> {
  let run = startedRun(call);
  const update = (change: Partial<LocalRun>) => {
    run = { ...run, ...change };
    onUpdate(run);
    return run;
  };
  const fail = (error: string) => update({ status: "failed", error, result: `The ${call.name} call failed: ${error}` });
  const args = parseArgs(call);
  if (!args) return fail("The call's arguments are not a JSON object.");
  const text = (key: string) => (typeof args[key] === "string" ? (args[key] as string) : "");
  const memory = context.memory ?? browserMemory;
  const { signal } = context;

  switch (call.name) {
    case RUN_JS: {
      const code = text("code");
      if (!code.trim()) return fail('The call needs non-empty "code".');
      const stopped = () => update({ status: "stopped", result: "Stopped before the code ran." });
      if (signal.aborted) return update({ code, ...stopped() });
      if (context.jsSafetyCheck) {
        update({ code, status: "checking" });
        const check = await checkJs(code, { settings: context.settings, signal, stream: context.stream, sleep: context.sleep });
        context.onCheck?.(check.figures, check.error);
        if (signal.aborted) return stopped();
        const verdict: JsCheck = { verdict: check.verdict, reason: check.reason };
        if (verdict.verdict === "unsafe") {
          return update({ status: "blocked", check: verdict, result: `Not run: the safety check blocked the code. ${verdict.reason}` });
        }
        update({ check: verdict });
      }
      update({ code, status: "running" });
      const outcome = await runJs(code, { signal, createWorker: context.createWorker });
      if (outcome.status === "stopped") return update({ status: "stopped", result: "Stopped while the code ran." });
      if (outcome.status === "timeout") {
        const error = `Stopped after ${JS_TIMEOUT_MS / 1000} s without finishing.`;
        return update({ status: "failed", error, result: error });
      }
      const logText = outcome.logs.length > 0 ? `\n\nConsole:\n${outcome.logs.join("\n")}` : "";
      if (outcome.status === "failed") {
        return update({ status: "failed", error: outcome.error, logs: outcome.logs, result: cut(`Error: ${outcome.error}${logText}`) });
      }
      return update({ status: "done", output: outcome.result, logs: outcome.logs, result: cut(`Result: ${outcome.result ?? "undefined"}${logText}`) });
    }
    case UPDATE_PLAN: {
      const steps = (Array.isArray(args.steps) ? args.steps : []).flatMap((s): PlanStep[] => {
        const item = (typeof s === "object" && s !== null ? s : {}) as { step?: unknown; status?: unknown };
        if (typeof item.step !== "string" || item.step.trim() === "") return [];
        const status = item.status === "in_progress" || item.status === "done" ? item.status : "pending";
        return [{ step: item.step.trim(), status }];
      });
      if (steps.length === 0) return fail('The call needs "steps": a list of { step, status }.');
      const done = steps.filter((s) => s.status === "done").length;
      return update({ status: "done", steps, result: `Plan updated: ${done} of ${steps.length} steps done.` });
    }
    case MEMORY_SAVE: {
      const body = text("body").trim();
      if (!body) return fail('The call needs a non-empty "body", with a short "title".');
      const saved = memory.save(text("title"), body);
      return update({ status: "done", note: saved, result: `Saved as ${saved.id}: "${saved.title}".` });
    }
    case MEMORY_READ: {
      const ids = (Array.isArray(args.ids) ? args.ids : [args.id]).filter((id): id is string => typeof id === "string" && id.trim() !== "");
      if (ids.length === 0) return fail('The call needs "ids": the notes to read, like ["m3"].');
      const wanted = ids.map((id) => id.trim());
      const found = memory.list().filter((n) => wanted.includes(n.id));
      const missing = wanted.filter((id) => !found.some((n) => n.id === id));
      if (found.length === 0) return fail(`No saved note has the id ${missing.join(", ")}.`);
      const texts = found.map((n) => `[${n.id}] ${n.title}\n${n.body}`);
      if (missing.length > 0) texts.push(`No saved note has the id ${missing.join(", ")}.`);
      return update({ status: "done", recalled: found, result: texts.join("\n\n") });
    }
    case MEMORY_DELETE: {
      const id = text("id").trim();
      if (!memory.remove(id)) return fail(`No saved note has the id "${id}".`);
      return update({ status: "done", deletedId: id, result: `Deleted ${id}.` });
    }
    case CREATE_FILE: {
      const name = text("name").trim().replace(/[\\/:*?"<>|]+/g, "_");
      if (!name) return fail('The call needs a non-empty "name".');
      if (typeof args.content !== "string") return fail('The call needs "content" as text.');
      const content = args.content;
      return update({
        status: "done",
        file: { name, content },
        result: `Created ${name} (${content.length} characters). The user can download it from your reply.`,
      });
    }
    case READ_FILE: {
      const name = text("name").trim();
      const file = context.attachments.find((a) => a.name === name);
      if (!file) {
        return fail(
          context.attachments.length > 0
            ? `No attached file is named "${name}". The files are: ${context.attachments.map((a) => `"${a.name}"`).join(", ")}.`
            : "The user has not attached any file.",
        );
      }
      const piece = readPiece(file, typeof args.offset === "number" ? args.offset : 0, typeof args.length === "number" ? args.length : READ_LIMIT);
      const more = piece.to < piece.total ? ` Call read_file with offset ${piece.to} for the rest.` : "";
      return update({
        status: "done",
        read: { name, from: piece.from, to: piece.to, total: piece.total },
        result: `${piece.text}\n\n[${name}: characters ${piece.from}–${piece.to} of ${piece.total}.${more}]`,
      });
    }
    default:
      return fail(`"${call.name}" is not a local tool.`);
  }
}
