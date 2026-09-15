import type { ToolDefinition } from "../../api/request.ts";
import type { ToolCall } from "../../api/sse.ts";
import { unknownToolError } from "../errors.ts";

// The web tools: `web_search` asks Tavily for pages matching a query,
// `web_fetch` reads one page through the Jina reader (r.jina.ai) as
// Markdown. Both run in the browser — the two services answer CORS — so
// ignis is not involved: the Tavily key stays in this browser's localStorage
// and goes only to Tavily. Without a key, a search asks search.tiago.zip (a
// public keyless JSON API), and when that fails reads DuckDuckGo Lite's
// results page through the Jina reader: both keyless, both less reliable
// than Tavily. Every call in a reply runs at once.

export const WEB_SEARCH_TOOL_NAME = "web_search";
export const WEB_FETCH_TOOL_NAME = "web_fetch";

export const WEB_SEARCH_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: WEB_SEARCH_TOOL_NAME,
    description: "Search the web. Returns the top results, each with its title, URL and an excerpt of the page.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "What to search for, as you would type it into a search engine." },
      },
      required: ["query"],
    },
  },
};

export const WEB_FETCH_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: WEB_FETCH_TOOL_NAME,
    description: "Read one web page. Returns its text as Markdown, cut to a fixed length.",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string", description: "The page's full http(s) URL." },
      },
      required: ["url"],
    },
  },
};

/** What the web tools add to the ignis system prompt of the conversation. */
export const WEB_IGNIS_PROMPT = `# Web
You can reach the web with two tools.
- \`web_search\` returns the top results for a query: title, URL and an excerpt. Use it for facts that may be recent, niche or that you are unsure of.
- \`web_fetch\` returns the text of one page. Use it when an excerpt is not enough, on URLs from the results or given by the user.
- Calls in the same reply run in parallel: search several angles, or read several pages, at once.
- Base your reply on what the tools returned and cite the URLs you used. If they found nothing useful, say so.`;

export const SEARCH_RESULTS = 5;
/** A web call with no answer by then fails, so a hung service never holds the turn. */
export const WEB_TIMEOUT_MS = 60_000;
/** A fetched page is cut to this many characters before it goes to the model. */
export const FETCH_LIMIT = 20_000;

export type WebStatus = "running" | "done" | "failed" | "stopped";

export type SearchResult = { title: string; url: string; content: string };

/** One web call as the Playground shows it and the model is told about it. */
export type WebRun = {
  callId: string;
  tool: typeof WEB_SEARCH_TOOL_NAME | typeof WEB_FETCH_TOOL_NAME;
  /** The query, or the URL. */
  target: string;
  status: WebStatus;
  results?: SearchResult[];
  /** The service a finished search asked. */
  via?: "tavily" | "tiago" | "duckduckgo";
  /** The page text as fetched, before the cut. */
  page?: string;
  error?: string;
};

export type WebTask = Pick<WebRun, "callId" | "tool" | "target">;

export const isWebTool = (name: string) => name === WEB_SEARCH_TOOL_NAME || name === WEB_FETCH_TOOL_NAME;

/**
 * A call the model made, read as a web task, or as a failed run whose error
 * explains what was wrong (a tool not in `available`, arguments without the
 * query or URL).
 */
export function parseWebCall(
  call: ToolCall,
  available: string[] = [WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME],
): { ok: true; task: WebTask } | { ok: false; run: WebRun } {
  const tool: WebRun["tool"] = call.name === WEB_FETCH_TOOL_NAME ? WEB_FETCH_TOOL_NAME : WEB_SEARCH_TOOL_NAME;
  const field = tool === WEB_FETCH_TOOL_NAME ? "url" : "query";
  const fail = (error: string, target = call.arguments) => ({
    ok: false as const,
    run: { callId: call.id, tool, target, status: "failed" as const, error },
  });
  if (!isWebTool(call.name) || !available.includes(call.name)) return fail(unknownToolError(call.name, available));
  let args: unknown;
  try {
    args = JSON.parse(call.arguments);
  } catch {
    return fail("The call's arguments are not valid JSON.");
  }
  const value = (typeof args === "object" && args !== null ? (args as Record<string, unknown>)[field] : undefined);
  if (typeof value !== "string" || value.trim() === "") return fail(`The call needs a non-empty "${field}".`);
  const target = value.trim();
  if (tool === WEB_FETCH_TOOL_NAME && !/^https?:\/\/\S+$/i.test(target)) {
    return fail(`"${target}" is not an http(s) URL.`, target);
  }
  return { ok: true, task: { callId: call.id, tool, target } };
}

export type RunWebOptions = {
  /** The Tavily key; without one, searches go to search.tiago.zip, then DuckDuckGo Lite. */
  tavilyKey: string | null;
  signal: AbortSignal;
  onUpdate: (run: WebRun) => void;
  fetch?: typeof fetch;
  timeoutMs?: number;
};

async function failure(res: Response, service: string): Promise<string> {
  const text = (await res.text().catch(() => "")).trim();
  let detail = text.slice(0, 300);
  try {
    const body = JSON.parse(text) as { detail?: { error?: string } | string; message?: string; error?: string };
    detail = (typeof body.detail === "object" ? body.detail?.error : body.detail) ?? body.message ?? body.error ?? detail;
  } catch {
    // Not JSON: the text itself.
  }
  return `${service} answered ${res.status}${detail ? `: ${detail}` : ""}`;
}

async function search(query: string, key: string, signal: AbortSignal, doFetch: typeof fetch): Promise<SearchResult[]> {
  const res = await doFetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({ query, max_results: SEARCH_RESULTS, search_depth: "basic" }),
    signal,
  });
  if (!res.ok) throw new Error(await failure(res, "Tavily"));
  const body = (await res.json()) as { results?: Partial<SearchResult>[] };
  return (body.results ?? []).map((r) => ({ title: r.title ?? "", url: r.url ?? "", content: r.content ?? "" }));
}

async function read(url: string, signal: AbortSignal, doFetch: typeof fetch): Promise<string> {
  const res = await doFetch(`https://r.jina.ai/${url}`, { signal });
  if (!res.ok) throw new Error(await failure(res, "The Jina reader"));
  const text = await res.text();
  // The reader answers 200 for a page that failed, with a warning ahead of the error page.
  const warning = /^Warning: Target URL returned error (\d{3})(?::\s*(.*))?$/m.exec(text);
  if (warning) throw new Error(`The page answered ${warning[1]}${warning[2] ? `: ${warning[2].trim()}` : ""}.`);
  return text;
}

/** DuckDuckGo's own result links are redirects carrying the target in `uddg`. */
function unwrapRedirect(url: string): string {
  try {
    const u = new URL(url);
    if (u.hostname.endsWith("duckduckgo.com") && u.pathname === "/l/") return u.searchParams.get("uddg") ?? url;
  } catch {
    // Not a URL: keep it as it is.
  }
  return url;
}

/** The display URL closing a result, `docs.rs/axum/latest/` with at most a date after it. */
const DISPLAY_URL = /^[\w.-]+\.[a-z]{2,}(\/\S*)?(\s+\S+)?$/i;

/**
 * The results on DuckDuckGo Lite's page, as the Jina reader writes it:
 * blocks of `N.[title](link)`, the excerpt, then the display URL.
 */
export function parseDuckDuckGoLite(markdown: string): SearchResult[] {
  const marker = markdown.indexOf("Markdown Content:");
  const body = marker === -1 ? markdown : markdown.slice(marker);
  const results: SearchResult[] = [];
  for (const block of body.split(/\n\s*\n/)) {
    const lines = block.split("\n").map((l) => l.trim());
    const head = /^\d+\.\s*\[(.*)\]\((https?:\/\/[^\s)]+)\)$/.exec(lines.find((l) => /^\d+\./.test(l)) ?? "");
    if (!head) continue;
    const content = lines
      .slice(lines.indexOf(head[0]) + 1)
      .filter((l) => l !== "" && !DISPLAY_URL.test(l))
      .join(" ")
      .replace(/\*\*/g, "");
    results.push({ title: head[1].replace(/\*\*/g, ""), url: unwrapRedirect(head[2]), content });
  }
  return results;
}

/** An excerpt written as HTML (`<strong>`, `&#x27;`), as plain text. */
function plainText(html: string): string {
  return html
    .replace(/<[^>]+>/g, "")
    .replace(/&#x([0-9a-f]+);/gi, (_, hex: string) => String.fromCodePoint(parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec: string) => String.fromCodePoint(Number(dec)))
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
}

type TiagoResult = { title?: string; url?: string; description?: string };

async function searchTiago(query: string, signal: AbortSignal, doFetch: typeof fetch): Promise<SearchResult[]> {
  const res = await doFetch("https://search.tiago.zip/api", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, type: "web", page: 0 }),
    signal,
  });
  if (!res.ok) throw new Error(await failure(res, "search.tiago.zip"));
  const body = (await res.json()) as { results?: { web?: { results?: TiagoResult[] } } };
  const results = (body.results?.web?.results ?? [])
    .filter((r) => r.url)
    .slice(0, SEARCH_RESULTS)
    .map((r) => ({ title: plainText(r.title ?? ""), url: r.url!, content: plainText(r.description ?? "") }));
  if (results.length === 0) throw new Error("search.tiago.zip returned no results.");
  return results;
}

const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err));

/** A search without a key: search.tiago.zip, then DuckDuckGo Lite through the Jina reader when that fails. */
async function keylessSearch(query: string, signal: AbortSignal, doFetch: typeof fetch): Promise<Pick<WebRun, "via" | "results">> {
  try {
    return { via: "tiago", results: await searchTiago(query, signal, doFetch) };
  } catch (first) {
    if (signal.aborted) throw first;
    try {
      return { via: "duckduckgo", results: await searchDuckDuckGo(query, signal, doFetch) };
    } catch (second) {
      if (signal.aborted) throw second;
      throw new Error(`${errorText(first)} Then: ${errorText(second)}`);
    }
  }
}

async function searchDuckDuckGo(query: string, signal: AbortSignal, doFetch: typeof fetch): Promise<SearchResult[]> {
  const page = await read(`https://lite.duckduckgo.com/lite/?q=${encodeURIComponent(query)}`, signal, doFetch);
  const results = parseDuckDuckGoLite(page).slice(0, SEARCH_RESULTS);
  if (results.length === 0) {
    throw new Error("DuckDuckGo Lite (through the Jina reader) returned no results it could read; it may be blocking the reader.");
  }
  return results;
}

/** Runs every task at once and resolves with each run finished (done, failed or stopped). */
export async function runWeb(tasks: WebTask[], options: RunWebOptions): Promise<WebRun[]> {
  const doFetch = options.fetch ?? fetch.bind(globalThis);
  const { signal } = options;
  return Promise.all(
    tasks.map(async (task): Promise<WebRun> => {
      const finish = (change: Pick<WebRun, "status"> & Partial<WebRun>) => {
        const run: WebRun = { ...task, ...change };
        options.onUpdate(run);
        return run;
      };
      if (signal.aborted) return finish({ status: "stopped" });
      options.onUpdate({ ...task, status: "running" });
      const timeoutMs = options.timeoutMs ?? WEB_TIMEOUT_MS;
      const timeout = AbortSignal.timeout(timeoutMs);
      const callSignal = AbortSignal.any([signal, timeout]);
      try {
        if (task.tool === WEB_FETCH_TOOL_NAME) return finish({ status: "done", page: await read(task.target, callSignal, doFetch) });
        return options.tavilyKey
          ? finish({ status: "done", via: "tavily", results: await search(task.target, options.tavilyKey, callSignal, doFetch) })
          : finish({ status: "done", ...(await keylessSearch(task.target, callSignal, doFetch)) });
      } catch (err) {
        if (signal.aborted) return finish({ status: "stopped" });
        if (timeout.aborted) return finish({ status: "failed", error: `No answer within ${Math.round(timeoutMs / 1000)} s.` });
        return finish({ status: "failed", error: err instanceof Error ? err.message : String(err) });
      }
    }),
  );
}

/** The tool result the model receives for a finished run. */
export function webToolResult(run: WebRun): string {
  if (run.status === "failed") return `The ${run.tool} call failed: ${run.error ?? "unknown error"}`;
  if (run.status === "stopped") return `The ${run.tool} call was stopped before it finished.`;
  if (run.tool === WEB_FETCH_TOOL_NAME) {
    const page = (run.page ?? "").trim();
    if (!page) return "The page returned no text.";
    return page.length > FETCH_LIMIT
      ? `${page.slice(0, FETCH_LIMIT)}\n\n[Cut: the page has ${page.length} characters, these are the first ${FETCH_LIMIT}.]`
      : page;
  }
  const results = run.results ?? [];
  if (results.length === 0) return `No results for "${run.target}".`;
  return results.map((r, i) => `${i + 1}. ${r.title}\n${r.url}\n${r.content.trim()}`).join("\n\n");
}

/** `3 web calls: 1 running, 2 done`. */
export function webSummary(runs: WebRun[]): string {
  const parts = (["running", "done", "failed", "stopped"] as const)
    .map((status) => [status, runs.filter((r) => r.status === status).length] as const)
    .filter(([, n]) => n > 0)
    .map(([status, n]) => `${n} ${status}`);
  return `${runs.length} web ${runs.length === 1 ? "call" : "calls"}${parts.length ? `: ${parts.join(", ")}` : ""}`;
}
