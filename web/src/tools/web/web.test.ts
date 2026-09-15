import { describe, expect, it } from "vitest";
import {
  FETCH_LIMIT,
  parseDuckDuckGoLite,
  parseWebCall,
  runWeb,
  type WebRun,
  type WebTask,
  webSummary,
  webToolResult,
} from "./web.ts";

const call = (name: string, args: string) => ({ id: "c", name, arguments: args });

describe("parseWebCall", () => {
  it("reads a query and a URL", () => {
    expect(parseWebCall(call("web_search", '{"query":" rust axum "}'))).toEqual({
      ok: true,
      task: { callId: "c", tool: "web_search", target: "rust axum" },
    });
    expect(parseWebCall(call("web_fetch", '{"url":"https://example.com/a"}'))).toEqual({
      ok: true,
      task: { callId: "c", tool: "web_fetch", target: "https://example.com/a" },
    });
  });

  it("fails bad JSON, a missing field, a non-http URL and a tool not available, with a reason", () => {
    for (const [c, available, error] of [
      [call("web_search", "{oops"), undefined, /not valid JSON/],
      [call("web_search", '{"q":"x"}'), undefined, /non-empty "query"/],
      [call("web_fetch", '{"url":"file:///etc/passwd"}'), undefined, /not an http\(s\) URL/],
      [call("web_fetch", '{"url":"https://a.b"}'), ["web_search"], /Unknown tool "web_fetch": the tools available are "web_search"/],
      [call("read_file", "{}"), [], /Unknown tool "read_file": no tools are available/],
    ] as const) {
      const parsed = parseWebCall(c, available ? [...available] : undefined);
      expect(parsed.ok).toBe(false);
      if (!parsed.ok) {
        expect(parsed.run.status).toBe("failed");
        expect(parsed.run.error).toMatch(error);
      }
    }
  });
});

// DuckDuckGo Lite's results page as r.jina.ai returned it on 2026-09-15, cut to two results.
const DDG_PAGE = `Title: rust axum middleware at DuckDuckGo

URL Source: https://lite.duckduckgo.com/lite/?q=rust+axum+middleware

Markdown Content:
1.[axum::middleware - Rust - Docs.rs](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.rs%2Faxum%2Flatest%2Faxum%2Fmiddleware%2Findex.html&rut=e14dc4a6)
Utilities for writing **middleware** Intro **axum** is unique in that it doesn't have its own bespoke **middleware** system.
docs.rs/axum/latest/axum/middleware/index.html

2.[Getting Started with Axum - Rust's Fastest-Growing Web Framework](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.rustfinity.com%2Fblog%2Faxum%2Drust%2Dtutorial&rut=0667b34e)
**Axum** has become the most popular **Rust** web framework.
www.rustfinity.com/blog/axum-rust-tutorial 2026-03-31T00:00:00.0000000
`;

describe("parseDuckDuckGoLite", () => {
  it("reads title, the unwrapped link and the excerpt of each result", () => {
    expect(parseDuckDuckGoLite(DDG_PAGE)).toEqual([
      {
        title: "axum::middleware - Rust - Docs.rs",
        url: "https://docs.rs/axum/latest/axum/middleware/index.html",
        content: "Utilities for writing middleware Intro axum is unique in that it doesn't have its own bespoke middleware system.",
      },
      {
        title: "Getting Started with Axum - Rust's Fastest-Growing Web Framework",
        url: "https://www.rustfinity.com/blog/axum-rust-tutorial",
        content: "Axum has become the most popular Rust web framework.",
      },
    ]);
  });

  it("finds nothing on a page that is not a results page", () => {
    expect(parseDuckDuckGoLite("Title: Please verify you are human\n\nMarkdown Content:\nSorry.")).toEqual([]);
  });
});

type Seen = { url: string; init?: RequestInit };

function fakeFetch(answer: (url: string) => Response, seen: Seen[] = []): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    seen.push({ url, init });
    return answer(url);
  }) as typeof fetch;
}

const search: WebTask = { callId: "s", tool: "web_search", target: "ignis" };
const read: WebTask = { callId: "f", tool: "web_fetch", target: "https://example.com/" };

describe("runWeb", () => {
  it("searches Tavily with the key and reads pages through the Jina reader", async () => {
    const seen: Seen[] = [];
    const doFetch = fakeFetch(
      (url) =>
        url.includes("tavily")
          ? Response.json({ results: [{ title: "T", url: "https://t", content: "excerpt", score: 1 }] })
          : new Response("# Example"),
      seen,
    );
    const updates: WebRun[] = [];
    const runs = await runWeb([search, read], {
      tavilyKey: "tvly-k",
      signal: new AbortController().signal,
      onUpdate: (r) => updates.push(r),
      fetch: doFetch,
    });
    expect(runs.map((r) => r.status)).toEqual(["done", "done"]);
    expect(runs[0].results).toEqual([{ title: "T", url: "https://t", content: "excerpt" }]);
    expect(runs[1].page).toBe("# Example");

    const tavily = seen.find((s) => s.url === "https://api.tavily.com/search")!;
    expect((tavily.init?.headers as Record<string, string>).Authorization).toBe("Bearer tvly-k");
    expect(JSON.parse(String(tavily.init?.body))).toMatchObject({ query: "ignis" });
    expect(seen.some((s) => s.url === "https://r.jina.ai/https://example.com/")).toBe(true);
    expect(updates.filter((u) => u.status === "running")).toHaveLength(2);
  });

  it("searches search.tiago.zip without a key, as plain text", async () => {
    const seen: Seen[] = [];
    const runs = await runWeb([search], {
      tavilyKey: null,
      signal: new AbortController().signal,
      onUpdate: () => {},
      fetch: fakeFetch(
        () =>
          Response.json({
            results: {
              web: {
                results: [
                  {
                    title: "GitHub - tokio-rs/axum",
                    url: "https://github.com/tokio-rs/axum",
                    description: "<strong>axum doesn&#x27;t have</strong> its own &amp; more",
                  },
                ],
              },
            },
          }),
        seen,
      ),
    });
    expect(runs[0]).toMatchObject({
      status: "done",
      via: "tiago",
      results: [{ title: "GitHub - tokio-rs/axum", url: "https://github.com/tokio-rs/axum", content: "axum doesn't have its own & more" }],
    });
    expect(seen.map((s) => s.url)).toEqual(["https://search.tiago.zip/api"]);
    expect(JSON.parse(String(seen[0].init?.body))).toEqual({ query: "ignis", type: "web", page: 0 });
  });

  it("falls back to DuckDuckGo Lite through the Jina reader when search.tiago.zip fails", async () => {
    const seen: Seen[] = [];
    const runs = await runWeb([{ ...search, target: "rust axum" }, read], {
      tavilyKey: null,
      signal: new AbortController().signal,
      onUpdate: () => {},
      fetch: fakeFetch(
        (url) =>
          url.includes("tiago")
            ? Response.json({ error: "the upstream search failed" }, { status: 502 })
            : new Response(url.includes("duckduckgo") ? DDG_PAGE : "page"),
        seen,
      ),
    });
    expect(runs[0]).toMatchObject({ status: "done", via: "duckduckgo" });
    expect(runs[0].results?.[0].url).toBe("https://docs.rs/axum/latest/axum/middleware/index.html");
    expect(runs[1]).toMatchObject({ status: "done", page: "page" });
    expect(seen.map((s) => s.url).sort()).toEqual([
      "https://r.jina.ai/https://example.com/",
      "https://r.jina.ai/https://lite.duckduckgo.com/lite/?q=rust%20axum",
      "https://search.tiago.zip/api",
    ]);
  });

  it("fails a keyless search when both services fail, with both reasons", async () => {
    const runs = await runWeb([search], {
      tavilyKey: null,
      signal: new AbortController().signal,
      onUpdate: () => {},
      fetch: fakeFetch((url) =>
        url.includes("tiago")
          ? Response.json({ error: "the upstream search failed" }, { status: 502 })
          : new Response("Title: Please verify you are human"),
      ),
    });
    expect(runs[0].status).toBe("failed");
    expect(runs[0].error).toMatch(/^search\.tiago\.zip answered 502: the upstream search failed Then: .*no results it could read/);
  });

  it("fails with the service's own error message", async () => {
    const runs = await runWeb([search], {
      tavilyKey: "bad",
      signal: new AbortController().signal,
      onUpdate: () => {},
      fetch: fakeFetch(() => Response.json({ detail: { error: "Unauthorized: missing or invalid API key." } }, { status: 401 })),
    });
    expect(runs[0].error).toBe("Tavily answered 401: Unauthorized: missing or invalid API key.");
  });

  it("fails a call with no answer in time, and a page the reader says failed", async () => {
    const hanging = ((_input: RequestInfo | URL, init?: RequestInit) =>
      new Promise((_resolve, reject) => init?.signal?.addEventListener("abort", () => reject(init.signal!.reason)))) as typeof fetch;
    const [late] = await runWeb([read], { tavilyKey: null, signal: new AbortController().signal, onUpdate: () => {}, fetch: hanging, timeoutMs: 20 });
    expect(late).toMatchObject({ status: "failed", error: "No answer within 0 s." });

    const page = "Title: \n\nURL Source: https://httpstat.us/502\n\nWarning: Target URL returned error 502: Bad Gateway\n\nMarkdown Content:\n502 Bad Gateway";
    const [bad] = await runWeb([read], {
      tavilyKey: null,
      signal: new AbortController().signal,
      onUpdate: () => {},
      fetch: fakeFetch(() => new Response(page)),
    });
    expect(bad).toMatchObject({ status: "failed", error: "The page answered 502: Bad Gateway." });
  });

  it("stops calls when the turn is stopped", async () => {
    const abort = new AbortController();
    abort.abort();
    const runs = await runWeb([search, read], {
      tavilyKey: "k",
      signal: abort.signal,
      onUpdate: () => {},
      fetch: fakeFetch(() => new Response("never")),
    });
    expect(runs.map((r) => r.status)).toEqual(["stopped", "stopped"]);
  });
});

describe("webToolResult and webSummary", () => {
  it("numbers search results and says when there are none", () => {
    const run: WebRun = { ...search, status: "done", results: [{ title: "A", url: "https://a", content: " one " }] };
    expect(webToolResult(run)).toBe("1. A\nhttps://a\none");
    expect(webToolResult({ ...run, results: [] })).toBe('No results for "ignis".');
  });

  it("cuts a long page and says so", () => {
    const page = "x".repeat(FETCH_LIMIT + 10);
    const result = webToolResult({ ...read, status: "done", page });
    expect(result.startsWith("x".repeat(FETCH_LIMIT) + "\n\n[Cut:")).toBe(true);
    expect(webToolResult({ ...read, status: "done", page: "short" })).toBe("short");
  });

  it("reports failed and stopped calls, and counts them", () => {
    const failed: WebRun = { ...search, status: "failed", error: "boom" };
    expect(webToolResult(failed)).toBe("The web_search call failed: boom");
    expect(webToolResult({ ...read, status: "stopped" })).toMatch(/stopped before it finished/);
    expect(webSummary([failed, { ...read, status: "done", page: "" }])).toBe("2 web calls: 1 done, 1 failed");
  });
});
