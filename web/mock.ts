import type { IncomingMessage } from "node:http";
import type { Plugin } from "vite";

// A fake ignis for `npm run dev:mock`: enough of /v1/models, streaming
// /v1/chat/completions and /metrics to work on the Playground without the
// shared GPU. Development only — never part of the build.
//
// Chat: honours the thinking controls (no reasoning for `reasoning_effort`
// "none" or `enable_thinking` false) and stops on a
// client disconnect; a last user message containing "/error" gets ignis's
// 503 "engine full" error instead of a stream.

function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    let raw = "";
    req.on("data", (part) => (raw += part));
    req.on("end", () => {
      try {
        resolve(JSON.parse(raw) as Record<string, unknown>);
      } catch {
        resolve({});
      }
    });
  });
}

export function mockIgnis(): Plugin {
  return {
    name: "ignis-mock",
    configureServer(server) {
      server.middlewares.use("/v1/models", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ object: "list", data: [{ id: "mock-model", object: "model", owned_by: "ignis" }] }));
      });

      server.middlewares.use("/v1/chat/completions", async (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end();
          return;
        }
        const body = await readJson(req);
        const messages = (body.messages as { role: string; content: string }[] | undefined) ?? [];
        const last = messages.filter((m) => m.role === "user").at(-1)?.content ?? "";
        if (last.includes("/error")) {
          res.statusCode = 503;
          res.setHeader("Content-Type", "application/json");
          res.end(
            JSON.stringify({
              error: { message: "the engine cannot admit the request right now (all lanes in use); retry", type: "server_error", code: "engine_full" },
            }),
          );
          return;
        }

        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        const id = `chatcmpl-mock-${Date.now()}`;
        const chunk = (delta: object, finish: string | null = null) =>
          `data: ${JSON.stringify({ id, object: "chat.completion.chunk", model: "mock-model", choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
        const thinkingOff = body.reasoning_effort === "none" || body.enable_thinking === false;
        const reasoning = thinkingOff ? [] : ["Thinking ", "about ", "it."];
        const content = ["Hello ", "from ", "the ", "mock ", "engine. ", "You ", "said: ", last];
        const pieces = [
          ...reasoning.map((t) => chunk({ reasoning_content: t })),
          ...content.map((t) => chunk({ content: t })),
          chunk({}, "stop"),
          `data: ${JSON.stringify({ id, object: "chat.completion.chunk", model: "mock-model", choices: [], usage: { prompt_tokens: 12 * messages.length, completion_tokens: reasoning.length + content.length, total_tokens: 12 * messages.length + reasoning.length + content.length } })}\n\n`,
          "data: [DONE]\n\n",
        ];
        let i = 0;
        const timer = setInterval(() => {
          if (i === pieces.length) {
            clearInterval(timer);
            res.end();
            return;
          }
          res.write(pieces[i++]);
        }, 120);
        req.on("close", () => clearInterval(timer));
      });

      server.middlewares.use("/metrics", (_req, res) => {
        res.setHeader("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
        res.end(
          [
            "# HELP ignis_build_info Constant build identity",
            "# TYPE ignis_build_info gauge",
            'ignis_build_info{version="0.1.0-mock"} 1',
            "# HELP ignis_scheduler_requests Current requests by scheduler state",
            "# TYPE ignis_scheduler_requests gauge",
            'ignis_scheduler_requests{state="waiting"} 0',
            'ignis_scheduler_requests{state="running"} 1',
            "# HELP ignis_generated_tokens_total Generated tokens on completed requests",
            "# TYPE ignis_generated_tokens_total counter",
            `ignis_generated_tokens_total ${Math.floor(Date.now() / 100) % 100000}`,
            "",
          ].join("\n"),
        );
      });
    },
  };
}
