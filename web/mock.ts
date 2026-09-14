import type { Plugin } from "vite";

// A fake ignis for `npm run dev:mock`: enough of /v1/models, streaming
// /v1/chat/completions and /metrics to work on the Playground without the
// shared GPU. Development only — never part of the build.
export function mockIgnis(): Plugin {
  return {
    name: "ignis-mock",
    configureServer(server) {
      server.middlewares.use("/v1/models", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify({ object: "list", data: [{ id: "mock-model", object: "model", owned_by: "ignis" }] }));
      });

      server.middlewares.use("/v1/chat/completions", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end();
          return;
        }
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        const id = `chatcmpl-mock-${Date.now()}`;
        const chunk = (delta: object, finish: string | null = null) =>
          `data: ${JSON.stringify({ id, object: "chat.completion.chunk", model: "mock-model", choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
        const pieces = [
          ...["Thinking ", "about ", "it."].map((t) => chunk({ reasoning_content: t })),
          ...["Hello ", "from ", "the ", "mock ", "engine."].map((t) => chunk({ content: t })),
          chunk({}, "stop"),
          `data: ${JSON.stringify({ id, object: "chat.completion.chunk", model: "mock-model", choices: [], usage: { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 } })}\n\n`,
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
        }, 60);
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
