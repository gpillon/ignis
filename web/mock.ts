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
//
// Agents: with the `agent` tool declared, a prompt containing "/agents"
// streams three agent calls (finish "tool_calls"); a request whose last
// message is a tool result answers from those results; an agent-lane
// request streams a longer report at its own pace.
//
// Web: with `web_search` declared, a prompt containing "/web" streams one
// search and one page read; an agent-lane request with the web tools whose
// task contains "/web" searches once before its report ("/agents /web …").
// The calls themselves run in the browser against the real services.
//
// Ask: with `ask_user` declared, a prompt containing "/ask" asks which team.

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
        // `max_model_len` (vLLM's name for the context length) feeds the
        // Playground's context bar, as ignis reports it.
        res.end(
          JSON.stringify({
            object: "list",
            data: [{ id: "mock-model", object: "model", owned_by: "ignis", max_model_len: 40960 }],
          }),
        );
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
        const tools = (body.tools as { function?: { name?: string } }[] | undefined) ?? [];
        const offersAgents = tools.some((t) => t.function?.name === "agent");
        const offersWeb = tools.some((t) => t.function?.name === "web_search");
        const offersAsk = tools.some((t) => t.function?.name === "ask_user");
        const lastRole = messages.at(-1)?.role;
        let content = ["Hello ", "from ", "the ", "mock ", "engine. ", "You ", "said: ", last];
        let calls: { tool: string; args: object }[] = [];
        let pace = 120;
        if (body.class === "agent" && offersWeb && lastRole === "user" && last.includes("/web")) {
          // An agent with web tools whose task mentions "/web" searches before it reports.
          content = ["Searching ", "first."];
          calls = [{ tool: "web_search", args: { query: last.replace(/^.*\/web/s, "").trim() || "ignis inference engine" } }];
        } else if (body.class === "agent") {
          const words = `Report for "${last.slice(0, 60)}". The mock agent looked at the task, checked three things and found the answer. Everything it needs is in the prompt, so the result is short and ready to merge.`;
          content = words.split(/(?<= )/);
          pace = 70 + Math.floor(Math.random() * 120);
        } else if ((offersAgents || offersWeb || offersAsk) && lastRole === "tool") {
          const results = messages.filter((m) => m.role === "tool").map((m) => `- ${m.content.slice(0, 80).replace(/\s+/g, " ")}`);
          content = ["The ", "tools ", "reported ", "back:\n\n", results.join("\n")];
        } else if (offersAgents && last.includes("/agents")) {
          content = ["I'll ", "split ", "this ", "into ", "three ", "parts."];
          calls = ["scheduler", "kv-cache", "telemetry"].map((name) => ({
            tool: "agent",
            args: { name, prompt: `Look at the ${name} part of: ${last.replace("/agents", "").trim()}` },
          }));
        } else if (offersAsk && last.includes("/ask")) {
          content = ["I ", "need ", "one ", "detail ", "first."];
          calls = [{ tool: "ask_user", args: { question: "Which team do you mean?", options: ["AS Roma", "SS Lazio"] } }];
        } else if (offersWeb && last.includes("/web")) {
          content = ["Let ", "me ", "look ", "that ", "up."];
          calls = [
            { tool: "web_search", args: { query: last.replace("/web", "").trim() || "ignis inference engine" } },
            { tool: "web_fetch", args: { url: "https://example.com/" } },
          ];
        }
        const completion = reasoning.length + content.length + calls.length;
        const pieces = [
          ...reasoning.map((t) => chunk({ reasoning_content: t })),
          ...content.map((t) => chunk({ content: t })),
          ...calls.map((c, index) =>
            chunk({ tool_calls: [{ index, id: `call_${Date.now()}_${index}`, type: "function", function: { name: c.tool, arguments: JSON.stringify(c.args) } }] }),
          ),
          chunk({}, calls.length ? "tool_calls" : "stop"),
          `data: ${JSON.stringify({ id, object: "chat.completion.chunk", model: "mock-model", choices: [], usage: { prompt_tokens: 12 * messages.length, completion_tokens: completion, total_tokens: 12 * messages.length + completion } })}\n\n`,
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
        }, pace);
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
