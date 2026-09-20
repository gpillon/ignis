import type { IncomingMessage } from "node:http";
import type { Plugin } from "vite";
import { mockDecide } from "./mockDecide.ts";
import { createMetricsSim } from "./mockMetrics.ts";

// A fake ignis for `npm run dev:mock`: enough of /v1/models, streaming
// /v1/chat/completions and /ui/metrics to work on the Playground without the
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
// Decide: /v1/decide answers every primitive from a hash of the question
// (mockDecide.ts); "/error" in a question's instructions fails that question
// alone, "/full" in the evidence refuses the whole request.
//
// Ask: with `ask_user` declared, a prompt containing "/ask" asks which team.
// Local: "/js", "/plan", "/file" and "/html" call run_js, update_plan and create_file.
// "/math" answers with formulas; "/long" streams a long reply for scrolling.

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
        const offers = (name: string) => tools.some((t) => t.function?.name === name);
        const offersLocal = offers("run_js") || offers("update_plan") || offers("create_file");
        const lastRole = messages.at(-1)?.role;
        let content = ["Hello ", "from ", "the ", "mock ", "engine. ", "You ", "said: ", last];
        let calls: { tool: string; args: object }[] = [];
        let pace = 120;
        if (last.includes("/long") && lastRole === "user") {
          // Long reasoning and a long answer, streamed fast: for checking how the transcript scrolls.
          const sentence = (n: number) => `This is sentence ${n} of a long stream, written to fill the transcript and wrap across several lines. `;
          reasoning.splice(0, reasoning.length, ...(thinkingOff ? [] : Array.from({ length: 60 }, (_, i) => sentence(i + 1))));
          content = Array.from({ length: 120 }, (_, i) => (i % 12 === 11 ? `${sentence(i + 1)}\n\n` : sentence(i + 1)));
          pace = 25;
        } else if (body.class === "agent" && offersWeb && lastRole === "user" && last.includes("/web")) {
          // An agent with web tools whose task mentions "/web" searches before it reports.
          content = ["Searching ", "first."];
          calls = [{ tool: "web_search", args: { query: last.replace(/^.*\/web/s, "").trim() || "ignis inference engine" } }];
        } else if (body.class === "agent") {
          const words = `Report for "${last.slice(0, 60)}". The mock agent looked at the task, checked three things and found the answer. Everything it needs is in the prompt, so the result is short and ready to merge.`;
          content = words.split(/(?<= )/);
          pace = 70 + Math.floor(Math.random() * 120);
        } else if ((offersAgents || offersWeb || offersAsk || offersLocal) && lastRole === "tool") {
          const results = messages.filter((m) => m.role === "tool").map((m) => `- ${m.content.slice(0, 80).replace(/\s+/g, " ")}`);
          content = ["The ", "tools ", "reported ", "back:\n\n", results.join("\n")];
        } else if (offersAgents && last.includes("/agents")) {
          // "/agents" splits the task three ways; "/agents 6" asks for six, which
          // is how the browser's connection limit is reproduced without a GPU
          // (GitHub #220).
          const asked = Number(/\/agents\s+(\d+)/.exec(last)?.[1]);
          const names = ["scheduler", "kv-cache", "telemetry", "sampler", "tokenizer", "router", "allocator", "logger"];
          const wanted = Number.isFinite(asked) ? Math.min(Math.max(asked, 1), names.length) : 3;
          content = ["I'll ", "split ", "this ", "into ", `${wanted} `, "parts."];
          calls = names.slice(0, wanted).map((name) => ({
            tool: "agent",
            args: { name, prompt: `Look at the ${name} part of: ${last.replace(/\/agents(\s+\d+)?/, "").trim()}` },
          }));
        } else if (offers("run_js") && last.includes("/js")) {
          content = ["Let ", "me ", "compute ", "it."];
          calls = [{ tool: "run_js", args: { code: "const squares = [1, 2, 3, 4].map((x) => x * x);\nconsole.log('squares', squares);\nreturn squares.reduce((a, b) => a + b);" } }];
        } else if (offers("update_plan") && last.includes("/plan")) {
          content = ["Here ", "is ", "the ", "plan."];
          calls = [
            {
              tool: "update_plan",
              args: { steps: [{ step: "Read the question", status: "done" }, { step: "Look things up", status: "in_progress" }, { step: "Write the answer", status: "pending" }] },
            },
          ];
        } else if (last.includes("/math") && lastRole === "user") {
          content = [
            "Hydrostatic pressure grows with depth:\n\n",
            "$$P = \\rho \\cdot g \\cdot h$$\n\n",
            "where $\\rho$ is the density, \\(g\\) the gravity and $h$ the depth. In display form:\n\n",
            "\\[\n\\int_0^h \\rho g \\, dz = \\rho g h\n\\]",
          ];
        } else if (offers("create_file") && last.includes("/html")) {
          content = ["Here ", "is ", "the ", "page."];
          calls = [
            {
              tool: "create_file",
              args: {
                name: "hello.html",
                content:
                  '<!doctype html>\n<html>\n<body style="font-family: sans-serif; padding: 24px">\n  <h1>Hello from ignis</h1>\n  <button onclick="this.textContent = \'Clicked\'">Click me</button>\n</body>\n</html>\n',
              },
            },
          ];
        } else if (offers("create_file") && last.includes("/file")) {
          content = ["I ", "wrote ", "it ", "to ", "a ", "file."];
          calls = [{ tool: "create_file", args: { name: "notes.md", content: "# Notes\n\n- one\n- two\n" } }];
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

      server.middlewares.use("/v1/decide", async (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end();
          return;
        }
        let raw = "";
        req.on("data", (part) => (raw += part));
        await new Promise((resolve) => req.on("end", resolve));
        const { status, body } = mockDecide(raw);
        res.statusCode = status;
        res.setHeader("Content-Type", "application/json");
        // One prefill, and the answers land together: the wait is the prefill,
        // not a stream.
        setTimeout(() => res.end(JSON.stringify(body)), 260);
      });

      // A live simulation (mockMetrics.ts), so the Monitor has traffic to draw.
      const metrics = createMetricsSim();
      server.middlewares.use("/ui/metrics", (_req, res) => {
        res.setHeader("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
        res.end(metrics.render());
      });
    },
  };
}
