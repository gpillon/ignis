import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { mockIgnis } from "./mock.ts";

// The Playground (GitHub #163, ADR 0026). `ignis-server --ui` serves the
// build under /ui/; in development, /v1 and /ui/metrics go to a running ignis
// (IGNIS_URL, from the shell or web/.env), or to the in-process mock with
// `npm run dev:mock`.
export default defineConfig(({ mode }) => {
  const target = loadEnv(mode, process.cwd(), "").IGNIS_URL || "http://127.0.0.1:8000";
  return {
    base: "/ui/",
    plugins: [react(), tailwindcss(), ...(mode === "mock" ? [mockIgnis()] : [])],
    server:
      mode === "mock"
        ? {}
        : {
            proxy: {
              "/v1": { target, changeOrigin: true },
              // The Playground's copy of the Prometheus exposition (ADR 0017).
              "/ui/metrics": { target, changeOrigin: true },
            },
          },
  };
});
