import { useEffect, useState } from "react";

type Model = { state: "loading" } | { state: "ready"; id: string } | { state: "error"; message: string };

// The Playground shell (GitHub #163): proves the page is served and can
// reach the engine. The chat itself is GitHub #164.
export function App() {
  const [model, setModel] = useState<Model>({ state: "loading" });

  useEffect(() => {
    fetch("/v1/models")
      .then(async (res) => {
        if (!res.ok) throw new Error(`GET /v1/models: ${res.status}`);
        const body = (await res.json()) as { data?: { id: string }[] };
        const id = body.data?.[0]?.id;
        if (!id) throw new Error("GET /v1/models: no model listed");
        setModel({ state: "ready", id });
      })
      .catch((err: unknown) => setModel({ state: "error", message: String(err) }));
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: "40rem", margin: "3rem auto", padding: "0 1rem" }}>
      <h1>Playground</h1>
      {model.state === "loading" && <p>Loading model…</p>}
      {model.state === "ready" && (
        <p>
          Model: <code>{model.id}</code>
        </p>
      )}
      {model.state === "error" && <p role="alert">{model.message}</p>}
    </main>
  );
}
