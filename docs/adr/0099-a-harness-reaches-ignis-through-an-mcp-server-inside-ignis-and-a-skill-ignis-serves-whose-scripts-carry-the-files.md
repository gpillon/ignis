---
status: accepted
---

# A harness reaches ignis through an MCP server inside ignis and a skill ignis serves, whose scripts carry the files

A coding agent's harness (opencode, qwen-code, pi, DeepSeek Harness) reaches ignis beyond chat through two things
ignis serves itself: an MCP server for the tools that need no file, and a skill whose scripts carry the files.

**The MCP server.** ignis serves the connector's server tools as an MCP server over Streamable HTTP, built on `rmcp`,
mounted at `/v1/mcp` behind the same key as the rest of `/v1` (ADR 0071), with rmcp's Host check turned off, as no
other route has one, and its Origin list left empty, its default.

- **Stateless.** Every request stands alone: no session, so an ignis restart strands no harness. rmcp's stateful
  mode was not taken: a session id dies with the process, and the harnesses recover from that differently or not at
  all. There is no server-to-client stream; stateless mode still carries a call's progress, in the answer to its own
  request.
- **A tool calls its route through the router, in process,** with the caller's `Authorization`. The call meets every
  gate a client's request meets: the key, the serving gate, the count of model use (ADR 0049), the body cap. A
  refusal is the route's own status, error JSON and `Retry-After`. A tool calling its handler directly would pass
  those gates by and rebuild each route's refusals by hand.
- **`/v1/mcp` itself is never held by the serving gate.** `initialize` and `tools/list` answer during a switch or a
  warm-up, and the connector can hold a call of its own (the image calls, one at a time) instead of meeting a `503`
  at the door; the tool's route is held, or refused, as it is for anyone.
- **A tool is declared where its endpoint is,** by an extension on the endpoint's `#[utoipa::path]`, and a key the
  extension does not know fails the table's build. The tool list is built once at start from the OpenAPI document
  (ADR 0036), sorted by name, so it is byte-identical whatever model is loaded, and a harness's prompt reuse
  survives it.

**The skill.** The tools that need a file (generating an image to a file, editing an image, transcribing audio) live
in a skill, `ignis`, whose Node scripts run in the harness's shell, beside the files, and call the existing
endpoints. ignis serves the skill from its own build, as it serves the Playground (ADR 0026), so the skill always
matches the server that gave it.

**Why files go through a skill.** MCP has no channel for a client to upload a file: the model writes a tool's
arguments, so a file's bytes would pass through the model, and a server reading the harness's paths works only on
the same machine. With the scripts, the server touches no file and ADR 0073's "ignis stores no files" holds.

## Considered Options

- **MCP roots, the server reading the harness's paths.** Rejected: it works only when server and harness share a
  disk, and DeepSeek Harness sends no roots.
- **The tool answers with a one-time upload URL and a curl recipe.** Rejected by the owner.
- **An upload route, or a file store.** Rejected: ADR 0073.
- **A native plugin per harness.** Rejected: four extension APIs to keep for what one MCP server and one skill,
  which all four read, already give.
- **A separate local connector process.** Rejected: one more binary to install on every client machine.
- **The skill as an npm package.** Rejected: an account and a supply chain to keep, and a version that drifts from
  the server's.
- **No skill.** Chosen first, in v1, and reversed once the remote case showed that the files can only be handled on
  the harness's side.

## Consequences

- **The tool list depends on the start configuration alone.** The tools that change the server are listed only when
  `connector.allow_changes` is on, and it is off by default; `connector.enabled` is on by default and, off, the MCP
  server and the skill's route answer 404. Both are config fields declared once like any other (ADR 0046), but they
  take effect at start only and cannot be patched.
- **`allow_changes` is not a security boundary:** an agent with a shell and the one key (ADR 0071) can patch the
  configuration itself.
- **`generate_image` returns a small preview, and only when the text model it returns to has vision on.** Without
  vision, an image left in the harness's history is refused `vision_disabled` on every later turn, so the tool
  answers without switching, once the image calls before it have ended, and names the skill's script. With vision,
  the images of a history that pass `vision.max_tokens` together are no longer refused on every later turn: the
  oldest are dropped (ADR 0102).
- **The connector's image calls run one at a time,** so several in one turn cost one switch out and one back.
- **No tool switches back.** The image model never runs beside a text model (ADR 0050); the next chat turn's
  implicit switch brings the text model back. The prompt state is lost on the way, until #655 keeps it across a
  switch.
- **`decide` answers on the 27B only.**
- **Transcription is the skill's alone:** the speech model runs beside the text model (ADR 0067), but its input is
  a file.
- **Results go as text in `content`.** An error is the server's error JSON, with `isError` set, then
  `Retry-After: <seconds>` as a second text when the route sent the header: DeepSeek Harness never shows
  `structuredContent` and drops embedded resources. The connector never retries.
- **Left open:** a keyless server on localhost stays open to DNS rebinding, on every route, not just this one.
  Closing it is a server-wide decision, not the connector's.
